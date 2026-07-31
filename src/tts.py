"""Kokoro text-to-speech wrapper (mlx-audio, Metal-accelerated).

Synthesis only -- playback (and barge-in interruption) is handled by the agent
so it can stop audio mid-sentence.

Besides audio, synthesis also returns a viseme timeline for lip-sync. Kokoro's
duration predictor already emits one frame count per phoneme as part of the
normal forward pass, so the timeline is free: we read the pipeline's
`pred_dur` instead of calling the model a second time or analysing the audio.
"""
from typing import List, Optional, Tuple

import re

import numpy as np

import config
import visemes

# (viseme, start_seconds, end_seconds)
Timeline = List[Tuple[str, float, float]]

# Where a too-long piece of text may be broken, best boundary first: a clause end
# reads better than a comma, and a comma better than a bare word break. Both
# ASCII and CJK/full-width punctuation, since Chinese replies contain neither
# spaces nor ASCII terminators.
_CHUNK_BOUNDARIES = (
    re.compile(r"[.!?\u3002\uff01\uff1f\u2026]+[\s\"')\]\u201d\u2019\u300d\u300f\uff09]*"),
    re.compile(r"[,;:\uff0c\u3001\uff1b\uff1a]+[\s\u201d\u2019\u300d\u300f\uff09]*"),
    re.compile(r"\s+"),
)


# Markdown decoration that Kokoro pronounces instead of ignoring: misaki
# phonemizes "*" as the word "asterisk", so an emphasised *boss* is spoken as
# "asterisk boss asterisk" (verified against the pipeline's own phoneme output).
# The system prompt tells the LLM to avoid these; stripping them here is what
# guarantees they are never heard.
_UNSPEAKABLE = re.compile(r"[*_`#]+|~~")


def speakable(text: str) -> str:
    """Drop markdown characters that TTS would read out as words."""
    return re.sub(r"[ \t]{2,}", " ", _UNSPEAKABLE.sub("", text)).strip()


def max_chunk_chars(lang_code: str) -> int:
    """Character budget for one Kokoro forward pass in `lang_code`."""
    return config.TTS_MAX_CHUNK_CHARS.get(lang_code,
                                          config.TTS_MAX_CHUNK_CHARS_DEFAULT)


def chunk_text(text: str, lang_code: str) -> List[str]:
    """Split `text` into pieces short enough for Kokoro to speak in full.

    Kokoro accepts at most `config.TTS_MAX_PHONEMES` phonemes per forward pass
    and mlx-audio truncates anything longer, which cuts the audio off mid-word.
    Its own chunker only splits on ASCII `.!?` at 400 characters, so a Chinese
    reply -- ~4 phonemes per character, and terminated by U+3002 rather than a
    period -- overshoots and loses its tail. Splitting here instead keeps whole
    clauses together and never hands the pipeline more than it can say.
    """
    limit = max_chunk_chars(lang_code)
    rest = text.strip()
    chunks: List[str] = []
    while len(rest) > limit:
        window = rest[:limit]
        cut = 0
        for pattern in _CHUNK_BOUNDARIES:
            ends = [m.end() for m in pattern.finditer(window)]
            if ends:
                cut = ends[-1]
                break
        if cut == 0:
            cut = limit  # no boundary at all: an unbroken run of characters
        piece = rest[:cut].strip()
        if piece:
            chunks.append(piece)
        rest = rest[cut:].lstrip()
    if rest:
        chunks.append(rest)
    return chunks


class KokoroTTS:
    def __init__(self, model_name: str = config.TTS_MODEL,
                 voice: str = config.TTS_VOICE,
                 speed: float = config.TTS_SPEED):
        # Imported lazily so importing this module is cheap.
        from mlx_audio.tts.utils import load_model
        print(f"[TTS] loading {model_name} (voice={voice}) ...")
        self.model = load_model(model_name)
        self.voice = voice
        self.speed = speed
        self.sample_rate = config.TTS_SAMPLE_RATE
        self.lang_code = config.TTS_LANG_CODE
        # Warm up so the first real response is not slow.
        try:
            self.synthesize("Ready.")
        except Exception as e:  # pragma: no cover
            print(f"[TTS] warmup skipped: {e}")

    # -- internals -------------------------------------------------------
    def _pipeline(self, lang_code: Optional[str] = None):
        """Kokoro's language pipeline (cached by the model, includes G2P)."""
        code = lang_code or self.lang_code
        return self.model._get_pipeline(code)

    def _voice_for_lang(self, lang: Optional[str]) -> str:
        """Return the appropriate voice for a Whisper language code.

        If `lang` maps to a different language than the default, use that
        language's configured voice. Otherwise use the instance default.
        """
        if lang is None:
            return self.voice
        lang_info = config.LANGUAGE_MAP.get(lang)
        if lang_info is None:
            return self.voice
        return lang_info["voice"]

    def _kokoro_lang_for(self, lang: Optional[str]) -> str:
        """Return the Kokoro pipeline code for a Whisper language code."""
        if lang is None:
            return self.lang_code
        lang_info = config.LANGUAGE_MAP.get(lang)
        if lang_info is None:
            return self.lang_code
        return lang_info["kokoro_lang"]

    # -- public API ------------------------------------------------------
    def synthesize(self, text: str, lang: Optional[str] = None) -> np.ndarray:
        """Return a mono float32 waveform at self.sample_rate for `text`."""
        audio, _ = self.synthesize_with_visemes(text, lang=lang)
        return audio

    def synthesize_with_visemes(self, text: str, lang: Optional[str] = None) -> Tuple[np.ndarray, Timeline]:
        """Return (waveform, viseme timeline) for `text`.

        `lang` is a Whisper language code (e.g. 'en', 'es', 'zh') used to
        select the correct Kokoro pipeline and voice. If None, uses the
        default configured language.

        Uses the pipeline directly rather than model.generate() because the
        pipeline's Result carries the phonemes and per-phoneme frame durations
        alongside the audio -- all from the one forward pass that generate()
        would have made anyway.
        """
        text = speakable(text)
        if not text:
            return np.zeros(0, dtype=np.float32), []

        kokoro_lang = self._kokoro_lang_for(lang)
        voice = self._voice_for_lang(lang)
        pipeline = self._pipeline(kokoro_lang)
        vocab = self.model.vocab

        chunks: List[np.ndarray] = []
        timeline: Timeline = []
        elapsed = 0.0  # seconds of audio emitted so far, to offset each chunk

        try:
            for segment in chunk_text(text, kokoro_lang):
                for result in pipeline(segment, voice=voice, speed=self.speed):
                    if result.audio is None:
                        continue
                    chunk = np.asarray(result.audio, dtype=np.float32).reshape(-1)
                    if chunk.size == 0:
                        continue
                    chunks.append(chunk)
                    if result.pred_dur is not None:
                        timeline.extend(
                            visemes.build_timeline(
                                result.phonemes, [int(d) for d in result.pred_dur],
                                vocab, offset_s=elapsed,
                            )
                        )
                    elapsed += chunk.size / self.sample_rate
        except Exception as e:
            # Never let lip-sync break speech: fall back to plain synthesis.
            print(f"[TTS] viseme path failed ({e}); falling back to plain synthesis")
            return self._synthesize_plain(text, kokoro_lang, voice), []

        if not chunks:
            return np.zeros(0, dtype=np.float32), []
        return np.concatenate(chunks), timeline

    def _synthesize_plain(self, text: str, lang_code: Optional[str] = None,
                          voice: Optional[str] = None) -> np.ndarray:
        """Original code path, used if the viseme-aware one ever fails."""
        v = voice or self.voice
        segments = self.model.generate(text=text, voice=v, speed=self.speed,
                                       lang_code=lang_code or self.lang_code)
        chunks = [np.asarray(s.audio, dtype=np.float32).reshape(-1) for s in segments]
        if not chunks:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(chunks)
