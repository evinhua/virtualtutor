"""Kokoro text-to-speech wrapper (mlx-audio, Metal-accelerated).

Synthesis only -- playback (and barge-in interruption) is handled by the agent
so it can stop audio mid-sentence.

Besides audio, synthesis also returns a viseme timeline for lip-sync. Kokoro's
duration predictor already emits one frame count per phoneme as part of the
normal forward pass, so the timeline is free: we read the pipeline's
`pred_dur` instead of calling the model a second time or analysing the audio.

Emotion. Kokoro takes no emotion input, so a mood is built from what can be
controlled: speaking rate (its own `speed`), pitch (resampling, with the rate
compensated so *only* pitch moves), loudness, the pauses around the sentence,
and how far the mouth travels toward each viseme. src/emotion.py decides the
mood; this module applies it.

The pitch shift is worth spelling out, because it is the one place where audio
and lip-sync could drift apart. Resampling by `r` multiplies pitch by `r` and
divides duration by `r`, which would speak the sentence faster as a side effect.
Synthesizing at `speed / r` cancels that: the duration ends up where the mood
asked for it, and only the pitch has moved. Kokoro's predicted durations are
computed at the synthesis speed, so the timeline is rescaled by the same `1/r`
and stays aligned to the millisecond.
"""
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import re

import numpy as np

import config
import emotion as emotion_mod
import visemes

# (viseme, start_seconds, end_seconds, level)
Timeline = List[Tuple]

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

# A bullet or numbered-list marker at the start of a line. The tutor is told not
# to write lists, but a bigger instruct model writes them anyway when it thinks
# it is being helpful, and "- " is read aloud as "dash".
_LIST_MARKER = re.compile(r"^[ \t]*(?:[-\u2022\u2013\u2014>]+|\d+[.)])[ \t]+", re.MULTILINE)

# A phonetic transcription between slashes, e.g. /ˈɡɾaθjas/. Read aloud these
# become the *names* of the symbols, which is worse than useless -- the whole
# point of "describe pronunciation as spoken syllables" in the system prompt.
# Only dropped when the slashes really do contain phonetic symbols.
_IPA_SPAN = re.compile(r"/[^/\n]{1,40}/")
_IPA_CHARS = re.compile(r"[\u0250-\u02af\u02b0-\u02ff\u0361\u03b8\u00e6\u0259]")

# Emoji and pictographs: misaki either drops them or reads a name.
_PICTOGRAPHS = re.compile(
    "[\U0001F000-\U0001FAFF\U00002600-\U000027BF\U0001F1E6-\U0001F1FF\ufe0f]"
)


def speakable(text: str) -> str:
    """Drop everything TTS would read out as words rather than speech."""
    text = _UNSPEAKABLE.sub("", text or "")
    text = _LIST_MARKER.sub("", text)
    text = _PICTOGRAPHS.sub("", text)
    text = _IPA_SPAN.sub(lambda m: "" if _IPA_CHARS.search(m.group(0)) else m.group(0), text)
    return re.sub(r"[ \t]{2,}", " ", text).strip()


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


# ---------------------------------------------------------------------------
# Prosody helpers (pure functions, so the maths is testable without a model)
# ---------------------------------------------------------------------------
def pitch_shift(audio: np.ndarray, ratio: float) -> np.ndarray:
    """Resample `audio` so its pitch is multiplied by `ratio`.

    The waveform also gets `ratio` times shorter; callers compensate by
    synthesizing at `speed / ratio` and rescaling the viseme timeline by
    `1 / ratio`. Linear interpolation is enough for the few percent the moods
    use -- Kokoro's 24 kHz output has little energy near Nyquist, so the
    resampling artefacts stay inaudible.
    """
    if audio.size == 0 or abs(ratio - 1.0) < 1e-3:
        return audio.astype(np.float32)
    n_out = max(1, int(round(audio.size / ratio)))
    idx = np.linspace(0.0, audio.size - 1, n_out)
    return np.interp(idx, np.arange(audio.size), audio).astype(np.float32)


def apply_gain(audio: np.ndarray, gain: float) -> np.ndarray:
    """Scale amplitude, backing off if that would clip."""
    if audio.size == 0 or abs(gain - 1.0) < 1e-3:
        return audio.astype(np.float32)
    out = audio.astype(np.float32) * float(gain)
    peak = float(np.max(np.abs(out)))
    if peak > 1.0:
        out = out / peak
    return out.astype(np.float32)


def pad_silence(audio: np.ndarray, lead_s: float, tail_s: float,
                sample_rate: int) -> np.ndarray:
    """Put `lead_s` of silence before `audio` and `tail_s` after it."""
    lead = np.zeros(max(0, int(round(lead_s * sample_rate))), dtype=np.float32)
    tail = np.zeros(max(0, int(round(tail_s * sample_rate))), dtype=np.float32)
    if lead.size == 0 and tail.size == 0:
        return audio.astype(np.float32)
    return np.concatenate([lead, audio.astype(np.float32), tail])


@dataclass
class Speech:
    """One synthesized sentence: what to play, what the mouth does, and the mood."""
    audio: np.ndarray
    timeline: Timeline = field(default_factory=list)
    prosody: emotion_mod.Prosody = field(default_factory=emotion_mod.Prosody)
    text: str = ""

    @property
    def emotion(self) -> str:
        return self.prosody.emotion


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
        return self.speak(text, lang=lang).audio

    def synthesize_with_visemes(self, text: str, lang: Optional[str] = None
                                ) -> Tuple[np.ndarray, Timeline]:
        """Return (waveform, viseme timeline) for `text`, spoken neutrally."""
        result = self.speak(text, lang=lang)
        return result.audio, result.timeline

    def speak(self, text: str, lang: Optional[str] = None,
              persona: Optional[str] = None,
              emotion: Optional[str] = None) -> Speech:
        """Synthesize `text` with the mood its content (or its cue) implies.

        `lang` is a Whisper language code ('en', 'es', 'zh') selecting the
        Kokoro pipeline and voice. `emotion` is a cue the model wrote, already
        pulled out of the token stream; without one the mood is read off the
        text and the persona.

        Uses the pipeline directly rather than model.generate() because the
        pipeline's Result carries the phonemes and per-phoneme frame durations
        alongside the audio -- all from the one forward pass that generate()
        would have made anyway.
        """
        prosody, text = emotion_mod.resolve(text, persona=persona, emotion=emotion)
        text = speakable(text)
        if not text:
            return Speech(np.zeros(0, dtype=np.float32), [], prosody, "")

        kokoro_lang = self._kokoro_lang_for(lang)
        voice = self._voice_for_lang(lang)

        try:
            audio, timeline = self._speak_chunks(text, kokoro_lang, voice, prosody)
        except Exception as e:
            # Never let prosody or lip-sync break speech.
            print(f"[TTS] expressive path failed ({e}); falling back to plain synthesis")
            return Speech(self._synthesize_plain(text, kokoro_lang, voice), [],
                          prosody, text)

        if audio.size == 0:
            return Speech(np.zeros(0, dtype=np.float32), [], prosody, text)

        audio = apply_gain(audio, prosody.gain)
        audio = pad_silence(audio, prosody.lead_s, prosody.tail_s, self.sample_rate)
        if prosody.lead_s:
            timeline = visemes.rescale(timeline, 1.0, prosody.lead_s)
        return Speech(audio, timeline, prosody, text)

    def _speak_chunks(self, text: str, kokoro_lang: str, voice: str,
                      prosody: emotion_mod.Prosody) -> Tuple[np.ndarray, Timeline]:
        """Synthesize every chunk of `text`, applying rate and pitch.

        Rate comes from Kokoro itself; pitch from resampling afterwards. The
        synthesis speed is divided by the pitch ratio so the resampling does not
        also change the speaking rate, and the timeline -- whose durations were
        predicted at that synthesis speed -- is rescaled by the same ratio.
        """
        pitch = float(prosody.pitch) or 1.0
        synth_speed = max(0.5, min(2.0, self.speed * prosody.speed / pitch))
        pipeline = self._pipeline(kokoro_lang)
        vocab = self.model.vocab

        chunks: List[np.ndarray] = []
        timeline: Timeline = []
        elapsed = 0.0  # seconds of audio emitted so far, to offset each chunk

        for segment in chunk_text(text, kokoro_lang):
            for result in pipeline(segment, voice=voice, speed=synth_speed):
                if result.audio is None:
                    continue
                chunk = np.asarray(result.audio, dtype=np.float32).reshape(-1)
                if chunk.size == 0:
                    continue
                chunk = pitch_shift(chunk, pitch)
                chunks.append(chunk)
                if result.pred_dur is not None:
                    spans = visemes.build_shaped_timeline(
                        result.phonemes, [int(d) for d in result.pred_dur],
                        vocab, intensity=prosody.intensity,
                    )
                    timeline.extend(visemes.rescale(spans, 1.0 / pitch, elapsed))
                elapsed += chunk.size / self.sample_rate

        if not chunks:
            return np.zeros(0, dtype=np.float32), []
        return np.concatenate(chunks), timeline

    def _synthesize_plain(self, text: str, lang_code: Optional[str] = None,
                          voice: Optional[str] = None) -> np.ndarray:
        """Original code path, used if the expressive one ever fails."""
        v = voice or self.voice
        segments = self.model.generate(text=text, voice=v, speed=self.speed,
                                       lang_code=lang_code or self.lang_code)
        chunks = [np.asarray(s.audio, dtype=np.float32).reshape(-1) for s in segments]
        if not chunks:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(chunks)
