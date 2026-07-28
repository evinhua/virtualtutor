"""Kokoro text-to-speech wrapper (mlx-audio, Metal-accelerated).

Synthesis only -- playback (and barge-in interruption) is handled by the agent
so it can stop audio mid-sentence.

Besides audio, synthesis also returns a viseme timeline for lip-sync. Kokoro's
duration predictor already emits one frame count per phoneme as part of the
normal forward pass, so the timeline is free: we read the pipeline's
`pred_dur` instead of calling the model a second time or analysing the audio.
"""
from typing import List, Tuple

import numpy as np

import config
import visemes

# (viseme, start_seconds, end_seconds)
Timeline = List[Tuple[str, float, float]]


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
    def _pipeline(self):
        """Kokoro's language pipeline (cached by the model, includes G2P)."""
        return self.model._get_pipeline(self.lang_code)

    # -- public API ------------------------------------------------------
    def synthesize(self, text: str) -> np.ndarray:
        """Return a mono float32 waveform at self.sample_rate for `text`."""
        audio, _ = self.synthesize_with_visemes(text)
        return audio

    def synthesize_with_visemes(self, text: str) -> Tuple[np.ndarray, Timeline]:
        """Return (waveform, viseme timeline) for `text`.

        Uses the pipeline directly rather than model.generate() because the
        pipeline's Result carries the phonemes and per-phoneme frame durations
        alongside the audio -- all from the one forward pass that generate()
        would have made anyway.
        """
        text = text.strip()
        if not text:
            return np.zeros(0, dtype=np.float32), []

        pipeline = self._pipeline()
        vocab = self.model.vocab

        chunks: List[np.ndarray] = []
        timeline: Timeline = []
        elapsed = 0.0  # seconds of audio emitted so far, to offset each chunk

        try:
            results = pipeline(text, voice=self.voice, speed=self.speed)
            for result in results:
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
            return self._synthesize_plain(text), []

        if not chunks:
            return np.zeros(0, dtype=np.float32), []
        return np.concatenate(chunks), timeline

    def _synthesize_plain(self, text: str) -> np.ndarray:
        """Original code path, used if the viseme-aware one ever fails."""
        segments = self.model.generate(text=text, voice=self.voice, speed=self.speed)
        chunks = [np.asarray(s.audio, dtype=np.float32).reshape(-1) for s in segments]
        if not chunks:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(chunks)
