"""Kokoro text-to-speech wrapper (mlx-audio, Metal-accelerated).

Synthesis only -- playback (and barge-in interruption) is handled by the agent
so it can stop audio mid-sentence.
"""
import numpy as np

import config


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
        # Warm up so the first real response is not slow.
        try:
            self.synthesize("Ready.")
        except Exception as e:  # pragma: no cover
            print(f"[TTS] warmup skipped: {e}")

    def synthesize(self, text: str) -> np.ndarray:
        """Return a mono float32 waveform at self.sample_rate for `text`."""
        text = text.strip()
        if not text:
            return np.zeros(0, dtype=np.float32)
        segments = self.model.generate(text=text, voice=self.voice, speed=self.speed)
        chunks = [np.asarray(s.audio, dtype=np.float32).reshape(-1) for s in segments]
        if not chunks:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(chunks)
