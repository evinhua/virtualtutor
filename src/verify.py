"""Offline self-test for VirtualTutor (no microphone required).

Checks:
  1. All pipeline imports load.
  2. Silero VAD loads and scores a frame.
  3. Kokoro TTS synthesizes speech to a wav file.
  4. mlx-whisper transcribes that wav back to text (STT round-trip).
  5. (optional) llama-server reachability.

Run: ./.venv/bin/python src/verify.py
"""
import sys

import numpy as np

import config


def check_imports():
    import sounddevice          # noqa: F401
    import torch                # noqa: F401
    import mlx_whisper          # noqa: F401
    from silero_vad import load_silero_vad  # noqa: F401
    from tts import KokoroTTS   # noqa: F401
    print("[1/5] imports OK")


def check_vad():
    import torch
    from silero_vad import load_silero_vad
    vad = load_silero_vad()
    frame = torch.zeros(config.FRAME_SIZE)
    prob = vad(frame, config.SAMPLE_RATE).item()
    assert 0.0 <= prob <= 1.0
    print(f"[2/5] Silero VAD OK (silence prob={prob:.3f})")


def check_tts_and_stt():
    import soundfile as sf
    import mlx_whisper
    from tts import KokoroTTS

    phrase = "Photosynthesis is how plants turn sunlight into energy."
    tts = KokoroTTS()
    audio = tts.synthesize(phrase)
    assert audio.size > 0, "TTS produced no audio"
    sf.write("samples/verify_tts.wav", audio, tts.sample_rate)
    print(f"[3/5] TTS OK ({audio.size/tts.sample_rate:.1f}s -> samples/verify_tts.wav)")

    # Whisper wants 16 kHz mono float32; resample from 24 kHz.
    audio16 = _resample(audio, tts.sample_rate, config.SAMPLE_RATE)
    result = mlx_whisper.transcribe(audio16, path_or_hf_repo=config.WHISPER_MODEL)
    text = result.get("text", "").strip()
    print(f"[4/5] STT OK -> transcribed: {text!r}")
    assert any(w in text.lower() for w in ("photosynthesis", "plant", "sunlight")), \
        "STT round-trip did not recover key words"


def check_llm():
    import requests
    try:
        base = config.LLAMA_SERVER_URL.replace("/v1/chat/completions", "/health")
        r = requests.get(base, timeout=2)
        print(f"[5/5] llama-server reachable (status {r.status_code})")
    except Exception as e:
        print(f"[5/5] llama-server not running (start it with ./scripts/start_server.sh): {e}")


def _resample(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out:
        return x.astype(np.float32)
    n_out = int(round(len(x) * sr_out / sr_in))
    idx = np.linspace(0, len(x) - 1, n_out)
    return np.interp(idx, np.arange(len(x)), x).astype(np.float32)


if __name__ == "__main__":
    check_imports()
    check_vad()
    check_tts_and_stt()
    check_llm()
    print("\nAll core checks passed.")
    sys.exit(0)
