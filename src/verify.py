"""Offline self-test for VirtualTutor (no microphone required).

Checks:
  1. All pipeline imports load.
  2. Silero VAD loads and scores a frame.
  3. Kokoro TTS synthesizes speech to a wav file.
  4. mlx-whisper transcribes that wav back to text (STT round-trip).
  5. The moods really do change the delivery (rate, pitch, articulation).
  6. (optional) LLM backend reachability.

Run: ./.venv/bin/python src/verify.py
"""
import sys

import numpy as np

import config
import emotion


def check_imports():
    import sounddevice          # noqa: F401
    import torch                # noqa: F401
    import mlx_whisper          # noqa: F401
    from silero_vad import load_silero_vad  # noqa: F401
    from tts import KokoroTTS   # noqa: F401
    print("[1/6] imports OK")


def check_vad():
    import torch
    from silero_vad import load_silero_vad
    vad = load_silero_vad()
    frame = torch.zeros(config.FRAME_SIZE)
    prob = vad(frame, config.SAMPLE_RATE).item()
    assert 0.0 <= prob <= 1.0
    print(f"[2/6] Silero VAD OK (silence prob={prob:.3f})")


def check_tts_and_stt():
    import soundfile as sf
    import mlx_whisper
    from tts import KokoroTTS

    phrase = "Photosynthesis is how plants turn sunlight into energy."
    tts = KokoroTTS()
    audio = tts.synthesize(phrase)
    assert audio.size > 0, "TTS produced no audio"
    sf.write("samples/verify_tts.wav", audio, tts.sample_rate)
    print(f"[3/6] TTS OK ({audio.size/tts.sample_rate:.1f}s -> samples/verify_tts.wav)")

    # Whisper wants 16 kHz mono float32; resample from 24 kHz.
    audio16 = _resample(audio, tts.sample_rate, config.SAMPLE_RATE)
    result = mlx_whisper.transcribe(audio16, path_or_hf_repo=config.WHISPER_MODEL)
    text = result.get("text", "").strip()
    print(f"[4/6] STT OK -> transcribed: {text!r}")
    assert any(w in text.lower() for w in ("photosynthesis", "plant", "sunlight")), \
        "STT round-trip did not recover key words"
    return tts


def check_emotions(tts):
    """An excited line must come out faster and wider-mouthed than a gentle one.

    Also checks the invariant that makes the pitch shift safe: the viseme
    timeline still ends where the speech does, so the mouth cannot drift.
    """
    line = "That is the phrase you want to remember."
    results = {}
    for mood in ("excited", "thoughtful"):
        speech = tts.speak(f"[{mood}] {line}")
        assert speech.audio.size > 0, f"{mood}: no audio"
        assert speech.timeline, f"{mood}: no viseme timeline"
        duration = speech.audio.size / tts.sample_rate
        speech_end = duration - speech.prosody.tail_s
        drift = abs(speech.timeline[-1][2] - speech_end)
        assert drift < 0.05, f"{mood}: lip-sync drifted {drift*1000:.0f} ms"
        levels = [span[3] for span in speech.timeline]
        assert max(levels) > 0, f"{mood}: timeline carries no articulation level"
        results[mood] = speech
        print(f"       {mood:<11} {duration:5.2f}s  "
              f"rate x{speech.prosody.speed:.2f}  pitch x{speech.prosody.pitch:.3f}  "
              f"mouth x{max(levels):.2f}  drift {drift*1000:.0f} ms")
    fast, slow = results["excited"], results["thoughtful"]
    assert fast.audio.size < slow.audio.size, "excited should be quicker than thoughtful"
    assert max(s[3] for s in fast.timeline) > max(s[3] for s in slow.timeline), \
        "excited should articulate wider than thoughtful"
    print(f"[5/6] moods OK ({len(emotion.EMOTIONS)} available)")


def check_llm():
    import requests
    try:
        r = requests.get(config.LLM_HEALTH_URL, timeout=2)
        print(f"[6/6] LLM backend '{config.LLM_BACKEND}' reachable "
              f"(status {r.status_code}, model {config.LLM_MODEL or 'server default'})")
    except Exception as e:
        print(f"[6/6] LLM backend not reachable ({config.LLM_START_HINT}): {e}")


def _resample(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out:
        return x.astype(np.float32)
    n_out = int(round(len(x) * sr_out / sr_in))
    idx = np.linspace(0, len(x) - 1, n_out)
    return np.interp(idx, np.arange(len(x)), x).astype(np.float32)


if __name__ == "__main__":
    check_imports()
    check_vad()
    tts = check_tts_and_stt()
    check_emotions(tts)
    check_llm()
    print("\nAll core checks passed.")
    sys.exit(0)
