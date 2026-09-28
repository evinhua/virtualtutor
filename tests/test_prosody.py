"""Tests for the audio side of an emotional delivery.

Three pure functions carry the risk: the pitch shift (a resample, so it changes
length and the viseme timeline has to follow), the gain (which must not clip) and
the pauses around a sentence. Nothing here loads a model.
"""
import numpy as np
import pytest

import tts


def tone(seconds: float = 0.5, freq: float = 200.0, sr: int = 24000) -> np.ndarray:
    t = np.arange(int(seconds * sr)) / sr
    return (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


# --- pitch shift ------------------------------------------------------------
@pytest.mark.parametrize("ratio", [0.94, 0.98, 1.02, 1.045, 1.08])
def test_pitch_shift_changes_length_by_the_ratio(ratio):
    audio = tone()
    out = tts.pitch_shift(audio, ratio)
    assert out.size == pytest.approx(audio.size / ratio, rel=1e-3)
    assert out.dtype == np.float32


def test_pitch_shift_raises_the_frequency():
    """A 200 Hz tone resampled by 1.05 has to come out near 210 Hz."""
    sr = 24000
    out = tts.pitch_shift(tone(1.0, 200.0, sr), 1.05)
    spectrum = np.abs(np.fft.rfft(out))
    peak_hz = float(np.fft.rfftfreq(out.size, 1 / sr)[np.argmax(spectrum)])
    assert peak_hz == pytest.approx(210.0, abs=2.0)


def test_pitch_shift_of_one_is_a_no_op():
    audio = tone(0.1)
    assert np.array_equal(tts.pitch_shift(audio, 1.0), audio)


def test_pitch_shift_of_empty_audio_is_empty():
    assert tts.pitch_shift(np.zeros(0, dtype=np.float32), 1.04).size == 0


def test_pitch_shift_keeps_the_signal_bounded():
    out = tts.pitch_shift(tone(), 0.95)
    assert np.max(np.abs(out)) <= 0.51


# --- gain -------------------------------------------------------------------
def test_gain_scales_amplitude():
    out = tts.apply_gain(tone(), 1.1)
    assert np.max(np.abs(out)) == pytest.approx(0.55, abs=0.01)


def test_gain_backs_off_rather_than_clipping():
    loud = (np.ones(100, dtype=np.float32) * 0.95)
    out = tts.apply_gain(loud, 1.15)
    assert np.max(np.abs(out)) <= 1.0


def test_gain_of_one_is_a_no_op():
    audio = tone(0.05)
    assert np.array_equal(tts.apply_gain(audio, 1.0), audio)


# --- pauses -----------------------------------------------------------------
def test_pad_silence_adds_exactly_the_requested_time():
    sr = 24000
    out = tts.pad_silence(tone(0.5, sr=sr), 0.12, 0.08, sr)
    assert out.size == int(0.5 * sr) + int(round(0.12 * sr)) + int(round(0.08 * sr))
    assert np.all(out[:int(0.12 * sr)] == 0.0)
    assert np.all(out[-int(0.08 * sr):] == 0.0)


def test_pad_silence_with_no_pauses_returns_the_same_audio():
    audio = tone(0.05)
    assert np.array_equal(tts.pad_silence(audio, 0.0, 0.0, 24000), audio)


# --- what the moods actually do together ------------------------------------
def test_rate_and_pitch_are_independent():
    """Synthesizing at speed/ratio and resampling by ratio must cancel out.

    This is the contract that keeps a pitch change from also speeding the
    sentence up -- and the reason the timeline is rescaled by 1/ratio.
    """
    import emotion
    for name in emotion.EMOTIONS:
        p = emotion.prosody_for(name)
        synth_speed = p.speed / p.pitch
        assert synth_speed * p.pitch == pytest.approx(p.speed)


# --- text the tutor must never have read out --------------------------------
def test_phonetic_transcription_is_not_read_out():
    """"/ˈɡɾaθjas/" spoken aloud is a list of symbol names."""
    assert tts.speakable("Gracias, said /\u02c8\u0261\u027ea\u03b8jas/ there") \
        == "Gracias, said there"


def test_plain_slashes_survive():
    assert tts.speakable("It is 24/7 in Madrid.") == "It is 24/7 in Madrid."


@pytest.mark.parametrize("listed,plain", [
    ("- primero\n- segundo", "primero\nsegundo"),
    ("1. uno\n2. dos", "uno\ndos"),
    ("\u2022 bullet", "bullet"),
])
def test_list_markers_are_removed(listed, plain):
    assert tts.speakable(listed) == plain


def test_emoji_are_dropped():
    assert tts.speakable("Well done \U0001F389") == "Well done"


def test_ordinary_hyphens_and_digits_are_kept():
    assert tts.speakable("A well-known 1-2 step") == "A well-known 1-2 step"
