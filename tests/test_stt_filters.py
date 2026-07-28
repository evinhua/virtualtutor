"""Tests for the Whisper anti-hallucination guards.

Whisper invents stock phrases when fed near-silence or noise. Three layers stop
that reaching the LLM: an energy gate, the model's own scores, and a blocklist.
"""
import numpy as np
import pytest

import config
import voice_agent as va


# --- Layer 1: energy gate ---------------------------------------------------
def test_rms_of_empty_buffer_is_zero():
    assert va.utterance_rms(np.zeros(0, dtype=np.float32)) == 0.0


def test_rms_of_constant_signal():
    assert va.utterance_rms(np.full(100, 0.5, dtype=np.float32)) == pytest.approx(0.5)


def test_silence_is_below_the_gate_and_speech_is_above():
    silence = np.random.default_rng(0).normal(0, 0.001, 8000).astype(np.float32)
    speech = np.random.default_rng(0).normal(0, 0.05, 8000).astype(np.float32)
    assert va.utterance_rms(silence) < config.STT_MIN_RMS
    assert va.utterance_rms(speech) > config.STT_MIN_RMS


# --- Layer 2: Whisper's own scores ------------------------------------------
def seg(no_speech, logprob):
    return {"no_speech_prob": no_speech, "avg_logprob": logprob}


def test_no_segments_is_inconclusive_and_passes():
    assert va.has_real_speech([]) is True
    assert va.has_real_speech(None) is True


def test_confident_speech_passes():
    assert va.has_real_speech([seg(0.01, -0.2)]) is True


def test_utterance_flagged_non_speech_throughout_is_rejected():
    assert va.has_real_speech([seg(0.95, -0.2), seg(0.99, -0.1)]) is False


def test_low_confidence_throughout_is_rejected():
    assert va.has_real_speech([seg(0.01, -3.0), seg(0.02, -2.5)]) is False


def test_one_good_segment_keeps_the_whole_utterance():
    """Mid-sentence segments must never be dropped -- that used to lose words."""
    assert va.has_real_speech([seg(0.99, -4.0), seg(0.02, -0.3), seg(0.9, -2.0)]) is True


def test_segments_missing_scores_are_treated_as_speech():
    assert va.has_real_speech([{}]) is True


# --- Layer 3: phrase blocklist ----------------------------------------------
@pytest.mark.parametrize("text", [
    "Thank you.", "thank you", "  Thanks for watching!  ", "Thanks.",
    "Please subscribe", "Bye bye.", "Okay.", "um", "Hmm...", ".", "", "   ",
])
def test_known_fillers_are_rejected(text):
    assert va.is_hallucination(text) is True


@pytest.mark.parametrize("text", [
    "What is photosynthesis?",
    "Thank you for explaining that, can you go slower?",
    "Okay, so why does the moon have phases?",
    "Bye means goodbye in English, right?",
])
def test_real_questions_pass(text):
    assert va.is_hallucination(text) is False


def test_blocklist_ignores_case_and_punctuation():
    assert va.is_hallucination("THANK YOU!!!") is True
    assert va.is_hallucination(None) is True
