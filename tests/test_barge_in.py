"""Tests for the barge-in decision that makes full duplex work.

The bug these cover: barge-in required N *consecutive* frames above the VAD and
RMS thresholds. Speech dips below both between words and on plosives, so the
counter kept restarting -- measured on synthesized speech at a normal speaking
level, "Stop." and "Wait!" never interrupted at all and a full sentence took
1.66 s, by which point the tutor had talked over the user. Evidence that decays
instead of resetting fires the same cases in 0.61-0.83 s.

Pure logic: no VAD model is loaded and no audio device is opened.
"""
import pytest

import config
from voice_agent import BargeInDetector

LOUD = config.BARGE_IN_MIN_RMS * 2
QUIET = config.BARGE_IN_MIN_RMS / 2


@pytest.fixture
def detector():
    return BargeInDetector()


def feed_all(detector, frames):
    """Feed (is_speech, rms) frames; return the 1-based frame that fired."""
    for i, (is_speech, rms) in enumerate(frames, 1):
        if detector.feed(is_speech, rms):
            return i
    return None


def test_the_threshold_matches_the_configured_duration():
    expected = int(config.BARGE_IN_SPEECH_DURATION
                   * config.SAMPLE_RATE / config.FRAME_SIZE)
    assert BargeInDetector().needed == expected


def test_sustained_speech_fires_after_the_configured_duration(detector):
    n = detector.needed
    assert feed_all(detector, [(True, LOUD)] * (n + 5)) == n


def test_it_does_not_fire_before_there_is_enough_evidence(detector):
    for _ in range(detector.needed - 1):
        assert detector.feed(True, LOUD) is False


def test_a_dip_between_words_does_not_start_over(detector):
    """The actual bug: one quiet frame used to reset the whole count."""
    n = detector.needed
    frames = []
    for _ in range(n):
        frames += [(True, LOUD), (True, LOUD), (False, QUIET)]
    fired = feed_all(detector, frames)
    assert fired is not None, "speech with natural dips must still interrupt"
    # Two frames gained, then BARGE_IN_DECAY given back: still net progress.
    assert fired <= 3 * n


def test_a_lone_spike_never_interrupts(detector):
    """A keypress or a burst of echo: loud, but over before it counts."""
    frames = []
    for _ in range(40):
        frames += [(True, LOUD)] + [(False, QUIET)] * 6
    assert feed_all(detector, frames) is None


def test_loud_noise_that_is_not_speech_never_interrupts(detector):
    assert feed_all(detector, [(False, LOUD * 10)] * 200) is None


def test_speech_below_the_echo_gate_never_interrupts(detector):
    """The RMS gate is what keeps the tutor's own voice from interrupting it."""
    assert feed_all(detector, [(True, QUIET)] * 200) is None


def test_it_fires_only_once_per_playback(detector):
    n = detector.needed
    assert feed_all(detector, [(True, LOUD)] * n) == n
    # Already interrupted: further speech is the utterance, not a new barge-in.
    assert all(detector.feed(True, LOUD) is False for _ in range(n * 2))
    assert detector.fired is True


def test_reset_arms_it_again(detector):
    feed_all(detector, [(True, LOUD)] * detector.needed)
    detector.reset()
    assert detector.fired is False
    assert detector.score == 0.0
    assert feed_all(detector, [(True, LOUD)] * detector.needed) == detector.needed


def test_evidence_cannot_bank_up_beyond_the_threshold(detector):
    """Long speech must not leave a surplus that outlives a later silence."""
    feed_all(detector, [(True, LOUD)] * (detector.needed * 4))
    detector.fired = False                     # as if a new reply started
    assert detector.score <= detector.needed


def test_silence_decays_the_score_to_zero(detector):
    detector.feed(True, LOUD)
    detector.feed(True, LOUD)
    for _ in range(100):
        detector.feed(False, QUIET)
    assert detector.score == 0.0
