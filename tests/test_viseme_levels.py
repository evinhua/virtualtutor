"""Tests for how far the mouth travels, on top of which shape it makes.

Each span carries a level: the sentence's emotional articulation times the
emphasis of the phoneme. It is applied by the renderer as openness, not as a
weight -- the browser normalises weights, so a uniform weight scale would
cancel out and nothing would look any different.
"""
import pytest

import visemes

FRAME = visemes.FRAME_MS / 1000.0  # 0.025 s


def vocab_for(phonemes, missing=()):
    return {p: i for i, p in enumerate(set(phonemes) - set(missing), start=1)}


def test_shaped_timeline_matches_the_plain_one_shape_for_shape():
    """The level channel must not change the shapes or their timing."""
    phonemes, pred_dur = "h\u025blO", [2, 4, 6, 2, 8, 3]
    vocab = vocab_for(phonemes)
    plain = visemes.build_timeline(phonemes, pred_dur, vocab)
    shaped = visemes.build_shaped_timeline(phonemes, pred_dur, vocab)
    assert [(v, s, e) for v, s, e, _ in shaped] == plain


def test_intensity_is_the_level_of_every_span():
    phonemes = "h\u025b"
    shaped = visemes.build_shaped_timeline(phonemes, [1, 2, 2, 1],
                                           vocab_for(phonemes), intensity=1.2)
    assert [round(level, 3) for _, _, _, level in shaped] == [1.2, 1.2, 1.2]


def test_default_intensity_is_one():
    phonemes = "h\u025b"
    shaped = visemes.build_shaped_timeline(phonemes, [0, 2, 2, 0], vocab_for(phonemes))
    assert all(level == pytest.approx(1.0) for _, _, _, level in shaped)


def test_primary_stress_widens_the_vowel_it_precedes():
    """"exACTly" should look spoken, not recited."""
    phonemes = "h\u02c8\u025bs"
    pred_dur = [0, 2, 1, 4, 2, 0]
    shaped = visemes.build_shaped_timeline(phonemes, pred_dur, vocab_for(phonemes))
    levels = {v: level for v, _, _, level in shaped}
    assert levels["E"] == pytest.approx(visemes.STRESS_LEVELS["\u02c8"])
    assert levels["KK"] == pytest.approx(1.0)     # unstressed consonant unchanged
    assert levels["SS"] == pytest.approx(1.0)     # emphasis does not carry over


def test_secondary_stress_is_gentler_than_primary():
    def level_of(mark):
        phonemes = f"{mark}\u025b"
        tl = visemes.build_shaped_timeline(phonemes, [0, 1, 4, 0], vocab_for(phonemes))
        return tl[0][3]
    assert 1.0 < level_of("\u02cc") < level_of("\u02c8")


def test_emphasis_and_emotion_multiply():
    phonemes = "\u02c8\u025b"
    tl = visemes.build_shaped_timeline(phonemes, [0, 1, 4, 0], vocab_for(phonemes),
                                       intensity=1.2)
    assert tl[0][3] == pytest.approx(1.2 * visemes.STRESS_LEVELS["\u02c8"])


def test_merged_neighbours_average_their_levels_by_duration():
    """A long unstressed vowel joined to a short stressed one is barely raised."""
    phonemes = "\u025b\u02c8e"          # both map to E, so they merge
    pred_dur = [0, 8, 1, 1, 0]
    tl = visemes.build_shaped_timeline(phonemes, pred_dur, vocab_for(phonemes))
    assert len(tl) == 1
    viseme, start, end, level = tl[0]
    assert viseme == "E"
    assert end - start == pytest.approx(10 * FRAME)
    assert 1.0 < level < visemes.STRESS_LEVELS["\u02c8"]


# --- rescale (used when audio is resampled for a pitch shift) ----------------
def test_rescale_stretches_and_offsets_while_keeping_levels():
    tl = [("KK", 0.0, 0.1, 1.2), ("E", 0.1, 0.4, 0.9)]
    out = visemes.rescale(tl, 0.5, 1.0)
    assert out == [("KK", 1.0, 1.05, 1.2), ("E", 1.05, 1.2, 0.9)]


def test_rescale_keeps_a_plain_timeline_plain():
    assert visemes.rescale([("E", 0.0, 0.2)], 2.0) == [("E", 0.0, 0.4)]


def test_rescale_of_nothing_is_nothing():
    assert visemes.rescale([]) == []


# --- wire format -------------------------------------------------------------
def test_wire_carries_the_level_when_there_is_one():
    tl = [("KK", 0.05, 0.15, 1.15), ("E", 0.15, 0.3, 0.9)]
    assert visemes.to_wire(tl) == [[50, "KK", 1.15], [150, "E", 0.9], [300, "sil"]]


def test_wire_stays_two_element_without_levels():
    """Old-style timelines must still produce the old wire format."""
    assert visemes.to_wire([("KK", 0.05, 0.15)]) == [[50, "KK"], [150, "sil"]]
