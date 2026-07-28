"""Tests for the phoneme -> viseme timeline used for lip-sync.

Durations are in Kokoro frames: one frame is 600 samples at 24 kHz = 25 ms.
`pred_dur` is always [<bos>, one per tokenized phoneme, <eos>], so a timeline
must span exactly sum(pred_dur) frames to stay in sync with the audio.
"""
import pytest

import visemes

FRAME = visemes.FRAME_MS / 1000.0  # 0.025 s


def vocab_for(phonemes, missing=()):
    """Fake Kokoro vocab: every phoneme maps to an id except those in `missing`."""
    return {p: i for i, p in enumerate(set(phonemes) - set(missing), start=1)}


def total(timeline):
    return timeline[-1][2] - timeline[0][1]


def test_frame_is_25_ms():
    assert visemes.FRAME_SAMPLES == 600
    assert visemes.FRAME_MS == pytest.approx(25.0)


@pytest.mark.parametrize("phoneme,expected", [
    ("p", "PP"), ("m", "PP"),
    ("f", "FF"), ("v", "FF"),
    ("θ", "TH"),
    ("t", "DD"), ("T", "DD"), ("n", "DD"),
    ("s", "SS"), ("z", "SS"),
    ("ʃ", "CH"), ("ʧ", "CH"),
    ("k", "KK"), ("h", "KK"),
    ("ɹ", "RR"),
    ("ɑ", "aa"), ("æ", "aa"),
    ("ɛ", "E"), ("ə", "E"),        # schwa must not fall through to silence
    ("ɪ", "ih"), ("j", "ih"),
    ("ɔ", "oh"),
    ("u", "ou"), ("w", "ou"),
    (" ", "sil"), (".", "sil"), ("?", "sil"),
    ("\u2764", "sil"),             # unknown symbol -> neutral, never a crash
])
def test_viseme_for(phoneme, expected):
    assert visemes.viseme_for(phoneme) == expected


def test_timeline_spans_the_audio_and_is_contiguous():
    phonemes = "hɛlO"                      # "hello"
    pred_dur = [2, 4, 6, 2, 8, 3]          # bos, h, ɛ, l, O, eos
    tl = visemes.build_timeline(phonemes, pred_dur, vocab_for(phonemes))

    assert [v for v, _, _ in tl] == ["KK", "E", "DD", "oh", "ou", "sil"]
    # Leading silence is the <bos> duration, and the span covers every frame.
    assert tl[0][1] == pytest.approx(2 * FRAME)
    assert tl[-1][2] == pytest.approx(sum(pred_dur) * FRAME)
    # No gaps and no overlaps between neighbouring shapes.
    for prev, nxt in zip(tl, tl[1:]):
        assert prev[2] == pytest.approx(nxt[1])


def test_diphthong_splits_into_two_shapes():
    phonemes = "I"
    tl = visemes.build_timeline(phonemes, [0, 8, 0], vocab_for(phonemes))
    assert [v for v, _, _ in tl] == ["aa", "ih"]
    first, second = tl
    assert first[2] - first[1] == pytest.approx(second[2] - second[1])


def test_identical_neighbours_are_merged():
    phonemes = "tdn"                        # all DD
    tl = visemes.build_timeline(phonemes, [0, 2, 2, 2, 0], vocab_for(phonemes))
    assert tl == [("DD", 0.0, pytest.approx(6 * FRAME))]


def test_single_frame_consonant_survives():
    """A 25 ms /d/ is real articulation, so it must not be absorbed away."""
    phonemes = "ɑdɑ"
    tl = visemes.build_timeline(phonemes, [0, 4, 1, 4, 0], vocab_for(phonemes))
    assert [v for v, _, _ in tl] == ["aa", "DD", "aa"]


def test_sub_frame_spans_are_absorbed_without_losing_time():
    """Halves of a one-frame diphthong are too short to render; time is kept."""
    phonemes = "dI"
    pred_dur = [0, 2, 1, 0]
    tl = visemes.build_timeline(phonemes, pred_dur, vocab_for(phonemes))
    assert [v for v, _, _ in tl] == ["DD"]
    assert total(tl) == pytest.approx(sum(pred_dur) * FRAME)


def test_stress_marks_donate_their_frames_to_the_following_vowel():
    phonemes = "hˈɛ"
    pred_dur = [1, 2, 1, 4, 1]
    tl = visemes.build_timeline(phonemes, pred_dur, vocab_for(phonemes))
    assert [v for v, _, _ in tl] == ["KK", "E", "sil"]
    kk, e, sil = tl
    assert kk[2] - kk[1] == pytest.approx(2 * FRAME)
    assert e[2] - e[1] == pytest.approx(5 * FRAME)   # 4 own + 1 from the stress mark
    assert sil[2] == pytest.approx(sum(pred_dur) * FRAME)


def test_out_of_vocabulary_phonemes_do_not_shift_the_alignment():
    """Only tokenized phonemes consume a duration slot."""
    phonemes = "h~ɛ"                        # '~' is not in the vocab
    vocab = vocab_for(phonemes, missing="~")
    tl = visemes.build_timeline(phonemes, [0, 2, 4, 0], vocab)
    assert [v for v, _, _ in tl] == ["KK", "E"]
    assert tl[1][2] - tl[1][1] == pytest.approx(4 * FRAME)


def test_offset_shifts_the_whole_timeline():
    phonemes = "hɛ"
    args = (phonemes, [1, 2, 2, 1], vocab_for(phonemes))
    base = visemes.build_timeline(*args)
    shifted = visemes.build_timeline(*args, offset_s=1.5)
    assert [v for v, _, _ in shifted] == [v for v, _, _ in base]
    for (_, s0, e0), (_, s1, e1) in zip(base, shifted):
        assert s1 == pytest.approx(s0 + 1.5)
        assert e1 == pytest.approx(e0 + 1.5)


@pytest.mark.parametrize("pred_dur", [None, [], [3], [1, 2]])
def test_degenerate_durations_yield_no_timeline(pred_dur):
    assert visemes.build_timeline("hɛ", pred_dur, vocab_for("hɛ")) == []


def test_to_wire_sends_starts_only_and_closes_with_silence():
    tl = [("KK", 0.05, 0.15), ("E", 0.15, 0.3)]
    assert visemes.to_wire(tl) == [[50, "KK"], [150, "E"], [300, "sil"]]


def test_to_wire_of_empty_timeline():
    assert visemes.to_wire([]) == []
