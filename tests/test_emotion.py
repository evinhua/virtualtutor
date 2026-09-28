"""Tests for the mood each spoken sentence is given.

The mood decides rate, pitch, loudness, the pauses around the sentence and how
wide the avatar articulates, so three things have to hold: a cue the model wrote
is obeyed and never spoken, the text's own signals are read correctly in all
three languages, and no combination of persona and mood can push the voice
outside the range where Kokoro still sounds like itself.
"""
import pytest

import config
import emotion


# --- explicit cues ----------------------------------------------------------
@pytest.mark.parametrize("text,expected", [
    ("[excited] That is exactly right!", "excited"),
    ("[gentle] Almost, it is la cuenta.", "gentle"),
    ("(thoughtful) Let me put it another way.", "thoughtful"),
    ("<curious> And where are you going?", "curious"),
    ("[EXCITED] shouting the cue", "excited"),
    ("[  proud  ] spaced out", "proud"),
])
def test_cue_is_recognised(text, expected):
    assert emotion.parse_cue(text)[0] == expected


def test_cue_is_removed_from_what_gets_spoken():
    found, text = emotion.parse_cue("[excited] Vamos a practicar!")
    assert found == "excited"
    assert text == "Vamos a practicar!"


def test_aliases_map_onto_a_real_mood():
    assert emotion.parse_cue("[happy] hola")[0] == "excited"
    assert emotion.parse_cue("[apologetic] lo siento")[0] == "gentle"
    assert emotion.canonical("Laughing") == "amused"


def test_brackets_that_are_not_moods_are_left_alone():
    """"(the bill)" is part of the sentence, not an instruction to the voice."""
    found, text = emotion.parse_cue("You ask for (the bill) politely.")
    assert found is None
    assert text == "You ask for (the bill) politely."


def test_only_the_first_cue_wins_but_all_are_stripped():
    found, text = emotion.parse_cue("[warm] Good. [proud] Very good.")
    assert found == "warm"
    assert "[" not in text and "]" not in text
    assert text == "Good. Very good."


def test_strip_cues_leaves_plain_text_untouched():
    assert emotion.strip_cues("Nothing to strip.") == "Nothing to strip."


# --- signals in the text ----------------------------------------------------
@pytest.mark.parametrize("text,expected", [
    ("Haha, that was a good try!", "amused"),
    ("\u54c8\u54c8\uff0c\u4f60\u5f88\u6709\u8da3\u3002", "amused"),
    ("Exactly, that is the one.", "proud"),
    ("\u592a\u597d\u4e86\uff0c\u4f60\u8bf4\u5f97\u5f88\u597d\u3002", "proud"),
    ("Muy bien, lo dijiste perfecto.", "proud"),
    ("Sorry, I was not clear.", "gentle"),
    ("No pasa nada, vamos otra vez.", "gentle"),
    ("\u522b\u62c5\u5fc3\uff0c\u6162\u6162\u6765\u3002", "gentle"),
    ("Hmm, let me think about that.", "thoughtful"),
    ("Well, it depends on the region\u2026", "thoughtful"),
    ("Wow, that is a great question!", "excited"),
    ("\u00a1Vamos a practicar!", "excited"),
    ("Where would you like to go?", "curious"),
    ("\u4f60\u60f3\u53bb\u54ea\u91cc\uff1f", "curious"),
    ("\u4f60\u559c\u6b22\u5417\uff1f", "curious"),
])
def test_signals_are_read_from_the_text(text, expected):
    assert emotion.detect(text) == expected


def test_plain_statement_falls_back_to_the_persona_baseline():
    assert emotion.detect("The word is la cuenta.", "secretary") == "deadpan"
    assert emotion.detect("The word is la cuenta.", "cheerleader") == "excited"
    assert emotion.detect("The word is la cuenta.", "tutor") == "warm"


def test_every_persona_has_a_baseline_mood_and_an_energy():
    for key in config.PERSONAS:
        assert emotion.PERSONA_EMOTION[key] in emotion.EMOTIONS
        assert 0.5 <= emotion.PERSONA_ENERGY[key] <= 1.5


def test_unknown_persona_still_gets_a_mood():
    assert emotion.detect("A statement.", "nobody") in emotion.EMOTIONS


def test_empty_text_does_not_crash():
    assert emotion.detect("", "tutor") == "warm"
    assert emotion.parse_cue("") == (None, "")


# --- prosody ----------------------------------------------------------------
def test_excited_is_faster_and_wider_than_thoughtful():
    fast = emotion.prosody_for("excited")
    slow = emotion.prosody_for("thoughtful")
    assert fast.speed > 1.0 > slow.speed
    assert fast.intensity > slow.intensity
    assert slow.lead_s > fast.lead_s      # a thoughtful line takes a breath first


def test_persona_energy_scales_the_deviation_without_flipping_it():
    """The secretary's gentle is still gentler than neutral, just less so.

    Energy scales how far a mood departs from neutral, so a high-energy persona
    is *more* gentle when it is gentle, not faster.
    """
    plain = emotion.prosody_for("gentle", "tutor")
    loud = emotion.prosody_for("gentle", "cheerleader")
    quiet = emotion.prosody_for("gentle", "secretary")
    for p in (plain, loud, quiet):
        assert p.speed < 1.0 and p.intensity < 1.0
    deviation = lambda p: 1.0 - p.speed        # noqa: E731
    assert deviation(quiet) < deviation(plain) < deviation(loud)


def test_neutral_persona_energy_changes_nothing():
    assert emotion.prosody_for("warm", "tutor") == emotion.EMOTIONS["warm"]


@pytest.mark.parametrize("name", sorted(emotion.EMOTIONS))
@pytest.mark.parametrize("persona", ["tutor", "cheerleader", "secretary", None])
def test_no_mood_leaves_the_safe_range(name, persona):
    """Beyond these bounds Kokoro stops sounding like the same speaker."""
    p = emotion.prosody_for(name, persona)
    for field, (low, high) in emotion.LIMITS.items():
        assert low <= getattr(p, field) <= high, f"{name}/{persona}: {field}"


def test_unknown_mood_is_neutral_rather_than_an_error():
    assert emotion.prosody_for("banana").emotion == emotion.NEUTRAL


# --- resolve: what the pipeline actually calls -------------------------------
def test_resolve_prefers_the_streamed_cue_over_the_text():
    prosody, text = emotion.resolve("Sorry, not quite.", persona="tutor",
                                    emotion="proud")
    assert prosody.emotion == "proud"
    assert text == "Sorry, not quite."


def test_resolve_falls_back_to_an_inline_cue_then_to_the_text():
    assert emotion.resolve("[gentle] Exactly right!")[0].emotion == "gentle"
    assert emotion.resolve("Exactly right!")[0].emotion == "proud"


def test_resolve_never_returns_a_cue_in_the_text():
    _, text = emotion.resolve("[amused] Nice pun.")
    assert text == "Nice pun."


# --- the prompt the model is given ------------------------------------------
def test_the_prompt_lists_exactly_the_moods_that_exist():
    """A cue the model is told to use but that we cannot apply would be spoken."""
    assert set(config.EMOTION_CUES) == set(emotion.EMOTIONS)
    for name in emotion.EMOTIONS:
        assert f"[{name}]" in config.EMOTION_PROMPT


@pytest.mark.parametrize("key", sorted(config.PERSONAS))
def test_every_persona_prompt_asks_for_a_mood_cue(key):
    prompt = config.build_system_prompt(key)
    assert config.EMOTION_PROMPT in prompt
    assert "not spoken" in prompt
