"""Tests for persona prompt composition.

Most personas are a personality layer on top of the shared tutoring rules; the
secretary brings its own base rules instead. Either way the parts that are not
personality -- the three subjects, short spoken answers, no markdown, replying
in the student's language -- must survive every persona.
"""
import pytest

import config


@pytest.mark.parametrize("key", sorted(config.PERSONAS))
def test_every_persona_has_menu_metadata(key):
    persona = config.PERSONAS[key]
    assert persona["name"] and persona["blurb"] and persona["style"]


@pytest.mark.parametrize("key", sorted(config.PERSONAS))
def test_prompt_keeps_the_base_rules_and_adds_the_style(key):
    prompt = config.build_system_prompt(key)
    assert config.persona_base_prompt(key) in prompt
    assert config.PERSONAS[key]["style"] in prompt


def test_shared_personas_are_built_on_the_shared_rules():
    """Only a persona that declares its own prompt leaves the shared base."""
    for key, persona in config.PERSONAS.items():
        if persona.get("prompt"):
            continue
        assert config.persona_base_prompt(key) == config.BASE_SYSTEM_PROMPT
        assert config.BASE_SYSTEM_PROMPT in config.build_system_prompt(key)


@pytest.mark.parametrize("key", sorted(config.PERSONAS))
def test_prompt_lets_the_tutor_name_its_own_persona(key):
    """Asked "who are you?", it should answer in character, by name."""
    prompt = config.build_system_prompt(key)
    name = config.PERSONAS[key]["name"]
    assert prompt.count(name) >= 2, "state the persona and how to introduce it"
    assert "who or what you are" in prompt
    assert "introduce yourself" in prompt


@pytest.mark.parametrize("key", sorted(config.PERSONAS))
def test_persona_name_lookup(key):
    assert config.persona_name(key) == config.PERSONAS[key]["name"]


def test_persona_name_falls_back_to_the_default():
    assert config.persona_name("nope") == config.PERSONAS[config.DEFAULT_PERSONA]["name"]


def test_unknown_persona_falls_back_to_the_default():
    assert config.build_system_prompt("nope") == config.build_system_prompt(config.DEFAULT_PERSONA)


def test_default_persona_exists():
    assert config.DEFAULT_PERSONA in config.PERSONAS


@pytest.mark.parametrize("key", sorted(config.PERSONAS))
def test_base_rules_are_tts_safe(key):
    rules = config.persona_base_prompt(key).lower()
    for constraint in ("no markdown", "no lists", "no emojis"):
        assert constraint in rules


@pytest.mark.parametrize("key", sorted(config.PERSONAS))
def test_base_rules_name_the_three_subjects(key):
    """The tutor teaches language, culture and travel -- not arbitrary homework."""
    rules = config.persona_base_prompt(key).lower()
    for topic in ("language", "culture", "travel"):
        assert topic in rules


@pytest.mark.parametrize("key", sorted(config.PERSONAS))
def test_pronunciation_guidance_stays_speakable(key):
    """Phonetic symbols or spelled-out letters are unusable through TTS."""
    rules = config.persona_base_prompt(key).lower()
    assert "spoken syllables" in rules
    assert "never as phonetic symbols" in rules


@pytest.mark.parametrize("key", sorted(config.PERSONAS))
def test_language_mirroring_survives_the_topic_change(key):
    rules = config.persona_base_prompt(key)
    assert "same language the student uses" in rules
    for language in ("Spanish", "Chinese"):
        assert language in rules


@pytest.mark.parametrize("key", sorted(config.PERSONAS))
def test_every_persona_inherits_the_subjects(key):
    """A personality changes the delivery, never what is being taught."""
    prompt = config.build_system_prompt(key).lower()
    for topic in ("language", "culture", "travel"):
        assert topic in prompt


def test_secretary_persona_is_offered():
    """The sassy secretary is a persona like any other: menu, API and prompt."""
    assert "secretary" in config.PERSONAS
    assert config.persona_name("secretary") == "The Sassy Secretary"
    prompt = config.build_system_prompt("secretary")
    assert "sassy" in prompt.lower()


def test_secretary_has_its_own_base_prompt():
    """It replaces the shared rules rather than layering on them."""
    base = config.persona_base_prompt("secretary")
    assert base == config.SECRETARY_SYSTEM_PROMPT
    assert base != config.BASE_SYSTEM_PROMPT
    prompt = config.build_system_prompt("secretary")
    assert config.BASE_SYSTEM_PROMPT not in prompt
    assert config.SECRETARY_SYSTEM_PROMPT in prompt


def test_secretary_prompt_is_written_in_its_own_frame():
    """Independent, but still a tutor: the boss/diary framing, and the lesson."""
    base = config.SECRETARY_SYSTEM_PROMPT.lower()
    assert "virtualtutor" in base
    assert "secretary" in base
    for word in ("boss", "diary", "agenda", "filed"):
        assert word in base


def test_secretary_style_stays_the_delivery_layer():
    """The style describes the voice; the subjects live in the prompt."""
    style = config.PERSONAS["secretary"]["style"]
    assert "deadpan" in style.lower()
    assert "affectionate and professional" in style

