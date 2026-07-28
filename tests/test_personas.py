"""Tests for persona prompt composition.

A persona is a personality layer on top of the shared tutoring rules, and the
rules (short spoken answers, no markdown) must survive every persona.
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
    assert config.BASE_SYSTEM_PROMPT in prompt
    assert config.PERSONAS[key]["style"] in prompt


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


def test_base_rules_are_tts_safe():
    rules = config.BASE_SYSTEM_PROMPT.lower()
    for constraint in ("no markdown", "no lists", "no emojis"):
        assert constraint in rules
