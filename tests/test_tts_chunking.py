"""Tests for the text chunking that keeps Kokoro from truncating a reply.

Kokoro synthesizes at most config.TTS_MAX_PHONEMES phonemes per forward pass and
mlx-audio truncates anything longer ("Truncating len(ps) == 657 > 510"), which
cuts the audio off mid-word. chunk_text() has to guarantee every piece fits,
prefer clause boundaries, and never lose or reorder text.
"""
import pytest

import config
import tts


ZH = "\u5149\u5408\u4f5c\u7528\u662f\u690d\u7269\u5229\u7528\u9633\u5149\u628a" \
     "\u4e8c\u6c27\u5316\u78b3\u548c\u6c34\u53d8\u6210\u7cd6\u7684\u8fc7\u7a0b\uff0c"


def test_short_text_is_one_chunk():
    assert tts.chunk_text("Plants use light.", "a") == ["Plants use light."]


@pytest.mark.parametrize("lang_code", ["a", "b", "e", "z"])
def test_every_chunk_fits_the_budget(lang_code):
    text = ZH * 40 if lang_code == "z" else "La planta usa la luz del sol " * 60
    limit = tts.max_chunk_chars(lang_code)
    chunks = tts.chunk_text(text, lang_code)
    assert chunks
    assert all(len(c) <= limit for c in chunks)


def test_chinese_budget_stays_under_the_phoneme_limit():
    """~4.13 phonemes per character measured with misaki zh, so leave headroom."""
    assert tts.max_chunk_chars("z") * 4.2 < config.TTS_MAX_PHONEMES


def test_unknown_pipeline_falls_back_to_the_default_budget():
    assert tts.max_chunk_chars("q") == config.TTS_MAX_CHUNK_CHARS_DEFAULT


def test_nothing_is_lost_or_reordered():
    text = ZH * 40
    assert "".join(tts.chunk_text(text, "z")) == text.strip()


def test_splits_on_a_clause_boundary_when_there_is_one():
    text = ("Uno dos tres cuatro. " * 30) + "final."
    for chunk in tts.chunk_text(text, "e"):
        assert chunk.endswith(".")


def test_falls_back_to_a_comma_then_a_word_break():
    comma = tts.chunk_text("palabra, " * 80, "e")
    assert all(c.endswith(",") for c in comma[:-1])

    words = tts.chunk_text("palabra " * 100, "e")
    assert all(" " not in c[-1] and not c.endswith("palabr") for c in words)
    assert "".join(words).replace(" ", "") == ("palabra" * 100)


def test_unbroken_run_is_hard_cut_rather_than_dropped():
    text = "x" * 900
    chunks = tts.chunk_text(text, "e")
    assert "".join(chunks) == text
    assert all(len(c) <= tts.max_chunk_chars("e") for c in chunks)


@pytest.mark.parametrize("blank", ["", "   ", "\n"])
def test_blank_text_yields_no_chunks(blank):
    assert tts.chunk_text(blank, "a") == []


# --- markdown that TTS would pronounce --------------------------------------
# misaki turns "*" into the word "asterisk", so *boss* is spoken as
# "asterisk boss asterisk". Emphasis has to be removed, not passed through.

def test_emphasis_markers_are_removed():
    assert tts.speakable('Well, *boss*, you say "La cuenta, por favor."') \
        == 'Well, boss, you say "La cuenta, por favor."'


@pytest.mark.parametrize("decorated,plain", [
    ("**very** important", "very important"),
    ("_quietly_ said", "quietly said"),
    ("use `hola` here", "use hola here"),
    ("~~wrong~~ right", "wrong right"),
    ("# Heading", "Heading"),
])
def test_every_markdown_flavour_is_stripped(decorated, plain):
    assert tts.speakable(decorated) == plain


def test_real_punctuation_is_kept():
    for text in ("¿Cómo estás?", "他说：“停下。”", "It costs $5, right?"):
        assert tts.speakable(text) == text


def test_stripping_does_not_leave_double_spaces():
    assert "  " not in tts.speakable("a * b * c")
