"""Tests for splitting the LLM token stream into speakable sentences.

Sentences are handed to TTS the moment they are complete, so the splitter has
to fire early without cutting inside numbers or abbreviations, and without
losing any text.
"""
import pytest

import voice_agent as va


def drain(text, chunk=3):
    """Feed `text` through next_sentence in small chunks, like a token stream."""
    buffer = ""
    out = []
    for i in range(0, len(text), chunk):
        buffer += text[i:i + chunk]
        while True:
            sentence, buffer = va.next_sentence(buffer)
            if not sentence:
                break
            out.append(sentence)
    if buffer.strip():          # the agent flushes the tail when the stream ends
        out.append(buffer.strip())
    return out


def test_incomplete_text_is_held_back():
    assert va.next_sentence("Photosynthesis is how") == (None, "Photosynthesis is how")


def test_terminator_alone_is_not_enough():
    """Punctuation must be followed by whitespace, so "3." can still become "3.14"."""
    assert va.next_sentence("The answer is 3.")[0] is None


def test_splits_on_sentence_end_and_keeps_the_remainder():
    sentence, rest = va.next_sentence("Plants use light. They make sugar")
    assert sentence == "Plants use light."
    assert rest == "They make sugar"


def test_split_keeps_the_terminator_for_prosody():
    for text in ("Ready! Now try", "Why? Because it", "Two things: first"):
        sentence, _ = va.next_sentence(text)
        assert sentence[-1] in "!?:"


def test_newline_ends_a_sentence_without_punctuation():
    sentence, rest = va.next_sentence("First step\nSecond step")
    assert (sentence, rest) == ("First step", "Second step")


def test_closing_quote_and_bracket_stay_with_the_sentence():
    sentence, _ = va.next_sentence('He said "stop." Then he')
    assert sentence == 'He said "stop."'


def test_decimals_do_not_split_the_stream():
    assert drain("Pi is 3.14159 exactly. Nice.") == ["Pi is 3.14159 exactly.", "Nice."]


def test_streamed_tokens_produce_whole_sentences():
    reply = ("Photosynthesis is how plants make food. They use sunlight, water "
             "and carbon dioxide! Does that make sense?")
    assert drain(reply) == [
        "Photosynthesis is how plants make food.",
        "They use sunlight, water and carbon dioxide!",
        "Does that make sense?",
    ]


def test_nothing_is_lost_or_duplicated():
    reply = "One. Two! Three? Four; five: six\nseven"
    joined = " ".join(drain(reply, chunk=1))
    assert joined.replace(" ", "") == reply.replace(" ", "").replace("\n", "")


@pytest.mark.parametrize("blank", ["", "   ", "\n", "\n\n"])
def test_blank_input_never_yields_a_sentence_to_speak(blank):
    sentence, _ = va.next_sentence(blank)
    assert not sentence


# --- CJK punctuation --------------------------------------------------------
# Chinese has no spaces and does not use ASCII terminators, so requiring
# "punctuation + whitespace" never fired: a whole reply reached TTS as one chunk
# and Kokoro truncated its tail ("Truncating len(ps) == 657 > 510").

ZH_FIRST = "\u5149\u5408\u4f5c\u7528\u662f\u690d\u7269\u5236\u9020\u98df\u7269\u7684\u65b9\u5f0f\u3002"
ZH_SECOND = "\u5b83\u9700\u8981\u9633\u5149\u3001\u6c34\u548c\u4e8c\u6c27\u5316\u78b3\u3002"


def test_chinese_full_stop_ends_a_sentence_without_whitespace():
    sentence, rest = va.next_sentence(ZH_FIRST + ZH_SECOND)
    assert sentence == ZH_FIRST
    assert rest == ZH_SECOND


@pytest.mark.parametrize("terminator", ["\u3002", "\uff01", "\uff1f", "\uff1b", "\uff1a", "\u2026"])
def test_every_cjk_terminator_splits(terminator):
    sentence, rest = va.next_sentence("\u4f60\u597d" + terminator + "\u518d\u89c1")
    assert sentence == "\u4f60\u597d" + terminator
    assert rest == "\u518d\u89c1"


def test_chinese_closing_quote_stays_with_the_sentence():
    text = "\u4ed6\u8bf4\uff1a\u201c\u505c\u4e0b\u3002\u201d\u7136\u540e"
    sentence, rest = va.next_sentence(text)
    assert sentence.endswith("\u201d")
    assert rest == "\u7136\u540e"


def test_chinese_comma_does_not_split_early():
    """A comma is a pause, not a sentence end -- TTS keeps the clause together."""
    assert va.next_sentence("\u690d\u7269\u9700\u8981\u9633\u5149\uff0c\u4e5f\u9700\u8981\u6c34")[0] is None


def test_streamed_chinese_produces_whole_sentences():
    assert drain(ZH_FIRST + ZH_SECOND, chunk=2) == [ZH_FIRST, ZH_SECOND]
