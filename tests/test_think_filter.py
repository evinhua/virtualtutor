"""Tests for stripping a reasoning model's <think> block out of the reply.

Qwen3 thinks before it answers. Thinking is switched off at the llama-server
(--reasoning-budget 0), but if the tags ever do arrive they must not be spoken,
shown in the transcript, or stored in the history -- and they arrive split across
streamed tokens, so the filter has to match tags it has only partly seen.
"""
import pytest

import voice_agent as va


def stream(text, chunk=3):
    """Feed `text` through ThinkFilter in small pieces, like a token stream."""
    f = va.ThinkFilter()
    out = "".join(f.feed(text[i:i + chunk]) for i in range(0, len(text), chunk))
    return out + f.flush()


def test_plain_reply_passes_through_unchanged():
    reply = "Plants use sunlight. Does that make sense?"
    assert stream(reply) == reply


def test_thinking_block_is_removed():
    assert stream("<think>The student asked about plants.</think>Plants use light.") \
        == "Plants use light."


def test_empty_thinking_block_from_reasoning_budget_zero():
    """--reasoning-budget 0 makes the template emit the tags with nothing inside."""
    assert stream("<think>\n\n</think>\n\nPlants use light.").strip() == "Plants use light."


@pytest.mark.parametrize("chunk", [1, 2, 3, 5, 7, 13])
def test_tags_split_across_token_boundaries(chunk):
    assert stream("<think>hmm, photosynthesis</think>It is how plants eat.",
                  chunk=chunk) == "It is how plants eat."


def test_text_before_the_block_is_kept():
    assert stream("Sure. <think>reasoning</think>Here it is.") == "Sure. Here it is."


def test_several_blocks_are_all_removed():
    assert stream("A<think>x</think>B<think>y</think>C") == "ABC"


def test_a_partial_tag_is_held_back_not_spoken():
    """Mid-stream the filter must not emit "<thi" -- TTS would try to say it."""
    f = va.ThinkFilter()
    assert f.feed("Hello <thi") == "Hello "
    assert f.feed("nk>secret</think> world") == " world"


def test_a_lookalike_that_never_becomes_a_tag_is_released():
    assert stream("The answer is < 5") == "The answer is < 5"
    assert stream("Compare a <thing> to a widget.") == "Compare a <thing> to a widget."


def test_unterminated_thinking_yields_nothing_to_speak():
    """A stream cut off inside a block leaves no half-thought in the transcript."""
    assert stream("<think>I was still working it out when") == ""


def test_nothing_inside_a_block_leaks_even_with_sentence_ends():
    """The block splits into sentences, so leaking it would be audible at once."""
    thought = "<think>First I check. Then I answer. Wait, no.</think>"
    assert stream(thought + "Yes.") == "Yes."
    assert "check" not in stream(thought + "Yes.")
