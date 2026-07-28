"""Tests for conversation history trimming.

Trimming happens in blocks so the token prefix after the system prompt stays
stable and llama.cpp can reuse its KV cache. The invariants that matter: the
system prompt survives, history resumes at a user turn, and trimming only
happens once the high water mark is passed.
"""
import pytest

import voice_agent as va


def history(n_turns):
    """System prompt + n alternating user/assistant messages."""
    msgs = [{"role": "system", "content": "sys"}]
    for i in range(n_turns):
        role = "user" if i % 2 == 0 else "assistant"
        msgs.append({"role": role, "content": f"{role} {i}"})
    return msgs


def trim(msgs, monkeypatch):
    monkeypatch.setattr(va, "conversation", msgs)
    va.trim_history()
    return va.conversation


@pytest.mark.parametrize("n", [0, 1, va.HISTORY_HIGH_WATER])
def test_short_history_is_untouched(n, monkeypatch):
    msgs = history(n)
    before = list(msgs)
    assert trim(msgs, monkeypatch) == before


def test_trimming_keeps_the_system_prompt_and_recent_turns(monkeypatch):
    msgs = trim(history(va.HISTORY_HIGH_WATER + 6), monkeypatch)
    assert msgs[0]["role"] == "system"
    assert msgs[0]["content"] == "sys"
    assert len(msgs) - 1 <= va.HISTORY_LOW_WATER + 1
    assert msgs[-1]["content"] == f"assistant {va.HISTORY_HIGH_WATER + 5}"


def test_history_resumes_at_a_user_turn(monkeypatch):
    """Never lead with a bare assistant reply -- the transcript must alternate."""
    for extra in range(1, 10):
        msgs = trim(history(va.HISTORY_HIGH_WATER + extra), monkeypatch)
        if len(msgs) > 1:
            assert msgs[1]["role"] == "user", f"broke with {extra} extra turns"


def test_repeated_trims_converge(monkeypatch):
    msgs = history(va.HISTORY_HIGH_WATER + 20)
    trim(msgs, monkeypatch)
    once = list(msgs)
    va.trim_history()
    assert msgs == once      # already below the water mark, so nothing more goes
