"""Tests for persona switching.

The bug these cover: a worker thread still unwinding from the previous session
made `is_running()` true, so `start_pipeline` returned early and the session
kept whichever persona it was first started with. Switching now always takes
effect, and a straggler thread is retired by the session generation instead of
blocking the next start.
"""
import queue
import threading
import time

import pytest

import config
import voice_agent as va


@pytest.fixture(autouse=True)
def restore_state():
    before = list(va.conversation), va.active_persona, va._generation
    yield
    va.conversation[:] = before[0]
    va.active_persona = before[1]
    va._generation = before[2]
    va.stop_event.clear()
    while True:
        try:
            va.event_q.get_nowait()
        except queue.Empty:
            break


def test_set_persona_rewrites_the_system_prompt():
    for key in config.PERSONAS:
        va.set_persona(key)
        prompt = va.conversation[0]["content"]
        assert config.PERSONAS[key]["style"] in prompt
        assert va.active_persona == key


def test_set_persona_keeps_the_conversation_history():
    va.conversation[:] = [{"role": "system", "content": "old"},
                          {"role": "user", "content": "hi"}]
    va.set_persona("jester")
    assert [m["role"] for m in va.conversation] == ["system", "user"]
    assert va.conversation[1]["content"] == "hi"


def test_unknown_persona_falls_back_to_the_default():
    assert va.set_persona("nope") == config.DEFAULT_PERSONA


def test_set_persona_announces_the_change():
    va.set_persona("explorer")
    events = []
    while True:
        try:
            events.append(va.event_q.get_nowait())
        except queue.Empty:
            break
    persona_events = [e for e in events if e["type"] == "persona"]
    assert persona_events, "the UI needs to know the persona changed"
    assert persona_events[-1]["key"] == "explorer"
    assert persona_events[-1]["name"] == config.PERSONAS["explorer"]["name"]


# --- session generation -----------------------------------------------------
def test_workers_of_the_current_session_keep_running():
    va.stop_event.clear()
    assert va.session_active(va._generation) is True


def test_a_stop_retires_the_current_session():
    va.stop_event.set()
    assert va.session_active(va._generation) is False


def test_a_new_session_retires_workers_of_the_old_one():
    """The straggler check: an old worker must not survive into a new session."""
    va.stop_event.clear()
    old = va._generation
    va._generation += 1                      # what start_pipeline does
    assert va.session_active(old) is False   # old worker exits
    assert va.session_active(va._generation) is True


def test_a_lingering_worker_cannot_block_the_next_start(monkeypatch):
    """stop_pipeline must clear _threads even if a thread outlives the join."""
    release = threading.Event()
    stuck = threading.Thread(target=release.wait, name="stuck-brain", daemon=True)
    stuck.start()
    monkeypatch.setattr(va, "_threads", [stuck])
    monkeypatch.setattr(va, "_input_stream", None)
    # Keep the test fast: the real join budget is 5 s.
    original_join = stuck.join
    monkeypatch.setattr(stuck, "join", lambda timeout=None: original_join(0.05))

    va.stop_pipeline()
    try:
        assert stuck.is_alive(), "the straggler is still running, as designed"
        assert va._threads == []
        assert va.is_running() is False, "a straggler must not look like a session"
    finally:
        release.set()
        stuck.join(timeout=2)


def test_stop_clears_the_interrupt_flag_it_used():
    """Left set, it would silence the next session's first reply."""
    va.stop_pipeline()
    assert not va.interrupt_event.is_set()


def test_stop_is_idempotent():
    va.stop_pipeline()
    va.stop_pipeline()
    assert va.is_running() is False
