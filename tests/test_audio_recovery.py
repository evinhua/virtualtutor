"""Output device failures must not end playback for the rest of the session.

macOS hands out ``PaErrorCode -9986`` ("Unspecified Audio Hardware Error") when
the default output device is switched, taken over by another app, or unplugged.
That used to raise out of `speak_worker` and kill the thread: the session stayed
"live" and listening but could never speak again. No device is opened here --
`sd.OutputStream` is replaced by fakes.
"""
import queue
import threading
import time

import numpy as np
import pytest
import sounddevice as sd

import voice_agent


class FakeStream:
    """Minimal stand-in for sd.OutputStream."""
    latency = 0.01

    def __init__(self, *, fail_on_start=False, fail_on_write=False, **kwargs):
        self._fail_on_start = fail_on_start
        self._fail_on_write = fail_on_write
        self.stopped = False
        self.closed = False
        self.written = 0
        self.aborted = False

    def start(self):
        if self._fail_on_start:
            raise sd.PortAudioError("Error starting stream: Internal PortAudio error", -9986)

    def write(self, block):
        if self._fail_on_write:
            raise sd.PortAudioError("Error writing to stream: Internal PortAudio error", -9986)
        self.written += len(block)

    def abort(self):
        self.aborted = True

    def close(self, ignore_errors=False):
        self.closed = True


def stream_factory(monkeypatch, outcomes):
    """Patch sd.OutputStream to hand out `outcomes` in order. Returns the list."""
    made = []
    pending = list(outcomes)

    def make(**kwargs):
        kind = pending.pop(0) if pending else "ok"
        stream = FakeStream(fail_on_start=kind == "start", fail_on_write=kind == "write")
        made.append(stream)
        return stream

    monkeypatch.setattr(sd, "OutputStream", make)
    return made


# --------------------------------------------------------------------------
# open_output_stream
# --------------------------------------------------------------------------
def test_open_retries_a_transient_failure(monkeypatch):
    made = stream_factory(monkeypatch, ["start", "ok"])
    out = voice_agent.open_output_stream(24000, backoff=(0.0, 0.0, 0.0))
    assert out is made[1]
    assert made[0].closed  # the failed one is not leaked


def test_open_rides_out_a_long_refusal(monkeypatch):
    """CoreAudio can refuse for many attempts in a row and then work."""
    made = stream_factory(monkeypatch, ["start"] * 8 + ["ok"])
    out = voice_agent.open_stream_with_retries(
        lambda: sd.OutputStream(), "output device", backoff=(0.0,) * 10)
    assert out is made[8]
    assert all(s.closed for s in made[:8])


def test_open_gives_up_without_raising(monkeypatch):
    made = stream_factory(monkeypatch, ["start", "start", "start"])
    assert voice_agent.open_output_stream(24000, backoff=(0.0, 0.0, 0.0)) is None
    assert len(made) == 3
    assert all(s.closed for s in made)


def test_the_mic_waits_longer_than_the_speaker():
    """A session is worth waiting for; a sentence spoken late is not."""
    assert sum(voice_agent.config.AUDIO_OPEN_BACKOFF) > 25
    assert sum(voice_agent.config.AUDIO_OPEN_BACKOFF_OUTPUT) < 3


# --------------------------------------------------------------------------
# memory_pressure_hint: -9986 also means "the kernel would not wire the buffer"
# --------------------------------------------------------------------------
def fake_memory(monkeypatch, free_pct, wired_pages=2_105_677, total=38654705664):
    """Stand in for `memory_pressure -Q` and `vm_stat`."""
    def run(cmd, **kwargs):
        class Result:
            stdout = (
                f"The system has {total} (2359296 pages with a page size of 16384).\n"
                f"System-wide memory free percentage: {free_pct}%\n"
                if cmd[0] == "memory_pressure" else
                f"Mach Virtual Memory Statistics: (page size of 16384 bytes)\n"
                f"Pages wired down:          {wired_pages}.\n")
        return Result()
    monkeypatch.setattr(voice_agent.subprocess, "run", run)


def test_no_hint_when_memory_is_fine(monkeypatch):
    fake_memory(monkeypatch, free_pct=67)
    assert voice_agent.memory_pressure_hint() is None


def test_hint_names_memory_when_wired_pages_have_eaten_the_machine(monkeypatch):
    # The state measured while a 25 GB model was loading: 3 % free, 32.1 GB wired.
    fake_memory(monkeypatch, free_pct=3)
    hint = voice_agent.memory_pressure_hint()
    assert hint is not None
    assert "3% of 36 GB free" in hint
    assert "32.1 GB of it wired" in hint
    assert "VT_LLM_CONTEXT" in hint  # tells the user what to actually do


def test_hint_survives_a_missing_vm_stat(monkeypatch):
    """The wired figure is a nicety; the free percentage is the decision."""
    def run(cmd, **kwargs):
        if cmd[0] == "vm_stat":
            raise OSError("no vm_stat here")
        class Result:
            stdout = ("The system has 38654705664 (x pages with a page size of 16384).\n"
                      "System-wide memory free percentage: 2%\n")
        return Result()
    monkeypatch.setattr(voice_agent.subprocess, "run", run)
    hint = voice_agent.memory_pressure_hint()
    assert hint and "2% of 36 GB free" in hint
    assert "of it wired" not in hint


def test_hint_never_raises_when_nothing_can_be_measured(monkeypatch):
    def boom(*a, **k):
        raise OSError("no memory_pressure here")
    monkeypatch.setattr(voice_agent.subprocess, "run", boom)
    assert voice_agent.memory_pressure_hint() is None


# --------------------------------------------------------------------------
# speak_worker
# --------------------------------------------------------------------------
class FakeTTS:
    sample_rate = 24000


@pytest.fixture
def session(monkeypatch):
    """Run speak_worker in its own generation, and tear it down after the test."""
    monkeypatch.setattr(voice_agent, "audio_q", queue.Queue())
    voice_agent.stop_event.clear()
    voice_agent.interrupt_event.clear()
    voice_agent.assistant_speaking.clear()
    voice_agent.synth_busy.clear()
    gen = voice_agent._generation
    thread = threading.Thread(
        target=voice_agent.speak_worker, args=(FakeTTS(), gen), daemon=True)
    thread.start()
    try:
        yield thread
    finally:
        voice_agent.stop_event.set()
        thread.join(timeout=2.0)
        voice_agent.stop_event.clear()


def put_sentence(samples=4096):
    voice_agent.audio_q.put((np.zeros(samples, dtype=np.float32), [], None))


def wait_until(predicate, timeout=2.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


def test_a_write_error_does_not_kill_the_thread(monkeypatch, session):
    made = stream_factory(monkeypatch, ["write", "ok"])

    put_sentence()
    assert wait_until(lambda: made and made[0].closed), "failing stream not discarded"
    assert session.is_alive()

    # The next sentence opens a fresh stream and is spoken in full.
    put_sentence(samples=4096)
    assert wait_until(lambda: len(made) == 2 and made[1].written == 4096)
    assert session.is_alive()


def test_an_unavailable_device_leaves_the_session_listening(monkeypatch, session):
    stream_factory(monkeypatch, ["start"] * 12)
    monkeypatch.setattr(voice_agent.time, "sleep", lambda s: None)
    monkeypatch.setattr(voice_agent.config, "AUDIO_OPEN_BACKOFF_OUTPUT", (0.0, 0.0, 0.0))

    put_sentence()
    assert wait_until(lambda: not voice_agent.assistant_speaking.is_set()
                      and voice_agent.audio_q.empty())
    assert session.is_alive()


def test_the_stream_is_opened_lazily(monkeypatch, session):
    made = stream_factory(monkeypatch, [])
    time.sleep(0.1)
    assert made == [], "no device should be touched before there is audio"
    put_sentence()
    assert wait_until(lambda: len(made) == 1 and made[0].written == 4096)
