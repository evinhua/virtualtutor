"""VirtualTutor - real-time interactive voice conversation pipeline.

Pipeline:  Microphone -> Silero VAD -> mlx-whisper (STT) -> llama.cpp (LLM)
           -> sentence buffer -> Kokoro (TTS) -> Speakers

Each stage runs in its own thread and communicates via queues, so the tutor
can keep listening while it thinks and speaks. Speaking while the tutor talks
triggers "barge-in": playback stops and the tutor listens to you instead.
"""
import json
import os
import queue
import re
import sys
import threading
import time

import numpy as np
import requests
import sounddevice as sd
import torch
from silero_vad import load_silero_vad

import config
import visemes
from tts import KokoroTTS

# ---------------------------------------------------------------------------
# Shared state
# ---------------------------------------------------------------------------
raw_q: "queue.Queue[np.ndarray]" = queue.Queue()      # 512-sample mic frames
utterance_q: "queue.Queue[np.ndarray]" = queue.Queue()  # complete user utterances
speak_q: "queue.Queue[str]" = queue.Queue()             # sentences to speak aloud
audio_q: "queue.Queue[tuple]" = queue.Queue()            # (audio, viseme timeline) ready to play

stop_event = threading.Event()          # global shutdown
assistant_speaking = threading.Event()  # set while TTS is playing
interrupt_event = threading.Event()     # set on barge-in to abort LLM + TTS
synth_busy = threading.Event()          # set while TTS synthesis is in flight

# UI events for the optional web frontend (see server.py). Bounded so a session
# running without a listener cannot grow without limit; oldest events are dropped.
event_q: "queue.Queue[dict]" = queue.Queue(maxsize=512)


def emit(kind: str, **data):
    """Publish a UI event, discarding the oldest if nobody is draining the queue."""
    event = {"type": kind, **data}
    try:
        event_q.put_nowait(event)
    except queue.Full:
        try:
            event_q.get_nowait()
            event_q.put_nowait(event)
        except (queue.Empty, queue.Full):
            pass

conversation = [{"role": "system", "content": config.SYSTEM_PROMPT}]
# Persona currently in force. Kept here (not just in the system prompt) so the
# CLI, the web UI and the transcript can all label replies correctly.
active_persona = config.PERSONA
# History is trimmed in blocks rather than every turn. Dropping the oldest
# message each turn changes the token prefix after the system prompt, which
# invalidates llama.cpp's KV cache and forces a full re-prefill of the whole
# conversation (measured: 511 ms / 179 tokens versus 35 ms / 1 token when the
# prefix is untouched). Trimming only once we exceed HISTORY_HIGH_WATER, and
# then cutting back to HISTORY_LOW_WATER, amortizes that one-off cost over
# many turns instead of paying it on every reply.
HISTORY_HIGH_WATER = 24  # messages after the system prompt before trimming
HISTORY_LOW_WATER = 12   # messages kept after a trim

SENTENCE_END = re.compile(r"[.!?;:]+[\s\"')\]]*\s|[\n]+")

# ---------------------------------------------------------------------------
# Session generation
# ---------------------------------------------------------------------------
# Workers are told which session they belong to and stop as soon as a new one
# starts. Without this, a worker still finishing a reply when the user pressed
# Stop could survive into the next session -- two speak_workers fighting over
# the output device, and a stale system prompt (i.e. the old persona) in play.
_generation = 0


def session_active(gen: int) -> bool:
    """True while `gen` is still the current session and no stop was requested."""
    return not stop_event.is_set() and gen == _generation


def set_persona(persona_key: str) -> str:
    """Switch persona, in or out of a running session. Returns the key applied.

    The system prompt is rewritten in place, so the next reply is in the new
    personality; history is kept so the conversation still makes sense.
    """
    global active_persona
    if persona_key not in config.PERSONAS:
        persona_key = config.DEFAULT_PERSONA
    active_persona = persona_key
    conversation[0]["content"] = config.build_system_prompt(persona_key)
    emit("persona", key=persona_key, name=config.persona_name(persona_key))
    return persona_key



# ---------------------------------------------------------------------------
# 1. Microphone capture
# ---------------------------------------------------------------------------
def audio_callback(indata, frames, time_info, status):
    if status:
        print(status, file=sys.stderr)
    raw_q.put(indata.copy().reshape(-1))


# ---------------------------------------------------------------------------
# 2. Voice Activity Detection
# ---------------------------------------------------------------------------
def vad_worker(gen: int):
    """Detect speech segments and enqueue complete utterances.

    While the assistant is speaking, detected speech triggers barge-in instead
    of being recorded, so the tutor stops talking and waits for the new query.
    """
    vad = load_silero_vad()
    max_silence = int(config.VAD_SILENCE_DURATION * config.SAMPLE_RATE / config.FRAME_SIZE)
    min_speech = int(config.VAD_MIN_SPEECH_DURATION * config.SAMPLE_RATE / config.FRAME_SIZE)
    preroll_len = max(1, int(config.VAD_PREROLL_DURATION * config.SAMPLE_RATE / config.FRAME_SIZE))
    # consecutive loud speech frames needed to treat sound as a real interruption
    barge_in_frames = int(0.4 * config.SAMPLE_RATE / config.FRAME_SIZE)

    from collections import deque
    preroll = deque(maxlen=preroll_len)  # recent frames captured before speech starts
    buffer: list[np.ndarray] = []
    speaking = False
    silence = 0
    speech_run = 0
    was_assistant_speaking = False
    # Emit a mic level roughly 10x/second rather than once per 512-sample frame.
    level_every = max(1, int(config.SAMPLE_RATE / config.FRAME_SIZE / 10))
    level_ticks = 0

    def flush_input():
        """Drop any buffered mic frames (e.g. echo picked up during playback)."""
        with raw_q.mutex:
            raw_q.queue.clear()

    while session_active(gen):
        try:
            frame = raw_q.get(timeout=0.2)
        except queue.Empty:
            continue

        # When the tutor JUST stopped speaking, discard echo/tail that leaked into
        # the mic during playback so it is not mistaken for (or merged into) speech.
        if was_assistant_speaking and not assistant_speaking.is_set():
            flush_input()
            preroll.clear(); buffer = []
            speaking = False; silence = 0; speech_run = 0
            vad.reset_states()
            was_assistant_speaking = False
            continue

        # --- While the tutor is speaking: half-duplex, optional barge-in ---
        if assistant_speaking.is_set():
            was_assistant_speaking = True
            if config.ENABLE_BARGE_IN:
                rms = float(np.sqrt(np.mean(np.square(frame))))
                prob = vad(torch.from_numpy(frame), config.SAMPLE_RATE).item()
                if prob >= config.VAD_THRESHOLD and rms >= config.BARGE_IN_MIN_RMS:
                    speech_run += 1
                    if speech_run >= barge_in_frames:
                        print("\n[VAD] barge-in detected -> interrupting tutor")
                        interrupt_event.set()
                        drain_queue(speak_q)
                        drain_queue(audio_q)
                        emit("interrupted")
                        speech_run = 0
                else:
                    speech_run = 0
            continue

        # --- Normal listening --------------------------------------------
        preroll.append(frame)
        prob = vad(torch.from_numpy(frame), config.SAMPLE_RATE).item()
        is_speech = prob >= config.VAD_THRESHOLD

        # Feed the UI a coarse input level (~10 Hz) to drive the background waves.
        level_ticks += 1
        if level_ticks >= level_every:
            level_ticks = 0
            emit("level", value=round(float(np.sqrt(np.mean(np.square(frame)))), 4))

        if is_speech:
            if not speaking:
                print("\n[VAD] speech detected, listening ...")
                emit("state", state="hearing")
                speaking = True
                # seed with pre-roll so the onset (first word) is not clipped
                buffer = list(preroll)
            else:
                buffer.append(frame)
            silence = 0
        elif speaking:
            buffer.append(frame)
            silence += 1
            if silence > max_silence:
                speaking = False
                silence = 0
                if len(buffer) >= min_speech:
                    audio = np.concatenate(buffer).astype(np.float32)
                    utterance_q.put(audio)
                    emit("state", state="thinking")
                else:
                    emit("state", state="listening")
                buffer = []
                preroll.clear()
                vad.reset_states()


# ---------------------------------------------------------------------------
# 3. STT + LLM (the "brain")
# ---------------------------------------------------------------------------
def utterance_rms(audio: np.ndarray) -> float:
    """Root-mean-square level of an utterance (0.0 for an empty buffer)."""
    if audio is None or audio.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(audio))))


def has_real_speech(segments) -> bool:
    """True if at least one Whisper segment looks like genuine speech.

    Used to reject a WHOLE utterance that is entirely non-speech. Individual
    mid-sentence segments are never dropped -- doing so previously lost words.
    An empty segment list is inconclusive, so it counts as speech and the
    phrase blocklist gets the final say.
    """
    segments = segments or []
    if not segments:
        return True
    return any(
        s.get("no_speech_prob", 0.0) <= config.STT_MAX_NO_SPEECH_PROB
        and s.get("avg_logprob", 0.0) >= config.STT_MIN_AVG_LOGPROB
        for s in segments
    )


def is_hallucination(text: str) -> bool:
    """True if `text` is one of Whisper's stock filler phrases.

    Compared after stripping punctuation and case, so "Thank you." and
    "thank you" both match.
    """
    normalized = re.sub(r"[^\w\s]", "", text or "").strip().lower()
    return normalized in config.STT_HALLUCINATION_PHRASES


def next_sentence(buffer: str):
    """Split the first complete sentence off `buffer`.

    Returns (sentence, remainder); sentence is None while the buffer does not
    yet hold a sentence end. Punctuation must be followed by whitespace or a
    newline, so "3.14" and mid-word colons do not split the stream early.
    """
    m = SENTENCE_END.search(buffer)
    if not m:
        return None, buffer
    return buffer[: m.end()].strip(), buffer[m.end():]


def transcribe(audio: np.ndarray) -> str:
    """Transcribe an utterance, guarding against Whisper hallucinations.

    Whisper fabricates stock phrases ("Thank you", "Thanks for watching") when
    given near-silence or noise. We defend in three layers: an energy gate, the
    model's own no-speech / confidence scores, and a phrase blocklist.
    """
    import mlx_whisper

    # Layer 1: energy gate -- ignore near-silent buffers entirely.
    rms = utterance_rms(audio)
    if rms < config.STT_MIN_RMS:
        print(f"[STT] skipped near-silence (rms={rms:.4f})")
        return ""

    # Robust decoding: greedy, no cross-segment priming (a hallucination amplifier).
    result = mlx_whisper.transcribe(
        audio,
        path_or_hf_repo=config.WHISPER_MODEL,
        temperature=0.0,
        condition_on_previous_text=False,
        no_speech_threshold=0.6,
        logprob_threshold=-1.0,
        compression_ratio_threshold=2.4,
    )

    # Layer 2: trust Whisper's own scores, but only to reject a WHOLE utterance
    # that is entirely non-speech (see has_real_speech).
    if not has_real_speech(result.get("segments")):
        print("[STT] rejected: entire utterance flagged non-speech/low-confidence")
        return ""
    text = result.get("text", "").strip()

    # Layer 3: blocklist of known hallucinated fillers.
    if is_hallucination(text):
        print(f"[STT] rejected hallucination: {text!r}")
        return ""

    return text


def stream_llm(prompt: str):
    """Stream the LLM reply, yielding complete sentences for TTS."""
    conversation.append({"role": "user", "content": prompt})
    payload = {
        "messages": conversation,
        "stream": True,
        "temperature": config.LLM_TEMPERATURE,
        "max_tokens": config.LLM_MAX_TOKENS,
    }
    sentence = ""
    full_reply = ""
    print(f"[{config.persona_name(active_persona)}]: ", end="", flush=True)
    try:
        resp = requests.post(config.LLAMA_SERVER_URL, json=payload, stream=True, timeout=120)
    except requests.RequestException as e:
        print(f"\n[LLM] cannot reach llama-server: {e}")
        emit("error", text="Cannot reach the language model server. "
                           "Start it with ./scripts/start_server.sh")
        conversation.pop()  # roll back the user turn
        return

    for line in resp.iter_lines():
        if interrupt_event.is_set() or stop_event.is_set():
            resp.close()
            break
        if not line:
            continue
        decoded = line.decode("utf-8").removeprefix("data: ").strip()
        if decoded == "[DONE]":
            break
        try:
            data = json.loads(decoded)
        except json.JSONDecodeError:
            continue
        token = data["choices"][0]["delta"].get("content", "")
        if not token:
            continue
        print(token, end="", flush=True)
        emit("assistant_delta", text=token)
        sentence += token
        full_reply += token
        complete, sentence = next_sentence(sentence)
        if complete:
            yield complete
    print()
    if sentence.strip() and not interrupt_event.is_set():
        yield sentence.strip()
    if full_reply.strip():
        conversation.append({"role": "assistant", "content": full_reply.strip()})
    emit("assistant_done")
    trim_history()


def trim_history():
    """Drop the oldest turns, but only once history grows past the high water mark.

    Keeps the system prompt at index 0 and always resumes at a "user" message so
    the transcript stays a valid user/assistant alternation.
    """
    if len(conversation) - 1 <= HISTORY_HIGH_WATER:
        return
    cut = len(conversation) - HISTORY_LOW_WATER
    # Advance to the next user turn so we never lead with a bare assistant reply.
    while cut < len(conversation) and conversation[cut]["role"] != "user":
        cut += 1
    if cut < len(conversation):
        del conversation[1:cut]


def brain_worker(gen: int):
    while session_active(gen):
        try:
            audio = utterance_q.get(timeout=0.2)
        except queue.Empty:
            continue

        interrupt_event.clear()
        text = transcribe(audio)
        # Transcription takes a second or two, in which the user may have
        # pressed Stop or switched session; do not start a reply into the void.
        if not session_active(gen):
            return
        if not text:
            emit("state", state="listening")
            continue
        print(f"\n[You]: {text}")
        emit("user", text=text)
        for sentence in stream_llm(text):
            if interrupt_event.is_set() or not session_active(gen):
                break
            speak_q.put(sentence)


# ---------------------------------------------------------------------------
# 4. TTS synthesis + playback (interruptible, pipelined)
# ---------------------------------------------------------------------------
def drain_queue(q: queue.Queue):
    """Discard everything currently queued (used on barge-in)."""
    with q.mutex:
        q.queue.clear()


def play_interruptible(out: sd.OutputStream, audio: np.ndarray):
    """Write `audio` to an already-open stream, stopping early on barge-in.

    Note: write() returns once PortAudio has accepted the samples, not once they
    have been heard -- roughly `out.latency` of audio is still buffered when the
    final write returns. Callers must account for that tail before treating the
    speaker as silent.
    """
    if audio.size == 0:
        return
    block = 2048
    if out.stopped:
        out.start()  # ~26 ms; only needed after a barge-in abort()
    for i in range(0, len(audio), block):
        if interrupt_event.is_set() or stop_event.is_set():
            break
        out.write(audio[i:i + block])


def synth_worker(tts: KokoroTTS, gen: int):
    """Synthesize queued sentences into audio, ahead of playback.

    Runs in its own thread so synthesis of the next sentence overlaps playback
    of the current one. Kokoro synthesizes at RTF ~0.05, so a sentence is ready
    far sooner than it takes to speak the previous one and the inter-sentence
    gap disappears entirely.

    Each item is (audio, viseme_timeline); the timeline comes from the same
    forward pass as the audio, so lip-sync costs nothing extra.
    """
    while session_active(gen):
        try:
            text = speak_q.get(timeout=0.2)
        except queue.Empty:
            continue
        if interrupt_event.is_set():
            continue
        # synth_busy stays set until the audio is queued, so the player never
        # sees "both queues empty" while a sentence is still being produced.
        synth_busy.set()
        try:
            audio, timeline = tts.synthesize_with_visemes(text)
            if audio.size and not interrupt_event.is_set():
                audio_q.put((audio, timeline))
        except Exception as e:
            print(f"\n[TTS] synthesis error: {e}")
        finally:
            synth_busy.clear()


def more_speech_coming() -> bool:
    """True while any sentence is queued, being synthesized, or ready to play."""
    return (not audio_q.empty()) or (not speak_q.empty()) or synth_busy.is_set()


def speak_worker(tts: KokoroTTS, gen: int):
    """Play synthesized audio through one long-lived output stream.

    One stream is reused for the whole session. Per sentence this avoids the
    ~190 ms that closing a stream spends draining its buffer (constructing a
    stream is only ~10 ms; the drain was the real cost), measured as 690 ms vs
    500 ms of wall time to play 500 ms of audio.
    """
    out = sd.OutputStream(samplerate=tts.sample_rate, channels=1, dtype="float32")
    out.start()
    # Monotonic time at which the audio already handed to PortAudio finishes.
    # Used to predict when the next sentence will actually be *heard*, so the
    # browser can start the mouth animation at the right moment.
    playhead = 0.0
    try:
        while session_active(gen):
            try:
                audio, timeline = audio_q.get(timeout=0.2)
            except queue.Empty:
                continue
            if interrupt_event.is_set():
                continue

            assistant_speaking.set()
            emit("state", state="speaking")

            # write() returns once PortAudio accepts the samples, so this
            # sentence starts being heard either after the buffer latency (if
            # the device is idle) or when the previous sentence finishes.
            now = time.monotonic()
            starts_at = max(now + out.latency, playhead)
            if timeline:
                emit("visemes",
                     start_in_ms=int(round((starts_at - now) * 1000)),
                     timeline=visemes.to_wire(timeline))
            playhead = starts_at + audio.size / tts.sample_rate

            play_interruptible(out, audio)

            if interrupt_event.is_set():
                # Barge-in: drop queued audio and cut playback immediately.
                # abort() discards PortAudio's ~121 ms buffer where stop() would
                # drain it (~190 ms), i.e. keep talking over the user.
                drain_queue(audio_q)
                out.abort()
                playhead = 0.0  # buffer discarded, nothing is queued to be heard
                assistant_speaking.clear()
                emit("state", state="listening")
                continue

            # Only release the mic once nothing else is coming AND the audio
            # still buffered in PortAudio has actually been heard. Clearing
            # earlier re-opens the mic while the speaker is still sounding,
            # which feeds the tutor's own tail back in as user speech.
            if not more_speech_coming():
                time.sleep(out.latency)
                if not more_speech_coming():
                    assistant_speaking.clear()
                    emit("state", state="listening")
    finally:
        out.close(ignore_errors=True)


# ---------------------------------------------------------------------------
# Persona selection
# ---------------------------------------------------------------------------
def select_persona() -> str:
    """Let the user pick a tutor persona at startup.

    Skips the prompt when VT_PERSONA is set to a valid persona or when stdin is
    not interactive (falls back to config.PERSONA / the default).
    """
    keys = list(config.PERSONAS.keys())

    # Honor an explicit, valid env-var choice without prompting.
    if os.environ.get("VT_PERSONA", "").strip().lower() in config.PERSONAS:
        chosen = config.PERSONA
        print(f"Persona (from VT_PERSONA): {config.PERSONAS[chosen]['name']}")
        return chosen

    # Non-interactive stdin (piped/automated): use the configured default.
    if not sys.stdin.isatty():
        return config.PERSONA

    print("Choose a tutor persona:\n")
    for i, key in enumerate(keys, 1):
        p = config.PERSONAS[key]
        default_tag = " (default)" if key == config.DEFAULT_PERSONA else ""
        print(f"  {i}. {p['name']}{default_tag}\n     {p['blurb']}")
    print()

    default_idx = keys.index(config.DEFAULT_PERSONA) + 1
    while True:
        try:
            raw = input(f"Enter number [1-{len(keys)}] (default {default_idx}): ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return config.DEFAULT_PERSONA
        if not raw:
            return config.DEFAULT_PERSONA
        if raw.isdigit() and 1 <= int(raw) <= len(keys):
            return keys[int(raw) - 1]
        print(f"  Please enter a number between 1 and {len(keys)}.")


# ---------------------------------------------------------------------------
# Pipeline lifecycle (used by both the CLI and the web server)
# ---------------------------------------------------------------------------
_threads: list[threading.Thread] = []
_input_stream = None
_tts: "KokoroTTS | None" = None
_lifecycle_lock = threading.Lock()


def is_running() -> bool:
    return any(t.is_alive() for t in _threads)


def load_tts() -> KokoroTTS:
    """Load Kokoro once and reuse it, so restarting does not re-pay model load."""
    global _tts
    if _tts is None:
        _tts = KokoroTTS()
    return _tts


def start_pipeline(persona_key: str = config.DEFAULT_PERSONA, reset_history: bool = True):
    """Start mic capture and all worker threads.

    If a session is already live this switches persona on it rather than doing
    nothing, so the caller's choice always takes effect.
    """
    global _threads, _input_stream, _generation
    with _lifecycle_lock:
        if is_running():
            set_persona(persona_key)
            return
        # Fresh state for a new session. Bumping the generation retires any
        # worker from a previous session that has not finished unwinding yet.
        _generation += 1
        gen = _generation
        stop_event.clear()
        interrupt_event.clear()
        assistant_speaking.clear()
        synth_busy.clear()
        for q in (raw_q, utterance_q, speak_q, audio_q):
            drain_queue(q)

        set_persona(persona_key)
        if reset_history:
            del conversation[1:]

        tts = load_tts()
        _threads = [
            threading.Thread(target=vad_worker, args=(gen,), name="vad", daemon=True),
            threading.Thread(target=brain_worker, args=(gen,), name="brain", daemon=True),
            threading.Thread(target=synth_worker, args=(tts, gen), name="synth", daemon=True),
            threading.Thread(target=speak_worker, args=(tts, gen), name="speak", daemon=True),
        ]
        for t in _threads:
            t.start()

        _input_stream = sd.InputStream(
            samplerate=config.SAMPLE_RATE, channels=config.CHANNELS,
            dtype="float32", blocksize=config.FRAME_SIZE, callback=audio_callback,
        )
        _input_stream.start()
        emit("state", state="listening")


def stop_pipeline():
    """Stop mic capture and shut the worker threads down.

    A worker can be deep inside a blocking call (Whisper decoding, or an LLM
    response still streaming), so joining is best-effort: `interrupt_event`
    unblocks playback and the HTTP stream, and any straggler is retired by the
    generation bump on the next start. `_threads` is cleared either way, so a
    slow shutdown can never make the next start a silent no-op -- that used to
    leave the session running under its original persona.
    """
    global _threads, _input_stream
    with _lifecycle_lock:
        stop_event.set()
        interrupt_event.set()
        if _input_stream is not None:
            try:
                _input_stream.stop()
                _input_stream.close()
            except Exception as e:
                print(f"[audio] input stream close: {e}")
            _input_stream = None
        deadline = time.monotonic() + 5.0
        for t in _threads:
            t.join(timeout=max(0.1, deadline - time.monotonic()))
        stragglers = [t.name for t in _threads if t.is_alive()]
        if stragglers:
            print(f"[pipeline] still unwinding in the background: {', '.join(stragglers)}")
        _threads = []
        assistant_speaking.clear()
        interrupt_event.clear()
        for q in (raw_q, utterance_q, speak_q, audio_q):
            drain_queue(q)
        emit("state", state="idle")


# ---------------------------------------------------------------------------
# Main (CLI)
# ---------------------------------------------------------------------------
def main():
    persona_key = select_persona()
    persona = config.PERSONAS[persona_key]

    print(f"\nInitializing VirtualTutor as {persona['name']} ...")
    start_pipeline(persona_key)

    print("\n\U0001F393  VirtualTutor is listening. Speak into your mic. Press Ctrl+C to quit.\n")
    try:
        while not stop_event.is_set():
            sd.sleep(200)
    except KeyboardInterrupt:
        print("\nShutting down ...")
    finally:
        stop_pipeline()


if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()
