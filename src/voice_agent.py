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

import numpy as np
import requests
import sounddevice as sd
import torch
from silero_vad import load_silero_vad

import config
from tts import KokoroTTS

# ---------------------------------------------------------------------------
# Shared state
# ---------------------------------------------------------------------------
raw_q: "queue.Queue[np.ndarray]" = queue.Queue()      # 512-sample mic frames
utterance_q: "queue.Queue[np.ndarray]" = queue.Queue()  # complete user utterances
speak_q: "queue.Queue[str]" = queue.Queue()             # sentences to speak aloud

stop_event = threading.Event()          # global shutdown
assistant_speaking = threading.Event()  # set while TTS is playing
interrupt_event = threading.Event()     # set on barge-in to abort LLM + TTS

conversation = [{"role": "system", "content": config.SYSTEM_PROMPT}]
MAX_HISTORY_TURNS = 12  # keep the system prompt + last N messages

SENTENCE_END = re.compile(r"[.!?;:]+[\s\"')\]]*\s|[\n]+")


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
def vad_worker():
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

    def flush_input():
        """Drop any buffered mic frames (e.g. echo picked up during playback)."""
        with raw_q.mutex:
            raw_q.queue.clear()

    while not stop_event.is_set():
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
                        with speak_q.mutex:
                            speak_q.queue.clear()
                        speech_run = 0
                else:
                    speech_run = 0
            continue

        # --- Normal listening --------------------------------------------
        preroll.append(frame)
        prob = vad(torch.from_numpy(frame), config.SAMPLE_RATE).item()
        is_speech = prob >= config.VAD_THRESHOLD

        if is_speech:
            if not speaking:
                print("\n[VAD] speech detected, listening ...")
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
                buffer = []
                preroll.clear()
                vad.reset_states()


# ---------------------------------------------------------------------------
# 3. STT + LLM (the "brain")
# ---------------------------------------------------------------------------
def transcribe(audio: np.ndarray) -> str:
    """Transcribe an utterance, guarding against Whisper hallucinations.

    Whisper fabricates stock phrases ("Thank you", "Thanks for watching") when
    given near-silence or noise. We defend in three layers: an energy gate, the
    model's own no-speech / confidence scores, and a phrase blocklist.
    """
    import mlx_whisper

    # Layer 1: energy gate -- ignore near-silent buffers entirely.
    rms = float(np.sqrt(np.mean(np.square(audio)))) if audio.size else 0.0
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
    # that is entirely non-speech. Never drop individual mid-sentence segments --
    # that is what previously lost words. If at least one segment is real speech,
    # keep the complete transcription.
    segments = result.get("segments", []) or []
    if segments:
        any_speech = any(
            s.get("no_speech_prob", 0.0) <= config.STT_MAX_NO_SPEECH_PROB
            and s.get("avg_logprob", 0.0) >= config.STT_MIN_AVG_LOGPROB
            for s in segments
        )
        if not any_speech:
            print("[STT] rejected: entire utterance flagged non-speech/low-confidence")
            return ""
    text = result.get("text", "").strip()

    # Layer 3: blocklist of known hallucinated fillers.
    normalized = re.sub(r"[^\w\s]", "", text).strip().lower()
    if normalized in config.STT_HALLUCINATION_PHRASES:
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
    print("[Tutor]: ", end="", flush=True)
    try:
        resp = requests.post(config.LLAMA_SERVER_URL, json=payload, stream=True, timeout=120)
    except requests.RequestException as e:
        print(f"\n[LLM] cannot reach llama-server: {e}")
        conversation.pop()  # roll back the user turn
        return

    for line in resp.iter_lines():
        if interrupt_event.is_set():
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
        sentence += token
        full_reply += token
        m = SENTENCE_END.search(sentence)
        if m:
            complete, sentence = sentence[: m.end()], sentence[m.end():]
            if complete.strip():
                yield complete.strip()
    print()
    if sentence.strip() and not interrupt_event.is_set():
        yield sentence.strip()
    if full_reply.strip():
        conversation.append({"role": "assistant", "content": full_reply.strip()})
    # Trim history (keep system prompt at index 0)
    if len(conversation) > MAX_HISTORY_TURNS + 1:
        del conversation[1:len(conversation) - MAX_HISTORY_TURNS]


def brain_worker():
    while not stop_event.is_set():
        try:
            audio = utterance_q.get(timeout=0.2)
        except queue.Empty:
            continue

        interrupt_event.clear()
        text = transcribe(audio)
        if not text:
            continue
        print(f"\n[You]: {text}")
        for sentence in stream_llm(text):
            if interrupt_event.is_set():
                break
            speak_q.put(sentence)


# ---------------------------------------------------------------------------
# 4. TTS playback (interruptible)
# ---------------------------------------------------------------------------
def play_interruptible(audio: np.ndarray, sample_rate: int):
    if audio.size == 0:
        return
    block = 2048
    with sd.OutputStream(samplerate=sample_rate, channels=1, dtype="float32") as out:
        for i in range(0, len(audio), block):
            if interrupt_event.is_set() or stop_event.is_set():
                break
            out.write(audio[i:i + block])


def speak_worker(tts: KokoroTTS):
    while not stop_event.is_set():
        try:
            text = speak_q.get(timeout=0.2)
        except queue.Empty:
            continue
        if interrupt_event.is_set():
            continue
        try:
            audio = tts.synthesize(text)
        except Exception as e:
            print(f"\n[TTS] synthesis error: {e}")
            continue
        assistant_speaking.set()
        try:
            play_interruptible(audio, tts.sample_rate)
        finally:
            # small tail so the last block flushes before we resume listening
            if not speak_q.qsize():
                assistant_speaking.clear()


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
# Main
# ---------------------------------------------------------------------------
def main():
    persona_key = select_persona()
    persona = config.PERSONAS[persona_key]
    # Apply the chosen persona to the conversation's system prompt.
    conversation[0]["content"] = config.build_system_prompt(persona_key)

    print(f"\nInitializing VirtualTutor as {persona['name']} ...")
    tts = KokoroTTS()

    threads = [
        threading.Thread(target=vad_worker, name="vad", daemon=True),
        threading.Thread(target=brain_worker, name="brain", daemon=True),
        threading.Thread(target=speak_worker, args=(tts,), name="speak", daemon=True),
    ]
    for t in threads:
        t.start()

    print("\n\U0001F393  VirtualTutor is listening. Speak into your mic. Press Ctrl+C to quit.\n")
    try:
        with sd.InputStream(samplerate=config.SAMPLE_RATE, channels=config.CHANNELS,
                            dtype="float32", blocksize=config.FRAME_SIZE,
                            callback=audio_callback):
            while not stop_event.is_set():
                sd.sleep(200)
    except KeyboardInterrupt:
        print("\nShutting down ...")
    finally:
        stop_event.set()


if __name__ == "__main__":
    main()
