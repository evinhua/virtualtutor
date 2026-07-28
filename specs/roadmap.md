# Roadmap

Priority order is top-down: each phase builds on the previous one.

## Phase 0 — Foundation (Done)

The current working pipeline.

- Threaded pipeline: microphone → Silero VAD → mlx-whisper → llama.cpp →
  sentence buffer → Kokoro TTS → speakers.
- End-of-speech detection with configurable silence duration and pre-roll so the
  first word is not clipped.
- Streaming LLM replies split into sentences and spoken as they arrive.
- Half-duplex playback with echo flushing; opt-in barge-in for headphones.
- Anti-hallucination guards for STT (energy gate, confidence scores, phrase
  blocklist).
- Central env-var-driven configuration (`config.py`).
- Offline verification script (`verify.py`) and setup/run scripts.

**Depends on:** nothing.

## Phase 1 — MVP hardening (v0.1)

Make the existing experience reliable and easy to adopt.

- Structured logging (levels, quiet/verbose modes) replacing ad-hoc prints.
- Graceful startup/shutdown and clear error messages when llama-server is down.
- Automated tests for the STT hallucination filters and sentence-splitting.
  *(Delivered: `tests/` runs 99 model-free unit tests in under a second, also
  covering the viseme timeline, history trimming and persona prompts.)*
- A single launcher that starts the LLM server and agent together.
  *(Delivered: `scripts/start_all.sh`.)*
- Device selection / listing for input and output audio devices.
- Documented troubleshooting guide (mic permissions, portaudio, echo).

**Depends on:** Phase 0.

## Phase 2 — Conversation quality (v0.2)

Improve the tutoring experience itself.

- Persistent conversation memory across sessions (save/restore history).
- Configurable tutor personas / subjects via prompt presets.
  *(Personas delivered: startup menu + `VT_PERSONA` env var — tutor, jester,
  cheerleader, explorer. Switchable mid-session from the web UI, and the tutor
  identifies itself by persona when asked.)*
- Smarter history management (token-aware trimming, summarization of old turns).
- Interruption-aware context so barge-in edits the ongoing turn cleanly.
- Latency instrumentation (per-stage timing) surfaced for tuning.

**Depends on:** Phase 1.

## Phase 3 — Broader reach (v0.3)

Widen who and how VirtualTutor can serve.

- Multilingual STT/TTS support (whisper multilingual + non-English Kokoro voices).
- Selectable quality tiers (small/medium Whisper, 7B/14B LLM) with guidance on
  memory tradeoffs.
- Optional lightweight UI (transcript view, push-to-talk, voice/model pickers).
- Full-duplex mode with software echo cancellation for speaker use without
  headphones.

**Depends on:** Phase 2.

## Phase 4 — Extensibility (Future)

- Pluggable backends (swap STT/LLM/TTS engines behind a common interface).
- Tool/skill hooks so the tutor can look things up or run exercises.
- Packaging as a distributable app or CLI installable outside the repo.
- Session analytics and progress tracking for learners.

**Depends on:** Phase 3.
