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
  *(Barge-in was reworked after it was reported as not interrupting at all: the
  gate required consecutive loud frames, which short interjections never produce,
  and a successful interruption discarded the words that caused it. Now scored
  with decay, and the interrupting audio seeds the new utterance.)*
- Anti-hallucination guards for STT (energy gate, confidence scores, phrase
  blocklist).
- Central env-var-driven configuration (`config.py`).
- Offline verification script (`verify.py`) and setup/run scripts.

**Depends on:** nothing.

## Phase 1 — MVP hardening (v0.1)

Make the existing experience reliable and easy to adopt.

- Structured logging (levels, quiet/verbose modes) replacing ad-hoc prints.
- Graceful startup/shutdown and clear error messages when llama-server is down.
  *(Mostly delivered: the UI surfaces an "LLM unreachable" error, stopping a
  session interrupts playback and a streaming reply instead of leaving threads to
  unwind silently, and the web server explains a busy port rather than dumping a
  traceback. Logging levels are still open.)*
- Automated tests for the STT hallucination filters and sentence-splitting.
  *(Delivered: `tests/` runs 171 tests in about two seconds, without loading
  models or opening audio devices — also covering the viseme timeline, history
  trimming, persona switching, runtime settings, the barge-in decision, the HTTP
  layer over a real socket, and the avatar asset contract.)*
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
  *(Half delivered: the interrupting utterance is now captured in full instead of
  being dropped, so the tutor answers what you actually said. The interrupted
  reply is still stored in history exactly as far as the LLM streamed it, which
  can include a sentence that was synthesized but never heard — trimming history
  to what was actually spoken is still open.)*
- Latency instrumentation (per-stage timing) surfaced for tuning.

**Depends on:** Phase 1.

## Phase 3 — Broader reach (v0.3)

Widen who and how VirtualTutor can serve.

- Multilingual STT/TTS support (whisper multilingual + non-English Kokoro voices).
  *(Delivered: Whisper multilingual auto-detects the spoken language, and the
  detected language routes through the pipeline to select the correct Kokoro
  pipeline and voice. Supported languages: English (`af_heart`), Spanish
  (`ef_dora`), and Chinese (`zf_xiaoxiao`). The LLM replies in the same language
  the student uses. All phonemes produced by the Spanish and Chinese G2P are
  already covered by the viseme map, so lip-sync works across all three
  languages. Per-language voices are overridable via `VT_TTS_VOICE_ES` and
  `VT_TTS_VOICE_ZH`, or changed mid-session from the Configuration dialog.)*
- Selectable quality tiers (small/medium Whisper, 7B/14B LLM) with guidance on
  memory tradeoffs.
- Optional lightweight UI (transcript view, push-to-talk, voice/model pickers).
  *(Delivered: local web UI with start/stop, live transcript, mic level, a
  persona picker that works mid-session, and a lip-synced avatar — either a
  drawn face or a photo one, whose mouth blends the strongest visemes so motion
  stays smooth. The bundled sprites are committed, so a clone shows the photo
  face. A Configuration dialog picks the voice per language and switches duplex
  mode, both applying immediately. Push-to-talk and a model picker are still
  open — a model change means reloading weights, which is not a live setting.)*
- Full-duplex mode with software echo cancellation for speaker use without
  headphones.
  *(Full duplex itself is delivered and switchable from the Configuration dialog,
  but it still relies on an RMS gate and therefore on headphones. Echo
  cancellation — the part that would make speakers usable — is open.)*

**Depends on:** Phase 2.

## Phase 4 — Extensibility (Future)

- Pluggable backends (swap STT/LLM/TTS engines behind a common interface).
- Tool/skill hooks so the tutor can look things up or run exercises.
- Packaging as a distributable app or CLI installable outside the repo.
- Session analytics and progress tracking for learners.
- 3D avatar: the viseme timeline already uses VRM/ARKit shape names, so it could
  drive a rigged head instead of the 2D mouth.
- Avatar builder that measures its own geometry (face detection or an
  interactive picker) instead of per-clip constants.

**Depends on:** Phase 3.
