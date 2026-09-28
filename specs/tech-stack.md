# Tech Stack

## Platform

- **Hardware:** Apple Silicon (tested on M3 Pro / 36 GB unified memory).
- **OS:** macOS.
- **Runtime:** Python 3.11 in a local virtual environment (`.venv`).

All four models sit in unified memory at once, which is what makes low-latency,
fully local operation possible.

## Language

- **Python 3.11** — single language for the whole pipeline. Chosen for its
  first-class support in the MLX and ML tooling ecosystem.

## Pipeline components

| Stage | Technology | Rationale |
|-------|-----------|-----------|
| VAD (voice activity detection) | Silero VAD v5 (`silero-vad`, via `torch`) | Accurate, lightweight, fully offline end-of-speech detection at 16 kHz. The same per-frame probability also decides barge-in in full duplex. |
| STT (speech-to-text) | `mlx-whisper` — `whisper-small` (multilingual) | Metal-accelerated Whisper on Apple Silicon; multilingual model auto-detects English, Spanish and Chinese. |
| LLM | **Ollama** serving `Qwen3.8-Uncensored:latest` (27 B Qwen3, Q4_K_M); `llama.cpp` + a local GGUF as a second backend (`VT_LLM_BACKEND=llamacpp`) | Both expose the same OpenAI-compatible streaming endpoint, so only the URL, the model name and how thinking is switched off differ (`config.llm_payload`). Ollama sends `reasoning_effort: "none"` per request, llama.cpp is started with `--reasoning-budget 0`; either way the agent also strips `<think>` blocks, because a spoken tutor cannot afford seconds of silent working-out. |
| Mood | `src/emotion.py` | Kokoro has no emotion input, so each sentence gets a mood — from a cue the model wrote, from signals in the text, or from the persona's baseline — which sets rate, pitch, loudness, the pauses around it and how wide the avatar articulates. |
| TTS (text-to-speech) | Kokoro-82M via `mlx-audio` (24 kHz, multilingual) | Small, natural-sounding, Metal-accelerated local TTS. Voices are listed per language in `config.AVAILABLE_VOICES` and selectable at runtime; the default is `af_heart` (EN), `ef_dora` (ES), `zf_xiaoxiao` (ZH). Text is chunked and stripped of markdown before synthesis — see *Nothing reaches the speaker unsayable* under Architecture. |
| Lip-sync | Kokoro's own duration predictor (`src/visemes.py`) | The frame count per phoneme falls out of the normal forward pass, so a frame-accurate viseme timeline costs no extra inference and no audio analysis. |
| Frontend | Python stdlib `ThreadingHTTPServer` + Server-Sent Events, static HTML/CSS/JS | A control surface and transcript view with zero added dependencies and no build step. Audio stays in Python, which preserves the half-duplex echo handling. |

## Frontend & avatar

- **No framework, no bundler.** `web/` is three static files served by
  `src/server.py`; the browser receives state, transcript deltas, mic level and
  viseme timelines over one SSE stream.
- **Settings are a dialog, not a restart.** A native `<dialog>` holds a voice
  dropdown per language and a half/full duplex switch, populated from
  `/api/config`. It has no Save button because the pipeline reads both settings
  at the point of use, so a change applies immediately; see *Runtime settings*
  under Architecture. Changes are broadcast over the same SSE stream, so a second
  tab does not go stale.
- **Viseme naming follows VRM/ARKit** (`aa`, `ih`, `ou`, `oh`, `E`, plus `PP`,
  `FF`, `TH`, `DD`, `SS`, `CH`, `KK`, `RR`), so the same timeline could drive a
  3D avatar later.
- **Two renderers.** A vector face is the default and needs no assets; if
  `web/avatar/manifest.json` exists, a photo avatar is used instead — one base
  face plus a mouth patch per viseme, feathered in through an elliptical mask.
- **The mouth has inertia.** A phoneme-accurate timeline contains many 25 ms
  spans, so switching shape per span (or restarting a crossfade on each change)
  flickers. Each viseme instead holds a weight that rises while it is the target
  (55 ms) and decays afterwards (95 ms), and the mouth is the weighted blend of
  the three strongest — a brief consonant only partly reaches its shape, as in
  real articulation. Patches are therefore stored unmasked so several can be
  averaged before the mask is applied once.
- **Animation runs on elapsed time,** not per-frame fractions, so it looks the
  same at 60 Hz and 120 Hz and cannot lurch when a tab is throttled.
- **Avatar authoring lives in `tools/`** and is not needed at runtime. Alignment
  uses `torch.grid_sample` and normalised cross-correlation over a scale/rotation
  grid rather than a face-landmark library, which keeps `requirements.txt`
  unchanged (`torch` is already there for Silero, `ffmpeg`/`ffprobe` for I/O).
  The search is coarse-to-fine inside a window: a full-resolution correlation
  per candidate pose would need tens of gigabytes.

## Key libraries & tools

- **`numpy`** — audio buffer math throughout the pipeline.
- **`sounddevice`** (+ system `portaudio`) — microphone capture and speaker
  playback; playback is chunked to allow mid-sentence interruption.
- **`soundfile`** — WAV read/write for the verification script.
- **`torch`** — backend required by Silero VAD.
- **`requests`** — streaming HTTP client for the backend's OpenAI-compatible API
  (Ollama by default, llama.cpp on request).
- **`misaki[en]`, `misaki[zh]`** — grapheme-to-phoneme text processing required by Kokoro for English and Chinese.
- **`espeak-ng`, `ffmpeg`** — system dependencies (installed via Homebrew).
- **spaCy `en_core_web_sm`** — downloaded on first run for text processing.

## Architecture

- **Threaded, queue-based pipeline.** Four worker threads — VAD, brain
  (STT+LLM), synthesis, and playback — communicate via `queue.Queue` and
  coordinate through `threading.Event` flags (`stop_event`,
  `assistant_speaking`, `interrupt_event`, `synth_busy`). Splitting synthesis
  from playback lets the next sentence be generated while the current one is
  still being spoken.
- **Streaming end to end.** LLM tokens are split into sentences on the fly and
  handed to TTS immediately, minimizing perceived latency. The splitter is
  language-aware: CJK terminators (`。！？；：…`) end a sentence on their own,
  because Chinese has no trailing space and no ASCII punctuation, and without
  them a whole Chinese reply reached TTS as a single block.
- **Nothing reaches the speaker unsayable.** Two guards sit in front of Kokoro,
  both of them because the failure is silent rather than an exception. Text is
  chunked to a per-pipeline character budget (`config.TTS_MAX_CHUNK_CHARS`), since
  mlx-audio truncates anything over 510 phonemes and simply drops the rest of the
  sentence — Chinese runs ~4.1 phonemes per character against Spanish's ~1.1, so
  one character budget cannot serve both. And markdown decoration is stripped
  (`tts.speakable()`), because misaki phonemizes `*` as the word "asterisk": an
  emphasised `*boss*` is otherwise spoken as "asterisk boss asterisk".
- **Reasoning is off, and filtered anyway.** Qwen3 thinks before answering by
  default, which is dead air in a voice loop. Thinking is closed immediately per
  request on Ollama (`reasoning_effort: "none"`) or at startup on llama.cpp
  (`--reasoning-budget 0`), and `ThinkFilter` strips `<think>` blocks from the
  token stream, matching tags split across tokens — Ollama keeps reasoning in a
  separate `reasoning` delta the agent never reads, but llama.cpp can land the
  tags in the reply text, so the client cannot assume they will not appear.
- **Mood cues never reach the speaker.** The model is asked to prefix a reply
  with `[excited]` or `[gentle]`; `CueFilter` pulls that out of the stream before
  anything is spoken, shown or stored, and drops cue-shaped words it invented
  (`[natural]`) rather than reading them aloud. A cue applies to the sentence it
  introduces, after which the text's own signals take over again.
- **Half-duplex by default.** The agent does not listen while speaking and
  flushes echo picked up during playback. Full duplex (barge-in) is opt-in and
  assumes headphones.
- **Barge-in is scored, not counted.** Interrupting used to require an unbroken
  run of frames above the VAD and RMS thresholds, which speech does not provide:
  it dips between words and on plosives, and each dip restarted the run. So
  evidence accumulates and decays instead (`BargeInDetector`,
  `BARGE_IN_SPEECH_DURATION`, `BARGE_IN_DECAY`), which tolerates the dips while
  still ignoring a lone spike, since a spike decays before it can reach the
  threshold.
- **An interruption keeps its own words.** Full duplex holds a longer pre-roll
  than a normal speech onset needs (`BARGE_IN_PREROLL_DURATION`, 1.2 s vs 0.4 s),
  because confirming a barge-in takes time; the new utterance is seeded from it,
  so the words that caused the interruption are transcribed rather than discarded
  with the echo. The echo flush is therefore confined to half duplex, where the
  mic really was muted — flushing in full duplex would throw away the
  interruption just captured.
- **Runtime settings are read at the point of use, not cached.** The TTS voice is
  resolved from `config.LANGUAGE_MAP` per synthesized sentence and the duplex
  mode from `config.ENABLE_BARGE_IN` on every mic frame, so the Configuration
  dialog can change either mid-session with no reload, no restart, and no state
  to keep in sync. The Kokoro pipeline follows from the voice name's first letter,
  so choosing a British voice switches the G2P to `b` as well rather than reading
  it with American pronunciation.
- **Sessions carry a generation number.** A worker can be blocked inside Whisper
  decoding or a streaming LLM response when the user presses Stop, so shutdown
  is best-effort: the generation bump retires any straggler instead of letting
  it shadow the next session or hold a stale system prompt.
- **UI events are a bounded queue.** `emit()` publishes to a 512-slot queue and
  drops the oldest event if nothing is draining it, so a CLI session with no
  browser attached cannot grow without limit.
- **Personas are prompt layers.** `config.build_system_prompt()` combines the
  shared tutoring rules — the subjects (language, culture, travel) and the
  spoken-output constraints — with a personality and the persona's name (so "who
  are you?" is answered in character); `set_persona()` rewrites the prompt in
  place, which makes mid-session switching possible while keeping the
  conversation. Five personas ship: tutor, jester, cheerleader, explorer,
  secretary.
- **The HTTP server keeps connections alive,** so every POST handler must consume
  its request body even when it ignores it. An unread body is parsed as the next
  request line and answered with 501, which showed up as the first Start after a
  Stop failing. `do_POST` reads the body once, before dispatch.

## Testing

- **`pytest`** (`requirements-dev.txt`, pinned), 231 tests: the viseme timeline
  (43), persona prompts and subjects (33), STT hallucination guards (26),
  sentence splitting including CJK (23), TTS chunking and markdown stripping (22),
  the HTTP layer (17), the runtime settings behind the Configuration dialog (16),
  the `<think>` filter (15), the barge-in decision (11), persona switching and
  session lifecycle (10), the avatar asset contract (9) and history trimming (6).
- **No models, no audio devices.** The suite runs in about two seconds. The HTTP
  tests drive the real handler over a socket on an ephemeral port, reusing one
  connection the way a browser does; avatar tests skip if `web/avatar/` has been
  deleted.
- **Regressions are pinned by reproducing them first** — reverting a fix must
  make its test fail, which is how the 501 and the absorbed one-frame phoneme
  were confirmed.
- **Audio behaviour is probed with a fake microphone.** Frames pushed into
  `raw_q` drive the real `vad_worker` with no audio device, and Kokoro supplies
  real speech to push through it. That is how the barge-in failure was
  reproduced — `"Stop."` and `"Wait!"` never interrupted at all, and a
  successful interruption still lost 41 % of the sentence — and how the fix was
  measured. These probes need models, so they stay out of the suite; the pure
  decision they informed (`BargeInDetector`) is unit-tested.

## Configuration & deployment

- **Configuration:** centralized in `src/config.py`, every value overridable via
  `VT_*` environment variables (models, voice, TTS language, per-language voices,
  persona, barge-in, LLM URL, web host/port). Multilingual voices can be
  overridden per language (`VT_TTS_VOICE_ES`, `VT_TTS_VOICE_ZH`), and an explicit
  env-var voice outside the curated list is added to it rather than ignored, so
  the dialog still offers it. `AVAILABLE_VOICES` in the same file is the only
  place the selectable voices are defined; `/api/config` serves it to the UI.
- **Deployment model:** local single-user. Both the LLM backend and the web
  UI bind to `127.0.0.1` only, with **no authentication** — anyone who can reach
  the web UI could start the microphone, so it must not be exposed to a network
  without adding access controls.
- **Scripts:** `scripts/download_model.sh` (fetch GGUF weights),
  `scripts/start_all.sh` (LLM backend + web frontend, Ctrl+C stops both),
  `scripts/start_server.sh` (Metal GPU LLM server), `scripts/start_agent.sh`
  (CLI voice agent), `scripts/start_web.sh` (web frontend).
- **Verification:** `src/verify.py` checks imports, VAD, TTS synthesis, a Whisper
  round-trip, that the moods change rate, pitch and articulation without
  drifting out of lip-sync, and LLM backend reachability — no microphone required.
- **Generated artifacts stay out of git** — GGUF weights and sample audio are
  gitignored. The photo avatar sprites are the exception: at ~200 KB they are
  committed, so a clone shows the real face without needing the source clip,
  which is not in the repo.

## Notable constraints

- First run downloads model weights (Kokoro, Whisper, spaCy, GGUF); all
  subsequent runs are offline except the local HTTP call.
- Real-time performance and simultaneous model residency assume Apple Silicon
  with ~36 GB unified memory. Larger models (whisper-medium, 14B Q5_K_M LLM) are
  supported but trade memory for quality.
- **Kokoro synthesizes at most 510 phonemes per forward pass** and mlx-audio
  truncates the excess with only a log warning, so the audio ends mid-sentence
  with nothing raised. Measured on one long Chinese reply: 20.45 s spoken, 9.10 s
  dropped. Chunking before the pipeline is the only defence, and the budget has to
  be per language because phoneme density differs by ~4x.
- **Full duplex needs headphones, and no threshold can fix that.** Nothing at
  frame level distinguishes the user's voice from the tutor's own voice returning
  through a speaker at the same level. `BARGE_IN_MIN_RMS` is the only guard:
  measured against the tutor's voice at speaker-bleed levels it ignores echo at
  RMS 0.010 but does interrupt at 0.025. Software echo cancellation is the actual
  fix and is on the roadmap.
- The bundled `ffmpeg` has no `drawtext` filter and no WebP encoder, so the
  avatar tools draw their own labels in numpy and ship JPEG sprites.
- Avatar sprites are tied to one source clip's geometry: the anchor and crop
  boxes in `tools/build_photo_avatar.py` are measured constants, so a different
  video needs them re-measured.
