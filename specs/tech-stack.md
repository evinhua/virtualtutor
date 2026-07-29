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
| VAD (voice activity detection) | Silero VAD v5 (`silero-vad`, via `torch`) | Accurate, lightweight, fully offline end-of-speech detection at 16 kHz. |
| STT (speech-to-text) | `mlx-whisper` — `whisper-small.en` | Metal-accelerated Whisper on Apple Silicon; small.en balances speed and accuracy for English. |
| LLM | `llama.cpp` server + Qwen2.5-7B-Instruct Q4_K_M (GGUF) | OpenAI-compatible streaming endpoint, strong 7B instruct model that fits comfortably in memory at Q4_K_M. |
| TTS (text-to-speech) | Kokoro-82M via `mlx-audio` (voice `af_heart`, 24 kHz) | Small, natural-sounding, Metal-accelerated local TTS. |
| Lip-sync | Kokoro's own duration predictor (`src/visemes.py`) | The frame count per phoneme falls out of the normal forward pass, so a frame-accurate viseme timeline costs no extra inference and no audio analysis. |
| Frontend | Python stdlib `ThreadingHTTPServer` + Server-Sent Events, static HTML/CSS/JS | A control surface and transcript view with zero added dependencies and no build step. Audio stays in Python, which preserves the half-duplex echo handling. |

## Frontend & avatar

- **No framework, no bundler.** `web/` is three static files served by
  `src/server.py`; the browser receives state, transcript deltas, mic level and
  viseme timelines over one SSE stream.
- **Viseme naming follows VRM/ARKit** (`aa`, `ih`, `ou`, `oh`, `E`, plus `PP`,
  `FF`, `TH`, `DD`, `SS`, `CH`, `KK`, `RR`), so the same timeline could drive a
  3D avatar later.
- **Two renderers.** A vector face is the default and needs no assets; if
  `web/avatar/manifest.json` exists, a photo avatar is used instead — one base
  face plus a mouth patch per viseme, feathered in through an elliptical mask.
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
- **`requests`** — streaming HTTP client for the llama.cpp OpenAI-compatible API.
- **`misaki[en]`** — grapheme-to-phoneme text processing required by Kokoro.
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
  handed to TTS immediately, minimizing perceived latency.
- **Half-duplex by default.** The agent does not listen while speaking and
  flushes echo picked up during playback; barge-in is opt-in for headphone use.
- **Sessions carry a generation number.** A worker can be blocked inside Whisper
  decoding or a streaming LLM response when the user presses Stop, so shutdown
  is best-effort: the generation bump retires any straggler instead of letting
  it shadow the next session or hold a stale system prompt.
- **UI events are a bounded queue.** `emit()` publishes to a 512-slot queue and
  drops the oldest event if nothing is draining it, so a CLI session with no
  browser attached cannot grow without limit.
- **Personas are prompt layers.** `config.build_system_prompt()` combines shared
  tutoring rules with a personality and the persona's name (so "who are you?" is
  answered in character); `set_persona()` rewrites the prompt in place, which
  makes mid-session switching possible while keeping the conversation.

## Testing

- **`pytest`** (`requirements-dev.txt`, pinned) covering the pure logic:
  viseme timeline, sentence splitting, STT hallucination guards, history
  trimming, persona prompts and switching, and the avatar asset contract.
- **No models, no audio devices.** The suite runs in about a second, and tests
  needing generated avatar sprites skip themselves when `web/avatar/` is absent.

## Configuration & deployment

- **Configuration:** centralized in `src/config.py`, every value overridable via
  `VT_*` environment variables (models, voice, TTS language, persona, barge-in,
  LLM URL, web host/port).
- **Deployment model:** local single-user. Both the llama.cpp server and the web
  UI bind to `127.0.0.1` only, with **no authentication** — anyone who can reach
  the web UI could start the microphone, so it must not be exposed to a network
  without adding access controls.
- **Scripts:** `scripts/download_model.sh` (fetch GGUF weights),
  `scripts/start_all.sh` (LLM backend + web frontend, Ctrl+C stops both),
  `scripts/start_server.sh` (Metal GPU LLM server), `scripts/start_agent.sh`
  (CLI voice agent), `scripts/start_web.sh` (web frontend).
- **Verification:** `src/verify.py` checks imports, VAD, TTS synthesis, a Whisper
  round-trip, and llama-server reachability — no microphone required.
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
- The bundled `ffmpeg` has no `drawtext` filter and no WebP encoder, so the
  avatar tools draw their own labels in numpy and ship JPEG sprites.
- Avatar sprites are tied to one source clip's geometry: the anchor and crop
  boxes in `tools/build_photo_avatar.py` are measured constants, so a different
  video needs them re-measured.
