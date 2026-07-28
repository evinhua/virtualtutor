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

- **Threaded, queue-based pipeline.** Three worker threads — VAD, brain
  (STT+LLM), and speech — communicate via `queue.Queue` and coordinate through
  `threading.Event` flags (`stop_event`, `assistant_speaking`, `interrupt_event`).
- **Streaming end to end.** LLM tokens are split into sentences on the fly and
  handed to TTS immediately, minimizing perceived latency.
- **Half-duplex by default.** The agent does not listen while speaking and
  flushes echo picked up during playback; barge-in is opt-in for headphone use.

## Configuration & deployment

- **Configuration:** centralized in `src/config.py`, every value overridable via
  `VT_*` environment variables (models, voice, LLM URL, barge-in, etc.).
- **Deployment model:** local single-user. The llama.cpp server binds to
  `127.0.0.1` only, with **no authentication** — it is intended for local use and
  must not be exposed to a network without adding access controls.
- **Scripts:** `scripts/download_model.sh` (fetch GGUF weights),
  `scripts/start_server.sh` (Metal GPU LLM server), `scripts/start_agent.sh`
  (voice agent).
- **Verification:** `src/verify.py` checks imports, VAD, TTS synthesis, a Whisper
  round-trip, and llama-server reachability — no microphone required.

## Notable constraints

- First run downloads model weights (Kokoro, Whisper, spaCy, GGUF); all
  subsequent runs are offline except the local HTTP call.
- Real-time performance and simultaneous model residency assume Apple Silicon
  with ~36 GB unified memory. Larger models (whisper-medium, 14B Q5_K_M LLM) are
  supported but trade memory for quality.
