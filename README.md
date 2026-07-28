# VirtualTutor 🎓

A continuous, real-time **voice conversation** tutor that runs fully on-device on
Apple Silicon. Speak to it, it thinks, and it talks back — with barge-in so you
can interrupt it any time.

```
Microphone ─PCM─▶ Silero VAD ─audio─▶ mlx-whisper ─text─▶ llama.cpp (Qwen2.5-7B)
                  (speech end)         (STT)                    │ token stream
                                                                ▼
Speakers ◀─PCM─ Kokoro TTS ◀─sentences─ sentence-split buffer ◀─┘
```

All four models sit in unified memory at once, giving low-latency responses.
Tested on **Apple M3 Pro / 36 GB / macOS**.

## Components

| Stage | Tech | Notes |
|-------|------|-------|
| VAD   | Silero VAD v5 (`silero-vad`) | detects end of speech, offline |
| STT   | `mlx-whisper` (`whisper-small.en`) | Metal-accelerated transcription |
| LLM   | `llama.cpp` server + Qwen2.5-7B-Instruct Q4_K_M | OpenAI-compatible, streaming |
| TTS   | Kokoro-82M via `mlx-audio` | 24 kHz, voice `af_heart` |

The pipeline is threaded: VAD, brain (STT+LLM), and speech run concurrently, and
sentences are streamed to TTS as soon as the LLM finishes each one to minimize
perceived latency.

## Setup

Already provisioned in this project, but to reproduce from scratch:

```bash
# 1. System dependencies
brew install portaudio ffmpeg espeak-ng llama.cpp

# 2. Python 3.11 virtual environment
/opt/homebrew/bin/python3.11 -m venv .venv
./.venv/bin/pip install -r requirements.txt

# 3. Download the LLM (~4.7 GB)
./scripts/download_model.sh
```

## Run

Everything with one command (recommended):

```bash
./scripts/start_all.sh          # then open http://127.0.0.1:8800
```

This starts the LLM backend, waits for the model to load, then starts the web
frontend. **Ctrl+C stops both.** If an LLM server is already running on the port
it is reused and left running on exit. The LLM's verbose log goes to
`$TMPDIR/virtualtutor-llm.log`; transcripts stay in the foreground.

### Or run the pieces separately

Open two terminals:

```bash
# Terminal 1 — LLM server (Metal GPU, local-only on 127.0.0.1:8080)
./scripts/start_server.sh

# Terminal 2 — the voice agent
./scripts/start_agent.sh
```

Then just talk. Pause for ~0.8 s and the tutor replies. By default it runs
**half-duplex** (it doesn't listen while speaking) to avoid the speaker echoing
into the mic. If you use headphones you can enable interrupting the tutor mid-
sentence with `VT_BARGE_IN=1 ./scripts/start_agent.sh`.

On startup the agent asks you to pick a **tutor persona** (personality). Choose
by number, or press Enter for the default. You can skip the menu by presetting
`VT_PERSONA=jester ./scripts/start_agent.sh`. Available personas:

| Key | Persona | Style |
|-----|---------|-------|
| `tutor` | The Patient Tutor (default) | Warm, calm and encouraging |
| `jester` | The Witty Jester | Clever puns, gentle teasing, fast-paced humor |
| `cheerleader` | The Enthusiastic Cheerleader | High energy, hyper-positive encouragement |
| `explorer` | The Curious Explorer | Treats every question like a fun mystery |

Quit with `Ctrl+C`.

## Web UI

A local web frontend is available as an alternative to the terminal: one
start/stop button, a live transcript, and an animated background whose colours
and wave shapes follow the conversation state and your voice level.

```bash
# Terminal 1 — LLM server
./scripts/start_server.sh

# Terminal 2 — web frontend
./scripts/start_web.sh          # then open http://127.0.0.1:8800
```

Audio stays on this machine: the microphone and speakers are driven by the same
Python pipeline as the CLI, and the browser is only the control surface and
transcript view. The server is built on the Python standard library
(`ThreadingHTTPServer` + Server-Sent Events), so it adds no dependencies.

| Env var | Default | Purpose |
|---------|---------|---------|
| `VT_WEB_HOST` | `127.0.0.1` | web UI bind address |
| `VT_WEB_PORT` | `8800` | web UI port |

**Security:** like the LLM server, the web UI binds to `127.0.0.1` and has **no
authentication**. Anyone who can reach it can start your microphone, so do not
bind it to `0.0.0.0` or expose it to a network without adding access control.

## Verify (no microphone needed)

```bash
./.venv/bin/python src/verify.py
```

Checks imports, VAD, TTS synthesis, a Whisper STT round-trip, and llama-server
reachability.

## Configuration

Everything is in `src/config.py` and can be overridden with env vars:

| Env var | Default | Purpose |
|---------|---------|---------|
| `VT_WHISPER_MODEL` | `mlx-community/whisper-small.en-mlx` | STT model |
| `VT_TTS_MODEL` | `prince-canuma/Kokoro-82M` | TTS model |
| `VT_TTS_VOICE` | `af_heart` | Kokoro voice (af_bella, am_adam, bf_emma, ...) |
| `VT_LLAMA_URL` | `http://localhost:8080/v1/chat/completions` | LLM endpoint |
| `VT_MODEL_FILE` | `Qwen2.5-7B-Instruct-Q4_K_M.gguf` | GGUF filename |
| `VT_PERSONA` | `tutor` | tutor personality: `tutor`, `jester`, `cheerleader`, `explorer` |
| `VT_BARGE_IN` | `0` | set `1` to allow interrupting the tutor (headphones only) |

With 36 GB you can step up quality: set `VT_WHISPER_MODEL=mlx-community/whisper-medium-mlx`
and use a 14B `Q5_K_M` GGUF for the LLM.

## Notes & limits

- The LLM server binds to `127.0.0.1` only (no external network exposure) and
  has no auth — it is intended for local single-user use.
- Runs half-duplex by default (does not listen while speaking) to prevent the
  speaker echoing into the mic and cutting off replies. Barge-in is opt-in via
  `VT_BARGE_IN=1` and works best with headphones.
- Each utterance is captured with a 0.4 s pre-roll so the first word is not
  clipped, and echo picked up during playback is flushed before listening resumes.
- First run downloads model weights (Kokoro, Whisper, spaCy `en_core_web_sm`,
  and the GGUF); subsequent runs are offline except the local HTTP call.

## Layout

```
virtualtutor/
├── requirements.txt
├── models/                 # GGUF weights (gitignored)
├── samples/                # verification wav output
├── scripts/
│   ├── download_model.sh
│   ├── start_all.sh        # backend + frontend, Ctrl+C stops both
│   ├── start_server.sh
│   ├── start_agent.sh
│   └── start_web.sh
├── specs/                  # mission, roadmap, tech stack
├── web/                    # web frontend (static, no build step)
│   ├── index.html
│   ├── styles.css
│   └── app.js
└── src/
    ├── config.py
    ├── tts.py
    ├── voice_agent.py
    ├── server.py           # local web server (stdlib only)
    └── verify.py
```
