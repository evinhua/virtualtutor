# VirtualTutor 🎓

A continuous, real-time **voice conversation** tutor that runs fully on-device on
Apple Silicon. Speak to it, it thinks, and it talks back — with barge-in so you
can interrupt it any time.

```
Microphone ─PCM─▶ Silero VAD ─audio─▶ mlx-whisper ─text─▶ llama.cpp (Qwen2.5-7B)
                  (speech end)         (STT)                    │ token stream
                                                                ▼
Speakers ◀─PCM─ Kokoro TTS ◀─sentences─ sentence-split buffer ◀─┘
                     │ phoneme durations
                     └──▶ viseme timeline ──▶ browser (lip-synced avatar)
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
| Lip-sync | Kokoro's own phoneme durations | no second model, frame-accurate |
| UI    | stdlib HTTP + SSE, static HTML/CSS/JS | optional; CLI works alone |

The pipeline runs four worker threads — VAD, brain (STT+LLM), synthesis, and
playback — communicating over queues, so the next sentence is synthesized while
the current one is still being spoken. Sentences go to TTS as soon as the LLM
finishes each one, to minimize perceived latency.

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

The persona is part of the tutor's identity, not just its tone: ask "who are
you?" and it answers in character ("I am VirtualTutor speaking as The Witty
Jester"). In the web UI the persona dropdown works **while a session is
running** — switching rewrites the system prompt in place, so the next reply
changes personality and the conversation so far is kept.

Quit with `Ctrl+C`.

## Web UI

A local web frontend is available as an alternative to the terminal: one
start/stop button, a live transcript, a lip-synced avatar, and an animated
background whose colours and wave shapes follow the conversation state and your
voice level.

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

| Endpoint | Purpose |
|----------|---------|
| `GET /api/personas` | available personas (key, name, blurb) |
| `GET /api/status` | whether a session is live, and the active persona |
| `GET /api/events` | SSE stream: state, transcript, mic level, visemes |
| `POST /api/start` | start listening — body `{"persona": "tutor"}` |
| `POST /api/persona` | switch persona, mid-session if one is running |
| `POST /api/stop` | stop listening |

| Env var | Default | Purpose |
|---------|---------|---------|
| `VT_WEB_HOST` | `127.0.0.1` | web UI bind address |
| `VT_WEB_PORT` | `8800` | web UI port |

**Security:** like the LLM server, the web UI binds to `127.0.0.1` and has **no
authentication**. Anyone who can reach it can start your microphone, so do not
bind it to `0.0.0.0` or expose it to a network without adding access control.

### Lip-sync

The avatar's mouth is driven by real phonemes, not by audio loudness, and it all
happens on-device. Kokoro is a StyleTTS2-style model whose duration predictor
emits one frame count per phoneme as part of the normal forward pass, and each
frame is exactly 600 samples at 24 kHz — 25.00 ms. So `src/visemes.py` reads
those durations, maps each IPA phoneme to a mouth shape, and builds a timeline
that lines up with the audio to the millisecond. No second model, no extra
inference, and no talking-head GPU work.

Audio keeps playing through Python rather than the browser, which preserves the
half-duplex echo handling. To stay in sync, the server sends the timeline with
the delay until the audio is actually audible (the PortAudio buffer plus
anything still playing), and the browser animates against that.

Viseme names follow the VRM/ARKit convention (`aa`, `ih`, `ou`, `oh`, `E` plus
consonant groups `PP`, `FF`, `TH`, `DD`, `SS`, `CH`, `KK`, `RR`), so the same
timeline could drive a 3D VRM avatar instead of the 2D canvas mouth.

| Env var | Default | Purpose |
|---------|---------|---------|
| `VT_TTS_LANG` | `a` | Kokoro language pipeline: `a` US English, `b` UK English |

### Photo avatar

The canvas face is the fallback. If `web/avatar/manifest.json` exists, the UI
instead shows a **real photo** whose mouth is swapped per viseme, built from any
short talking-head video:

```bash
# 1. Find mouth shapes: contact sheets of every frame, numbered
./.venv/bin/python tools/mouth_contact_sheet.py VIDEO \
    --box 245,590,250,165 --ref 480 --near 30 --min-ncc 0.80 --out /tmp/sheets

# 2. Note one frame number per viseme in a picks file, then export
./.venv/bin/python tools/build_photo_avatar.py VIDEO \
    --picks tools/picks.avatar1.json --out web/avatar --size 512
```

`tools/picks.avatar1.json` is the mapping used for the bundled example. Reload
the page and the photo appears; delete `web/avatar/` to go back to the drawing.

Three things make this look like one person rather than a flickering slideshow:

- **The head is pinned.** A hand-held clip drifts (measured on the sample:
  46 px horizontally, 148 px vertically), so every exported frame is warped to
  one canonical pose. Alignment template-matches the eyes-and-nose-bridge patch
  — rigid under speech — over scale and rotation, with no face-landmark library:
  just `torch.grid_sample` and normalised cross-correlation, searched
  coarse-to-fine in a window so it costs seconds, not gigabytes.
- **Only the mouth moves.** Everything outside a feathered ellipse comes from a
  single base face, so the eyes never blink or glance sideways when a phoneme
  changes. `mouth_contact_sheet.py --near` also restricts candidate frames to
  those already close to the canonical pose, which keeps resampling minimal.
- **It stays small.** Shipping 14 whole faces meant 2.3 MB of near-identical
  pixels; shipping one base face plus 14 mouth patches as JPEG is ~170 KB. The
  browser applies the same feathered ellipse the builder used (verified to agree
  within 0.4 % per pixel), so the composite is identical either way.

Both the sprites and the source video stay out of git: `web/avatar/` is
generated, so a fresh clone shows the drawn face until you build your own.

## Verify (no microphone needed)

```bash
./.venv/bin/python src/verify.py
```

Checks imports, VAD, TTS synthesis, a Whisper STT round-trip, and llama-server
reachability.

## Tests

Unit tests cover the pure logic that is easy to break and hard to notice: the
viseme timeline, the sentence splitter feeding TTS, the Whisper
anti-hallucination guards, history trimming, persona prompts and switching, and
the photo avatar's asset contract. They load no models and open no audio
devices, so the suite (127 tests) runs in about a second.

```bash
./.venv/bin/pip install -r requirements-dev.txt   # once
./.venv/bin/python -m pytest
```

Tests that need the generated avatar sprites skip themselves when
`web/avatar/` has not been built.

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
├── requirements-dev.txt    # test-only dependencies
├── pytest.ini
├── models/                 # GGUF weights (gitignored)
├── samples/                # verification wav output
├── scripts/
│   ├── download_model.sh
│   ├── start_all.sh        # backend + frontend, Ctrl+C stops both
│   ├── start_server.sh
│   ├── start_agent.sh
│   └── start_web.sh
├── specs/                  # mission, roadmap, tech stack
├── tests/                  # unit tests (no models loaded)
├── tools/                  # avatar authoring (not needed at runtime)
│   ├── mouth_contact_sheet.py
│   ├── build_photo_avatar.py
│   └── picks.avatar1.json
├── web/                    # web frontend (static, no build step)
│   ├── index.html
│   ├── styles.css
│   ├── app.js
│   └── avatar/             # generated photo sprites (gitignored)
└── src/
    ├── config.py
    ├── tts.py
    ├── visemes.py          # IPA -> mouth shape timeline for lip-sync
    ├── voice_agent.py
    ├── server.py           # local web server (stdlib only)
    └── verify.py
```
