# VirtualTutor 🎓

A continuous, real-time **voice conversation** tutor that runs fully on-device on
Apple Silicon. Speak to it, it thinks, and it talks back — and in full duplex you
can cut it off mid-sentence. It tutors **language learning, culture and travel**:
useful phrases and how they are pronounced, how a custom actually works, and what
to do once you are there. Supports **English**, **Spanish**, and **Chinese** with
automatic language detection.

```
Microphone ─PCM─▶ Silero VAD ─audio─▶ mlx-whisper ─text─▶ llama.cpp (Qwen3-8B)
                  (speech end)         (STT)                    │ token stream
                                        │ lang detect           ▼
Speakers ◀─PCM─ Kokoro TTS ◀─sentences─ sentence-split buffer ◀─┘
                     │ phoneme durations
                     └──▶ viseme timeline ──▶ browser (lip-synced avatar)
```

All four models sit in unified memory at once, giving low-latency responses.
Tested on **Apple M3 Pro / 36 GB / macOS**.

## What it teaches

The shared system prompt in `src/config.py` scopes the tutor to three connected
subjects, and every persona inherits them:

- **Language** — useful words and phrases, grammar in plain terms, and
  pronunciation given as spoken syllables. Phonetic symbols are pointless here:
  the tutor's words go through TTS, so `/ˈkwen.ta/` would be read out as symbols.
- **Culture** — customs, etiquette, food, festivals and everyday life where the
  language is spoken, as something regional and changing rather than a list of
  national traits.
- **Travel** — planning a trip, getting around, ordering a meal, asking for
  directions, being a considerate guest.

Ask it something else and it answers in a sentence, then offers a way back. It
will not write code or markdown even when asked directly, because everything it
says is spoken: a fenced code block reaches the speaker as "backtick backtick
backtick". For the same reason `src/tts.py` strips markdown decoration before
synthesis — misaki phonemizes `*` as the word "asterisk", so an emphasised
`*boss*` would otherwise be spoken as "asterisk boss asterisk".

## Components

| Stage | Tech | Notes |
|-------|------|-------|
| VAD   | Silero VAD v5 (`silero-vad`) | detects end of speech and barge-in, offline |
| STT   | `mlx-whisper` (`whisper-small`, multilingual) | Metal-accelerated transcription, auto-detects language |
| LLM   | `llama.cpp` server + Qwen3-8B Q5_K_M | OpenAI-compatible, streaming, thinking disabled |
| TTS   | Kokoro-82M via `mlx-audio` | 24 kHz, multilingual (EN/ES/ZH), voice switchable live |
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

# 3. Download the LLM (~5.9 GB)
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
sentence with `VT_BARGE_IN=1 ./scripts/start_agent.sh` — see
[Barge-in](#barge-in).

On startup the agent asks you to pick a **tutor persona** (personality). Choose
by number, or press Enter for the default. You can skip the menu by presetting
`VT_PERSONA=jester ./scripts/start_agent.sh`. Available personas:

| Key | Persona | Style |
|-----|---------|-------|
| `tutor` | The Patient Tutor (default) | Warm, calm and encouraging |
| `jester` | The Witty Jester | Clever puns, gentle teasing, fast-paced humor |
| `cheerleader` | The Enthusiastic Cheerleader | High energy, hyper-positive encouragement |
| `explorer` | The Curious Explorer | Treats every question like a fun mystery |
| `secretary` | The Sassy Secretary | Deadpan office wit, runs the session like your diary |

The persona is part of the tutor's identity, not just its tone: ask "who are
you?" and it answers in character ("I am VirtualTutor speaking as The Witty
Jester"). In the web UI the persona dropdown works **while a session is
running** — switching rewrites the system prompt in place, so the next reply
changes personality and the conversation so far is kept.

Quit with `Ctrl+C`.

### Multilingual

VirtualTutor automatically detects and responds in **English**, **Spanish**, and
**Chinese**. Just speak in your language — no configuration needed.

| Language | Whisper detection | Default Kokoro voice | Pipeline |
|----------|------------------|----------------------|----------|
| English  | `en` (automatic) | `af_heart`   | `a` (US) |
| Spanish  | `es` (automatic) | `ef_dora`    | `e`      |
| Chinese  | `zh` (automatic) | `zf_xiaoxiao`| `z`      |

Each language's voice can be changed while a session runs — see
[Configuration dialog](#configuration-dialog).

How it works:
1. Multilingual Whisper transcribes your speech and detects which language you're speaking.
2. The LLM replies in the same language.
3. Kokoro TTS synthesizes the reply using the language-appropriate voice and G2P pipeline.
4. Lip-sync visemes work for all three languages — the phoneme-to-mouth-shape mapping covers Spanish and Chinese IPA.

Language switching is seamless: speak Spanish mid-conversation and the tutor
switches to Spanish on its next reply.

Two things had to be language-aware for a reply to be spoken *in full*. The
sentence splitter now recognises CJK terminators (`。！？；：…`), which have no
trailing space and are what Chinese actually ends a sentence with — without them
a whole Chinese reply reached TTS as one block. And Kokoro synthesizes at most
510 phonemes per forward pass, silently truncating the rest
(`WARNING:root:Truncating len(ps) == 657 > 510`), so `src/tts.py` splits text on
clause boundaries first, with a per-pipeline character budget: Chinese runs
~4.1 phonemes per character against Spanish's ~1.1, so the same 400-character
block mlx-audio uses for both overshoots badly in Chinese. Measured on one long
Chinese reply, the unchunked path spoke 20.45 s and dropped the remaining 9.10 s;
chunked, it speaks all 29.55 s.

## Web UI

A local web frontend is available as an alternative to the terminal: one
start/stop button, a live transcript, a lip-synced avatar, a Configuration dialog
for voices and duplex mode, and an animated background whose colours and wave
shapes follow the conversation state and your voice level.

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
| `GET /api/config` | voices offered per language, plus the duplex mode |
| `GET /api/events` | SSE stream: state, transcript, mic level, visemes |
| `POST /api/start` | start listening — body `{"persona": "tutor"}` |
| `POST /api/persona` | switch persona, mid-session if one is running |
| `POST /api/config` | set voices / duplex mode — body `{"voices": {"en": "bf_emma"}, "duplex": "full"}` |
| `POST /api/stop` | stop listening |

| Env var | Default | Purpose |
|---------|---------|---------|
| `VT_WEB_HOST` | `127.0.0.1` | web UI bind address |
| `VT_WEB_PORT` | `8800` | web UI port |

**Security:** like the LLM server, the web UI binds to `127.0.0.1` and has **no
authentication**. Anyone who can reach it can start your microphone, so do not
bind it to `0.0.0.0` or expose it to a network without adding access control.

### Configuration dialog

The **Configuration** button in the panel header opens a dialog with the two
settings worth changing without a restart:

- **Voices** — one dropdown per language (English, Spanish, Chinese) listing the
  Kokoro voices for that language. English offers both American (`af_*`, `am_*`)
  and British (`bf_*`, `bm_*`) speakers; picking a British voice switches the G2P
  pipeline to `b` as well, since a British voice read with American
  pronunciation is worse than either on its own.
- **Duplex mode** — a switch between half duplex (default: the mic is muted while
  the tutor speaks) and full duplex (keep listening, so you can interrupt it).
  Full duplex is the same thing as `VT_BARGE_IN=1` and wants headphones — see
  [Barge-in](#barge-in) for what interrupting actually does.

There is no Save button because there is nothing to save: the voice is resolved
per synthesized sentence and the duplex mode is read on every mic frame, so both
apply immediately — mid-session included. A voice change lands on the tutor's
next sentence; one already synthesized keeps the old voice. Changes are
broadcast over the SSE stream, so a second browser tab does not go stale.

The dialog is populated from `/api/config`, which reads `AVAILABLE_VOICES` in
`src/config.py` — that file stays the only place the voice list is defined. A
voice picked for the first time is downloaded from Hugging Face (a few hundred
KB); after that it is cached like every other model.

### Barge-in

In full duplex the mic stays open while the tutor talks, and speaking over it
stops it mid-sentence. Two details decide whether that feels right:

**How much speech counts as an interruption.** Not one loud frame — a keypress
or a door would stop the tutor. But not an unbroken run of loud frames either,
which is what this used to require: speech dips below the threshold between
words and on plosives, and every dip restarted the count. Measured on
synthesized speech at a normal speaking level, that never fired *at all* for
`"Stop."` or `"Wait!"`, and took 1.66 s on a whole sentence — long enough that
the tutor talked over you and you gave up. So evidence now accumulates and
*decays* rather than resetting (`BARGE_IN_SPEECH_DURATION`, `BARGE_IN_DECAY`),
which fires the same four phrases in 0.61–0.83 s while a lone spike still cannot
reach the threshold before it decays away.

**What happens to the words you interrupted with.** They are kept. The frames
leading up to the interruption seed the new utterance
(`BARGE_IN_PREROLL_DURATION`, 1.2 s — longer than the 0.4 s pre-roll used for a
normal speech onset, because confirming a barge-in takes up to ~0.85 s). Before,
they were discarded along with the echo flush, so `"Wait, stop, I do not
understand"` reached the STT as `"I do not understand"` at best — and a short
`"Stop."` left nothing above the minimum utterance length, so the tutor went
quiet and never answered.

The one thing no frame-level heuristic can do is tell your voice from the
tutor's own voice coming back through a speaker at the same level. That is what
`BARGE_IN_MIN_RMS` guards, and why full duplex asks for headphones: echo quieter
than the gate is ignored, echo louder than it interrupts the tutor with itself.

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

The mouth has **inertia** rather than switching shape per span. A timeline of
real phonemes contains plenty of 25 ms spans, and snapping to each one — or
restarting a crossfade on every change — flickers, because a fade longer than
the span never finishes. Instead every viseme keeps a weight that rises toward 1
while it is the target (55 ms) and decays afterwards (95 ms, slower, as closing
is in speech), and the rendered mouth is the weighted blend of the three
strongest. A fleeting consonant therefore only gets part of the way to its
shape, which is what articulation actually does: measured on one sentence, brief
`PP`, `FF`, `CH`, `KK` and `TH` spans peak at 0.37–0.50 weight while sustained
vowels reach 1.0. Weights advance against real frame deltas, so motion is
identical on 60 Hz and 120 Hz displays.

Rendered pixel change inside the mouth region, per frame, on the same sentence:

| | mean | p95 | max |
|---|---|---|---|
| per-span crossfade | 2.74 | 7.10 | 10.99 |
| weighted blend | 1.31 | 3.20 | 4.11 |

The patches are stored unmasked and the feathered ellipse is applied once to the
blend, so averaging several shapes stays identical to the single-patch composite
the builder produced.

Viseme names follow the VRM/ARKit convention (`aa`, `ih`, `ou`, `oh`, `E` plus
consonant groups `PP`, `FF`, `TH`, `DD`, `SS`, `CH`, `KK`, `RR`), so the same
timeline could drive a 3D VRM avatar instead of the 2D canvas mouth.

| Env var | Default | Purpose |
|---------|---------|---------|
| `VT_TTS_LANG` | `a` | fallback Kokoro pipeline (`a` US English, `b` UK English) for a voice whose name does not imply one |

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

The bundled `web/avatar/` sprites are committed — 200 KB for the whole face, so a
clone shows the photo avatar straight away. `tools/picks.avatar1.json` is the
mapping they were built from. Delete `web/avatar/` to go back to the drawing, or
rebuild from your own clip with the two commands above; the source video itself
stays out of the repo.

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

The source clip stays out of git; the sprites built from it are committed, so
the photo avatar works from a clone with no extra steps.

## Verify (no microphone needed)

```bash
./.venv/bin/python src/verify.py
```

Checks imports, VAD, TTS synthesis, a Whisper STT round-trip, and llama-server
reachability.

## Tests

Unit tests cover the pure logic that is easy to break and hard to notice: the
viseme timeline, the sentence splitter feeding TTS, the chunking that keeps
Kokoro from truncating a long reply, the markdown stripped before speech, the
`<think>` filter, the Whisper
anti-hallucination guards, history trimming, persona prompts and switching, the
runtime settings behind the Configuration dialog (voice per language, the
pipeline each voice implies, duplex mode), the barge-in decision and the photo
avatar's asset contract. They load no models and open no audio devices, so the
suite (231 tests) runs in
about two seconds.

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
| `VT_WHISPER_MODEL` | `mlx-community/whisper-small-mlx` | STT model (multilingual) |
| `VT_TTS_MODEL` | `prince-canuma/Kokoro-82M` | TTS model |
| `VT_TTS_VOICE` | `af_heart` | Kokoro voice for English |
| `VT_TTS_VOICE_ES` | `ef_dora` | Kokoro voice for Spanish |
| `VT_TTS_VOICE_ZH` | `zf_xiaoxiao` | Kokoro voice for Chinese |
| `VT_LLAMA_URL` | `http://localhost:8080/v1/chat/completions` | LLM endpoint |
| `VT_MODEL_FILE` | `Qwen_Qwen3-8B-Q5_K_M.gguf` | GGUF filename |
| `VT_MODEL_REPO` | `bartowski/Qwen_Qwen3-8B-GGUF` | Hugging Face repo the GGUF is downloaded from |
| `VT_PERSONA` | `tutor` | tutor personality: `tutor`, `jester`, `cheerleader`, `explorer`, `secretary` |
| `VT_BARGE_IN` | `0` | set `1` to allow interrupting the tutor (headphones only) |

With 36 GB you can step up quality: set `VT_WHISPER_MODEL=mlx-community/whisper-medium-mlx`
and use a 14B `Q5_K_M` GGUF for the LLM.

### Thinking mode

Qwen3 is a hybrid reasoning model: asked a question it works the answer out
inside `<think> ... </think>` before replying. For a voice tutor that is dead air
of several seconds, so `start_server.sh` passes `--reasoning-budget 0`, which
tells the chat template to close thinking immediately. Measured on this machine,
the first *spoken* sentence leaves the LLM 1.1–1.9 s after the prompt.

The agent also strips `<think>` blocks out of the token stream
(`ThinkFilter` in `src/voice_agent.py`), matching tags that arrive split across
tokens. That guard matters because whether the tags reach the client depends on
llama.cpp's `--reasoning-format`: with the default they are parsed out into
`reasoning_content`, but with `--reasoning-format none` an empty
`<think></think>` pair lands in the reply text — and TTS would try to pronounce
it. Filtering also keeps the tags out of the transcript and out of the history
that gets re-prefilled every turn.

## Notes & limits

- The LLM server binds to `127.0.0.1` only (no external network exposure) and
  has no auth — it is intended for local single-user use.
- Runs half-duplex by default (does not listen while speaking) to prevent the
  speaker echoing into the mic and cutting off replies. Barge-in is opt-in via
  `VT_BARGE_IN=1` or the Configuration dialog, and needs headphones.
- Each utterance is captured with a 0.4 s pre-roll so the first word is not
  clipped. In half duplex the echo picked up during playback is flushed before
  listening resumes; in full duplex there is nothing to flush, and the audio
  leading up to a barge-in is kept so the interrupting words are transcribed.
- First run downloads model weights (Kokoro, Whisper, spaCy `en_core_web_sm`,
  and the GGUF); subsequent runs are offline except the local HTTP call.
- The prompt asks for 1–3 sentences and mostly gets them in English; Chinese
  replies often run longer. They are spoken in full either way, since text is
  chunked under Kokoro's phoneme limit, but expect a paragraph rather than a line.
- Cultural and travel claims are the LLM's, and an 8B model states wrong ones
  confidently — in testing it told me not to use the left hand in Spain, which is
  not a Spanish custom. Treat it as conversation practice, not a guidebook.

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
│   └── avatar/             # photo sprites (committed, ~200 KB)
└── src/
    ├── config.py
    ├── tts.py              # Kokoro synthesis, chunked under the phoneme limit
    ├── visemes.py          # IPA -> mouth shape timeline for lip-sync
    ├── voice_agent.py      # pipeline threads, sentence split, <think> filter
    ├── server.py           # local web server (stdlib only)
    └── verify.py
```
