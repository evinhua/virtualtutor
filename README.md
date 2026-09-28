# VirtualTutor 🎓

A continuous, real-time **voice conversation** tutor that runs fully on-device on
Apple Silicon. Speak to it, it thinks, and it talks back — and in full duplex you
can cut it off mid-sentence. It tutors **language learning, culture and travel**:
useful phrases and how they are pronounced, how a custom actually works, and what
to do once you are there. Supports **English**, **Spanish**, and **Chinese** with
automatic language detection, and it speaks **with emotion** — each sentence gets
a mood that changes its rate, pitch, loudness and how wide the avatar articulates.

```
Microphone ─PCM─▶ Silero VAD ─audio─▶ mlx-whisper ─text─▶ Ollama (Qwen3.8-Uncensored)
                  (speech end)         (STT)                    │ token stream
                                        │ lang detect           ▼
                                                        mood + sentence split
                                                                │
Speakers ◀─PCM─ Kokoro TTS ◀─rate/pitch/gain─────────────────────┘
                     │ phoneme durations + articulation
                     └──▶ viseme timeline ──▶ browser (lip-synced avatar)
```

The LLM runs in Ollama; the other three models sit in unified memory next to it.
Tested on **Apple M3 Pro / 36 GB / macOS**.

## What it teaches

The shared system prompt in `src/config.py` scopes the tutor to three connected
subjects, and every persona teaches them — four by inheriting the shared prompt,
the secretary through its own (see [Personas](#run)):

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
| LLM   | **Ollama** + `Qwen3.8-Uncensored:latest` | native `/api/chat`, streaming, context pinned, thinking disabled |
| Mood  | `src/emotion.py` | cue from the model or signals in the text → rate, pitch, gain, articulation |
| TTS   | Kokoro-82M via `mlx-audio` | 24 kHz, multilingual (EN/ES/ZH), voice switchable live |
| Lip-sync | Kokoro's own phoneme durations | no second model, frame-accurate |
| UI    | stdlib HTTP + SSE, static HTML/CSS/JS | optional; CLI works alone |

The pipeline runs four worker threads — VAD, brain (STT+LLM), synthesis, and
playback — communicating over queues, so the next sentence is synthesized while
the current one is still being spoken. Sentences go to TTS as soon as the LLM
finishes each one, to minimize perceived latency.

llama.cpp is still supported as a second backend (`VT_LLM_BACKEND=llamacpp`);
see [LLM backend](#llm-backend).

## Setup

Already provisioned in this project, but to reproduce from scratch:

```bash
# 1. System dependencies
brew install portaudio ffmpeg espeak-ng ollama

# 2. Python 3.11 virtual environment
/opt/homebrew/bin/python3.11 -m venv .venv
./.venv/bin/pip install -r requirements.txt

# 3. Make sure the LLM is in Ollama (pulls it if it is missing)
./scripts/download_model.sh
```

`Qwen3.8-Uncensored:latest` is the default (a 27 B Qwen3 at Q4_K_M, ~17 GB on
disk). Point VirtualTutor at anything else `ollama list` shows with
`VT_LLM_MODEL`; nothing else has to change, since both backends speak the same
OpenAI-compatible protocol. A model you built locally with `ollama create` cannot
be pulled on another machine — the script says so instead of failing obscurely.

## Run

Everything with one command (recommended):

```bash
./scripts/start_all.sh          # then open http://127.0.0.1:8800
```

This makes sure Ollama is serving, loads the model into memory before the first
question rather than during it, then starts the web frontend. **Ctrl+C stops
what it started.** An Ollama that was already running is reused and left running
on exit — which on macOS is the normal case, since it runs as a background
service. Its log goes to `$TMPDIR/virtualtutor-llm.log`; transcripts stay in the
foreground.

### Or run the pieces separately

Open two terminals:

```bash
# Terminal 1 — LLM backend (loads the model; exits if Ollama already serves it)
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

Four personas are a style paragraph layered on the shared `BASE_SYSTEM_PROMPT`.
The **secretary is independent**: it carries its own base prompt
(`SECRETARY_SYSTEM_PROMPT`) that *replaces* the shared one, because its frame is
different rather than merely louder — the student is the boss, the lesson is an
appointment in their diary, and a mistake gets filed rather than gently
corrected. Any persona can do this by adding a `prompt` key next to its `style`.

What an independent prompt may not drop is the part that is not personality, so
it restates in its own voice the three subjects, pronunciation as spoken
syllables, no markdown or code, 1–3 sentences, and replying in the student's
language. `tests/test_personas.py` checks those invariants against *every*
persona's base prompt, not just the shared one, so a new prompt cannot quietly
lose the rules that keep a reply speakable.

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

### Emotion

Kokoro has no emotion input: one voice, one flat delivery. So a mood is built out
of the things that *can* be controlled without a second model — and the same mood
drives the face, so what you hear and what you see agree.

| | what it changes |
|---|---|
| rate | Kokoro's own `speed`, ±10 % |
| pitch | a resample, ±5 %, with the rate compensated so only pitch moves |
| loudness | gain, and the pauses before and after the sentence |
| articulation | how far the mouth travels toward each viseme |

Nine moods: `neutral`, `warm`, `excited`, `amused`, `curious`, `gentle`,
`thoughtful`, `proud`, `deadpan`. Where one comes from, in order:

1. **The model says so.** It is asked to prefix a reply with a cue — `[gentle]
   Almost, it is "la cuenta".` The cue is stripped out of the token stream
   (`CueFilter` in `src/voice_agent.py`) before anything is spoken, shown or
   stored, and a cue the model invented (`[natural]`, `[soft voice]`) is dropped
   rather than read aloud. A cue applies to the sentence it introduces; later
   sentences are read on their own, so "[warm] Say la cuenta. Want to try it?"
   comes out warm and then curious.
2. **Signals in the text**, in all three languages: `!!`/`¡`/`太棒了` → excited,
   a question → curious, "sorry"/"lo siento"/"别担心" → gentle, "exactly"/"muy
   bien"/"太好了" → proud, an ellipsis or "hmm" → thoughtful.
3. **The persona's baseline**: the cheerleader is excited, the secretary deadpan,
   the patient tutor warm. A persona also has an *energy* that scales how far
   each mood departs from neutral, so the secretary's gentle is milder than the
   cheerleader's rather than a different shape.

Measured on "That is the phrase you want to remember.":

| mood | spoken length | rate | pitch | widest mouth | lip-sync drift |
|------|---------------|------|-------|--------------|----------------|
| excited | 2.53 s | ×1.10 | ×1.045 | ×1.38 | 0 ms |
| thoughtful | 3.17 s | ×0.90 | ×0.980 | ×1.03 | 0 ms |

The pitch shift is the one thing that could have broken lip-sync, because
resampling changes duration as well as pitch. Synthesizing at `speed / ratio`
cancels that — only the pitch moves — and the timeline, whose durations Kokoro
predicted at that synthesis speed, is rescaled by the same `1 / ratio`. The drift
column above is that invariant checked end to end by `src/verify.py`.

Moods stay deliberately narrow (`emotion.LIMITS`). Past roughly ±10 % rate and
±5 % pitch, Kokoro stops sounding like the same speaker in a different mood and
starts sounding like a worse model.

### LLM backend

The default is **Ollama**, talking to `Qwen3.8-Uncensored:latest` over its
**native** `/api/chat` endpoint — not the OpenAI-compatible one, for one
specific reason: `/v1/chat/completions` silently ignores `options`, and
`options.num_ctx` is the only way to tell Ollama how big a context to load the
model with. Without it Ollama sizes the KV cache for the model's full trained
length, 131072 for Qwen3, which makes this model **25 GB resident instead of
17 GB** — and on a 36 GB machine that is the difference between a session that
can open the microphone and one that cannot (see
[Notes & limits](#notes--limits)). llama.cpp keeps the OpenAI dialect, where the
same settings are startup flags. The differences all live in
`config.llm_payload`, `config.llm_chunk` and `scripts/llm_env.sh`:

| | Ollama (default) | llama.cpp |
|---|---|---|
| endpoint | `http://localhost:11434/api/chat` | `http://localhost:8080/v1/chat/completions` |
| model | named per request (`VT_LLM_MODEL`) | the one GGUF the server was started with |
| context | `options.num_ctx` per request (`VT_LLM_CONTEXT`) | `--ctx-size` at startup |
| thinking off | `think: false` per request | `--reasoning-budget 0` at startup |
| stream shape | JSON lines, `message.content`, `done` flag | `data:` chunks, `choices[0].delta.content` |

```bash
# use llama.cpp instead
VT_LLM_BACKEND=llamacpp ./scripts/download_model.sh   # fetches the GGUF
VT_LLM_BACKEND=llamacpp ./scripts/start_all.sh
```

Pinning the context in the request, rather than only when the model is warmed,
is what makes it stick: a reply streamed without it makes Ollama **evict the
pinned runner and reload the model** at its own default, which is how a session
that started fine ran out of memory two turns later.

Disabling thinking matters more than it sounds. Asked a question, Qwen3 reasons
first, and on Ollama that reasoning arrives in a separate field:
the reply text stays clean either way, but the first spoken word waits several
seconds for a monologue nobody hears. With `think: false` the first
token arrives **2.1 s** after the prompt on a warm model (measured, M3 Pro; the
27 B model at Q4_K_M needs ~9 s to load at an 8192 context, which is why
`start_all.sh` loads it before the first question). The `<think>` filter in
`src/voice_agent.py` stays as the belt-and-braces guard for a backend that leaks
the tags into the text instead.

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

On top of *which* shape and *when*, each span carries **how far** the mouth
travels toward it. Two things move it, and both are sent with the timeline as
`[[start_ms, viseme, level], ...]` — a two-element entry still means level 1, so
the wire format stayed backwards compatible:

- **Emphasis.** Kokoro's phoneme string carries its own stress marks, and a vowel
  after `ˈ` is articulated 15 % wider than the same vowel unstressed (`ˌ` 7 %).
  Those marks used to be silently absorbed into the following phoneme's duration;
  now they also shape it. When two spans merge, their levels are averaged by
  duration, so a long unstressed vowel next to a brief stressed one is barely
  raised.
- **Mood.** The sentence's articulation multiplier, so an excited line is spoken
  *and* mouthed wide and a thoughtful one is closer to a mumble.

The level is applied as openness rather than as weight, because the renderer
normalises weights — scaling them all would cancel out and nothing would move.
The drawn face opens its mouth further (height fully, width at 35 %, as in
speech); the photo avatar, which cannot be stretched, mixes a little of the
wide-open `aa` patch or the closed `sil` patch into the blend. Measured on a
sustained vowel at 200 px:

| articulation | drawn mouth opening | photo blend |
|---|---|---|
| ×1.3 (excited) | 19.1 px | `aa` share 0.62 → 0.67 |
| ×1.0 | 14.6 px | — |
| ×0.7 (thoughtful) | 10.2 px | `sil` share 0.38 → 0.49 |

Transition times also follow the speaking rate: attack and release are scaled by
the sentence's `pace`, so a fast sentence articulates crisply and a slow one
softly. Measured 30 ms into the same vowel, the mouth is 4.46 px open at pace
1.25 against 3.76 px at 0.8. The drawn face additionally raises and slants its
brows per mood, and the avatar's ring takes the mood's hue, so the mood is
visible and not only audible.

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

Checks imports, VAD, TTS synthesis, a Whisper STT round-trip, that the moods
really do change rate, pitch and articulation without drifting out of lip-sync,
and LLM backend reachability.

## Tests

Unit tests cover the pure logic that is easy to break and hard to notice: the
viseme timeline and its articulation levels, the sentence splitter feeding TTS,
the chunking that keeps Kokoro from truncating a long reply, the mood read off a
cue or off the text in three languages, the pitch-shift and gain maths, the
markdown, phonetics and pseudo-cues stripped before speech, the `<think>` filter,
the request each LLM backend gets, the Whisper anti-hallucination guards, history
trimming, persona prompts (including the invariants an independent prompt must
keep) and switching, the runtime settings behind the
Configuration dialog (voice per language, the pipeline each voice implies, duplex
mode), the barge-in decision, recovery from an output-device error and from a
malformed LLM stream, the language-switch hint that has to ride in the system
prompt, the memory-pressure diagnosis behind a -9986 stream failure, and the
photo avatar's asset contract. They load no models and open no audio devices, so
the suite (405 tests) runs in about two seconds.

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
| `VT_LLM_BACKEND` | `ollama` | `ollama` or `llamacpp` |
| `VT_LLM_MODEL` | `Qwen3.8-Uncensored:latest` | model name sent to Ollama (anything in `ollama list`) |
| `VT_LLM_CONTEXT` | `8192` | context the model is loaded with (Ollama backend) |
| `VT_LLM_KEEP_ALIVE` | `30m` | how long Ollama keeps that runner loaded |
| `VT_OLLAMA_HOST` | `http://localhost:11434` | where Ollama is serving |
| `VT_LLAMACPP_HOST` | `http://localhost:8080` | where llama-server is serving |
| `VT_LLAMA_URL` | derived from the backend | full chat-completions URL, if you need to override it |
| `VT_MODEL_FILE` | `Qwen_Qwen3-8B-Q5_K_M.gguf` | GGUF filename (llama.cpp backend) |
| `VT_MODEL_REPO` | `bartowski/Qwen_Qwen3-8B-GGUF` | Hugging Face repo the GGUF is downloaded from |
| `VT_PERSONA` | `tutor` | tutor personality: `tutor`, `jester`, `cheerleader`, `explorer`, `secretary` |
| `VT_BARGE_IN` | `0` | set `1` to allow interrupting the tutor (headphones only) |

With 36 GB you can step up STT quality with
`VT_WHISPER_MODEL=mlx-community/whisper-medium-mlx`, or trade the 27 B model for
a smaller one if you want lower latency: `VT_LLM_MODEL=qwen3:8b`.

### Thinking mode

Qwen3 is a hybrid reasoning model: asked a question it works the answer out
before replying. For a voice tutor that is dead air of several seconds, so each
request carries `think: false` on Ollama (`--reasoning-budget 0` at
startup on llama.cpp), which tells the chat template to close thinking
immediately. Measured on this machine, the first token then arrives 2.1 s after
the prompt on a warm 27 B model.

The agent also strips `<think>` blocks out of the token stream
(`ThinkFilter` in `src/voice_agent.py`), matching tags that arrive split across
tokens. Ollama keeps reasoning in a separate `thinking` field that the agent
simply does not read, but llama.cpp with `--reasoning-format none` lands an empty
`<think></think>` pair in the reply text — and TTS would try to pronounce it.
Filtering also keeps the tags out of the transcript and out of the history that
gets re-prefilled every turn.

## Notes & limits

- Ollama and the web UI listen on localhost only and have no auth — this is a
  local single-user setup.
- The default model is large for this machine, and how it is *loaded* matters
  more than its size on disk. Left to itself Ollama sizes the KV cache for the
  model's full trained length — 131072 for Qwen3 — which makes the same 27 B
  Q4_K_M model **25 GB resident and leaves 2 % of memory free**, at which point
  CoreAudio can no longer wire a stream buffer and no session can open the
  microphone. Both `scripts/llm_env.sh` (when it warms the model) and every
  request the agent streams therefore pin the context: at `VT_LLM_CONTEXT=8192`
  the same model is **17 GB resident with 24 % free**, and loads in 9 s rather
  than 20. The pin rides in the request, through Ollama's native `/api/chat`, so
  it needs no Ollama restart and no `OLLAMA_CONTEXT_LENGTH` in the service's
  environment — and because it is in *every* request, a reply can no longer
  evict the pinned runner and reload the model at the server's own default,
  which is how a session that started fine used to run out of memory two turns
  later.
- `VT_LLM_MODEL` can also point at something smaller (`qwen3:8b`), which is the
  better answer if you want headroom rather than the biggest model that fits.
- It is an *uncensored* (abliterated) model. The system prompt still scopes it to
  language, culture and travel and still forbids markdown, but it will follow a
  request an aligned model would refuse. Bear that in mind before handing the mic
  to someone else.
- Runs half-duplex by default (does not listen while speaking) to prevent the
  speaker echoing into the mic and cutting off replies. Barge-in is opt-in via
  `VT_BARGE_IN=1` or the Configuration dialog, and needs headphones.
- macOS reports one error, `PaErrorCode -9986` / `Unspecified Audio Hardware
  Error`, for two unrelated causes, and the more likely one here is **memory**,
  not the device. CoreAudio wires (`mlock`) a 64 KB buffer per stream, and on a
  36 GB machine running a 25 GB model that allocation fails:
  `HALB_SharedBuffer::Lock: mlock failed: byte size 65536, errno 35` →
  `StartIOThread: the IO thread failed to start, Error: 2003329396` (`'what'`).
  Measured here, warming `Qwen3.8-Uncensored` at its default 131072 context took
  the machine to 3 % free with 32.1 GB of 36 GB wired and swap at 19.2 of 20 GB;
  every microphone open failed for the two minutes that lasted, while the mic
  itself was fine. So opening is retried on a backoff — ~30 s for the mic, which
  only delays the start of a session, and ~1.5 s for the speaker, where waiting
  means a sentence spoken late — and when it still fails, the app asks
  `memory_pressure` and says so instead of blaming the device. To avoid the spike
  rather than wait it out: a smaller `VT_LLM_CONTEXT`, a smaller
  `VT_LLM_MODEL`, or simply a minute between "model loaded" and pressing Start.
- The other cause is a genuinely unavailable device. `afplay
  /System/Library/Sounds/Ping.aiff` separates the two: if that fails too, it is
  `coreaudiod` rather than VirtualTutor. A stray client can also hold the mic —
  `log show --last 30s --predicate 'process == "coreaudiod"' | grep -c
  "node=-Input"` is non-zero while the input engine is running, so if nothing of
  yours is recording, something is squatting on the device.
- Changing the audio device is tolerated but not followed. PortAudio enumerates
  devices once, so a default device that changed since import used to fail the
  session; the list is now re-read when a session starts, and a device error
  mid-sentence drops that sentence and reopens on the next one instead of killing
  playback for the rest of the session. Switching to a device that appeared
  *during* a session still needs a stop and start, since re-enumerating would
  invalidate the open mic stream.
- Each utterance is captured with a 0.4 s pre-roll so the first word is not
  clipped. In half duplex the echo picked up during playback is flushed before
  listening resumes; in full duplex there is nothing to flush, and the audio
  leading up to a barge-in is kept so the interrupting words are transcribed.
- First run downloads model weights (Kokoro, Whisper, spaCy `en_core_web_sm`);
  the LLM comes from Ollama. Subsequent runs are offline except the local HTTP call.
- Mood cues depend on the model following an instruction. When it writes none,
  the mood is read off the text instead, so delivery is never flat — it is just
  less specific.
- The prompt asks for 1–3 sentences and mostly gets them in English; Chinese
  replies often run longer. They are spoken in full either way, since text is
  chunked under Kokoro's phoneme limit, but expect a paragraph rather than a line.
- Cultural and travel claims are the LLM's, and it states wrong ones
  confidently — in testing an 8B model told me not to use the left hand in Spain,
  which is not a Spanish custom. Treat it as conversation practice, not a guidebook.

## Layout

```
virtualtutor/
├── requirements.txt
├── requirements-dev.txt    # test-only dependencies
├── pytest.ini
├── models/                 # GGUF weights for the llama.cpp backend (gitignored)
├── samples/                # verification wav output
├── scripts/
│   ├── llm_env.sh          # backend, host and model, shared by the scripts
│   ├── download_model.sh   # ollama pull (or a GGUF for llama.cpp)
│   ├── start_all.sh        # backend + frontend, Ctrl+C stops both
│   ├── start_server.sh     # Ollama
│   ├── start_llamacpp_server.sh
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
    ├── emotion.py          # mood -> rate, pitch, gain, articulation
    ├── tts.py              # Kokoro synthesis, chunked under the phoneme limit
    ├── visemes.py          # IPA -> mouth shape timeline, with articulation levels
    ├── voice_agent.py      # pipeline threads, sentence split, think/cue filters
    ├── server.py           # local web server (stdlib only)
    └── verify.py
```
