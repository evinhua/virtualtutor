# Mission

## Project

**VirtualTutor** — a continuous, real-time voice conversation tutor that runs
fully on-device on Apple Silicon.

## One-line description

Speak to it, it thinks, and it talks back — a low-latency, offline voice tutor
built entirely from local models in unified memory.

## Problem statement

Learning through conversation is powerful, but most voice AI tutors depend on
cloud APIs. That creates three problems:

- **Privacy** — spoken audio and transcripts leave the user's machine.
- **Latency & connectivity** — round-trips to remote servers add delay and
  require a stable network.
- **Cost & lock-in** — per-request pricing and vendor dependence.

VirtualTutor removes all three by running the full speech pipeline (VAD → STT →
LLM → TTS) locally on Apple Silicon, with all four models resident in unified
memory for responsive, natural back-and-forth conversation.

## Target audience

- **Learners** who want a patient, always-available conversational tutor for
  explaining concepts out loud.
- **Privacy-conscious users** who want their voice and conversations to stay on
  their own machine.
- **Developers & researchers** exploring fully local, real-time voice agent
  pipelines on Apple Silicon (MLX, llama.cpp).

## Success criteria

- **Fully offline** operation after the first model download — no cloud calls
  beyond the local HTTP endpoint.
- **Low perceived latency** — the tutor begins speaking as soon as the first
  sentence is ready, not after the full reply is generated.
- **Natural turn-taking** — reliable end-of-speech detection (~0.8 s pause), and
  optional barge-in that a short interjection is enough to trigger and that keeps
  the words you interrupted with, instead of stopping the tutor and then having
  nothing to answer.
- **Clean transcription** — no hallucinated filler phrases reaching the LLM.
- **Settings you can change while talking** — the voice for each language applies
  from the tutor's next sentence and the duplex mode from the next mic frame, with
  no restart and no lost conversation.
- **A face that matches the voice** — lip-sync driven by the actual phonemes
  being spoken, accurate to the millisecond, with no additional model, and
  moving smoothly rather than snapping between shapes.
- **A personality you can change** — personas are part of the tutor's identity,
  selectable at startup and switchable mid-conversation.
- **Multilingual without configuration** — speak English, Spanish or Chinese and
  the tutor detects the language, replies in it, and speaks it back with the
  correct voice and lip-sync.
- **Start and stop are dependable** — a session can be stopped and restarted at
  any moment, including mid-reply, without a stuck thread or a stale persona.
- **Runs within 36 GB** unified memory on an Apple M3 Pro with all models loaded
  simultaneously.

## Core principles

- **On-device first** — privacy and independence from the cloud are
  non-negotiable defaults.
- **Low latency by design** — stream and pipeline every stage; never wait for a
  whole response when a sentence will do.
- **Concurrency without collision** — VAD, brain (STT+LLM), synthesis and
  playback run in separate threads communicating via queues, and a session that
  is shutting down must never bleed into the next one.
- **Robustness over cleverness** — defensive layers (energy gates, confidence
  thresholds, phrase blocklists, echo flushing) keep the experience stable.
- **Conversational, spoken-first output** — short, plain-language replies suited
  to text-to-speech, never markdown or code blocks.
- **Configurable, not hardcoded** — every model and behavior knob is overridable
  via environment variables, and the settings worth changing mid-conversation are
  read where they are used rather than captured at startup, so a dialog can change
  them live without a restart or a second copy of the state.
- **Reuse what the models already compute** — the mouth is animated from
  Kokoro's own duration predictor rather than a second model, and the avatar is
  aligned with the tensor library already in the dependency list.
- **No dependency the feature does not need** — the UI is stdlib HTTP and static
  files, with no framework, bundler, or build step.
- **The fragile logic is tested** — timeline construction, sentence splitting,
  transcription guards, history trimming, persona switching, runtime settings, the
  barge-in decision and the HTTP layer are covered by fast tests that need no
  models, and a fix for a reported bug starts by reproducing it. Where the bug is
  in audio behavior, the reproduction drives the real worker through a fake
  microphone rather than a mocked one.
- **Claims are measured** — smoothness, alignment quality, asset size and how long
  an interruption takes to register are reported as numbers rather than
  adjectives, so a change can be shown to be an improvement.
