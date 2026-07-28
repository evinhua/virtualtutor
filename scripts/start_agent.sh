#!/usr/bin/env bash
# Start the VirtualTutor voice agent (mic -> VAD -> STT -> LLM -> TTS -> speakers).
set -euo pipefail
cd "$(dirname "$0")/.."
exec ./.venv/bin/python src/voice_agent.py
