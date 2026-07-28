#!/usr/bin/env bash
# Start the VirtualTutor web frontend (local only, no authentication).
# The LLM server must be running separately: ./scripts/start_server.sh
set -euo pipefail
cd "$(dirname "$0")/.."

exec ./.venv/bin/python src/server.py
