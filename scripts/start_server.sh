#!/usr/bin/env bash
# Start the LLM backend: Ollama, serving the model VirtualTutor talks to.
#
# Ollama is talked to through its native /api/chat endpoint, which is
# the only one that accepts options.num_ctx -- so src/voice_agent.py pins the
# context (and thinking off) per request; see config.llm_payload.
#
# If Ollama is already serving -- on macOS it usually is, as a background
# service -- this only loads the model and exits. Otherwise it runs `ollama
# serve` in the foreground, so Ctrl+C stops it.
#
# For the llama.cpp backend instead: VT_LLM_BACKEND=llamacpp, and see
# ./scripts/start_llamacpp_server.sh.
set -euo pipefail
cd "$(dirname "$0")/.."
# shellcheck source=scripts/llm_env.sh
. ./scripts/llm_env.sh

if [ "$VT_LLM_BACKEND" = "llamacpp" ]; then
  exec ./scripts/start_llamacpp_server.sh
fi

if ! command -v ollama >/dev/null 2>&1; then
  echo "ollama is not installed. Install it with: brew install ollama" >&2
  exit 1
fi

if llm_ready; then
  echo "Ollama already serving on ${LLM_HOST}."
  if ! ollama list | grep -F "$VT_LLM_MODEL" >/dev/null; then
    echo "Model '${VT_LLM_MODEL}' is not in \`ollama list\`." >&2
    echo "Pull or import it first: ./scripts/download_model.sh" >&2
    exit 1
  fi
  echo "Loading ${VT_LLM_MODEL} into memory at a ${VT_LLM_CONTEXT}-token context ..."
  llm_warm
  echo "Ready. Model stays loaded for 30 minutes of idleness."
  exit 0
fi

echo "Starting Ollama on ${LLM_HOST} (Ctrl+C to stop) ..."
exec ollama serve
