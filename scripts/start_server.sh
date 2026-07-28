#!/usr/bin/env bash
# Start the llama.cpp OpenAI-compatible server with full Metal GPU offload.
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL_FILE="${VT_MODEL_FILE:-Qwen2.5-7B-Instruct-Q4_K_M.gguf}"
MODEL_PATH="models/${MODEL_FILE}"
PORT="${VT_PORT:-8080}"

if [[ ! -f "$MODEL_PATH" ]]; then
  echo "Model not found at $MODEL_PATH"
  echo "Run: ./scripts/download_model.sh"
  exit 1
fi

# -ngl 99 : offload all layers to the Apple Metal GPU
# -c 4096 : context window
# --host 127.0.0.1 keeps the server local-only (no external exposure)
exec llama-server \
  -m "$MODEL_PATH" \
  --host 127.0.0.1 \
  --port "$PORT" \
  -ngl 99 \
  -c 4096 \
  --jinja
