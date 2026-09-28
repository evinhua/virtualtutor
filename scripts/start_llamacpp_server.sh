#!/usr/bin/env bash
# Start the llama.cpp OpenAI-compatible server with full Metal GPU offload.
#
# This is the alternative backend, used when VT_LLM_BACKEND=llamacpp. The
# default backend is Ollama; see ./scripts/start_server.sh.
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL_FILE="${VT_MODEL_FILE:-Qwen_Qwen3-8B-Q5_K_M.gguf}"
MODEL_PATH="models/${MODEL_FILE}"
PORT="${VT_PORT:-8080}"

if [[ ! -f "$MODEL_PATH" ]]; then
  echo "Model not found at $MODEL_PATH"
  echo "Run: VT_LLM_BACKEND=llamacpp ./scripts/download_model.sh"
  exit 1
fi

# -ngl 99 : offload all layers to the Apple Metal GPU
# -c 4096 : context window
# --host 127.0.0.1 keeps the server local-only (no external exposure)
# --reasoning-budget 0 : Qwen3 is a hybrid reasoning model and thinks before it
#   answers by default. In a voice loop that is dead air (and, if the tags leak,
#   a chain of thought read aloud), so the template is told to close thinking
#   immediately. Harmless on a model without a reasoning template.
exec llama-server \
  -m "$MODEL_PATH" \
  --host 127.0.0.1 \
  --port "$PORT" \
  -ngl 99 \
  -c 4096 \
  --jinja \
  --reasoning-budget 0
