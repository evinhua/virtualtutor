#!/usr/bin/env bash
# Make sure the LLM VirtualTutor talks to is available locally.
#
# Default backend is Ollama, so this checks `ollama list` and pulls the model if
# it is missing. A model you imported yourself (from a GGUF, via `ollama create`)
# is not in the registry and cannot be pulled -- the script says so rather than
# failing with a bare error.
#
# With VT_LLM_BACKEND=llamacpp it downloads a GGUF from Hugging Face instead.
set -euo pipefail
cd "$(dirname "$0")/.."
# shellcheck source=scripts/llm_env.sh
. ./scripts/llm_env.sh

if [ "$VT_LLM_BACKEND" = "llamacpp" ]; then
  MODEL_REPO="${VT_MODEL_REPO:-bartowski/Qwen_Qwen3-8B-GGUF}"
  MODEL_FILE="${VT_MODEL_FILE:-Qwen_Qwen3-8B-Q5_K_M.gguf}"
  DEST="models/${MODEL_FILE}"

  if [[ -f "$DEST" ]]; then
    echo "Model already present: $DEST"
    exit 0
  fi

  echo "Downloading ${MODEL_REPO}/${MODEL_FILE} -> ${DEST}"
  # huggingface_hub was installed with the python deps; use its CLI.
  ./.venv/bin/python - "$MODEL_REPO" "$MODEL_FILE" <<'PY'
import sys
from huggingface_hub import hf_hub_download
repo, fname = sys.argv[1], sys.argv[2]
path = hf_hub_download(repo_id=repo, filename=fname, local_dir="models")
print("Downloaded to:", path)
PY
  echo "Done."
  exit 0
fi

if ! command -v ollama >/dev/null 2>&1; then
  echo "ollama is not installed. Install it with: brew install ollama" >&2
  exit 1
fi

# grep reads all of the output rather than exiting early: with `set -o
# pipefail`, a `grep -q` that closes the pipe first makes ollama exit 141 and the
# model look missing.
if ollama list | grep -F "$VT_LLM_MODEL" >/dev/null; then
  echo "Model already available in Ollama: ${VT_LLM_MODEL}"
  exit 0
fi

echo "Pulling ${VT_LLM_MODEL} ..."
if ! ollama pull "$VT_LLM_MODEL"; then
  cat >&2 <<MSG

Could not pull '${VT_LLM_MODEL}'.

If this is a model you built locally (for example with \`ollama create\` from a
GGUF), it only exists on the machine that made it -- recreate it there, or point
VirtualTutor at a model you do have:

  ollama list
  VT_LLM_MODEL='<name from that list>' ./scripts/start_all.sh
MSG
  exit 1
fi
echo "Done."
