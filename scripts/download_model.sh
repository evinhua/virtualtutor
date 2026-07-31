#!/usr/bin/env bash
# Download the LLM GGUF used by llama-server.
# Default: Qwen3-8B Q5_K_M (~5.9 GB) - stronger than Qwen2.5-7B and still fast
# enough for voice on an M3 Pro. Qwen3 is a hybrid reasoning model; thinking is
# switched off at the server (see start_server.sh), which voice needs anyway.
set -euo pipefail
cd "$(dirname "$0")/.."

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
