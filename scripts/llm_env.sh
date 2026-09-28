#!/usr/bin/env bash
# Shared LLM backend settings, sourced by the other scripts.
#
# Two backends are supported and both speak the OpenAI-compatible protocol the
# agent uses; only the URL, the model name and how thinking is disabled differ.
# Keep the defaults in step with src/config.py.
#
#   ollama   (default) -- a model already pulled into Ollama
#   llamacpp           -- llama-server with a local GGUF

VT_LLM_BACKEND="${VT_LLM_BACKEND:-ollama}"
VT_LLM_MODEL="${VT_LLM_MODEL:-Qwen3.8-Uncensored:latest}"
# Context the model is loaded with. Ollama otherwise uses the model's trained
# length -- 131072 for Qwen3 -- and sizes its KV cache for it, which on a 36 GB
# machine is the difference between a session that starts and one that cannot
# open the microphone: measured here, the same model is 25 GB resident at 131072
# and 17 GB at 8192, leaving 2 % of memory free against 28 %. CoreAudio has to
# wire a 64 KB buffer per stream, and at 2 % free the kernel refuses
# ("mlock failed ... errno 35"), which surfaces as PaErrorCode -9986.
VT_LLM_CONTEXT="${VT_LLM_CONTEXT:-8192}"

if [ "$VT_LLM_BACKEND" = "llamacpp" ]; then
  LLM_HOST="${VT_LLAMACPP_HOST:-http://localhost:8080}"
  LLM_PORT="${VT_PORT:-8080}"
  LLM_PROBE_URL="${LLM_HOST}/health"
  LLM_PROBE_MATCH='"ok"'
  LLM_START_SCRIPT="./scripts/start_llamacpp_server.sh"
else
  LLM_HOST="${VT_OLLAMA_HOST:-http://localhost:11434}"
  LLM_PORT="${VT_OLLAMA_PORT:-11434}"
  LLM_PROBE_URL="${LLM_HOST}/api/version"
  LLM_PROBE_MATCH='version'
  LLM_START_SCRIPT="./scripts/start_server.sh"
fi

llm_ready() {
  curl -fsS -m 2 "$LLM_PROBE_URL" 2>/dev/null | grep -q "$LLM_PROBE_MATCH"
}

# Load the model into memory (and keep it there) so the first spoken reply is
# not waiting on a 17 GB read. This is also where the context length is pinned:
# the runner Ollama starts here is the one every later request reuses, so
# num_ctx set at load time governs the whole session. Harmless on llama.cpp,
# which has no such call (its context is a --ctx-size flag at startup).
llm_warm() {
  [ "$VT_LLM_BACKEND" = "ollama" ] || return 0
  curl -fsS -m 600 "${LLM_HOST}/api/chat" \
    -d "{\"model\":\"${VT_LLM_MODEL}\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"think\":false,\"stream\":false,\"keep_alive\":\"30m\",\"options\":{\"num_predict\":1,\"num_ctx\":${VT_LLM_CONTEXT}}}" \
    >/dev/null
}
