#!/usr/bin/env bash
# Start everything VirtualTutor needs: the llama.cpp LLM backend and the web
# frontend. Ctrl+C stops both.
#
# The LLM's own log is noisy, so it goes to a file; the web server and the voice
# pipeline (transcripts, state) stay in the foreground.
#
# Both services listen on 127.0.0.1 only and have NO authentication -- this is a
# local single-user setup. See README for details.
#
# Usage: ./scripts/start_all.sh
set -uo pipefail
cd "$(dirname "$0")/.."

LLM_PORT="${VT_PORT:-8080}"
WEB_HOST="${VT_WEB_HOST:-127.0.0.1}"
WEB_PORT="${VT_WEB_PORT:-8800}"
LLM_LOG="${TMPDIR:-/tmp}/virtualtutor-llm.log"
LLM_TIMEOUT="${VT_LLM_TIMEOUT:-180}"   # seconds to wait for model load

llm_pid=""
web_pid=""
started_llm=0
shutting_down=0

port_busy() { lsof -nP -iTCP:"$1" -sTCP:LISTEN >/dev/null 2>&1; }

stop_pid() {
  # Graceful TERM, escalate to KILL if it refuses to go.
  local pid="$1" name="$2"
  [ -n "$pid" ] || return 0
  kill -0 "$pid" 2>/dev/null || return 0
  printf '  stopping %s (pid %s) ...\n' "$name" "$pid"
  kill -TERM "$pid" 2>/dev/null || true
  local i
  for i in $(seq 1 40); do
    kill -0 "$pid" 2>/dev/null || return 0
    sleep 0.25
  done
  printf '  %s did not exit, forcing\n' "$name"
  kill -KILL "$pid" 2>/dev/null || true
}

cleanup() {
  # Runs from both the signal trap and EXIT; only do the work once.
  [ "$shutting_down" -eq 1 ] && return
  shutting_down=1
  printf '\nStopping VirtualTutor ...\n'
  stop_pid "$web_pid" "web frontend"
  if [ "$started_llm" -eq 1 ]; then
    stop_pid "$llm_pid" "LLM server"
  elif [ -n "$llm_pid" ]; then
    printf '  leaving the LLM server running (it was already up before this script)\n'
  fi
  wait 2>/dev/null
  printf 'All stopped.\n'
}

trap cleanup INT TERM
trap cleanup EXIT

# --- 1. LLM backend ---------------------------------------------------------
if port_busy "$LLM_PORT"; then
  echo "LLM server already running on port $LLM_PORT -- reusing it."
else
  echo "Starting LLM server on 127.0.0.1:$LLM_PORT (log: $LLM_LOG) ..."
  ./scripts/start_server.sh > "$LLM_LOG" 2>&1 &
  llm_pid=$!
  started_llm=1

  printf 'Loading model '
  ready=0
  for _ in $(seq 1 "$LLM_TIMEOUT"); do
    if ! kill -0 "$llm_pid" 2>/dev/null; then
      echo
      echo "LLM server exited during startup. Last lines of $LLM_LOG:"
      tail -15 "$LLM_LOG"
      exit 1
    fi
    if curl -s "http://127.0.0.1:$LLM_PORT/health" 2>/dev/null | grep -q '"ok"'; then
      ready=1
      break
    fi
    printf '.'
    sleep 1
  done
  echo
  if [ "$ready" -ne 1 ]; then
    echo "LLM server did not become ready within ${LLM_TIMEOUT}s. See $LLM_LOG"
    exit 1
  fi
  echo "LLM server ready."
fi

# --- 2. Web frontend --------------------------------------------------------
if port_busy "$WEB_PORT"; then
  echo "Port $WEB_PORT is already in use -- stop whatever is on it, or set VT_WEB_PORT."
  exit 1
fi

echo "Starting web frontend ..."
./scripts/start_web.sh &
web_pid=$!

sleep 1
if ! kill -0 "$web_pid" 2>/dev/null; then
  echo "Web frontend failed to start."
  exit 1
fi

cat <<BANNER

  VirtualTutor is up.

    Open:  http://${WEB_HOST}:${WEB_PORT}
    LLM :  127.0.0.1:${LLM_PORT}  (log: ${LLM_LOG})

  Local only, no authentication. Press Ctrl+C to stop everything.

BANNER

# --- 3. Supervise -----------------------------------------------------------
# Poll rather than `wait` so Ctrl+C is handled promptly and so we notice if
# either service dies on its own.
while :; do
  if ! kill -0 "$web_pid" 2>/dev/null; then
    echo "Web frontend exited."
    break
  fi
  if [ "$started_llm" -eq 1 ] && ! kill -0 "$llm_pid" 2>/dev/null; then
    echo "LLM server exited. Last lines of $LLM_LOG:"
    tail -15 "$LLM_LOG"
    break
  fi
  sleep 1
done
