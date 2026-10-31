#!/usr/bin/env bash
# run.sh — start the Llama RPC Farm web UI.
#
# - uses the project .venv (falls back to system python3)
# - frees port 8199 if a stale instance is bound
# - runs detached (setsid, nohup fallback on macOS) so it survives terminal close / logout
# - idempotent: safe to run repeatedly; it just (re)starts the app
#
# stop with:  fuser -k 8199/tcp   (or the dashboard's Restart button)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PORT=8199
LOG=/tmp/llama-web.log

log() { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }

# already running? report and exit
if curl -s -m 2 "http://127.0.0.1:$PORT/" 2>/dev/null | grep -qi "llama-swap\|Llama RPC Farm"; then
  echo "Already running at http://127.0.0.1:$PORT (restart via dashboard or: fuser -k $PORT/tcp)"
  exit 0
fi

# venv python, then fall back to system python3
if [ -x "$SCRIPT_DIR/.venv/bin/python" ]; then
  PY="$SCRIPT_DIR/.venv/bin/python"
elif [ -x "$SCRIPT_DIR/.venv/Scripts/python.exe" ]; then
  PY="$SCRIPT_DIR/.venv/Scripts/python.exe"
else
  log "No .venv found — run ./install.sh first (falling back to system python3)"
  PY="$(command -v python3)"
fi

# free a stale port without pkill -f self-match issues
fuser -k "$PORT"/tcp >/dev/null 2>&1 || true
sleep 1

log "Starting web UI (python: $PY)"
# setsid (Linux) detaches from this terminal; macOS lacks it -> nohup+disown
if command -v setsid >/dev/null 2>&1; then
  setsid -f "$PY" "$SCRIPT_DIR/src/app.py" > "$LOG" 2>&1
else
  nohup "$PY" "$SCRIPT_DIR/src/app.py" > "$LOG" 2>&1 &
  disown
fi

# wait for it to come up (max ~10s)
for _ in $(seq 1 10); do
  sleep 1
  if curl -s -m 2 "http://127.0.0.1:$PORT/" 2>/dev/null | grep -qi "llama-swap\|Llama RPC Farm"; then
    echo "Web UI running at http://127.0.0.1:$PORT"
    echo "Logs: $LOG"
    exit 0
  fi
done

echo "UI did not answer on :$PORT yet — check $LOG" >&2
tail -n 20 "$LOG" 2>/dev/null || true
exit 1
