#!/usr/bin/env bash
# restart.sh — one entry point for restarting the farm.
#
#   ./restart.sh            # FULL: stop llama-swap + every farm-managed
#                           #       llama/RPC process on ALL machines, then
#                           #       restart the swap daemon and the web UI.
#   ./restart.sh --full     # same as above (explicit)
#   ./restart.sh --ui       # UI-ONLY: restart just the web app on :8199.
#                           #          Leaves llama-swap and loaded models
#                           #          (and their VRAM) untouched.
#
# `run.sh` remains start-if-not-running (it never restarts anything).
#
# Safe: pkill patterns cannot match this script. Port 8199 is freed with
# fuser and the old daemon is also killed by its `src/app.py` command line.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PORT=8199
LOG=/tmp/llama-web.log

usage() {
  sed -n '2,12p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
}

MODE=full
case "${1:-}" in
  ""|--full|full) MODE=full ;;
  --ui|ui)        MODE=ui ;;
  -h|--help|help) usage; exit 0 ;;
  *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
esac

if [ -x "$SCRIPT_DIR/.venv/bin/python" ]; then
  PY="$SCRIPT_DIR/.venv/bin/python"
elif [ -x "$SCRIPT_DIR/.venv/Scripts/python.exe" ]; then
  PY="$SCRIPT_DIR/.venv/Scripts/python.exe"
else
  PY="$(command -v python3)"
fi

free_ui_port() {
  # fuser kills the :8199 owner without pkill -f matching this script.
  fuser -k "$PORT"/tcp >/dev/null 2>&1 || echo "port $PORT free"
  sleep 1
  # belt+braces: fuser can miss (not installed / different netns) and a stale
  # daemon from an old checkout may serve stale UI. This script's own command
  # line ("restart.sh") does not contain "src/app.py", so it cannot self-match.
  pkill -f "src/app.py" 2>/dev/null || true
  sleep 1
  if (ss -ltn 2>/dev/null || netstat -ltn 2>/dev/null) | grep -q ":$PORT "; then
    echo "WARNING: port $PORT still bound after kill — old daemon may serve stale UI"
    fuser -k "$PORT"/tcp >/dev/null 2>&1 || true
    sleep 1
  fi
}

start_ui() {
  echo "Starting web UI (python: $PY)..."
  # setsid survives terminal close; nohup is the macOS fallback.
  if command -v setsid >/dev/null 2>&1; then
    setsid -f "$PY" "$SCRIPT_DIR/src/app.py" > "$LOG" 2>&1
  else
    nohup "$PY" "$SCRIPT_DIR/src/app.py" > "$LOG" 2>&1 &
  fi
  sleep 2
  if curl -s -m 3 "http://127.0.0.1:$PORT/" 2>/dev/null | grep -qi "llama-swap\|Llama RPC Farm"; then
    echo "Web UI running at http://127.0.0.1:$PORT"
  else
    echo "Web UI starting... check $LOG"
    tail -n 20 "$LOG" 2>/dev/null || true
  fi
  V=$(curl -s -m 5 "http://127.0.0.1:$PORT/" 2>/dev/null | grep -o 'app\.js?v=[^"]*' | head -1 || true)
  [ -n "$V" ] && echo "Asset: /static/$V"
  echo "Logs: $LOG"
}

stop_swap_local() {
  echo "Stopping llama-swap and local llama-server..."
  # A bare pkill is not a true restart: llama-swap exits 0 under systemd
  # (Restart=on-failure), so the unit is bounced explicitly when present.
  if command -v systemctl >/dev/null 2>&1 && \
     systemctl --user list-unit-files 2>/dev/null | grep -q "^llama-swap.service"; then
    systemctl --user stop llama-swap 2>/dev/null || true
    pkill llama-server 2>/dev/null || echo "llama-server not running"
  elif command -v pkill >/dev/null 2>&1; then
    pkill llama-swap 2>/dev/null || echo "llama-swap not running"
    pkill llama-server 2>/dev/null || echo "llama-server not running"
  else
    killall llama-swap 2>/dev/null || echo "llama-swap not running"
    killall llama-server 2>/dev/null || echo "llama-server not running"
  fi
}

stop_all_llama() {
  # llama-swap's daemon shutdown does not run each model's cmdStop, so
  # remote-local llama-servers and per-port ggml-rpc-servers outlive it.
  # Stop every farm-managed process on every machine (best-effort:
  # offline boxes must not block the restart).
  [ -n "$PY" ] && [ -f "$SCRIPT_DIR/src/app.py" ] || return 0
  echo "Stopping farm llama processes on all machines..."
  ( cd "$SCRIPT_DIR" && "$PY" - <<'PYEOF' ) || true
import sys
sys.path.insert(0, "src")
import config_loader as cfg
import rpc_manager
import swap_manager
for m in cfg.machines.all():
    try:
        rpc_manager.stop_rpc(m["name"])
    except Exception as e:
        print(f"[restart] stop_rpc {m.get('name')}: {e}")
try:
    for r in swap_manager.stop_custom_runtimes():
        print(f"[restart] custom {r.get('model_id')}: "
              f"{'stopped' if r.get('killed') else r.get('error') or 'still running'}")
except Exception as e:
    print(f"[restart] custom runtimes: {e}")
PYEOF
}

start_swap() {
  if command -v systemctl >/dev/null 2>&1 && \
     systemctl --user list-unit-files 2>/dev/null | grep -q "^llama-swap.service"; then
    echo "Relaunching llama-swap via systemd..."
    systemctl --user start llama-swap
    sleep 1
  fi
}

if [ "$MODE" = "ui" ]; then
  echo "Restarting web UI only (llama-swap and models left running)..."
  free_ui_port
  start_ui
  echo "Serving checkout: $SCRIPT_DIR"
  echo "Done (UI-only)."
  exit 0
fi

echo "Full restart: stopping daemons/models on all machines..."
stop_swap_local
stop_all_llama
free_ui_port
start_swap
start_ui
echo "Serving checkout: $SCRIPT_DIR"
echo "Done (full)."
