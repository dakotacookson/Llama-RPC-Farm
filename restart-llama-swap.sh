#!/bin/bash
# restart-llama-swap.sh - kills llama processes then restarts web UI
# Safe: uses pkill without -f so script path does NOT match itself
# Uses fuser to free port 8199 (avoids pkill -f self-match hang)

set -e

echo "Stopping llama-swap and llama-server (if running)..."
# llama-swap runs under a systemd --user unit (Restart=on-failure). A bare
# pkill delivers SIGTERM, swap exits CLEANLY (status 0) and systemd does
# NOT restart it — leaving :8080 dead until Start All. A true restart must
# bounce the unit: stop removes the unit cgroup so swap cannot linger,
# start relaunches it under systemd supervision.
if command -v systemctl >/dev/null 2>&1 && \
   systemctl --user list-unit-files 2>/dev/null | grep -q "^llama-swap.service"; then
  systemctl --user stop llama-swap 2>/dev/null || true
  pkill llama-server 2>/dev/null || echo "llama-server not running"
else
  # macOS / no unit: fall back to process kills by exact name
  if command -v pkill >/dev/null 2>&1; then
    pkill llama-swap 2>/dev/null || echo "llama-swap not running"
    pkill llama-server 2>/dev/null || echo "llama-server not running"
  else
    killall llama-swap 2>/dev/null || echo "llama-swap not running"
    killall llama-server 2>/dev/null || echo "llama-server not running"
  fi
fi
# free web UI port - kills old python src/app.py without matching this script
fuser -k 8199/tcp > /dev/null 2>&1 || echo "port 8199 free"
sleep 1
# belt+braces: fuser can miss (not installed / different netns) and any
# stale daemon from an old checkout may be the one bound to :8199 serving
# STALE ui files. Kill any app daemon directly so a restart always
# replaces the server process.
# NOTE: pkill -f matches full command lines; this script's own command line
# is ".../restart-llama-swap.sh" which does NOT contain "src/app.py", so it
# cannot match itself.
pkill -f "src/app.py" 2>/dev/null || true
sleep 1
if (ss -ltn 2>/dev/null || netstat -ltn 2>/dev/null) | grep -q ":8199 "; then
  echo "WARNING: port 8199 still bound after kill — old daemon may still serve stale UI"
  fuser -k 8199/tcp > /dev/null 2>&1 || true
  sleep 1
fi

# Relaunch swap under systemd (true restart — unit was stopped above).
# llama-swap picks up its config at launch, so the app's boot-reconcile
# push lands before/while it starts; --watch-config covers later edits.
if command -v systemctl >/dev/null 2>&1 && \
   systemctl --user list-unit-files 2>/dev/null | grep -q "^llama-swap.service"; then
  echo "Relaunching llama-swap via systemd..."
  systemctl --user start llama-swap
  sleep 1
fi

echo "Starting web UI with venv python..."
# Resolve project root from this script's location; prefer project .venv,
# fall back to current python3. No hardcoded paths.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -x "$SCRIPT_DIR/.venv/bin/python" ]; then
  PY="$SCRIPT_DIR/.venv/bin/python"
elif [ -x "$SCRIPT_DIR/.venv/Scripts/python.exe" ]; then
  PY="$SCRIPT_DIR/.venv/Scripts/python.exe"
else
  PY="$(command -v python3)"
fi
# Use setsid so it survives terminal close (nohup fallback for macOS,
# which lacks setsid); direct python path avoids venv activate issues
if command -v setsid >/dev/null 2>&1; then
  setsid -f "$PY" "$SCRIPT_DIR/src/app.py" > /tmp/llama-web.log 2>&1
else
  nohup "$PY" "$SCRIPT_DIR/src/app.py" > /tmp/llama-web.log 2>&1 &
fi

sleep 2

if curl -s -m 3 http://127.0.0.1:8199/ 2>/dev/null | grep -qi "llama-swap\|Llama RPC Farm"; then
  echo "Web UI running at http://127.0.0.1:8199"
else
  echo "Web UI starting... check /tmp/llama-web.log"
  tail -n 20 /tmp/llama-web.log 2>/dev/null || true
fi
# Cache-bust check: index.html carries /static/app.js?v=<mtime>-<boot>, where
# <boot> is unique per daemon start. A changed v= proves the browser can no
# longer serve the pre-restart cached app.js. Also verify the served app.js
# actually contains the latest fix marker.
V=$(curl -s -m 5 http://127.0.0.1:8199/ 2>/dev/null | grep -o 'app\.js?v=[^"]*' | head -1 || true)
if [ -n "$V" ]; then
  echo "Asset: /static/$V"
  if curl -s -m 10 "http://127.0.0.1:8199/static/$V" 2>/dev/null | grep -q "STAR topology"; then
    echo "Fix verified: served app.js contains the all-legs debug fix"
  else
    echo "WARNING: served app.js lacks the all-legs fix marker — stale checkout or old daemon?"
  fi
else
  echo "WARNING: could not read asset version from / (UI may still be booting)"
fi
echo "Serving checkout: $SCRIPT_DIR"
echo "Logs: /tmp/llama-web.log"
echo "Done."
