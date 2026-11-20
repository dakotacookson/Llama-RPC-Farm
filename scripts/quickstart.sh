#!/usr/bin/env bash
# One-command solo quickstart: install + engines + starter model + launch + test.
#
#   curl -fsSL https://raw.githubusercontent.com/dakotacookson/Llama-RPC-Farm/main/scripts/quickstart.sh | bash
#
# Existing checkout? It is reused. Pass flags through to src/quickstart.py:
#   ./scripts/quickstart.sh --recipe qwen3-4b
set -euo pipefail

REPO="${LLAMA_FARM_REPO:-https://github.com/dakotacookson/Llama-RPC-Farm.git}"
DEST="${LLAMA_FARM_DIR:-$HOME/Llama-RPC-Farm}"
APP_URL="${FARM_URL:-http://127.0.0.1:8199}"

say() { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }

if ! command -v git >/dev/null 2>&1 && [ ! -d "$DEST/.git" ]; then
  echo "ERROR: git is required for the source quickstart." >&2
  exit 1
fi

if [ -d "$DEST/.git" ]; then
  say "Using existing checkout: $DEST"
else
  say "Cloning Llama RPC Farm into $DEST"
  git clone --depth 1 "$REPO" "$DEST"
fi

cd "$DEST"

if [ ! -x ".venv/bin/python" ] && [ ! -x ".venv/Scripts/python.exe" ]; then
  say "Installing the farm app (one time)"
  ./install.sh
fi

if ! curl -fsS --max-time 3 "$APP_URL/api/machines" >/dev/null 2>&1; then
  say "Starting the farm UI"
  ./run.sh >/dev/null
  for _ in $(seq 1 60); do
    curl -fsS --max-time 3 "$APP_URL/api/machines" >/dev/null 2>&1 && break
    sleep 1
  done
fi

PY="$DEST/.venv/bin/python"
[ -x "$PY" ] || PY="$DEST/.venv/Scripts/python.exe"

say "Running quickstart"
exec "$PY" "$DEST/src/quickstart.py" "$@"
