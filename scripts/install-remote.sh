#!/usr/bin/env bash
# One-line source installer:
#   curl -fsSL https://raw.githubusercontent.com/dakotacookson/Llama-RPC-Farm/main/scripts/install-remote.sh | bash
set -euo pipefail

REPO="${LLAMA_FARM_REPO:-https://github.com/dakotacookson/Llama-RPC-Farm.git}"
DEST="${LLAMA_FARM_DIR:-$HOME/Llama-RPC-Farm}"

if ! command -v git >/dev/null 2>&1; then
  echo "Llama RPC Farm needs git to install from source." >&2
  exit 1
fi

if [ -d "$DEST/.git" ]; then
  echo "==> Updating existing checkout at $DEST"
  git -C "$DEST" pull --ff-only
else
  if [ -e "$DEST" ] && [ -n "$(ls -A "$DEST" 2>/dev/null || true)" ]; then
    echo "ERROR: $DEST exists and is not an empty directory." >&2
    echo "Move it aside or set LLAMA_FARM_DIR to a different path." >&2
    exit 1
  fi
  echo "==> Cloning Llama RPC Farm into $DEST"
  git clone --depth 1 "$REPO" "$DEST"
fi

cd "$DEST"
exec ./install.sh
