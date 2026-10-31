#!/usr/bin/env bash
# install.sh — one-shot installer for Llama RPC Farm.
#
# - ensures a suitable Python 3.11+ (downloads a local build via uv if the
#   system python is too old or missing — no sudo needed)
# - creates .venv in the project root
# - installs everything from requirements.txt
#
# Safe to re-run; idempotent.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"
REQ="$SCRIPT_DIR/requirements.txt"
UV_DIR="$SCRIPT_DIR/.uv-bin"
MIN_MINOR=11   # we need >= 3.11

log() { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }
die() { printf '\n\033[1;31mERROR: %s\033[0m\n' "$*" >&2; exit 1; }

python_ok() {
  # $1 = python binary; succeeds if version >= 3.$MIN_MINOR
  local v
  v="$("$1" -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")' 2>/dev/null)" || return 1
  local major minor
  major="${v%%.*}"; minor="${v##*.}"
  [ "$major" -eq 3 ] && [ "$minor" -ge "$MIN_MINOR" ]
}

# ---------------------------------------------------------------- python
PY=""
for cand in python3 python3.13 python3.12 python3.11 python; do
  if command -v "$cand" >/dev/null 2>&1 && python_ok "$(command -v "$cand")"; then
    PY="$(command -v "$cand")"
    break
  fi
done

if [ -z "$PY" ]; then
  log "No Python >= 3.$MIN_MINOR found — downloading a local build with uv"
  mkdir -p "$UV_DIR"
  UV="$UV_DIR/uv"
  if [ ! -x "$UV" ]; then
    ARCH="$(uname -m)"
    case "$(uname -s)/$ARCH" in
      Linux/x86_64)  UV_ARCH="x86_64-unknown-linux-gnu" ;;
      Linux/aarch64|Linux/arm64) UV_ARCH="aarch64-unknown-linux-gnu" ;;
      Darwin/x86_64) UV_ARCH="x86_64-apple-darwin" ;;
      Darwin/arm64|Darwin/aarch64) UV_ARCH="aarch64-apple-darwin" ;;
      *) die "unsupported platform/arch: $(uname -s) $ARCH" ;;
    esac
    curl -fSL --retry 3 -o "$UV.tar.gz" \
      "https://github.com/astral-sh/uv/releases/latest/download/uv-$UV_ARCH.tar.gz"
    tar -xzf "$UV.tar.gz" -C "$UV_DIR" --strip-components=1
    rm -f "$UV.tar.gz"
  fi
  "$UV" python install 3.12
  "$UV" venv --python 3.12 "$VENV_DIR"
  PY="$VENV_DIR/bin/python"
else
  log "Using Python: $PY ($("$PY" -V 2>&1))"
fi

# ---------------------------------------------------------------- venv
if [ ! -x "$VENV_DIR/bin/python" ] && [ ! -x "$VENV_DIR/Scripts/python.exe" ]; then
  log "Creating virtualenv at $VENV_DIR"
  "$PY" -m venv "$VENV_DIR"
fi

# pip bootstrap (venvs created by uv-managed pythons may lack it)
VENV_PY="$VENV_DIR/bin/python"
if [ -x "$VENV_DIR/Scripts/python.exe" ]; then VENV_PY="$VENV_DIR/Scripts/python.exe"; fi
"$VENV_PY" -m pip --version >/dev/null 2>&1 || "$VENV_PY" -m ensurepip >/dev/null 2>&1 || true

log "Installing requirements"
"$VENV_PY" -m pip install --upgrade pip -q
"$VENV_PY" -m pip install -r "$REQ"

log "Done."
echo
echo "Next steps:"
echo "  $SCRIPT_DIR/run.sh          # start the web UI (background, survives logout)"
echo "  http://127.0.0.1:8199"
