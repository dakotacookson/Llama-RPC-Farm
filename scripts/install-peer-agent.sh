#!/usr/bin/env bash
# Install/update the peer agent on THIS Linux host and enable it at boot.
#
# Usage:
#   PEER_AGENT_TOKEN='long-random-token' ./scripts/install-peer-agent.sh [wheel]
#
# With no wheel argument, installs from ./peer_agent in this checkout.
# The systemd --user unit is enabled and linger is turned on so the agent
# starts after reboot even when the user is not logged in.
#
# A suitable Python (>= 3.10) is required. If the system one is too old or
# lacks venv/pip (Ubuntu 18.04/Jetson, minimal Debian), a user-local Python
# 3.11 is downloaded via uv — no sudo.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TOKEN="${PEER_AGENT_TOKEN:-}"
WHEEL="${1:-}"
AGENT_HOME="$HOME/.llama-peer-agent"
VENV="$AGENT_HOME/venv"
ENV_DIR="$HOME/.config/llama-peer-agent"
ENV_FILE="$ENV_DIR/agent.env"
UNIT_DIR="$HOME/.config/systemd/user"
UNIT="$UNIT_DIR/llama-peer-agent.service"
MIN_MINOR=10

python_ok() {
  local v major minor
  v="$("$1" -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")' 2>/dev/null)" || return 1
  major="${v%%.*}"; minor="${v##*.}"
  [ "$major" -eq 3 ] && [ "$minor" -ge "$MIN_MINOR" ]
}

find_python() {
  local cand p
  for cand in python3 python3.13 python3.12 python3.11 python3.10; do
    p="$(command -v "$cand" 2>/dev/null || true)"
    [ -n "$p" ] && python_ok "$p" && { printf '%s\n' "$p"; return 0; }
  done
  return 1
}

ensure_uv() {
  local uv="$AGENT_HOME/.uv-bin/uv"
  # A previously downloaded uv may be unusable on this glibc (e.g. an old
  # Ubuntu/Jetson). Verify it actually runs before reusing it.
  if [ -x "$uv" ] && "$uv" --version >/dev/null 2>&1; then
    printf '%s\n' "$uv"; return 0
  fi
  rm -f "$uv"
  local tmp
  case "$(uname -s)/$(uname -m)" in
    # musl builds are static and also run on old glibc (Ubuntu 18.04 /
    # Jetson L4T); the gnu uv build requires glibc >= 2.28.
    Linux/x86_64) UV_ARCH="x86_64-unknown-linux-musl" ;;
    Linux/aarch64|Linux/arm64) UV_ARCH="aarch64-unknown-linux-musl" ;;
    *) echo "error: unsupported platform/arch for Python bootstrap: $(uname -s) $(uname -m)" >&2; return 1 ;;
  esac
  command -v curl >/dev/null 2>&1 || { echo "error: curl is required to bootstrap Python" >&2; return 1; }
  command -v tar  >/dev/null 2>&1 || { echo "error: tar is required to bootstrap Python" >&2; return 1; }
  mkdir -p "$AGENT_HOME/.uv-bin"
  tmp="$AGENT_HOME/.uv-bin/.tmp.$$"
  rm -rf "$tmp"; mkdir -p "$tmp"
  echo "==> Downloading uv ($UV_ARCH)" >&2
  curl -fLsS --retry 3 -o "$tmp/uv.tar.gz" \
    "https://github.com/astral-sh/uv/releases/latest/download/uv-$UV_ARCH.tar.gz"
  tar -xzf "$tmp/uv.tar.gz" -C "$tmp" --strip-components=1
  mv "$tmp/uv" "$uv"
  chmod +x "$uv"
  rm -rf "$tmp"
  printf '%s\n' "$uv"
}

if [[ -z "$TOKEN" ]]; then
  echo "error: PEER_AGENT_TOKEN is required (export it or pass it inline)" >&2
  exit 2
fi
if ! command -v systemctl >/dev/null 2>&1; then
  echo "error: systemctl is required (Linux systemd user service)" >&2
  exit 2
fi

mkdir -p "$AGENT_HOME" "$ENV_DIR" "$UNIT_DIR"
chmod 700 "$AGENT_HOME" "$ENV_DIR"
if [ -x "$VENV/bin/python" ] && ! python_ok "$VENV/bin/python"; then
  echo "==> Replacing too-old virtualenv at $VENV"
  rm -rf "$VENV"
fi

export UV_PYTHON_INSTALL_DIR="$AGENT_HOME/.uv-python"

PY="$(find_python || true)"
UV=""
if [ -z "$PY" ]; then
  echo "==> No Python >= 3.$MIN_MINOR found — bootstrapping Python 3.11 with uv"
  UV="$(ensure_uv)"
  "$UV" python install 3.11
  PY="$("$UV" python find 3.11)"
fi

USE_UV=0
if [ -n "$UV" ]; then
  USE_UV=1
else
  if [ ! -x "$VENV/bin/python" ]; then
    if ! "$PY" -m venv "$VENV" 2>/dev/null; then
      USE_UV=1
    fi
  fi
  if [ "$USE_UV" = "0" ] && ! "$VENV/bin/python" -m pip --version >/dev/null 2>&1; then
    USE_UV=1
  fi
fi
if [ "$USE_UV" = "1" ]; then
  [ -n "$UV" ] || UV="$(ensure_uv)"
  # --seed keeps a real pip in the venv for the agent's self-update path.
  rm -rf "$VENV"
  "$UV" venv --seed --python "$PY" "$VENV"
  if [[ -n "$WHEEL" ]]; then
    "$UV" pip install --python "$VENV/bin/python" --force-reinstall --no-deps "$WHEEL"
  else
    "$UV" pip install --python "$VENV/bin/python" --force-reinstall --no-deps "$ROOT/peer_agent"
  fi
else
  if [[ -n "$WHEEL" ]]; then
    "$VENV/bin/python" -m pip install --force-reinstall --no-deps "$WHEEL"
  else
    "$VENV/bin/python" -m pip install --force-reinstall --no-deps "$ROOT/peer_agent"
  fi
fi

umask 077
printf 'PEER_AGENT_TOKEN=%s\n' "$TOKEN" > "$ENV_FILE"
chmod 600 "$ENV_FILE"
cat > "$UNIT" <<EOF
[Unit]
Description=Llama RPC Farm peer agent
After=network-online.target
Wants=network-online.target

[Service]
EnvironmentFile=%h/.config/llama-peer-agent/agent.env
ExecStart=%h/.llama-peer-agent/venv/bin/llama-peer-agent --host 0.0.0.0 --port 8198
Restart=on-failure
RestartSec=5

[Install]
WantedBy=default.target
EOF
chmod 644 "$UNIT"

systemctl --user daemon-reload
systemctl --user enable --now llama-peer-agent
loginctl enable-linger "$USER" 2>/dev/null || true
sleep 1
systemctl --user is-enabled llama-peer-agent
systemctl --user is-active llama-peer-agent
"$VENV/bin/llama-peer-agent" --version
loginctl show-user "$USER" -p Linger --value 2>/dev/null || true
