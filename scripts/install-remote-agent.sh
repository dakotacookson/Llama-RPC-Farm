#!/usr/bin/env bash
# One-line peer-agent installer for Linux/macOS:
#   curl -fsSL https://raw.githubusercontent.com/dakotacookson/Llama-RPC-Farm/main/scripts/install-remote-agent.sh | bash
#
# Optional: PEER_AGENT_TOKEN=... to pre-seed the token. Without it the
# agent starts unpaired and the farm app pairs it during discovery.
#
# No suitable Python? The installer downloads a user-local Python 3.11 via
# uv (no sudo). This is what makes old distros (Ubuntu 18.04 / Jetson,
# 32-bit-free Bookworm) work without python3-venv/pip pre-installed.
set -euo pipefail

REPO="${LLAMA_FARM_REPO:-dakotacookson/Llama-RPC-Farm}"
ROOT="$HOME/.llama-peer-agent"
VENV="$ROOT/venv"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

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
  local uv="$ROOT/.uv-bin/uv"
  # A previously downloaded uv may be unusable on this glibc (e.g. an old
  # Ubuntu/Jetson). Verify it actually runs before reusing it.
  if [ -x "$uv" ] && "$uv" --version >/dev/null 2>&1; then
    printf '%s\n' "$uv"; return 0
  fi
  rm -f "$uv"
  local arch tmp
  case "$(uname -s)/$(uname -m)" in
    # musl builds are static and also run on old glibc (Ubuntu 18.04 /
    # Jetson L4T); the gnu uv build requires glibc >= 2.28.
    Linux/x86_64) UV_ARCH="x86_64-unknown-linux-musl" ;;
    Linux/aarch64|Linux/arm64) UV_ARCH="aarch64-unknown-linux-musl" ;;
    Darwin/x86_64) UV_ARCH="x86_64-apple-darwin" ;;
    Darwin/arm64|Darwin/aarch64) UV_ARCH="aarch64-apple-darwin" ;;
    *) echo "error: unsupported platform/arch for Python bootstrap: $(uname -s) $(uname -m)" >&2; return 1 ;;
  esac
  command -v curl >/dev/null 2>&1 || { echo "error: curl is required to bootstrap Python" >&2; return 1; }
  command -v tar  >/dev/null 2>&1 || { echo "error: tar is required to bootstrap Python" >&2; return 1; }
  mkdir -p "$ROOT/.uv-bin"
  tmp="$ROOT/.uv-bin/.tmp.$$"
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

mkdir -p "$ROOT"
if [ -x "$VENV/bin/python" ] && ! python_ok "$VENV/bin/python"; then
  echo "==> Replacing too-old virtualenv at $VENV"
  rm -rf "$VENV"
fi

# Keep the bootstrapped Python inside the agent home so it cannot vanish
# with an unrelated uv cache/setting.
export UV_PYTHON_INSTALL_DIR="$ROOT/.uv-python"

PY="$(find_python || true)"
UV=""
if [ -z "$PY" ]; then
  echo "==> No Python >= 3.$MIN_MINOR found — bootstrapping Python 3.11 with uv"
  UV="$(ensure_uv)"
  "$UV" python install 3.11
  PY="$("$UV" python find 3.11)"
fi

WHEEL_URL="$("$PY" - "$REPO" <<'PY'
import json, sys, urllib.request
repo = sys.argv[1]
req = urllib.request.Request(
    f"https://api.github.com/repos/{repo}/releases?per_page=1",
    headers={"Accept": "application/vnd.github+json", "User-Agent": "llama-farm-installer"})
with urllib.request.urlopen(req, timeout=30) as r:
    releases = json.load(r)
for rel in releases:
    for asset in rel.get("assets", []):
        name = asset.get("name", "")
        if name.startswith("llama_rpc_farm_agent-") and name.endswith(".whl"):
            print(asset["browser_download_url"])
            raise SystemExit
raise SystemExit("no agent wheel asset found in the latest release")
PY
)"

WHEEL_NAME="$(basename "$WHEEL_URL")"
echo "==> Downloading peer agent wheel ($WHEEL_NAME)"
# pip validates the wheel filename, so keep the release basename instead of
# renaming it to a generic agent.whl (which pip rejects).
curl -fsSL --retry 3 "$WHEEL_URL" -o "$TMP/$WHEEL_NAME"

echo "==> Installing into $VENV"
USE_UV=0
if [ -n "$UV" ]; then
  USE_UV=1
else
  if ! "$PY" -m venv "$VENV" 2>/dev/null; then
    USE_UV=1
  elif ! "$VENV/bin/python" -m pip --version >/dev/null 2>&1; then
    USE_UV=1
  fi
fi
if [ "$USE_UV" = "1" ]; then
  [ -n "$UV" ] || UV="$(ensure_uv)"
  # --seed gives the venv its own pip; the agent's /agent/update uses
  # `python -m pip install` and would fail in a pip-less uv venv.
  rm -rf "$VENV"
  "$UV" venv --seed --python "$PY" "$VENV"
  "$UV" pip install --python "$VENV/bin/python" \
    --quiet --force-reinstall --no-deps "$TMP/$WHEEL_NAME"
else
  "$VENV/bin/python" -m pip install \
    --quiet --force-reinstall --no-deps "$TMP/$WHEEL_NAME"
fi

if [[ -n "${PEER_AGENT_TOKEN:-}" ]]; then
  ENV_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/llama-peer-agent"
  mkdir -p "$ENV_DIR"
  umask 077
  printf 'PEER_AGENT_TOKEN=%s\n' "$PEER_AGENT_TOKEN" > "$ENV_DIR/agent.env"
  chmod 600 "$ENV_DIR/agent.env"
fi

echo "==> Installing the user service"
"$VENV/bin/llama-peer-agent" --install-service
if command -v systemctl >/dev/null 2>&1; then
  export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}"
  systemctl --user daemon-reload 2>/dev/null || true
  systemctl --user enable --now llama-peer-agent 2>/dev/null || true
  loginctl enable-linger "$USER" 2>/dev/null || true
fi

echo
echo "Peer agent installed."
echo "  version: $("$VENV/bin/llama-peer-agent" --version 2>/dev/null || echo unknown)"
echo "  port:    8198"
echo "The farm app will pair it automatically when it discovers this host."
