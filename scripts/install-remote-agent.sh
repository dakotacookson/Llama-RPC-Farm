#!/usr/bin/env bash
# One-line peer-agent installer for Linux/macOS:
#   curl -fsSL https://raw.githubusercontent.com/dakotacookson/Llama-RPC-Farm/main/scripts/install-remote-agent.sh | bash
#
# Optional: PEER_AGENT_TOKEN=... to pre-seed the token. Without it the
# agent starts unpaired and the farm app pairs it during discovery.
set -euo pipefail

REPO="${LLAMA_FARM_REPO:-dakotacookson/Llama-RPC-Farm}"
ROOT="$HOME/.llama-peer-agent"
VENV="$ROOT/venv"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

if ! command -v python3 >/dev/null 2>&1; then
  echo "error: python3 is required" >&2
  exit 1
fi

WHEEL_URL="$(python3 - "$REPO" <<'PY'
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

echo "==> Downloading peer agent wheel"
curl -fsSL --retry 3 "$WHEEL_URL" -o "$TMP/agent.whl"

echo "==> Installing into $VENV"
mkdir -p "$ROOT"
if [[ ! -x "$VENV/bin/python" ]]; then
  python3 -m venv "$VENV"
fi
"$VENV/bin/python" -m pip install --quiet --force-reinstall --no-deps "$TMP/agent.whl"

if [[ -n "${PEER_AGENT_TOKEN:-}" ]]; then
  ENV_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/llama-peer-agent"
  mkdir -p "$ENV_DIR"
  umask 077
  printf 'PEER_AGENT_TOKEN=%s\n' "$PEER_AGENT_TOKEN" > "$ENV_DIR/agent.env"
  chmod 600 "$ENV_DIR/agent.env"
fi

echo "==> Installing the user service"
"$VENV/bin/llama-peer-agent" --install-service

echo
echo "Peer agent installed."
echo "  version: $("$VENV/bin/llama-peer-agent" --version 2>/dev/null || echo unknown)"
echo "  port:    8198"
echo "The farm app will pair it automatically when it discovers this host."
