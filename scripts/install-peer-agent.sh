#!/usr/bin/env bash
# Install/update the peer agent on THIS Linux host and enable it at boot.
#
# Usage:
#   PEER_AGENT_TOKEN='long-random-token' ./scripts/install-peer-agent.sh [wheel]
#
# With no wheel argument, installs from ./peer_agent in this checkout.
# The systemd --user unit is enabled and linger is turned on so the agent
# starts after reboot even when the user is not logged in.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TOKEN="${PEER_AGENT_TOKEN:-}"
WHEEL="${1:-}"
VENV="$HOME/.llama-peer-agent/venv"
ENV_DIR="$HOME/.config/llama-peer-agent"
ENV_FILE="$ENV_DIR/agent.env"
UNIT_DIR="$HOME/.config/systemd/user"
UNIT="$UNIT_DIR/llama-peer-agent.service"

if [[ -z "$TOKEN" ]]; then
  echo "error: PEER_AGENT_TOKEN is required (export it or pass it inline)" >&2
  exit 2
fi
if ! command -v systemctl >/dev/null 2>&1; then
  echo "error: systemctl is required (Linux systemd user service)" >&2
  exit 2
fi

mkdir -p "$HOME/.llama-peer-agent" "$ENV_DIR" "$UNIT_DIR"
chmod 700 "$HOME/.llama-peer-agent" "$ENV_DIR"
if [[ ! -x "$VENV/bin/python" ]]; then
  python3 -m venv "$VENV"
fi
if [[ -n "$WHEEL" ]]; then
  "$VENV/bin/python" -m pip install --force-reinstall --no-deps "$WHEEL"
else
  "$VENV/bin/python" -m pip install --force-reinstall --no-deps "$ROOT/peer_agent"
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
