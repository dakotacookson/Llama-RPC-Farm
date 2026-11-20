#!/usr/bin/env bash
# Proxmox VE helper: install Llama RPC Farm inside a Debian/Ubuntu VM or LXC.
# Run as root inside the guest:
#   curl -fsSL .../packaging/proxmox/install-llama-rpc-farm.sh | bash
set -euo pipefail

VERSION="${LLAMA_FARM_VERSION:-latest}"
ARCH="$(dpkg --print-architecture 2>/dev/null || echo amd64)"
case "$ARCH" in
  amd64) DEB_ARCH=amd64 ;;
  arm64) DEB_ARCH=arm64 ;;
  *) echo "unsupported architecture: $ARCH" >&2; exit 1 ;;
esac

if [ "$VERSION" = "latest" ]; then
  URL="$(curl -fsSL https://api.github.com/repos/dakotacookson/Llama-RPC-Farm/releases/latest \
    | grep -o "https://[^\"]*llama-rpc-farm_[^\"]*_${DEB_ARCH}.deb" | head -1)"
else
  URL="https://github.com/dakotacookson/Llama-RPC-Farm/releases/download/release-${VERSION}/llama-rpc-farm_${VERSION}_${DEB_ARCH}.deb"
fi
[ -n "$URL" ] || { echo "could not resolve the .deb asset" >&2; exit 1; }

apt-get update
apt-get install -y curl ca-certificates
curl -fL "$URL" -o /tmp/llama-rpc-farm.deb
apt-get install -y /tmp/llama-rpc-farm.deb
rm -f /tmp/llama-rpc-farm.deb

echo
echo "Llama RPC Farm installed."
echo "UI: http://$(hostname -I | awk '{print $1}'):8199"
echo "Gateway: http://$(hostname -I | awk '{print $1}'):8090"
echo
echo "For GPU/RPC peers, install the peer agent package on each host."
