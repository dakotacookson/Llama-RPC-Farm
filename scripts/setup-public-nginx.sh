#!/usr/bin/env bash
# Expose the Llama RPC Farm API through nginx on TCP 443.
#
#   sudo ./scripts/setup-public-nginx.sh \
#     --domain api.example.com --email you@example.com
#
# Or bring your own certificate:
#   sudo ./scripts/setup-public-nginx.sh \
#     --domain api.example.com \
#     --cert /etc/ssl/farm/fullchain.pem \
#     --key  /etc/ssl/farm/privkey.pem
#
# This exposes ONLY :8090 (the OpenAI/Anthropic gateway), not the UI :8199.
# Keep the UI on Tailscale/WireGuard or add a separate, access-controlled UI
# vhost yourself. Raw :8090/:8199/:8198/RPC ports should not be forwarded.
set -euo pipefail

DOMAIN=""
EMAIL=""
UPSTREAM="http://127.0.0.1:8090"
CERT=""
KEY=""
ENV_FILE=""
NGINX_SITE="llama-rpc-farm-public"
USE_CERTBOT=0

usage() {
  sed -n '2,18p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
  exit 2
}
while [ $# -gt 0 ]; do
  case "$1" in
    --domain) DOMAIN="${2:-}"; shift 2 ;;
    --email) EMAIL="${2:-}"; shift 2 ;;
    --upstream) UPSTREAM="${2:-}"; shift 2 ;;
    --cert) CERT="${2:-}"; shift 2 ;;
    --key) KEY="${2:-}"; shift 2 ;;
    --env-file) ENV_FILE="${2:-}"; shift 2 ;;
    -h|--help) usage ;;
    *) echo "unknown option: $1" >&2; usage ;;
  esac
done
[ -n "$DOMAIN" ] || usage

if [ "$(id -u)" -ne 0 ]; then
  echo "run as root (sudo)" >&2
  exit 1
fi
command -v nginx >/dev/null 2>&1 || {
  echo "nginx is not installed; install it first (apt install nginx)" >&2
  exit 1
}
if [ -z "$CERT" ] || [ -z "$KEY" ]; then
  command -v certbot >/dev/null 2>&1 || {
    echo "certbot is required unless --cert/--key are supplied" >&2
    exit 1
  }
  [ -n "$EMAIL" ] || {
    echo "--email is required for Let's Encrypt" >&2
    exit 1
  }
  CERT="/etc/letsencrypt/live/$DOMAIN/fullchain.pem"
  KEY="/etc/letsencrypt/live/$DOMAIN/privkey.pem"
  USE_CERTBOT=1
fi

# Resolve the target user's data dir (sudo -E / SUDO_USER, not root's home).
TARGET_USER="${SUDO_USER:-$(id -un)}"
TARGET_HOME="$(getent passwd "$TARGET_USER" | cut -d: -f6 || true)"
[ -n "$TARGET_HOME" ] || TARGET_HOME="$HOME"
if [ -z "$ENV_FILE" ]; then
  ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
  if [ -d "$ROOT/config" ] && [ -f "$ROOT/VERSION" ]; then
    ENV_FILE="$ROOT/config/runtime.env"
  elif [ -n "${LLAMA_FARM_DATA_DIR:-}" ]; then
    ENV_FILE="$LLAMA_FARM_DATA_DIR/config/runtime.env"
  elif [ "$(uname -s)" = "Darwin" ]; then
    ENV_FILE="$TARGET_HOME/Library/Application Support/LlamaRPCFarm/config/runtime.env"
  else
    ENV_FILE="${XDG_CONFIG_HOME:-$TARGET_HOME/.config}/llama-rpc-farm/config/runtime.env"
  fi
fi
mkdir -p "$(dirname "$ENV_FILE")"
touch "$ENV_FILE"
chown "$TARGET_USER" "$ENV_FILE" 2>/dev/null || true

set_env() {
  key="$1"; val="$2"; file="$3"
  if grep -q "^[#[:space:]]*$key=" "$file"; then
    sed -i "s|^[#[:space:]]*$key=.*|$key=$val|" "$file"
  else
    printf '\n%s=%s\n' "$key" "$val" >> "$file"
  fi
}
set_env FARM_TRUSTED_PROXIES 127.0.0.1 "$ENV_FILE"
set_env FARM_REQUIRE_AUTH 1 "$ENV_FILE"
set_env FARM_GATEWAY_REQUIRE_KEYS 1 "$ENV_FILE"
chown "$TARGET_USER" "$ENV_FILE" 2>/dev/null || true

SITE_AVAIL="/etc/nginx/sites-available/$NGINX_SITE.conf"
SITE_ENABLED="/etc/nginx/sites-enabled/$NGINX_SITE.conf"
WWW="/var/www/llama-rpc-farm-acme"
mkdir -p "$WWW/.well-known/acme-challenge"

write_http_bootstrap() {
  cat > "$SITE_AVAIL" <<NGINX
server {
    listen 80;
    listen [::]:80;
    server_name $DOMAIN;
    location /.well-known/acme-challenge/ { root $WWW; }
    location / { return 404; }
}
NGINX
  ln -sf "$SITE_AVAIL" "$SITE_ENABLED"
  nginx -t
  systemctl reload nginx 2>/dev/null || nginx -s reload
}

write_https() {
  cat > "$SITE_AVAIL" <<NGINX
map \$http_upgrade \$connection_upgrade {
    default upgrade;
    ''      close;
}

server {
    listen 80;
    listen [::]:80;
    server_name $DOMAIN;
    location /.well-known/acme-challenge/ { root $WWW; }
    location / { return 301 https://\$host\$request_uri; }
}

server {
    listen 443 ssl http2;
    listen [::]:443 ssl http2;
    server_name $DOMAIN;

    ssl_certificate     $CERT;
    ssl_certificate_key $KEY;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_session_cache shared:SSL:10m;
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;
    add_header X-Content-Type-Options nosniff always;

    client_max_body_size 0;
    proxy_read_timeout 7200;
    proxy_send_timeout 7200;

    location / {
        proxy_pass $UPSTREAM;
        proxy_http_version 1.1;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
        proxy_buffering off;
        proxy_cache off;
        proxy_set_header Connection "";
    }
}
NGINX
  nginx -t
  systemctl reload nginx 2>/dev/null || nginx -s reload
}

if [ "$USE_CERTBOT" -eq 1 ]; then
  if [ ! -f "$CERT" ]; then
    echo "==> requesting certificate for $DOMAIN"
    write_http_bootstrap
    certbot certonly --webroot -w "$WWW" -d "$DOMAIN" \
      --non-interactive --agree-tos --email "$EMAIL"
  fi
fi
write_https

cat <<NEXT

Public API configured.

  URL:      https://$DOMAIN/v1
  upstream: $UPSTREAM
  exposed:  :8090 only (not UI :8199, agent :8198, or RPC ports)
  env file: $ENV_FILE

The script set FARM_TRUSTED_PROXIES / FARM_REQUIRE_AUTH /
FARM_GATEWAY_REQUIRE_KEYS in that file (source and packaged builds read it).

Restart the app:
  source checkout:  ./restart.sh --ui
  packaged Linux:   systemctl --user restart llama-rpc-farm
  packaged macOS:   launchctl kickstart -k gui/$(id -u)/com.llamarpc.farm.app
  packaged Windows: schtasks /End /TN LlamaRPCFarm ; schtasks /Run /TN LlamaRPCFarm

Do not expose raw :8090/:8199/:8198/RPC ports to the internet.
NEXT
