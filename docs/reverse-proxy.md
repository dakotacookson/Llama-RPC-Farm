# Reverse proxy / public access

The farm UI (`:8199`) and gateway (`:8090`) are HTTP services. If they must
be reachable outside a trusted LAN, terminate TLS at nginx (or another
proxy) and keep the raw ports private.

## Required app environment

```bash
FARM_TRUSTED_PROXIES=127.0.0.1   # proxy peer allowed to set forwarded headers
FARM_REQUIRE_AUTH=1              # no keys yet => remote callers are restricted
FARM_GATEWAY_REQUIRE_KEYS=1      # :8090 refuses keyless non-console callers
FARM_COOKIE_SECURE=1             # login cookie Secure even if proto is unclear
```

Without these, a proxy on the same host can make every request look like the
loopback operator console.

## nginx

A complete sample is in `deploy/nginx/llama-rpc-farm.conf`.

Key rules:

- `proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for`
- `proxy_set_header X-Forwarded-Proto $scheme`
- `proxy_buffering off` for the gateway (SSE / long generations)
- `proxy_read_timeout 7200` or higher
- TLS 1.2/1.3 + HSTS

## What is safe to expose

| Port | Exposure |
|---|---|
| `443` (UI/gateway via nginx) | yes, with access keys + login |
| `8199` raw | no |
| `8090` raw | no |
| `8198` raw agent HTTP | no — use agent HTTPS/mTLS (`8197`) or VPN |
| `5399` RPC mTLS tunnel | optional, per peer; prefer WireGuard |
| `50052`, `51000–51499` raw RPC | never |
| `52000–52499` remote-local proxy | never |

The peer agent now supports an HTTPS/mTLS listener on `8197` once certs are
installed. The farm switches a machine to HTTPS automatically when RPC
security is enabled; after the agent restarts, its legacy HTTP listener binds
loopback only. Firewall `8198` from untrusted networks.

## First login

With `FARM_REQUIRE_AUTH=1`, create the first admin account from the local
console (loopback/VPN) before exposing the UI. The public UI then requires a
login/access key.

## RPC

Do not forward raw RPC. Enable `rpc_security: mtls` (Settings → RPC security),
then forward only each peer's TCP `5399` if you insist on TCP exposure. Better:
use WireGuard/Tailscale and forward the VPN port only.
