# Public API access (443 → nginx → :8090)

Two supported ways to use the farm from outside your LAN:

1. **Tailscale / WireGuard (recommended, no public ports)** — use the farm
   UI and API over the VPN. This is the default personal setup.
2. **Public HTTPS API** — forward **only TCP 443** to nginx, and let nginx
   proxy to the gateway on `127.0.0.1:8090`.

Do not port-forward raw `:8090`, `:8199`, `:8198`, or RPC ports.

## Option 1: VPN

```text
http://<tailscale-hostname>:8199     # UI + control plane
http://<tailscale-hostname>:8090/v1  # clients
```

No nginx or 443 setup is required. Tailscale encrypts and authenticates the
transport. Use farm access keys for API clients and the normal login for UI.

## Option 2: public API on 443

Requirements:

- a DNS name pointing at the public IP, e.g. `api.example.com`
- nginx and certbot on the farm host
- router forwarding **443 only** (80 temporarily for Let's Encrypt)

Run:

```bash
sudo ./scripts/setup-public-nginx.sh \
  --domain api.example.com \
  --email you@example.com
```

Or bring your own certificate:

```bash
sudo ./scripts/setup-public-nginx.sh \
  --domain api.example.com \
  --cert /etc/ssl/farm/fullchain.pem \
  --key  /etc/ssl/farm/privkey.pem
```

The script:

- configures an HTTP ACME vhost and obtains a Let's Encrypt certificate
- configures HTTPS on 443, HSTS, SSE-safe buffering, and long timeouts
- proxies only to `http://127.0.0.1:8090`
- does **not** expose the UI, agent, or RPC ports

Access keys stay in `config/settings.yaml`, which the app keeps owner-only
(`0600`). The script writes security flags into the correct runtime env file
automatically:

```bash
FARM_TRUSTED_PROXIES=127.0.0.1
FARM_REQUIRE_AUTH=1
FARM_GATEWAY_REQUIRE_KEYS=1
```

- Source checkout: `config/runtime.env`
- Packaged Linux: `~/.config/llama-rpc-farm/config/runtime.env`
- Packaged macOS: `~/Library/Application Support/LlamaRPCFarm/config/runtime.env`
- Packaged Windows: `%APPDATA%\LlamaRPCFarm\config\runtime.env`
- Override with `--env-file /path/to/runtime.env`

Then restart:

```bash
# source checkout
./restart.sh --ui

# packaged Linux
systemctl --user restart llama-rpc-farm

# packaged macOS
launchctl kickstart -k gui/$(id -u)/com.llamarpc.farm.app

# packaged Windows
schtasks /End /TN LlamaRPCFarm
schtasks /Run /TN LlamaRPCFarm
```

Packaged binaries read `config/runtime.env` at startup from their data
directory, so no service-file edit is required.

Clients use:

```bash
curl https://api.example.com/v1/models \
  -H "Authorization: Bearer sk-farm-..."
```

`/v1/models` is empty without a key once keys exist, and generate paths
return `401`. Create keys in **Access**.

## What the script does not do

- It does not forward the UI. If you must expose `:8199`, add a separate
  vhost with an IP allowlist or client-certificate requirement; the admin UI
  is much more sensitive than the API.
- It does not configure your router, DNS, or firewall.
- It does not enable RPC mTLS or expose RPC ports.
- It does not set `FARM_COOKIE_SECURE`; only set that if the UI itself is
  HTTPS-only (it breaks login cookies over plain-HTTP Tailscale).
