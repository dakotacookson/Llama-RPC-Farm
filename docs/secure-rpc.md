# Secure RPC transport (optional mTLS)

llama.cpp RPC has no authentication and no TLS. The farm cannot fix that
inside llama.cpp, so it offers an optional **mutual-TLS TCP wrapper** around
the existing RPC streams.

Off by default:

```yaml
# config/settings.yaml
rpc_security: off      # off | mtls
```

## What changes — and what does not

Unchanged:

- RPC port allocation and hashing
- per-model / per-peer port numbers
- `rpc.state.<port>` device records
- `--rpc` order and `RPC<N>` numbering
- start/stop scripts, `cmdStop`, reaper, co-serve behavior

Changed only when `rpc_security: mtls`:

| Side | Listener | Target |
|---|---|---|
| Peer agent | `0.0.0.0:5399` (one mTLS tunnel, client cert required) | `127.0.0.1:<requested rpc port>` |
| Peer agent API | `0.0.0.0:8197` HTTPS/mTLS (same certs) | agent API |
| Master app | `127.0.0.1:<existing rpc port>` | `<peer-ip>:5399` (mTLS client cert) |

`--rpc` keeps the **same port number** and switches the host from the peer
IP to `127.0.0.1`:

```text
before: --rpc "10.0.0.53:51203"
mtls:   --rpc "127.0.0.1:51203"
```

The master proxy sends the target RPC port inside the TLS connection; the
peer tunnel routes only to `50052`, `51000–51499`, or `52000–52499`.

**Only one port per peer needs forwarding: TCP 5399.** Dynamic per-model
RPC ports are never exposed.

## Enable / disable

UI: **Settings → RPC security**.

API:

```bash
curl -X POST http://127.0.0.1:8199/api/rpc/security/enable \
  -H 'Content-Type: application/json' -d '{"reload": true}'
curl http://127.0.0.1:8199/api/rpc/security/status
curl -X POST http://127.0.0.1:8199/api/rpc/security/disable \
  -H 'Content-Type: application/json' -d '{"reload": true}'
```

Enabling requires every configured peer to be online **and running an agent
build that includes the `/tls/*` endpoints** (agent 2.0.51+ / the current
rolling build). The farm generates a CA, a master client cert, and one
server cert per peer, then installs the peer cert through the agent's typed
`/tls/install` endpoint. If any peer fails, security stays **off**. Update
old agents from the Machines page first.

After enabling/disabling, llama-swap reloads and models must restart to pick
up the new `--rpc` targets. Stop/Launch the model (or restart the farm) if a
currently loaded model was started before the change.

Enabling also installs the same certs on the peer agent and starts its
HTTPS/mTLS listener on `8197`; the farm then talks to that agent over HTTPS.
Restart each agent once after enabling so its legacy HTTP `8198` listener
binds loopback only.

## Security properties

- RPC streams are encrypted with TLS 1.2+.
- The peer requires a client certificate signed by the farm CA.
- The master requires a server certificate signed by the same CA.
- Certificates and keys are written `0600`.
- Hostname verification is disabled because peer IPs can drift; identity is
  the CA + client certificate. Reissue certs after changing a peer IP if you
  want strict SAN checking.
- This protects **RPC transport only**. Keep the UI (`:8199`), gateway
  (`:8090`), and agent (`:8198`) on a trusted LAN/VPN and use farm access
  keys. Never port-forward RPC directly.

## Limits

- The app process must stay running to keep the master-side client proxies
  alive. If the app is down, already-running model RPC connections may keep
  working, but new connections/reloads will not.
- A peer agent restart drops its TLS listeners until the next `/rpc/ensure`
  (model load) runs.
- One TLS tunnel per peer (port 5399) carries every per-model RPC stream;
  the peer routes to the requested port after mutual authentication.
- There is an extra userspace TLS hop on both sides. Benchmark before using
  it on a bandwidth-sensitive deployment.
- WireGuard/Tailscale remains the recommended choice when possible: one VPN
  port, no per-RPC mapping, kernel-speed encryption.
