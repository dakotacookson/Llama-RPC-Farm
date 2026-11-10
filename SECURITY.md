# Security policy

## Scope

**Llama RPC Farm is a homelab tool for trusted networks.** It is designed
for a LAN or a VPN such as Tailscale/WireGuard, not for exposure to the
public internet.

The most important fact: **llama.cpp's RPC server is unauthenticated by
design.** Anyone who can reach a peer's `ggml-rpc-server` port can use that
machine's GPU/RAM. The farm assumes the network is the boundary.

Do **not** port-forward:

- the RPC server ports (`50052` and the per-model rolled ports),
- the UI/API (`8199`),
- the view gateway (`8090`),
- or the peer-agent port (`8198`).

If you need remote access, put the whole farm behind a VPN or an
authenticated reverse proxy.

## Supported versions

Security fixes land on the current `main` line and the newest release
line. Older rolling builds are retained but not separately patched. Use the
newest [release](https://github.com/dakotacookson/Llama-RPC-Farm/releases)
when reporting an issue.

## Reporting a vulnerability

Please **do not open a public issue** for a security problem.

Use GitHub's private vulnerability reporting on this repository:

1. Open the [Security tab](https://github.com/dakotacookson/Llama-RPC-Farm/security).
2. Click **Report a vulnerability**.
3. Include reproduction steps, affected version, OS, and any logs with
   secrets/IPs/tokens redacted.

If private reporting is unavailable, open a minimal issue asking the
maintainer for a private channel — without exploit details.

Useful reports include:

- bypasses of the `:8090` access-key model gate,
- path escapes in the peer agent's file endpoints,
- shell/flag injection in generated scripts or launch commands,
- secret leakage into configs, scripts, API responses, or logs,
- an agent endpoint that can execute arbitrary commands.

## What the app touches

- HTTP to peer agents on machines you add (Bearer-token auth, typed
  endpoints, no remote shell).
- Downloads from official GitHub releases for llama.cpp, llama-swap, and
  the peer agent.
- Model downloads from Hugging Face when you ask for them.

Everything else stays inside the project folder (or the platform data dir
for packaged builds). The farm does not modify your existing llama.cpp or
llama-swap configs.

## Secrets model

- One git-ignored `.env` vault at the project root holds Fernet-encrypted
  values (`FARM_SECRET_KEY`, `FARM_SALT`, `HF_TOKEN_ENC`,
  `AGENT_TOKEN_ENC`, `MACHINE_<ID>_AGENT_TOKEN_ENC`).
- Agent tokens and the HF token are decrypted in memory only at the point
  of use. Generated scripts read `AGENT_TOKEN_<NAME>` from a host-side
  `swap-secrets.env` sidecar (mode 0600); tokens are never baked into
  configs or scripts.
- API responses expose only `agent_token_set` / `hf_token_set` booleans,
  never plaintext values.
- `config/settings.yaml` holds access keys and is per-install,
  git-ignored user data. Keep it mode `0600` on a shared machine.

Full details: [docs/secrets.md](docs/secrets.md).

## Hardening checklist

- [ ] Farm reachable only on LAN/VPN; no public port-forward.
- [ ] `.env`, `config/machines.yaml`, and `config/settings.yaml` are mode
      `0600` on multi-user boxes.
- [ ] Peer agents use per-machine tokens (or the app-managed farm token),
      not a shared password.
- [ ] Keep llama.cpp, llama-swap, and the peer agent on current releases.
- [ ] Redact tokens, IPs, and usernames before posting logs.
- [ ] Use per-user access keys instead of handing out an admin key.

## Out of scope

- Public multi-tenant hosting.
- Denial of service from trusted-LAN peers.
- Vulnerabilities in llama.cpp or llama-swap themselves — report those
  upstream (this project only downloads and drives their official builds).
