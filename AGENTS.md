# AGENTS.md — Llama RPC Farm

Guidance for AI coding agents working in this repo. Only non-obvious, verified facts.

## Pre-publication checklist (BEFORE making this repo public)

Run ALL of these against every commit reachable from `main`
(`git rev-list --all`), not just HEAD — a HEAD-only scan misses history:

1. Secrets: machine credentials (compare against the maintainer's
   LOCAL machines.yaml / vault — never paste the real values into
   tracked files; describe them as patterns only, e.g. "password shown
   in ~/.hermes skill or machines vault"), the real access keys
   (zazi/dad keys in the local settings — check by pattern, not value),
   OpenAI-style `sk-` + Hugging Face `hf_` tokens longer than test
   stubs, and `.env`/`swap-secrets.env`/`machines.yaml` content.
2. Identifiers: LAN/Tailscale IPs that match the maintainer's REAL
   addresses (pull them from local `config/machines.yaml` + Tailscale
   status at scan time and compare; examples in tests must be generic
   `10.0.0.x`/`100.100.x.x` fakes), MAC addresses present in the local
   `config/machines.yaml` (compare each match — fake fixtures like
   `de:ad:be:ef:12:34` / `bc:24:11:de:ad:be` are fine, real NIC MACs are
   not), mount names specific to the maintainer's setup, home-path
   usernames.
3. QA/agent state: `.qe/`, `qe-result.json`, `.worktrees/`, `.playwright-mcp/`,
   generated tune/usage artifacts — all git-ignored in HEAD; verify
   `git ls-files` shows none.
4. History shape: if any of the above EVER got committed, prefer
   squashing to a fresh single commit + force push (cheap while the repo
   is young) over filter-repo surgery.
5. Then run `git gc --prune=now` after reflog expiry so local dangling
   blobs don't linger.

Known-safe residuals (do not chase): `10.0.0.x` inside test fixtures where
machine names are invented (`aibox1` with fake MACs), generic
`100.100.x.x` placeholders, doc examples (`sk-farm-XXXX`), and
test-stub keys (`sk-dad-test-…`).

## What this is

FastAPI web UI (`src/app.py`, port **8199**) that manages a llama.cpp RPC
cluster. Peers are reached **only** through the peer agent; agents ship as
their own `.deb`/`.pkg`/`setup.exe` packages (no remote shell, no
sshpass). A Linux master
runs llama-swap + llama-server, peers run `ggml-rpc-server` or native
`llama-server` via the agent. Vanilla-JS frontend (`ui/`), YAML configs
(`config/`). The app keeps its state under this folder (or the platform
data dir for packaged builds); engine/agent installs use standard user
locations.

## AI agent quick playbook: install & manage

For a human walkthrough see `docs/setup.md`; the agent-relevant facts:

```bash
./install.sh          # Linux/macOS — idempotent: venv + requirements.txt
install.bat           # Windows equivalent (cmd.exe)
./run.sh              # start UI detached on :8199 (Linux/macOS); run.bat on Windows
./restart.sh --ui     # restart ONLY the web UI (models/VRAM untouched)
./restart.sh          # FULL restart: stop llama-swap + all farm llama/RPC on every machine, restart daemon + UI
```

- First run needs NOTHING pre-existing except Python 3.11+ (install.sh
  fetches a local build via uv if missing, no sudo). `install.sh` is safe
  to re-run; `run.sh` is safe to re-run (idempotent, won't kill model
  processes — only the stale :8199 owner).
- Config state is ALL in `config/*.yaml` at runtime (machines, groups,
  models/, stores, swap). Everything except `*.example.yaml`, `swap_groups`
  templates, and icons is git-ignored per-install user data; the app
  creates defaults for missing files (`src/settings_persister.py` and
  `config_loader.py` tolerate empty/missing YAML).
- Peer agent: per-machine `agent_port` (default 8198) and a vault-only
  agent token. Agents are installed from their own platform packages; the
  Machines page only adds their IP/port/token. Updates are **release-driven**:
  the app live-checks GitHub for the highest published release, installs the
  release wheel for pip agents (`/agent/update`) or the raw platform binary
  for standalone agents (`/agent/update-binary`), then verifies `/version`.
  Local source/bundled versions must never decide whether a peer updates.
  Linux installs should enable linger; Windows uses a boot Scheduled Task
  (`LlamaPeerAgent`); macOS uses the `.pkg` launchd agent.
  Packaged binaries self-register at next login; an agent with no token
  opens a bounded pairing window and the app auto-enrolls it with the
  auto-managed farm token. Windows agent adds TCP 8198/UDP 8199 firewall
  rules.
- Secrets NEVER go in YAML or generated scripts: HF token + agent tokens
  live in the Fernet vault (`.env`, keys `FARM_SECRET_KEY`, `FARM_SALT`,
  `HF_TOKEN_ENC`, `AGENT_TOKEN_ENC`, `MACHINE_<ID>_AGENT_TOKEN_ENC`). The
  farm token is app-managed (auto-generated, no UI override); per-machine
  tokens are set via the Machines UI/API (`agent_token_set` flags only; no
  plaintext in responses) or `scripts/set-agent-token.py <machine>`.
- Managing models/machines as an agent: use the HTTP API at `:8199`
  (complete route reference: `docs/api.md`) rather than hand-editing YAML
  — the running app holds in-memory state and YAML CRUD from outside is
  invisible until restart.
- llama-swap serving (`:8080`) + view gateway (`:8090`,
  `src/swap_gateway.py`) are the client-facing endpoints; model list per
  client is filtered by views (`config/settings.yaml`) / access keys.
  The gateway is embedded — app startup runs it in-process and takes over
  `:8090` by stopping a stale standalone `swap_gateway.py` listener (so
  gateway code can never outlive the app); unrelated port owners are left
  alone. `FARM_GATEWAY_DISABLE=1` opts out for a deliberately external
  gateway.
- Never kill a peer's `ggml-rpc-server` / host `llama-server` blindly:
  stop them via the swap/UI lifecycle paths — llama-server processes hold
  model state a user may be serving from right now.
- The app does NOT hot-reload (see below): after any `src/*.py` edit an
  agent MUST restart. Use `./restart.sh` (full: stops llama-swap + every
  farm llama/RPC process on all machines, restarts daemon + UI) or
  `./restart.sh --ui` (web app only; leaves models/VRAM running).
  `run.sh` only starts the UI when it is not already running. systemd unit
  owners can also: `systemctl --user restart llama-swap`.

## Commands

```bash
.venv/bin/python src/app.py                 # run the app (http://localhost:8199)
.venv/bin/python -m pytest tests            # all tests (fast, fully stubbed, ~3s)
.venv/bin/python -m pytest tests/test_swap_manager.py::test_stop_targets_correct_machine  # single test
./restart.sh                               # FULL: kills llama-swap + all farm llama/RPC on every machine, restarts daemon + UI
./restart.sh --ui                          # UI-only restart (models/VRAM untouched)
./restart-llama-swap.sh                    # backwards-compatible alias for ./restart.sh --full
./run.sh                                    # (or run.bat) start UI without killing llama processes
./farmctl status                            # CLI over the :8199 API (docs/cli.md)
./farmctl models | start | stop             # configs / daemon start (no model load) / daemon stop
./farmctl launch --model ID | unload ID     # warm ONE model / free ONE model
./farmctl doctor | snapshot
```

`farmctl` is API-only (never reads config or agents directly). A bare
`launch` is invalid on purpose — llama-swap loads on demand, so `start`
starts the daemon and `launch --model` warms one model.

- `docs/` has a curated index (docs/README.md): setup, architecture, api,
  secrets, cross-platform, rpc-model, models-and-variants, peer-agent,
  release, visualizer, tuner, development. Read the relevant one before
  large changes; AGENTS.md holds only traps.
- No lint/typecheck configured — `python3 -m py_compile`, `node --check
  ui/static/app.js`, and `bash -n` on touched shell scripts are the
  de-facto syntax gates.
- The app does NOT hot-reload. After editing `src/*.py` you must restart
  the daemon or your change won't be exercised by a live test against
  :8199. Frontend `app.js` IS cache-busted automatically
  (`asset_v` = mtime+boot); a hard reload after restart is still wise.
- Chat testing is a native tab (same UI, `GET /api/chat/models` + streaming `POST /api/chat` proxied to llama-swap) — chainlit was removed, no :8191 service exists.
- Tests are all unit-stubbed (peer-agent calls faked) — no real hosts
  contacted, safe to run anywhere.

## Verification order that matters

1. `python3 -m py_compile` every edited `src/*.py`
2. `node --check ui/static/app.js` if JS touched
3. `pytest tests` — several tests assert EXACT rendered script/CLI strings;
   a behavior change usually means updating the test's expected string too
   (e.g. `tests/test_render_rpc.py`, `tests/test_port_roll.py` assert
   literal PowerShell/bash snippets).

## Architecture facts agents get wrong

- **Master = the machine running the app.** `local_node.is_local()` decides;
  everything else is a peer. llama-swap runs on the host machine
  (`swap_manager.get_host()`), not necessarily the master role.
- **Machine identity = MAC, not IP.** DHCP re-leases drift IPs;
  `rpc_manager._sync_machine_ip()` auto-heals via ARP (only for machines
  with a stored `mac`, and only after the new IP answers on the agent port). Code that
  targets peers should tolerate the config IP being stale.
- **RPC numbering has ONE source of truth:** `rpc_manager.compute_rpc_map()`
  (and its mirror `_rpc_legend`). Peer order = `--rpc` order; per peer,
  GPUs by index then CPU iff RAM selected. The visualizer resolves
  `RPC<N>` backend tokens via `/api/rpc-map` — never add a second
  numbering scheme.
- **Model variants share the base model's GGUF filename.** Any code mapping
  filename → servers (`buildFlowModelMaps` in app.js) must not let variant
  data overwrite base-model keys — this caused wrong machines to light up
  in the visualizer once.
- **Device tokens are vendor-prefixed:** `CUDA0/1..`, `ROCm0..`, `SYCL0..`,
  plain `Metal` (no index) for Apple. `src/gpu_vendor.py` owns this; don't
  hardcode `CUDA`.
- **Flags live in three synced views** in the model editor: common inputs
  (`#flag-ctx-size`, `#flag-cache-k/v`), chip rows (`#flags-container`),
  raw text (`#flags-text`). Editing any one must sync the others live
  (see `setCtxViews` / `setKvCacheViews` / `upsertFlagTextLine` in app.js).
  ctx-size also answers to the `--c` alias.
- **Sched events are anonymous.** GGML sched log lines carry no model id;
  attribution in `handleSchedEvent` uses freshest-streaming-model policy.
  RPC backend tokens are matched against `/api/rpc-map` — unknown ids
  light nothing (no modulo guessing).

## OS coverage (linux / windows / macos)

The app runs on any of the three; peers can be any mix. When touching
anything that runs shell commands:

- **There is no SSH or remote-shell transport.** Every peer operation is a
  typed peer-agent HTTP call (`/rpc/*`, `/llama/*`, `/files/*`,
  `/service/*`, `/install/*`, `/metrics`, `/system/*`). Do not add SSH,
  sshpass, paramiko, or a remote shell command path.
- Windows agents use PowerShell/CIM internally; `%USERPROFILE%` is
  expanded by the agent/installer, never by a remote cmd.exe shell.
- macOS has **no pgrep, no pkill, no setsid** — use port probes /
  `killall` / `ps -axc -o pid,comm` / nohup. Grep for `os_type == "macos"`
  before assuming a POSIX branch is fine.
- Local process exec: Windows needs `shell=True` string commands or
  `Popen` with `creationflags` (DETACHED_PROCESS); POSIX uses
  `["bash","-c",cmd]` + `start_new_session=True`. Never pass a LIST with
  `shell=True` on Windows (cmd.exe list2cmdline mangles quotes).
- Port-targeted process kills must anchor the port with a trailing space
  (`-p $port `, and PowerShell `' ' + $CommandLine + ' ' -like '* -p N *'`)
  so 5109 never matches 51096.
- llama.cpp prebuilts: Linux CUDA builds need glibc ≥ 2.38 (Ubuntu 24.04+);
  Vulkan builds only 2.34. `llamacpp_installer.py` probes `ldd --version`
  and falls back Vulkan-first. POSIX tarballs need `--strip-components=1`
  (they nest in `llama-<tag>/`); Windows zips are flat.
- **Linux arm64 prebuilts (Raspberry Pi) need glibc ≥ 2.38** — CPU,
  Vulkan and CUDA arm64 assets are all built on Ubuntu 24.04. Bookworm
  (2.36) must source-build; `prebuilt_blocker()` handles this. A Pi is a
  CPU-only RPC peer (`ggml-rpc-server` exposes one `CPU` device). Arch is
  resolved by `resolve_arch()` before asset/glibc decisions because a
  fresh machine has no `detected.arch`; hardware detect persists `arch`
  and `board` (from `/proc/device-tree/model`) and must merge, not
  replace, the existing `detected` dict. Release assets:
  `...-linux-arm64`, `..._arm64.deb`, `llama-peer-agent-linux-arm64`.

## Config / state model

- `config/machines.yaml`, `config/groups.yaml`, `config/models/*.yaml` are
  YAML CRUD'd at runtime — hand edits are fine but the running app holds
  its own in-memory state; don't edit while it's serving without restarting.
- Model paths use `store://` aliases (`config/stores.yaml` maps alias →
  real path per OS) — never store absolute model paths.
- **Secrets**: HF token + peer-agent tokens live ONLY in the Fernet vault
  (`.env`, git-ignored). Generated scripts read `AGENT_TOKEN_<NAME>` from
  the `swap-secrets.env` sidecar. Never bake a token into YAML, generated
  scripts, or API responses (`agent_token_set` flags only). There are
  injection tests (`test_shell_injection.py`, `test_script_injection.py`)
  — flag values are `shlex.quote`d; keep it that way.
- Generated scripts are chmod 0700, vault `.env` 0600 — don't loosen
  permissions or write secrets at world-readable paths.
- User-visible strings/comments in this codebase are often load-bearing
  documentation (long "why" comments citing llama.cpp internals). Preserve
  them when editing; add one when you learn a non-obvious upstream fact.

## Gotchas

- Two config copies exist: `config/` (source of truth) and whatever the
  app pushed to the swap host (`~/.config/llama-swap/`) — don't hand-edit
  the pushed one, it gets overwritten on next launch.
- `config/machines.yaml` machine `id` and `name` can DIVERGE (historical
  rename: id `aibox2`, name `aibox3`). Nearly all code keys on `name`.
  Don't assume they're equal, don't "fix" one without the other.
- `rpc_manager._render_rpc` output is a bash f-string: literal `{`/`}`
  for the generated script must be written `{{`/`}}`. Easy to confuse
  with PowerShell `{ }` inside the same string.
- llama-swap's log endpoints replay buffered history on connect — any new
  consumer of `/logs/stream/*` must prime silently (~1.5s) before
  forwarding, or old activity flares up the UI on every page load.
- **`detected` must be MERGED, never replaced.** Hardware detect and the
  120 s watcher (`routes/hardware.py`) both write it from live stats, which
  do not carry `arch`/`board`/`driver_version`. Replacing the dict wiped a
  Pi's identity (Linux icon regression, wrong prebuilt gate) and floored
  sub-GB RAM to 0. Use `stats_collector.ram_gb()` for RAM (0.4 GB stays
  0.4). Any new writer must merge into the existing dict.
- **CPU/RAM-only placements are first-class**: model chips are
  `<machine>-RAM`, exposure mode `ram`, launch `-ngl 0`; `rpc_manager`
  gives a GPU-less peer its own `cpu` RPC group. `tuner.estimate` falls
  back to a RAM budget (`_cpu_ram_budget`) when no accelerator chips are
  selected — don't reintroduce a GPU-only early raise there.
- Ports: rpc-servers get per-model rolled ports (`_roll_port_if_busy`);
  Windows excludes Hyper-V/WSL reserved ranges. Base 50052 is legacy.