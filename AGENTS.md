# AGENTS.md — Llama RPC Farm

Guidance for AI coding agents working in this repo. Only non-obvious, verified facts.

## Pre-publication checklist (BEFORE making this repo public)

Run ALL of these against every commit reachable from `main`
(`git rev-list --all`), not just HEAD — a HEAD-only scan misses history:

1. Secrets: machine SSH passwords (compare against the maintainer's
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
cluster over SSH: a Linux master runs llama-swap + llama-server, Windows/Linux
peers run `ggml-rpc-server`. Vanilla-JS frontend (`ui/`), YAML configs
(`config/`). Nothing outside this folder is ever modified.

## AI agent quick playbook: install & manage

For a human walkthrough see `docs/setup.md`; the agent-relevant facts:

```bash
./install.sh          # Linux/macOS — idempotent: venv + requirements.txt
install.bat           # Windows equivalent (cmd.exe)
./run.sh              # start UI detached on :8199 (Linux/macOS); run.bat on Windows
fuser -k 8199/tcp     # stop the UI (Linux) — then ./run.sh to restart
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
- Secrets NEVER go in YAML or generated scripts: machine SSH passwords +
  HF token live in the Fernet vault (`.env`, keys `FARM_SECRET_KEY`,
  `FARM_SALT`, `HF_TOKEN_ENC`, `MACHINE_<ID>_PASSWORD_ENC`). Agents set
  passwords via the Machines UI/API (`password_set` flags only; no
  plaintext in responses) or `scripts/migrate_secrets_to_env.py`.
- Managing models/machines as an agent: use the HTTP API at `:8199`
  (complete route reference: `docs/api.md`) rather than hand-editing YAML
  — the running app holds in-memory state and YAML CRUD from outside is
  invisible until restart.
- llama-swap serving (`:8080`) + view gateway (`:8090`,
  `src/swap_gateway.py`) are the client-facing endpoints; model list per
  client is filtered by views (`config/settings.yaml`) / access keys.
- Never kill a peer's `ggml-rpc-server` / host `llama-server` blindly:
  stop them via the swap/UI lifecycle paths — llama-server processes hold
  model state a user may be serving from right now.
- The app does NOT hot-reload (see below): after any `src/*.py` edit an
  agent MUST restart the daemon (`run.sh` re-starts only :8199; full
  swap+UI restart = `./restart-llama-swap.sh`; systemd unit owners:
  `systemctl --user restart llama-swap`, `sudo systemctl restart swap-gateway`).

## Commands

```bash
.venv/bin/python src/app.py                 # run the app (http://localhost:8199)
.venv/bin/python -m pytest tests            # all tests (fast, fully stubbed, ~3s)
.venv/bin/python -m pytest tests/test_swap_manager.py::test_stop_targets_correct_machine  # single test
./restart-llama-swap.sh                     # kills llama-swap/llama-server + old app, restarts UI
./run.sh                                    # (or run.bat) start UI without killing llama processes
```

- `docs/` has a curated index (docs/README.md): architecture, api,
  secrets, cross-platform, rpc-model, visualizer, tuner, development.
  Read the relevant one before large changes; AGENTS.md holds only traps.
- No lint/typecheck configured — `python3 -m py_compile`, `node --check
  ui/static/app.js`, and `bash -n` on touched shell scripts are the
  de-facto syntax gates.
- The app does NOT hot-reload. After editing `src/*.py` you must restart
  the daemon or your change won't be exercised by a live test against
  :8199. Frontend `app.js` IS cache-busted automatically
  (`asset_v` = mtime+boot); a hard reload after restart is still wise.
- Chat testing is a native tab (same UI, `GET /api/chat/models` + streaming `POST /api/chat` proxied to llama-swap) — chainlit was removed, no :8191 service exists.
- Tests are all unit-stubbed (SSH faked in `tests/conftest.py`) — no real
  hosts contacted, safe to run anywhere.

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
  with a stored `mac`, and only after the new IP answers SSH). Code that
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

- Windows peers: OpenSSH default shell is cmd.exe — remote commands are
  `powershell -NoProfile -Command "..."` wrappers; `%USERPROFILE%` expands
  only under cmd.exe.
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

## Config / state model

- `config/machines.yaml`, `config/groups.yaml`, `config/models/*.yaml` are
  YAML CRUD'd at runtime — hand edits are fine but the running app holds
  its own in-memory state; don't edit while it's serving without restarting.
- Model paths use `store://` aliases (`config/stores.yaml` maps alias →
  real path per OS) — never store absolute model paths.
- **Secrets**: machine SSH passwords + HF token live ONLY in the Fernet
  vault (`.env`, git-ignored). Generated scripts take `$SSH_PASS` from the
  environment (`scripts/farm-run.sh` wrapper or `swap-secrets.env`
  sidecar). Never bake a password into YAML, generated scripts, or API
  responses (`password_set` flags only). There are injection tests
  (`test_shell_injection.py`, `test_script_injection.py`) — flag values
  are `shlex.quote`d; keep it that way.
- Generated scripts are chmod 0700, vault `.env` 0600 — they drive SSH
  sessions via `$SSH_PASS`; don't loosen permissions or write secrets at
  world-readable paths.
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
- Ports: rpc-servers get per-model rolled ports (`_roll_port_if_busy`);
  Windows excludes Hyper-V/WSL reserved ranges. Base 50052 is legacy.