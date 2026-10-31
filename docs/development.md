# Development guide

Also read `../AGENTS.md` (traps & gotchas) and `docs/architecture.md`
for orientation. This file is the practical how-to.

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt    # Windows: .venv/Scripts/pip
.venv/bin/python src/app.py                  # http://localhost:8199
```

First run auto-creates the vault (`.env`) and registers the local
machine as master. Chat testing is a native tab in the same UI (no extra service).

## Running / restarting

- The app does **not** hot-reload Python. After editing `src/*.py`,
  restart: `./restart-llama-swap.sh` (kills llama processes + the old
  app, restarts with the venv python, verifies the UI answers and the
  served `app.js` is current). `./run.sh` (POSIX) / `run.bat` (Windows)
  start it without killing llama processes.
- Frontend `app.js` is cache-busted automatically: `index.html` links
  `app.js?v={{asset_v}}` where `asset_v` = file mtime + daemon boot id.
  A hard reload after restart is still wise during development.

## Testing

```bash
.venv/bin/python -m pytest tests                       # all (~3s)
.venv/bin/python -m pytest tests/test_swap_manager.py  # one file
.venv/bin/python -m pytest tests/test_port_roll.py::test_render_kill_patterns_anchored
```

- All tests are unit-stubbed: `tests/conftest.py` replaces
  `rpc_manager.cfg` with an in-memory machine map (`stub_cfg` fixture —
  populate it with `stub_cfg.update({...})`) and fakes SSH. No real host
  is ever contacted.
- **String-assertion style:** several tests assert EXACT rendered
  bash/PowerShell snippets from `rpc_manager` / `swap_manager`. If you
  change rendering behavior, expect to update the expected strings in
  `test_render_rpc.py`, `test_port_roll.py`, `test_swap_manager.py`.
  When building the fixture by hand: `_split_servers` needs
  `cfg.machines.require/get/all`.
- Shell-injection tests (`test_shell_injection.py`,
  `test_script_injection.py`) fail on any unquoted metacharacter from
  user-controlled values — a feature; keep the quoting.

## Syntax gates (no linter configured)

```bash
python3 -m py_compile src/*.py          # or each edited file
node --check ui/static/app.js
bash -n restart-llama-swap.sh run.sh    # if touched
```

## Verification checklist for changes

1. `py_compile` edited Python
2. `node --check ui/static/app.js` if JS touched
3. `pytest tests`
4. If backend behavior: restart the daemon, then exercise the live app
   on :8199 (the running process holds its own config state).
5. Live endpoints worth curling during work:
   - `GET /api/machines`, `GET /api/rpc-map`, `GET /api/swap/loaded`
   - llama-swap feeds: `curl -N http://127.0.0.1:8080/logs/stream/proxy`

## Common pitfalls (details in AGENTS.md / other docs)

- Editing `config/*.yaml` while the app serves — in-memory state wins
  until restart.
- `rpc_manager._render_rpc` f-string: literal braces in generated bash
  must be `{{`/`}}`; PowerShell `{ }` sits inside the same string.
- Port-anchored kill patterns need the trailing-space anchor; tests
  enforce the exact padded PowerShell form.
- Never assume pgrep/pkill/setsid exist (macOS), or that `%USERPROFILE%`
  expands under PowerShell (`$env:USERPROFILE` there).
- Tests must never contact real hosts — everything goes through the
  `stub_cfg` fixture + fakes. Don't introduce live SSH in tests.

## Conventions

- Long "why" comments citing llama.cpp internals are load-bearing docs —
  preserve and extend them.
- Machine `name` is the key everywhere (`id` can diverge historically).
- Model paths always `store://` aliases; device tokens always vendor-
  prefixed via `gpu_vendor.py`.
- Secrets only via `secrets_vault`; API returns `*_set` flags only.

## Releases / distribution

No packaging configured — it runs from the checkout. If distributing:
rotate `.env` secrets, keep `THIRD-PARTY-LICENSES.md` + the LGPL-2.1
text (paramiko), keep paramiko as a pip dependency (see README legal
section).