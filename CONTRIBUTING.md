# Contributing to Llama RPC Farm

Thanks for helping make llama.cpp clustering easier for everyone! This is
an open, maintainer-governed project — all contributions go through pull
requests.

## Governance (short version)

- **Every PR is reviewed and must be approved by the project maintainer**
  ([@dakotacookson](https://github.com/dakotacookson)) before merge.
  There are currently no other moderators; when moderators are added this
  section will change.
- Maintainers reserve the right to close PRs that duplicate existing work,
  break the project's promises (see below), or need too much rework — but
  feedback will always be given.

## Project promises (what a PR must not break)

1. **"Nothing outside the folder is ever modified"** — the app must not
   touch the user's llama.cpp/llama-swap configs or anything outside the
   project directory.
2. **No secrets in configs, scripts, or API responses** — machine SSH
   passwords and tokens live in the encrypted vault only. If you touch
   shell generation, there are injection tests you must keep green
   (`tests/test_shell_injection.py`, `tests/test_script_injection.py`).
3. **All generated scripts are vault-fed at runtime** (`scripts/farm-run.sh`
   pattern) — nothing baked into disk.
4. **Cross-platform correctness** — Linux/Windows/macOS peers must all
   survive your change: Windows OpenSSH runs cmd.exe (wrap
   PowerShell), macOS has no `pgrep`/`pkill`/`setsid`. See AGENTS.md.

## Getting started

```bash
./install.sh        # or install.bat (creates .venv + requirements)
./run.sh            # http://localhost:8199
```

Read `docs/README.md` (curated index) and `AGENTS.md` (gotchas the code
encodes) before your first PR — the string-assertion test style and the
single-source-of-truth rules catch most newcomers.

## Development workflow

1. Fork the repo; branch off `main`.
2. For every changed file run the syntax gates:
   - Python: `python3 -m py_compile src/<file>.py`
   - frontend: `node --check ui/static/app.js`
   - generated shell: `bash -n <script>`
3. Run the test suite: `.venv/bin/python -m pytest tests` — it is fully
   stubbed (no real SSH hosts are contacted) and fast.
4. Tests assert exact rendered script/CLI strings; any behavior change in
   script generation usually means updating the test's expected string too.
5. Open a PR with a short "what + why". Screenshots of the UI are welcome
   for visible changes.

## Great first contributions

- 🧪 **Hardware matrix testing** (see README "Supported hardware"): real
  reports for AMD ROCm, Apple Silicon/metal, CPU-RAM-sharing or Intel XPU
  setups — "works on my box" reports, logs, and PRs that fix what breaks
  on them are gold.
- Docs improvements (especially for platforms beyond the maintainer's).
- Test coverage in areas the string tests don't reach.

## Code of conduct

Be kind, review people's setups without condescension, and remember
everyone's home network is different. Reports of sensitive info in logs
should be redacted by the poster before publishing.

## License note for contributors

Your contributions are licensed under the project's `LICENSE` (Apache-2.0)
when you submit a PR. The wrapped engines (llama.cpp, llama-swap) remain
MIT under their own upstream terms.