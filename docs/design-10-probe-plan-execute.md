# Design 10 — probe / plan / execute split (testability requirement #3)

Task t_bd9422d1 · 2026-09-24 · implements requirement #3 (unit-testability,
card t_b4dd1cda comment 200) on top of design 01.

## The three layers

```
bootstrap.py (entry, later phase)
  ├─ detect_system_info(run)   -> SystemInfo            (env.py, real impls)
  ├─ probe(info, fs, run, ...) -> Plan                  (probe.py, PURE)
  └─ Executor.execute(plan)    -> summary dict          (execute.py, dumb runner)
```

- **env.py** — `SystemInfo` value object (os/arch/gpu_vendor/glibc/driver/vram/wsl)
  + two injection interfaces: `RunCmd` (command probes → `RunResult`) and `FS`
  (existence/read/binary-smoke `file_exists_and_runs`). Real implementations
  (`RealRunCmd`, `RealFS`) live here; probe.py never imports them.
- **probe.py** — pure decision logic. ZERO subprocess/os.system/direct-disk
  calls; every observation arrives via the injected `SystemInfo` + `FS` +
  `RunCmd`. Restructures design 01's §4 (detection ladder — as pure functions
  over already-captured facts), §5 (presence probes, skip/upgrade/repair vs the
  `installer-state.yaml` manifest), §6 (settings-prefill merge rules a/b/c:
  write only if empty / stale / `--force-settings`).
- **execute.py** — one handler per plan step, no decisions. Downloads are
  stub-able (`downloader` injected; prod wiring is a later phase). Settings
  writes go through the comment-preserving ruamel round-trip pattern (§6).
  Failure capture (operator req #4) also lives here: every step exception is
  wrapped into the `llama-rpc-farm-failure-envelope/1` JSON bundle with
  secret redaction — one AI-facing report format for installs and launches.

## Plan step types (typed, not dicts)

`SkipTool(name, reason, found_path, version)` · `InstallTool(name, source,
version, backend, dest, extra)` · `RepairTool(InstallTool)` (§5 broken
`--version`) · `WriteSettings(key, value, file, merge_reason)` ·
`ManifestUpdate(tool, entry)` · `PlanNote(kind, tool, message)` (refusals,
manual actions, capability statements — never silence).

## Backend parity (operator clarification)

Both engines are planned on every viable backend: `VLLM_BACKEND` /
`LLCPP_BACKEND` matrices in probe.py encode CUDA/ROCm/Metal(MLX)/Vulkan/CPU
per OS. Native-Windows vLLM emits an explicit `PlanNote(refused)` citing the
WSL2 path; vLLM-Metal is opt-in (`--with-vllm-metal`). `ENGINE_DEFAULTS` +
`tool_defaults()` carry per-backend device-env (`CUDA_VISIBLE_DEVICES` /
`HIP_VISIBLE_DEVICES` / none on Metal) and CUDA-only flag lists.

## Hermetic test contract

`tests/installer/` runs offline, no root, <30s: pyfakefs pytest plugin (dev dep,
in requirements.txt) for the filesystem + a dict-backed `RunCmd` stub.
Matrix coverage: Linux/CUDA, Linux/ROCm, Linux/CPU, Windows/CUDA, macOS/MLX.
Golden idempotency: apply plan A (skips) → re-probe → all-skip. One
`@pytest.mark.slow` placeholder integration test (skipped by default) reserves
the real network/asset roundtrip for the downloader phase.