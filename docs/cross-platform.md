# Cross-platform support (linux / windows / macos)

The app itself runs on any of the three; peers can be any mix. Every code
path that executes shell commands has OS branches. The rule: **before
assuming a POSIX branch is fine, grep for `os_type == "macos"` and
`os == "windows"`** — most shell-command sites are triple-handled.

## Remote command execution

- Windows peers: OpenSSH default shell is **cmd.exe**. Remote commands
  are either bare cmd commands (`taskkill /F /IM ...`) or
  `powershell -NoProfile -Command "..."` wrappers. `%USERPROFILE%`
  expands only under cmd.exe; under PowerShell use `$env:USERPROFILE`.
- POSIX peers: `tail`, `pgrep`, `pkill`, `setsid`, `ss`, `/dev/tcp`.
- **macOS has no `pgrep`, `pkill`, `setsid`, or `systemctl`.** Use port
  probes, `killall` (exact binary name), `ps -axc -o pid,comm` (BSD ps:
  `-c` shows command basename), and `nohup ... &`. Where a fallback
  exists it's inside the same command string (`nproc || sysctl -n hw.ncpu`).

## Local process execution

- POSIX: `subprocess.Popen(["bash", "-c", cmd], start_new_session=True)`.
- Windows: `Popen(cmd, shell=True, creationflags=CREATE_NEW_PROCESS_GROUP
  | DETACHED_PROCESS)` — **never pass a LIST with `shell=True`** (cmd.exe
  list2cmdline mangles quotes). The installer's `run_local` instead uses a
  list with `shell=False` on all platforms (its comment explains why).

## Port-anchored process kills

Kill patterns MUST anchor the port with trailing space so 5109 never
matches a co-served model on 51096:

- pgrep/pkill: `ggml-[r]pc-server.*-p $port ` (trailing space; the
  `[r]` keeps the pattern from matching itself).
- PowerShell: `(' ' + $_.CommandLine + ' ') -like '* -p $port *'` — pad
  both sides so a port at the END of a command line still matches
  (unpadded `*-p $port *` misses that case; both live kill code and the
  generated scripts do this — keep them in sync, tests assert the
  rendered string).

## Process management per OS

| Task | linux | windows | macos |
|---|---|---|---|
| rpc-server stop (port) | `pkill -f "...-p N "` | CIM kill by cmdline | (peers are linux/win today; use killall by name) |
| llama-swap running? | `pgrep -f "llama-[s]wap"` | `Get-Process` count | port probe |
| llama-swap stop | systemctl + pkill | `taskkill /F /IM llama-swap.exe` | `killall` + `ps -axc` kill |
| daemon start | systemd → setsid fallback | `Win32_Process.Create` via PowerShell | nohup (no systemctl/setsid) |
| llama-server local | Popen bash detached | Popen shell detached | Popen bash detached |
| llama-server remote (SSH) | `nohup ... & echo PID=$!` | `Start-Process cmd /c ...` (no nohup/$! on cmd.exe) | same as linux |

## Paths

- POSIX: `~`, `$HOME`; logs in `$HOME` root (`rpc-server-<port>.log`,
  `llama-server.log`, `llama-swap.log`).
- Windows: `%USERPROFILE%` (cmd.exe) / `$env:USERPROFILE` (PowerShell);
  state files use `$env:TEMP` when /tmp is unavailable.
- The installer writes to `~/llama.cpp/bin` (POSIX) or
  `%USERPROFILE%\llama.cpp\bin` (Windows). Legacy layouts
  (`~/llama.cpp/build/bin`, flat `llama.cpp\llama-server.exe`) are probed
  and honored by `_detect_bin_on_machine` — don't break those probes.

## llama.cpp prebuilts (llamacpp_installer.py)

Verified against upstream release b11056:

| Target | Asset | Notes |
|---|---|---|
| macOS arm64/x64 | `bin-macos-<kind>.tar.gz` | Metal built in |
| Linux + NVIDIA (glibc ≥ 2.38, e.g. Ubuntu 24.04) | `bin-ubuntu-cuda-13.3-x64` + matching `cudart-...ubuntu-cuda-13.3` | CUDA builds need GLIBC 2.38 |
| Linux + NVIDIA (glibc 2.34–2.37, e.g. Ubuntu 22.04) | `bin-ubuntu-vulkan-x64` | only needs 2.34; devices are `Vulkan0..` |
| Linux + AMD | `bin-ubuntu-rocm-10.0` | |
| Linux + Intel | `bin-ubuntu-sycl-fp16` | |
| Windows + NVIDIA | `bin-win-cuda-12.4-x64` + `cudart-...12.4` zip | zips are FLAT (no strip) |
| Windows + AMD/Intel/CPU | rocm / sycl / cpu zips | |

Hard-learned rules baked into the installer:

- POSIX tarballs nest in `llama-<tag>/` → extract with
  `--strip-components=1` or binaries land one dir deep.
- `releases/latest/download/...` resolves to the newest SEMVER tag (may
  carry no binaries) — download from the **per-tag** URL of the newest
  release that has bin assets instead.
- `cudart-*` assets don't carry the `llama-` prefix; the asset inventory
  must include them or the "missing assets" check falsely aborts.
- glibc is probed via `ldd --version`; detected arch is persisted into
  `machines.yaml.detected.arch`.

## llama-swap prebuilts (swap_installer.py)

llama-swap is a single-host tool: it runs ONLY on the swap host, so there
is no fleet install — one endpoint, `POST /api/install-llama-swap?name=`.

Flow (mirrors the llama.cpp installer):

1. **Detect first** — existing installs are always skipped:
   - local machine: `shutil.which` + `~/.local/bin` / `/usr/local/bin` /
     `/usr/bin` (POSIX) or `%USERPROFILE%\llama-swap\llama-swap.exe` (win)
   - remote: same probe over SSH
2. **Install if missing** — official release asset from
   `mostlygeek/llama-swap` (verified against v256):

| Target | Asset | Notes |
|---|---|---|
| linux amd64/arm64 | `llama-swap_<num>_<os>_<kind>.tar.gz` | FLAT tarball — only the binary is extracted into `~/.local/bin` |
| macos (darwin) amd64/arm64 | same pattern, `darwin` | |
| windows amd64 | `llama-swap_<num>_windows_amd64.zip` → `%USERPROFILE%\llama-swap\` | flat zip |

Hard-learned rules baked in:

- Asset names carry the tag NUMBER without the "v" (`v256` release →
  `llama-swap_256_...`) — building names from the raw tag fails.
- The tarball contains docs at root too; extract only the `llama-swap`
  member (no `--strip-components` games).
- Download from the **per-tag** URL (same rule as llama.cpp installer).
- On success, `swap_manager.save_settings(binary=...)` is updated so
  `launch()` uses the new binary immediately.

UI entry points: **Settings** tab ("Install llama-swap (this host)",
below the llama.cpp install buttons) and **Launch** tab's llama-swap card
("Install llama-swap" — targets the selected swap host). llama-swap is
single-host software — there is deliberately NO "install on all nodes"
for it.

## Stats collection

- linux: `/proc/stat|loadavg|meminfo`, `nproc`, `nvidia-smi`/`rocm-smi`.
- windows: `Get-CimInstance` (CPU via `Win32_PerfFormattedData_PerfOS_Processor`,
  RAM/cores via `Win32_Processor`/`Win32_OperatingSystem`), `nvidia-smi`,
  `Get-PSDrive`. **Get-Counter was removed** — its module autoload + two
  samples with a 500ms sleep cost ~3.5s per poll (the whole Windows
  stats blob went 4.9s → ~1.0s); the WMI counter is locale-independent
  and tracks Task Manager within a few points.
- macos: `sysctl`/`top`/`vm_stat`/`system_profiler` (no nvidia-smi;
  Apple GPUs reported by `system_profiler SPDisplaysDataType`).

Performance rules learned the hard way:

- **Fast-fail offline hosts:** a closed SSH port must be detected with a
  0.5s socket probe before any paramiko connect (a full connect timeout
  is ~6s of dead time per offline machine per cycle).
- **Never let one slow host block the dashboard:** `/api/stats` waits
  for every collection; the dashboard uses `/api/stats/stream` (NDJSON,
  per-machine lines as they complete) so each card paints independently.
- When adding a Windows PS blob, measure it end-to-end over SSH first —
  individual cmdlets are ~0.1–0.8s but module autoloads inside one
  session can silently add seconds.

## Testing implications

`tests/conftest.py` stubs SSH entirely, so OS branches are exercised by
unit tests, not live hosts. When adding an OS branch, add its literal
command assertions to the matching test file (`test_swap_manager.py`,
`test_port_roll.py`, `test_render_rpc.py` all assert exact strings).