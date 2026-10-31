"""Machine-type-aware llama.cpp installer.

Picks the correct official prebuilt for each machine (os + gpu_vendor +
arch) and installs it over SSH (or locally):

  macOS arm64/x64     -> bin-macos-*  (Metal built in)
  Linux + AMD (ROCm)  -> bin-ubuntu-rocm-<ver>-<arch>.tar.gz
  Linux + NVIDIA      -> bin-ubuntu-cuda-13.3-<arch>.tar.gz when the
      target's glibc is new enough (>= 2.38, e.g. Ubuntu 24.04), with the
      matching cudart tarball; older glibc (e.g. Ubuntu 22.04 = 2.35)
      falls back to bin-ubuntu-vulkan-<arch>.tar.gz, whose binaries need
      only glibc 2.34 (Vulkan devices Vulkan0/Vulkan1/...; the app's
      GPU-token layer maps vendor prefixes at detection time).
  Linux + CPU         -> bin-ubuntu-<arch>.tar.gz
  Windows + NVIDIA    -> bin-win-cuda-<ver>-<arch>.zip + matching cudart zip
  Windows + AMD       -> bin-win-rocm-<ver>-<arch>.zip
  Windows + CPU       -> bin-win-cpu-<arch>.zip (Vulkan zip as fallback)

Install destination: ~/llama.cpp/bin (POSIX) or
%USERPROFILE%\\llama.cpp\\bin (Windows). The machine's llama_bin is
updated on success. Never touches anything else.

Tarball layout caveat: POSIX prebuilts extract into a versioned
subdir (llama-<tag>/), so we --strip-components=1 to land binaries
flat inside the dest dir (the app's bin paths expect bin/llama-server).
Windows zips are already flat. cudart-...-ubuntu tarballs ship a
cudart-<tag>-.../ subdir — flattened the same way (they only carry
libcudart/libcublas .so files).
"""
import json
import os
import urllib.request

GITHUB_API = "https://api.github.com/repos/ggml-org/llama.cpp/releases"
GITHUB_LATEST_DL = "https://github.com/ggml-org/llama.cpp/releases/latest/download"
LINUX_VER = "10.0"   # ubuntu-rocm-<ver>
WIN_CUDA_VER = "12.4"
WIN_ROCM_VER = "10.0"
LINUX_CUDA_VER = "13.3"   # ubuntu-cuda-<ver>; 12.8 kept as fallback below
LINUX_CUDA_VER_OLD = "12.8"
DEST_POSIX = "~/llama.cpp/bin"
DEST_WIN = "%USERPROFILE%\\llama.cpp\\bin"

# CUDA linux prebuilts are built against a newer glibc (2.38 via the
# release toolchain) while their Vulkan siblings only need 2.34 —
# measured on b11056: vulkan binaries max GLIBC_2.34 (runs on Ubuntu
# 22.04's 2.35), cuda binaries' libggml-base/libllama-* need 2.38.
GLIBC_CUDA_MIN = (2, 38)

# ---------- Windows Firewall rules ----------
# Program-scoped firewall rules match the LISTENING exe's exact path.
# The 2026-10 llama.cpp upgrade moved the Windows binaries from the
# llama.cpp\ root into llama.cpp\bin\, so the pre-upgrade allow rule
# (llama.cpp\ggml-rpc-server.exe) stopped matching and the default
# inbound policy silently DROPPED the peer's RPC port: SYNs vanish
# with no RST, every llama-server --rpc connect hung for minutes per
# attempt, and winbox-backed models "never loaded" (observed 2026-10-26).
# install_on re-ensures the rules against the ACTUAL installed path, so
# any future upgrade that moves the exe heals the rules by itself.
WIN_FW_EXES = ("ggml-rpc-server.exe", "llama-server.exe")
WIN_FW_RULE_PREFIX = "Llama RPC Farm - "


def _win_fw_ensure_ps():
    """PowerShell body ensuring an inbound Allow rule exists for every
    farm exe at the installed path. Contains NO double quotes: run_remote
    wraps the body as `powershell -NoProfile -Command "..."` and escapes
    embedded quotes, so single-quoted PS strings are the only safe form.
    Prints `exe:state` tokens (ok|added|failed) for the caller to parse."""
    names = ",".join(f"'{n}'" for n in WIN_FW_EXES)
    return (
        f"$names=@({names}); $r=@(); "
        "foreach($n in $names){ "
        "$p=Join-Path $env:USERPROFILE ('llama.cpp\\bin\\'+$n); "
        "$ok=@(Get-NetFirewallApplicationFilter -Program $p "
        "-ErrorAction SilentlyContinue | Get-NetFirewallRule "
        "-ErrorAction SilentlyContinue | Where-Object { $_.Enabled "
        "-eq 'True' -and $_.Direction -eq 'Inbound' -and $_.Action "
        "-eq 'Allow' }); "
        "if($ok.Count -gt 0){ $r += $n+':ok'; continue }; "
        f"try{{ New-NetFirewallRule -DisplayName ('{WIN_FW_RULE_PREFIX}'+$n) "
        "-Direction Inbound -Action Allow -Program $p -Profile Private,Public "
        "-ErrorAction Stop | Out-Null; $r += $n+':added' }catch{ "
        "$r += $n+':failed' }}; Write-Output ($r -join ' ')")


def _win_fw_remove_ps():
    """PowerShell body removing ONLY the rules install_on created
    (DisplayName prefix match), never a user's own rules."""
    names = ",".join(f"'{n}'" for n in WIN_FW_EXES)
    return (
        f"$names=@({names}); foreach($n in $names){{ "
        f"try{{ Get-NetFirewallRule -DisplayName ('{WIN_FW_RULE_PREFIX}'+$n) "
        "-ErrorAction SilentlyContinue | Remove-NetFirewallRule "
        "-ErrorAction Stop }catch{}}; Write-Output fw-rules-removed")


def _ensure_win_firewall(run, log=lambda m: None):
    """Idempotently ensure the firewall rules exist after a Windows
    install/upgrade. Returns a short note for the install report
    ('' when everything was already in place).

    Non-fatal by design: a failed rule add (e.g. non-admin session)
    returns a WARNING note instead of failing the install — the binaries
    are fine, only inbound reachability is at risk, and the operator can
    add the rule manually from the report text."""
    try:
        code, out = run(_win_fw_ensure_ps(), timeout=60)
    except Exception as e:
        log(f"firewall ensure skipped: {e}")
        return ""
    added, failed, seen = [], [], False
    for tok in (out or "").split():
        exe, sep, st = tok.partition(":")
        if sep and exe in WIN_FW_EXES:
            seen = True
            if st == "added":
                added.append(exe)
                log(f"firewall: allow rule added for {exe}")
            elif st == "failed":
                failed.append(exe)
            # 'ok' = rule already present — nothing to report
    if failed:
        return ("WARNING: could not add firewall allow rule for "
                + ", ".join(failed) + " (needs an admin session) — "
                "inbound RPC/llama-server ports will be unreachable")
    if added:
        return "firewall allow rules added for " + ", ".join(added)
    if not seen:
        return ("WARNING: firewall rule check gave no result "
                f"(exit {code}) — verify an inbound allow rule for "
                "llama.cpp\\bin\\ggml-rpc-server.exe manually")
    return ""


def _fetch_latest_assets(limit=30):
    """(tag, [asset names], download_base) of the newest release that has
    bin assets. download_base is the per-release URL prefix — the
    'releases/latest/download' shortcut resolves to the newest SEMVER tag
    (e.g. v0.4.1, which carries no bin assets), NOT the newest release
    with binaries, so it must never be used for bin downloads."""
    req = urllib.request.Request(
        f"{GITHUB_API}?per_page={limit}",
        headers={"Accept": "application/vnd.github+json",
                 "User-Agent": "llama-rpc-farm"})
    with urllib.request.urlopen(req, timeout=30) as r:
        releases = json.loads(r.read().decode())
    for rel in releases:
        # include the cudart companions (they don't carry the llama-
        # prefix: cudart-llama-bin-win-*.zip / cudart-llama-*-ubuntu-*)
        names = [a["name"] for a in rel.get("assets", [])
                 if (("llama-" in a["name"] and "bin-" in a["name"])
                     or a["name"].startswith("cudart-"))]
        if names:
            base = f"https://github.com/ggml-org/llama.cpp/releases/download/{rel['tag_name']}"
            return rel["tag_name"], names, base
    return None, [], GITHUB_LATEST_DL


def _glibc_version(cmd_prefix, run):
    """(major, minor) glibc of the target machine, or None when unknown.
    Uses `ldd --version` (works on glibc AND musl systems' error path)."""
    code, out = run("ldd --version 2>&1 | head -1")
    text = (out or "").strip()
    import re
    m = re.search(r"(\d+)\.(\d+)", text)
    if not m:
        return None
    try:
        return (int(m.group(1)), int(m.group(2)))
    except ValueError:
        return None


def _pick_assets(machine, tag, assets, glibc=None):
    """Choose release asset name(s) for a machine. Returns
    (asset_names, dest, backend_note, is_win_zip) or raises ValueError."""
    os_type = machine.get("os", "linux")
    vendor = (machine.get("gpu_vendor") or "nvidia").strip().lower()
    arch = (machine.get("detected") or {}).get("arch") or "x86_64"
    is_arm = arch in ("arm64", "aarch64")
    if os_type == "macos":
        # Apple hardware is overwhelmingly arm64; only pick x64 when the
        # probe explicitly reported Intel
        kind = "x64" if arch in ("x86_64", "x64") and vendor != "apple" else "arm64"
        return [f"llama-{tag}-bin-macos-{kind}.tar.gz"], DEST_POSIX, \
            "Metal backend included", False
    if os_type == "windows":
        arch_kind = "arm64" if is_arm else "x64"
        if vendor == "amd":
            return [f"llama-{tag}-bin-win-rocm-{WIN_ROCM_VER}-{arch_kind}.zip"], \
                DEST_WIN, "HIP/ROCm backend", True
        if vendor == "nvidia" and not is_arm:
            cud = f"cudart-llama-bin-win-cuda-{WIN_CUDA_VER}-{arch_kind}.zip"
            return [f"llama-{tag}-bin-win-cuda-{WIN_CUDA_VER}-{arch_kind}.zip",
                    cud], DEST_WIN, f"CUDA {WIN_CUDA_VER} backend (+cudart)", True
        if vendor == "intel":
            return [f"llama-{tag}-bin-win-sycl-{arch_kind}.zip"], DEST_WIN, \
                "SYCL backend (oneAPI runtime bundled)", True
        if vendor == "nvidia" and is_arm:
            return [f"llama-{tag}-bin-win-cuda-13.4-arm64.zip",
                    f"cudart-llama-bin-win-cuda-13.4-arm64.zip"], DEST_WIN, \
                "CUDA 13.4 backend (+cudart)", True
        return [f"llama-{tag}-bin-win-cpu-{arch_kind}.zip"], DEST_WIN, \
            "CPU backend (Vulkan zip available as fallback)", True
    # linux (incl. macOS-less POSIX)
    arch_kind = "arm64" if is_arm else "x64"
    if vendor == "amd":
        return [f"llama-{tag}-bin-ubuntu-rocm-{LINUX_VER}-{arch_kind}.tar.gz"], \
            DEST_POSIX, "HIP/ROCm backend", False
    if vendor == "intel":
        return [f"llama-{tag}-bin-ubuntu-sycl-fp16-{arch_kind}.tar.gz"], \
            DEST_POSIX, "SYCL backend (oneAPI runtime bundled)", False
    if vendor == "nvidia":
        # CUDA prebuilts exist upstream now — but need glibc >= 2.38
        # (measured: cuda libllama-*/libggml-base require GLIBC_2.38,
        # vulkan siblings only 2.34). Prefer CUDA when the target is new
        # enough, else fall back to Vulkan.
        if glibc is not None and glibc >= GLIBC_CUDA_MIN:
            cud = f"cudart-llama-{tag}-bin-ubuntu-cuda-{LINUX_CUDA_VER}-{arch_kind}.tar.gz"
            return [f"llama-{tag}-bin-ubuntu-cuda-{LINUX_CUDA_VER}-{arch_kind}.tar.gz",
                    cud], DEST_POSIX, \
                f"CUDA {LINUX_CUDA_VER} backend (+cudart)", False
        note = "Vulkan backend on NVIDIA (devices Vulkan0..; glibc-new "
        note += f"targets get CUDA {LINUX_CUDA_VER} instead)"
        return [f"llama-{tag}-bin-ubuntu-vulkan-{arch_kind}.tar.gz"], \
            DEST_POSIX, note, False
    return [f"llama-{tag}-bin-ubuntu-{arch_kind}.tar.gz"], DEST_POSIX, \
        "CPU backend", False


def plan(machine):
    """What WOULD be installed for this machine: asset names + backend.
    No side effects; used by the UI before installing."""
    try:
        tag, assets, _base = _fetch_latest_assets()
    except Exception as e:
        return {"ok": False, "error": f"cannot reach GitHub releases: {e}"}
    try:
        names, dest, note, _zip = _pick_assets(machine, tag, assets)
    except ValueError as e:
        return {"ok": False, "error": str(e)}
    missing = [n for n in names if n not in assets]
    return {"ok": True, "tag": tag, "assets": names, "dest": dest,
            "backend": note, "missing": missing}


def install_on(machine, log=lambda m: None):
    """Install the right prebuilt on a machine (local or via SSH).
    Returns dict {ok, machine, bin, detail}."""
    import config_loader as cfg
    from ssh_manager import SSHManager, SSHError
    import local_node

    name = machine.get("name", "?")
    is_local = local_node.is_local(machine)
    is_win = machine.get("os", "linux") == "windows"

    def run_local(cmd, timeout=600):
        import subprocess
        # list + shell=False on every platform: passing a LIST with
        # shell=True on Windows routes it through cmd.exe (list2cmdline),
        # whose quote-stripping mangles the embedded PowerShell quotes.
        proc = subprocess.run(
            ["powershell", "-NoProfile", "-Command", cmd] if is_win
            else ["bash", "-c", cmd],
            capture_output=True, text=True, timeout=timeout,
            shell=False)
        return proc.returncode, proc.stdout + proc.stderr

    def run_remote(cmd, timeout=600):
        ssh = SSHManager(machine["ip"], machine["username"],
                         machine["password"], port=machine.get("port", 22))
        if is_win:
            # Windows peers' OpenSSH default shell is cmd.exe — a raw
            # PowerShell body dies there ('$ErrorActionPreference' is not
            # recognized...). Hand it to cmd as a 'powershell -NoProfile
            # -Command' wrapper, same escaping as rpc_manager (AGENTS.md
            # OS coverage): embedded double quotes become \" so PS parses
            # them as literals inside the outer quoted command.
            cmd = ('powershell -NoProfile -Command "'
                   + cmd.replace('"', '\\"') + '"')
        code, out, err = ssh.run(cmd, timeout=timeout)
        return code, out + err

    run = run_local if is_local else run_remote
    try:
        # arch matters for asset choice; probe it when not detected yet
        if not ((machine.get("detected") or {}).get("arch")):
            if is_win:
                code, out = run("echo $env:PROCESSOR_ARCHITECTURE")
                a = (out or "").strip().lower()
                machine.setdefault("detected", {})["arch"] = \
                    "arm64" if "arm" in a else "x86_64"
            else:
                code, out = run("uname -m")
                machine.setdefault("detected", {})["arch"] = \
                    (out or "").strip() or "x86_64"
            # persist the arch probe so future plans skip re-probing
            try:
                import config_loader as _cfg
                _cfg.machines.update(name, {"detected": machine["detected"]})
            except Exception:
                pass
    except (SSHError, OSError) as e:
        return {"ok": False, "machine": name, "detail": f"probe failed: {e}"}
    try:
        tag, assets, base = _fetch_latest_assets()
        glibc = None if is_win else _glibc_version(None, run)
        names, dest, note, _zip = _pick_assets(machine, tag, assets, glibc)
        plan_info = {"ok": True, "tag": tag, "assets": names, "dest": dest,
                     "backend": note,
                     "missing": [n for n in names if n not in assets]}
    except Exception as e:
        return {"ok": False, "machine": name,
                "detail": f"cannot reach GitHub releases: {e}"}
    if not plan_info.get("ok"):
        return {"ok": False, "machine": name, "detail": plan_info["error"]}
    if plan_info["missing"]:
        return {"ok": False, "machine": name,
                "detail": "release assets not found upstream: "
                          + ", ".join(plan_info["missing"])}
    urls = [f"{base}/{n}" for n in plan_info["assets"]]
    try:
        if is_win:
            # download + expand each zip into the dest dir
            ps = ["$ErrorActionPreference='Stop'",
                  f"$d='{DEST_WIN}'",
                  "New-Item -ItemType Directory -Force -Path $d | Out-Null"]
            for i, u in enumerate(urls):
                ps.append(
                    f"Invoke-WebRequest -Uri '{u}' -OutFile \"$env:TEMP\\llamabin{i}.zip\"")
                ps.append(
                    f"Expand-Archive -Force \"$env:TEMP\\llamabin{i}.zip\" $d")
                ps.append(f"Remove-Item \"$env:TEMP\\llamabin{i}.zip\"")
            code, out = run("; ".join(ps))
        else:
            # curl each asset, extract FLAT into ~/llama.cpp/bin
            # (tarballs nest inside llama-<tag>/; --strip-components=1
            # removes it so bin/llama-server is the real path)
            cmds = ["mkdir -p \"$HOME/llama.cpp/bin\""]
            for i, u in enumerate(urls):
                cmds.append(
                    f"curl -fSL --retry 3 -o /tmp/llamabin{i}.tgz '{u}'")
                cmds.append(
                    f"tar -xzf /tmp/llamabin{i}.tgz -C \"$HOME/llama.cpp/bin\" "
                    "--strip-components=1")
                cmds.append(f"rm -f /tmp/llamabin{i}.tgz")
            cmds.append(
                "chmod +x \"$HOME/llama.cpp/bin/\"* 2>/dev/null; true")
            code, out = run(" && ".join(cmds))
        if code != 0:
            return {"ok": False, "machine": name,
                    "detail": f"install failed (exit {code}): {out[-500:]}"}
    except (SSHError, OSError) as e:
        return {"ok": False, "machine": name, "detail": f"transfer failed: {e}"}

    bin_dir = DEST_WIN if is_win else DEST_POSIX
    bin_path = (f"{bin_dir}\\llama-server.exe" if is_win
                else f"{bin_dir}/llama-server")
    # Windows: the upgrade just (re)wrote the exes — re-ensure the
    # firewall rules match the CURRENT exe path before the machine is
    # trusted to serve again (see the WIN_FW block comment above).
    fw_note = _ensure_win_firewall(run, log) if is_win else ""
    try:
        cfg.machines.update(name, {"llama_bin": bin_dir})
    except Exception:
        pass
    detail = (f"installed {', '.join(plan_info['assets'])} "
              f"({plan_info['tag']}) to {bin_dir}")
    if fw_note:
        detail += f"; {fw_note}"
    return {"ok": True, "machine": name, "bin": bin_path,
            "backend": plan_info["backend"], "detail": detail}


# ---------- dynamic setup classification ----------
# Never pattern-match the update path off a path string: probe the
# machine itself. A machine added five minutes ago must classify the
# same as one that's been here a year, and the flags it was built with
# (CUDA/ROCm/Vulkan, arch list, RPC/FA/NCCL) decide how it updates —
# observed 2026-10-24: a stock Vulkan prebuilt installed over aibox1's
# custom CUDA build broke every --device CUDA* launch on the farm.

def _machine_run(machine):
    """(run, is_local) for a machine — bash -c locally, raw SSH
    remotely (POSIX; Windows peers never take the source path)."""
    import local_node
    from ssh_manager import SSHManager
    is_local = local_node.is_local(machine)

    def run_local(cmd, timeout=7200):
        import subprocess
        proc = subprocess.run(["bash", "-c", cmd], capture_output=True,
                              text=True, timeout=timeout, shell=False)
        return proc.returncode, proc.stdout + proc.stderr

    def run_remote(cmd, timeout=7200):
        ssh = SSHManager(machine["ip"], machine["username"],
                         machine["password"], port=machine.get("port", 22))
        code, out, err = ssh.run(cmd, timeout=timeout)
        return code, out + err

    return (run_local if is_local else run_remote), is_local


def probe_setup(machine, log=lambda m: None):
    """Dynamically classify a machine's llama.cpp install by probing it.

    Returns dict:
      kind:     "source"   -> llama_bin sits inside a real llama.cpp
                             source tree (git repo + CMakeCache govern
                             it); updates REBUILD it in place so every
                             custom flag in the cache is preserved.
               "prebuilt" -> installer-managed binary dir (or POSIX
                             machine with no working source tree);
                             updates install the newest prebuilt chosen
                             dynamically from os/gpu_vendor/arch/glibc.
               "missing"  -> no usable binary; first install = prebuilt.
      flags:   parsed CMakeCache entries (source kind) for the job log
               (GGML_CUDA/HIP/VULKAN/RPC, CUDA arch list, compiler) so
               the user sees exactly what their update will keep."""
    out = {"machine": machine.get("name", "?"),
           "kind": "prebuilt", "llama_bin": None,
           "build_dir": None, "src_root": None, "flags": {}}
    lb = (machine.get("llama_bin") or "").strip()
    if not lb or machine.get("os", "linux") == "windows":
        # Windows can't remote-source-build (no MSVC toolchain
        # assumption) and an empty llama_bin just needs a prebuilt.
        out["kind"] = "missing" if not lb else out["kind"]
        if not lb:
            out["kind"] = "missing"
        return out
    run, _is_local = _machine_run(machine)
    try:
        # Resolve the bin dir ON the machine (tilde/env paths only mean
        # something there) and test what actually exists.
        code, probe = run(
            rf'd="{lb}"; d="${{d/\~/$HOME}}"; d="${{d%/}}"; d="${{d%bin}}"; '
            rf'd="${{d%/}}"; '
            f'[ -d "$d" ] || exit 3; echo "D=$d"; '
            f'[ -f "$d/CMakeCache.txt" ] && echo CACHE || echo NOCACHE; '
            f'[ -f "$d/bin/llama-server" ] && echo BIN || echo NOBIN')
        probe = (probe or "").strip().splitlines()
        if code == 3:
            out["kind"] = "missing"
            return out
        has_cache = "CACHE" in probe
        # Resolved-on-the-target build dir (tilde/$VAR expanded there):
        # every later command (git, cmake, version probe) must use the
        # machine's own expansion — "~/..." inside quotes stays literal
        # in bash and cd/grep silently miss (observed on aibox3).
        resolved = next((l[2:] for l in probe if l.startswith("D=")), None)
        if not resolved:
            return out          # unresolvable path — treat as prebuilt
        out["build_dir"] = resolved
        out["llama_bin"] = (resolved + "/bin") if "BIN" in probe else lb
        if not has_cache:
            return out          # prebuilt dir (installer layout) or bare dir
        # CMakeCache exists: is a git source tree behind it?
        code, gitout = run(
            f'cd "{out["build_dir"]}" && git rev-parse --show-toplevel 2>/dev/null')
        root = (gitout or "").strip()
        if code != 0 or not root.startswith("/"):
            return out          # cache but no repo — treat as prebuilt
        out["src_root"] = root
        # Read the flags this build carries, for the log + sanity.
        cmd = (f'grep -E "^(GGML_CUDA:BOOL|GGML_HIP:BOOL|GGML_VULKAN:BOOL|'
               f'GGML_RPC:BOOL|GGML_CUDA_FA:BOOL|GGML_CUDA_NCCL:BOOL|'
               f'CMAKE_CUDA_ARCHITECTURES:|CMAKE_CUDA_COMPILER:FILEPATH|'
               f'GGML_CUDA_COMPRESSION_MODE:STRING)" '
               f'"{out["build_dir"]}/CMakeCache.txt" 2>/dev/null')
        _c, flags_blob = run(cmd)
        for line in (flags_blob or "").splitlines():
            if ":" in line:
                k, _, v = line.partition(":")
                v = v.strip()
                if "=" in v:
                    # strip the cmake type prefix (BOOL=/FILEPATH=/…)
                    v = v.split("=", 1)[1]
                out["flags"][k.strip()] = v
        out["kind"] = "source"
        return out
    except Exception as e:
        log(f"probe failed on {machine.get('name', '?')}: {e}")
        return out          # unreachable -> prebuilt attempt is harmless


def update_source_on(machine, probe=None, log=lambda m: None):
    """Update a machine's llama.cpp SOURCE build in place (POSIX only).

    Never regenerates the build config: git fetch + fast-forward to
    origin/master, then cmake --build the EXISTING build dir — its
    CMakeCache keeps every custom flag (CUDA/ROCm backend, arch list,
    FA/NCCL/RPC toggles), so the update cannot change the build setup.
    Refuses (no force, no rebase) when the tree has local-only commits
    or dirty tracked files — silently merging a patch stack is how
    custom flags get lost.

    Build parallelism is RAM-aware: nvcc spikes ~2GB per job, so jobs
    = min(cores, available_ram // 2GB), floor 1 — a small peer (3GB
    RAM) builds with -j1 instead of OOMing. Full build log is left at
    /tmp/llamacpp-build.log on the target for postmortems."""
    name = machine.get("name", "?")
    if probe is None:
        probe = probe_setup(machine, log=log)
    if probe.get("kind") != "source":
        return {"ok": False, "machine": name,
                "detail": f"no source tree behind "
                          f"{probe.get('llama_bin') or machine.get('llama_bin')}"}
    build_dir = probe["build_dir"]
    src_root = probe["src_root"]
    run, _is_local = _machine_run(machine)

    try:
        code, out = run(f"cd '{src_root}' && git fetch origin master "
                        "2>&1 | tail -1")
        if code != 0:
            return {"ok": False, "machine": name,
                    "detail": f"git fetch failed: {(out or '')[-300:]}"}
        code, out = run(f"cd '{src_root}' && (git merge-base --is-ancestor "
                        "HEAD origin/master && echo CLEAN-ANCESTOR "
                        "|| echo LOCAL-ONLY)")
        if "CLEAN-ANCESTOR" not in (out or ""):
            return {"ok": False, "machine": name,
                    "detail": "source tree has local-only commits — "
                              "update refused, merge manually"}
        code, out = run(f"cd '{src_root}' && git status --porcelain "
                        "| grep -v '^??' | head -5")
        if (out or "").strip():
            return {"ok": False, "machine": name,
                    "detail": "tracked files modified: "
                              + (out or "").strip()[:200]}
        code, out = run(f"cd '{src_root}' && git checkout origin/master "
                        "2>&1 | tail -1")
        if code != 0:
            return {"ok": False, "machine": name,
                    "detail": f"git checkout failed: {(out or '')[-300:]}"}
        fl = probe.get("flags") or {}
        kept = ", ".join(f"{k}={v}" for k, v in sorted(fl.items())[:6]) or "cache defaults"
        log(f"fast-forwarded {name} to origin/master — rebuilding "
            f"{build_dir} in place (preserving {kept})")

        # RAM-aware parallelism (see docstring): available MB col 7 of free
        code, out = run("nproc; free -m | awk '/^Mem:/{print $7}'")
        parts = (out or "").split()
        try:
            cores = int(parts[0])
        except (ValueError, IndexError):
            cores = 2
        try:
            ram_avail_mb = int(parts[1])
        except (ValueError, IndexError):
            ram_avail_mb = 4096
        jobs = max(1, min(cores, ram_avail_mb // 2048))
        log(f"building with -j{jobs} (cores={cores}, "
            f"available_ram={ram_avail_mb}MB)")

        build_cmd = (f"cd '{src_root}' && cmake --build '{build_dir}' "
                     f"-j {jobs} > /tmp/llamacpp-build.log 2>&1; "
                     "_ec=$?; tail -3 /tmp/llamacpp-build.log; exit $_ec")
        code, out = run(build_cmd, timeout=7200)
        if code != 0:
            return {"ok": False, "machine": name,
                    "detail": f"build failed (exit {code}): "
                              f"{(out or '')[-400:]}"}

        bin_path = f"{probe.get('llama_bin') or build_dir + '/bin'}/llama-server"
        code, out = run(f"'{bin_path}' --version 2>&1 | head -1")
        version = (out or "").strip()
        log(f"{name} now: {version}")
        return {"ok": True, "machine": name, "bin": bin_path,
                "detail": f"source build updated to {version}"}
    except Exception as e:
        return {"ok": False, "machine": name,
                "detail": f"source update failed: {e}"}
