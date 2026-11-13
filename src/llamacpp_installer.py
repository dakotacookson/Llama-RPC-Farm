"""Machine-type-aware llama.cpp installer.

Picks the correct official prebuilt for each machine (os + gpu_vendor +
arch) and installs it through the peer agent (or locally):

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
import agent_client
import local_node
import urllib.request


GITHUB_API = "https://api.github.com/repos/ggml-org/llama.cpp/releases"
GITHUB_LATEST_DL = "https://github.com/ggml-org/llama.cpp/releases/latest/download"
LINUX_VER = "10.0"   # ubuntu-rocm-<ver>
WIN_CUDA_VER = "12.4"
WIN_ROCM_VER = "10.0"
# CUDA 12.4 GA needs >= 551.61 on Windows. Older drivers keep working with
# the Vulkan prebuilt (same llama.cpp/RPC feature set, no CUDA).
WIN_CUDA_MIN_DRIVER = (551, 61)
LINUX_CUDA_VER = "13.4"   # ubuntu-cuda-<ver>; upstream also ships 12.8
LINUX_CUDA_VER_OLD = "12.8"   # fallback when the newest CUDA asset is absent
DEST_POSIX = "~/llama.cpp/bin"
DEST_WIN = "%USERPROFILE%\\llama.cpp\\bin"
# Source rebuilds only need the farm's binaries — building EVERY target
# (tests, examples, benches) made upgrades needlessly slow. Unknown targets
# fall back to the minimal pair in update_source_on / the peer agent.
SOURCE_BUILD_TARGETS = (
    "llama-server", "ggml-rpc-server", "llama-cli", "llama-completion",
    "llama-quantize", "llama-fit-params", "llama-bench", "llama-tokenize",
)

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


def _fetch_release_asset_sets(limit=30):
    """[(tag, [asset names], base)] newest-first, but ONLY releases that
    contain a real llama.cpp binary asset (a `llama-*-bin-*` file).

    Upstream uploads a release's assets over several minutes (cudart
    companions first, main bins later). Treating a cudart-only release as
    "the latest" made installs ask for `llama-bNNN-bin-...` before it
    existed. Those partial releases are skipped here."""
    _hdrs = {"Accept": "application/vnd.github+json",
             "User-Agent": "llama-rpc-farm"}
    _tok = (os.environ.get("GITHUB_TOKEN")
            or os.environ.get("GH_TOKEN") or "").strip()
    if _tok:
        _hdrs["Authorization"] = f"Bearer {_tok}"
    req = urllib.request.Request(
        f"{GITHUB_API}?per_page={limit}", headers=_hdrs)
    with urllib.request.urlopen(req, timeout=30) as r:
        releases = json.loads(r.read().decode())
    out = []
    for rel in releases:
        names = [a["name"] for a in rel.get("assets", [])
                 if (("llama-" in a["name"] and "bin-" in a["name"])
                     or a["name"].startswith("cudart-"))]
        mains = [n for n in names if n.startswith("llama-") and "-bin-" in n]
        if not mains:
            continue  # still uploading / cudart-only
        base = (f"https://github.com/ggml-org/llama.cpp/releases/download/"
                f"{rel['tag_name']}")
        out.append((rel["tag_name"], names, base))
    return out


def _fetch_latest_assets(limit=30):
    """(tag, [asset names], download_base) of the newest release that has
    real llama.cpp binaries (cudart-only/uploading releases skipped)."""
    sets = _fetch_release_asset_sets(limit)
    if sets:
        return sets[0]
    return None, [], GITHUB_LATEST_DL


def _choose_release_for(machine, releases, glibc=None, driver_version=None):
    """Newest release that actually has EVERY asset this machine needs.
    Falls back to the newest release's missing list for a clear error."""
    fallback = None
    for tag, names, base in releases:
        try:
            picked = _pick_assets(machine, tag, names, glibc, driver_version)
        except ValueError:
            continue
        assets, dest, note, is_zip = picked
        missing = [n for n in assets if n not in names]
        sel = {"tag": tag, "assets": assets, "dest": dest,
               "backend": note, "zip": is_zip, "base": base,
               "missing": missing}
        if not missing:
            return sel
        if fallback is None:
            fallback = sel
    if fallback is not None:
        return fallback
    raise ValueError("no llama.cpp release assets found")


def _driver_tuple(text):
    """'551.61' / '551.61.0' -> (551, 61); None when unknown."""
    import re
    m = re.search(r"(\d+)\.(\d+)", str(text or ""))
    try:
        return (int(m.group(1)), int(m.group(2))) if m else None
    except (TypeError, ValueError):
        return None


def _windows_nvidia_driver(machine, run=None, is_local=True):
    """Detected NVIDIA driver tuple on a Windows machine, or None."""
    dv = ((machine.get("detected") or {}).get("driver_version")
          or machine.get("driver_version") or "")
    if dv:
        return _driver_tuple(dv)
    if not is_local and agent_client.should_use_agent(machine):
        try:
            sd = agent_client.system_detail(machine) or {}
            return _driver_tuple(sd.get("driver_version") or sd.get("driver"))
        except Exception:
            return None
    if run is not None and is_local:
        try:
            _c, out = run("nvidia-smi --query-gpu=driver_version "
                          "--format=csv,noheader")
        except Exception:
            return None
        first = (out or "").strip().splitlines()
        return _driver_tuple(first[0] if first else "")
    return None


def _glibc_from_text(text):
    import re
    m = re.search(r"(\d+)\.(\d+)", str(text or ""))
    try:
        return (int(m.group(1)), int(m.group(2))) if m else None
    except (TypeError, ValueError):
        return None


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


def _vendor_of(machine):
    """Effective GPU vendor: recorded field first, then detected device
    names. A blank/mis-detected vendor must never send an NVIDIA box to
    the CPU asset."""
    import gpu_vendor
    v = gpu_vendor.normalize_vendor((machine or {}).get("gpu_vendor"))
    if v:
        return v
    det = (machine or {}).get("detected") or {}
    names = " ".join(str(n) for n in (det.get("gpu_names") or [])).lower()
    if "nvidia" in names:
        return "nvidia"
    if "amd" in names or "radeon" in names or "instinct" in names:
        return "amd"
    if "intel" in names:
        return "intel"
    if "apple" in names:
        return "apple"
    return "cpu"


_BACKEND_EXPECT = {"nvidia": "CUDA", "amd": "ROCm", "intel": "SYCL",
                  "apple": "Metal", "cpu": "CPU"}


def _device_summary(text):
    """Backends llama-server reports in --list-devices (CPU if none)."""
    t = (text or "").lower()
    found = [n for n in ("CUDA", "ROCm", "HIP", "SYCL", "Vulkan", "Metal")
             if n.lower() in t]
    return ",".join(found) or "CPU"


def _device_warning(machine, text):
    """Non-fatal warning when the installed build doesn't expose the
    accelerator the machine's specs call for (or a CPU-classified host
    still exposes Vulkan, which llama.cpp will auto-use)."""
    vendor = _vendor_of(machine)
    got = _device_summary(text)
    exp = _BACKEND_EXPECT.get(vendor, "CPU")
    if exp != "CPU" and exp.lower() not in got.lower():
        return (f"expected {exp} device for {vendor} host but "
                f"--list-devices shows: {got}")
    if exp == "CPU" and "vulkan" in got.lower():
        return (f"CPU-classified host exposes Vulkan ({got}); llama.cpp "
                f"will use it automatically — consider a CPU-only build "
                f"if it is unstable")
    return ""


# Upstream CUDA prebuilts carry cubins for these SM levels only (verified
# on b12368: sm_86/89/103/120/121, PTX compute_121 only). Older cards
# (P100 sm_60, GTX 1080 sm_61, RTX 2060/2060S sm_75) get no kernels, so a
# prebuilt install there cannot work — those hosts must build from source.
# Update this set when upstream's release build matrix changes.
PREBUILT_CUDA_ARCHS = frozenset({86, 89, 103, 120, 121})


def _cap_int(text):
    """'8.6' -> 86; junk -> 0."""
    try:
        major, minor = str(text).strip().split(".")
        return int(major) * 10 + int(minor)
    except (TypeError, ValueError, AttributeError):
        return 0


def machine_cuda_archs(machine):
    """SM levels this machine's GPUs need, from detected compute_cap."""
    det = (machine or {}).get("detected") or {}
    out = []
    for cap in (det.get("compute_cap") or []):
        v = _cap_int(cap)
        if v:
            out.append(v)
    return out


def prebuilt_cuda_covers(machine):
    """True when the upstream CUDA prebuilt has kernels for every GPU.
    Unknown capabilities do not block (Windows detects none)."""
    caps = machine_cuda_archs(machine)
    if not caps:
        return True
    return all(c in PREBUILT_CUDA_ARCHS for c in caps)


def prebuilt_cuda_error(machine):
    """Non-empty reason when a CUDA prebuilt cannot run on this host."""
    if _vendor_of(machine) != "nvidia" or prebuilt_cuda_covers(machine):
        return ""
    need = ",".join(f"sm_{c}" for c in machine_cuda_archs(machine))
    have = ",".join(f"sm_{c}" for c in sorted(PREBUILT_CUDA_ARCHS))
    return (f"upstream CUDA prebuilt has kernels only for {have}; this host "
            f"needs {need} → use the source build (engine_mode=source)")


def backend_pref(machine):
    """Explicit backend pin: auto|cpu|cuda|rocm|sycl|vulkan|metal."""
    p = (machine or {}).get("engine_backend") or "auto"
    p = str(p).strip().lower()
    return p if p in ("auto", "cpu", "cuda", "rocm", "sycl",
                      "vulkan", "metal") else "auto"


def _machine_glibc(machine):
    if (machine or {}).get("os", "linux") != "linux":
        return None
    try:
        import local_node
        if local_node.is_local(machine):
            import subprocess
            out = subprocess.run(["ldd", "--version"], capture_output=True,
                                 text=True, timeout=10).stdout
            return _glibc_from_text(out)
    except Exception:
        pass
    try:
        import agent_client
        sd = agent_client.system_detail(machine) or {}
        return _glibc_from_text(sd.get("glibc"))
    except Exception:
        return None


def prebuilt_blocker(machine, glibc=None):
    """Reason a prebuilt cannot run here, or '' when it can.
    Only the CUDA path has hard OS/arch/glibc requirements."""
    os_type = (machine or {}).get("os", "linux")
    if os_type in ("windows", "macos"):
        return ""
    pref = backend_pref(machine)
    if pref in ("cpu", "vulkan", "rocm", "sycl", "metal"):
        return ""
    if _vendor_of(machine) != "nvidia":
        return ""
    err = prebuilt_cuda_error(machine)
    if err:
        return err
    if glibc is None:
        glibc = _machine_glibc(machine)
    if glibc is not None and glibc < GLIBC_CUDA_MIN:
        return (f"CUDA prebuilt needs glibc >= {GLIBC_CUDA_MIN[0]}."
                f"{GLIBC_CUDA_MIN[1]}; this host has {glibc[0]}.{glibc[1]} "
                f"→ use the source build")
    return ""


def decide_engine_plan(machine, probe=None, glibc=None):
    """SOURCE OF TRUTH: prebuilt only when this host can actually run it,
    otherwise source. engine_mode=source/prebuilt overrides.

    Returns {mode, reason, prebuilt_blocker, source_available, probe}.
    """
    pref = (machine.get("engine_mode") or "auto").strip().lower()
    if probe is None:
        try:
            probe = probe_setup(machine)
        except Exception:
            probe = None
    source_ok = bool(probe and probe.get("kind") == "source")
    blocker = prebuilt_blocker(machine, glibc)
    if pref == "source":
        return {"mode": "source", "reason": "engine_mode=source",
                "prebuilt_blocker": blocker, "source_available": source_ok,
                "probe": probe}
    if pref == "prebuilt":
        return {"mode": "prebuilt", "reason": "engine_mode=prebuilt",
                "prebuilt_blocker": blocker, "source_available": source_ok,
                "probe": probe}
    if not blocker:
        return {"mode": "prebuilt",
                "reason": "prebuilt is compatible with this host's "
                          "OS/GPU/glibc",
                "prebuilt_blocker": "", "source_available": source_ok,
                "probe": probe}
    if source_ok:
        return {"mode": "source", "reason": blocker,
                "prebuilt_blocker": blocker, "source_available": True,
                "probe": probe}
    return {"mode": "source",
            "reason": blocker + " (no source tree detected)",
            "prebuilt_blocker": blocker, "source_available": False,
            "probe": probe}


def _first_present(assets, groups):
    """First (names, note) whose assets ALL exist upstream. Falls back to
    the first group so the caller reports the missing asset (upstream
    rotates asset versions) instead of crashing."""
    for names, note in groups:
        if names and all(n in assets for n in names):
            return names, note
    return groups[0][0], groups[0][1]


def _pick_assets(machine, tag, assets, glibc=None, driver_version=None):
    """Choose release asset name(s) for a machine. Returns
    (asset_names, dest, backend_note, is_win_zip) or raises ValueError.

    Backend is chosen from the machine's real specs (vendor + arch + OS +
    glibc/driver), with an ordered fallback chain when upstream lacks an
    asset: CUDA -> Vulkan -> CPU (NVIDIA), ROCm -> Vulkan -> CPU (AMD),
    SYCL -> CPU (Intel), Metal (macOS), CPU (no accelerator)."""
    os_type = machine.get("os", "linux")
    vendor = _vendor_of(machine)
    pref = backend_pref(machine)
    arch = (machine.get("detected") or {}).get("arch") or "x86_64"
    is_arm = arch in ("arm64", "aarch64")
    if os_type == "macos":
        # Apple Silicon -> arm64; an Intel Mac (x86_64) -> x64.
        kind = "x64" if arch in ("x86_64", "x64") and not is_arm else "arm64"
        return [f"llama-{tag}-bin-macos-{kind}.tar.gz"], DEST_POSIX, \
            "Metal backend included", False
    if os_type == "windows":
        arch_kind = "arm64" if is_arm else "x64"
        if pref == "cpu":
            names, note = _first_present(assets, [
                ([f"llama-{tag}-bin-win-cpu-{arch_kind}.zip"],
                 "CPU backend (pinned)")])
            return names, DEST_WIN, note, True
        if pref == "vulkan":
            names, note = _first_present(assets, [
                ([f"llama-{tag}-bin-win-vulkan-{arch_kind}.zip"],
                 "Vulkan backend (pinned)"),
                ([f"llama-{tag}-bin-win-cpu-{arch_kind}.zip"],
                 "CPU backend (no Vulkan asset)")])
            return names, DEST_WIN, note, True
        if pref == "cuda":
            vendor = "nvidia"
        elif pref == "rocm":
            vendor = "amd"
        elif pref == "sycl":
            vendor = "intel"
        if vendor == "amd":
            groups = [
                ([f"llama-{tag}-bin-win-rocm-{WIN_ROCM_VER}-{arch_kind}.zip"],
                 "HIP/ROCm backend"),
                ([f"llama-{tag}-bin-win-vulkan-{arch_kind}.zip"],
                 "Vulkan backend (no ROCm asset upstream)"),
                ([f"llama-{tag}-bin-win-cpu-{arch_kind}.zip"],
                 "CPU backend (no ROCm/Vulkan asset upstream)"),
            ]
        elif vendor == "intel":
            groups = [
                ([f"llama-{tag}-bin-win-sycl-{arch_kind}.zip"],
                 "SYCL backend (oneAPI runtime bundled)"),
                ([f"llama-{tag}-bin-win-cpu-{arch_kind}.zip"],
                 "CPU backend (no SYCL asset upstream)"),
            ]
        elif vendor == "nvidia" and is_arm:
            groups = [
                ([f"llama-{tag}-bin-win-cuda-13.4-arm64.zip",
                  "cudart-llama-bin-win-cuda-13.4-arm64.zip"],
                 "CUDA 13.4 backend (+cudart)"),
                ([f"llama-{tag}-bin-win-cpu-{arch_kind}.zip"],
                 "CPU backend (no CUDA arm64 asset upstream)"),
            ]
        elif vendor == "nvidia":
            if (driver_version is not None
                    and driver_version < WIN_CUDA_MIN_DRIVER):
                # Old driver can't run the CUDA prebuilt at all — do not
                # fall through to CUDA: Vulkan, else CPU.
                groups = [
                    ([f"llama-{tag}-bin-win-vulkan-{arch_kind}.zip"],
                     f"Vulkan backend (NVIDIA driver "
                     f"{driver_version[0]}.{driver_version[1]} is older "
                     f"than CUDA {WIN_CUDA_VER}'s minimum)"),
                    ([f"llama-{tag}-bin-win-cpu-{arch_kind}.zip"],
                     "CPU backend (old NVIDIA driver, no Vulkan asset "
                     "upstream)"),
                ]
            else:
                groups = [
                    ([f"llama-{tag}-bin-win-cuda-{WIN_CUDA_VER}-{arch_kind}.zip",
                      f"cudart-llama-bin-win-cuda-{WIN_CUDA_VER}-{arch_kind}.zip"],
                     f"CUDA {WIN_CUDA_VER} backend (+cudart)"),
                    ([f"llama-{tag}-bin-win-vulkan-{arch_kind}.zip"],
                     "Vulkan backend (no CUDA asset upstream)"),
                    ([f"llama-{tag}-bin-win-cpu-{arch_kind}.zip"],
                     "CPU backend (no CUDA/Vulkan asset upstream)"),
                ]
        else:
            groups = [([f"llama-{tag}-bin-win-cpu-{arch_kind}.zip"],
                       "CPU backend")]
        names, note = _first_present(assets, groups)
        return names, DEST_WIN, note, True
    # linux / other POSIX
    arch_kind = "arm64" if is_arm else "x64"
    if pref == "cpu":
        names, note = _first_present(assets, [
            ([f"llama-{tag}-bin-ubuntu-{arch_kind}.tar.gz"],
             "CPU backend (pinned)")])
        return names, DEST_POSIX, note, False
    if pref == "vulkan":
        names, note = _first_present(assets, [
            ([f"llama-{tag}-bin-ubuntu-vulkan-{arch_kind}.tar.gz"],
             "Vulkan backend (pinned)"),
            ([f"llama-{tag}-bin-ubuntu-{arch_kind}.tar.gz"],
             "CPU backend (no Vulkan asset)")])
        return names, DEST_POSIX, note, False
    if pref == "cuda":
        vendor = "nvidia"
    elif pref == "rocm":
        vendor = "amd"
    elif pref == "sycl":
        vendor = "intel"
    if vendor == "amd":
        groups = [
            ([f"llama-{tag}-bin-ubuntu-rocm-{LINUX_VER}-{arch_kind}.tar.gz"],
             "HIP/ROCm backend"),
            ([f"llama-{tag}-bin-ubuntu-vulkan-{arch_kind}.tar.gz"],
             "Vulkan backend (no ROCm asset upstream)"),
            ([f"llama-{tag}-bin-ubuntu-{arch_kind}.tar.gz"],
             "CPU backend (no ROCm/Vulkan asset upstream)"),
        ]
    elif vendor == "intel":
        groups = [
            ([f"llama-{tag}-bin-ubuntu-sycl-fp16-{arch_kind}.tar.gz"],
             "SYCL backend (oneAPI runtime bundled)"),
            ([f"llama-{tag}-bin-ubuntu-{arch_kind}.tar.gz"],
             "CPU backend (no SYCL asset upstream)"),
        ]
    elif vendor == "nvidia":
        groups = []
        # CUDA linux prebuilts need glibc >= 2.38; the Vulkan sibling only
        # needs 2.34. Try newest CUDA then the known-good older one.
        if glibc is not None and glibc >= GLIBC_CUDA_MIN:
            for _cv in (LINUX_CUDA_VER, LINUX_CUDA_VER_OLD):
                groups.append(
                    ([f"llama-{tag}-bin-ubuntu-cuda-{_cv}-{arch_kind}.tar.gz",
                      f"cudart-llama-{tag}-bin-ubuntu-cuda-{_cv}-{arch_kind}.tar.gz"],
                     f"CUDA {_cv} backend (+cudart)"))
        groups += [
            ([f"llama-{tag}-bin-ubuntu-vulkan-{arch_kind}.tar.gz"],
             "Vulkan backend (no CUDA asset usable on this glibc)"),
            ([f"llama-{tag}-bin-ubuntu-{arch_kind}.tar.gz"],
             "CPU backend (no CUDA/Vulkan asset upstream)"),
        ]
    else:
        groups = [([f"llama-{tag}-bin-ubuntu-{arch_kind}.tar.gz"],
                   "CPU backend")]
    names, note = _first_present(assets, groups)
    return names, DEST_POSIX, note, False


def plan(machine):
    """What WOULD be installed for this machine: asset names + backend.
    No side effects; used by the UI before installing."""
    try:
        releases = _fetch_release_asset_sets()
    except Exception as e:
        return {"ok": False, "error": f"cannot reach GitHub releases: {e}"}
    if not releases:
        return {"ok": False, "error": "no llama.cpp release assets found"}
    try:
        sel = _choose_release_for(
            machine, releases,
            driver_version=_driver_tuple(
                (machine.get("detected") or {}).get("driver_version")
                or machine.get("driver_version")))
    except ValueError as e:
        return {"ok": False, "error": str(e)}
    return {"ok": True, "tag": sel["tag"], "assets": sel["assets"],
            "dest": sel["dest"], "backend": sel["backend"],
            "missing": sel["missing"]}


def install_on(machine, log=lambda m: None):
    """Install the right prebuilt on a machine (local or via the agent).
    Returns dict {ok, machine, bin, detail}."""
    import config_loader as cfg
    import local_node
    import subprocess   # TimeoutExpired is a SubprocessError, NOT an OSError

    name = machine.get("name", "?")
    is_local = local_node.is_local(machine)
    is_win = machine.get("os", "linux") == "windows"
    # The source-of-truth gate: refuse a prebuilt that this host cannot
    # actually run (CUDA arch coverage / glibc), unless an explicit
    # backend pin (vulkan/cpu/...) makes it viable.
    blocker = prebuilt_blocker(machine)
    if blocker:
        return {"ok": False, "machine": name, "detail": blocker}

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
        raise RuntimeError("remote install is agent-only; use the peer agent")

    run = run_local if is_local else run_remote
    try:
        # arch matters for asset choice; probe it when not detected yet
        if not ((machine.get("detected") or {}).get("arch")):
            if not is_local:
                try:
                    arch = (agent_client.info(machine) or {}).get("arch")
                except Exception:
                    arch = None
                machine.setdefault("detected", {})["arch"] = arch or "x86_64"
            elif is_win:
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
    except (RuntimeError, OSError, subprocess.TimeoutExpired) as e:
        return {"ok": False, "machine": name, "detail": f"probe failed: {e}"}
    try:
        if is_win:
            glibc = None
        elif is_local:
            glibc = _glibc_version(None, run)
        else:
            try:
                sd = agent_client.system_detail(machine) or {}
                glibc = _glibc_from_text(sd.get("glibc"))
            except Exception:
                glibc = None
        driver = None
        if is_win and (machine.get("gpu_vendor") or "").lower() == "nvidia":
            driver = _windows_nvidia_driver(machine, run=run,
                                            is_local=is_local)
        releases = _fetch_release_asset_sets()
        if not releases:
            return {"ok": False, "machine": name,
                    "detail": "no llama.cpp release assets found"}
        sel = _choose_release_for(machine, releases, glibc, driver)
        base = sel["base"]
        plan_info = {"ok": True, "tag": sel["tag"],
                     "assets": sel["assets"], "dest": sel["dest"],
                     "backend": sel["backend"],
                     "missing": sel["missing"]}
    except Exception as e:
        return {"ok": False, "machine": name,
                "detail": f"cannot reach GitHub releases: {e}"}
    if not plan_info.get("ok"):
        return {"ok": False, "machine": name, "detail": plan_info["error"]}
    if plan_info["missing"]:
        return {"ok": False, "machine": name,
                "detail": "release assets not found upstream: "
                          + ", ".join(plan_info["missing"])}
    # Never silently replace a CUDA-configured machine with a Vulkan
    # prebuilt. Upstream CUDA builds need glibc >= 2.38; on older hosts
    # _pick_assets falls back to Vulkan, which cannot serve --device
    # CUDA*/--mmproj-device CUDA* (this broke aibox3). Such machines must
    # update from a CUDA source build instead.
    backend_note = (plan_info.get("backend") or "").lower()
    tokens = str(machine.get("gpu") or "").upper()
    if (not is_win and "vulkan" in backend_note
            and (machine.get("gpu_vendor") or "").lower() == "nvidia"
            and "CUDA" in tokens):
        return {"ok": False, "machine": name,
                "detail": "this host's only glibc-compatible prebuilt is "
                          "Vulkan, but the machine is configured for CUDA "
                          "device tokens. Point llama_bin at a CUDA source "
                          "build (e.g. ~/llama.cpp/build/bin) so the Update "
                          "button rebuilds it, or switch the machine/model "
                          "device tokens to Vulkan."}
    urls = [f"{base}/{n}" for n in plan_info["assets"]]
    if not is_local and agent_client.should_use_agent(machine):
        try:
            dest = "~/llama.cpp/bin"
            agent_client.install_artifact(
                machine, urls, dest,
                archive="zip" if is_win else "tar.gz",
                # Windows release zips are flat (no top-level directory);
                # stripping one component discarded every member.
                strip_components=0 if is_win else 1,
                mode=None if is_win else 0o755,
                # CLEAN install: stage + verify + swap, so stale backend
                # plugins from a previous build can't survive the upgrade.
                replace=True)
            bin_name = "llama-server.exe" if is_win else "llama-server"
            rpc_name = "ggml-rpc-server.exe" if is_win else "ggml-rpc-server"
            v = agent_client.install_verify(machine, f"{dest}/{bin_name}")
            if v.get("returncode") != 0:
                return {"ok": False, "machine": name,
                        "detail": f"installed but llama-server verify failed: "
                                  f"{v.get('output', '')[-300:]}"}
            # RPC builds disagree on --version; --help is the reliable
            # load/run probe (and detects missing shared-library symlinks).
            rpc_ok = False
            rpc_out = ""
            for args in (["--help"], ["--version"]):
                try:
                    rv = agent_client.install_verify(
                        machine, f"{dest}/{rpc_name}", args, timeout=35) or {}
                    rpc_out = str(rv.get("output") or "")
                    if rv.get("returncode") == 0:
                        rpc_ok = True
                        break
                except Exception as e:
                    rpc_out = str(e)
                    if "not allowed" in rpc_out:
                        # Older agents cannot verify RPC; presence is the
                        # best available check until the agent is updated.
                        try:
                            rst = agent_client.file_stat(
                                machine, f"{dest}/{rpc_name}") or {}
                            rpc_ok = bool(rst.get("exists")
                                          and rst.get("is_file"))
                        except Exception:
                            rpc_ok = False
                        break
            if not rpc_ok:
                return {"ok": False, "machine": name,
                        "detail": f"installed but llama RPC verify failed "
                                  f"({rpc_name}): {rpc_out[-300:]}"}
            dev_note, dev_warn = "", ""
            try:
                dv = agent_client.install_verify(
                    machine, f"{dest}/{bin_name}", ["--list-devices"],
                    timeout=45) or {}
                dev_text = str(dv.get("output") or "")
                dev_note = _device_summary(dev_text)
                dev_warn = _device_warning(machine, dev_text)
            except Exception:
                pass
            bin_path = f"{dest}/{bin_name}"
            try:
                cfg.machines.update(name, {"llama_bin": dest})
            except Exception:
                pass
            return {"ok": True, "machine": name, "bin": bin_path,
                    "backend": plan_info["backend"],
                    "devices": dev_note, "warning": dev_warn,
                    "detail": f"installed {', '.join(plan_info['assets'])} "
                              f"({plan_info['tag']}) to {dest} (llama-server + "
                              f"ggml-rpc-server)"
                              + (f"; devices: {dev_note}" if dev_note else "")
                              + (f"; WARNING: {dev_warn}" if dev_warn else "")}
        except Exception as e:
            return {"ok": False, "machine": name,
                    "detail": f"agent install failed: {e}"}
    # download+extract budget: CUDA+cuDART archives are ~1 GB total; a
    # flat 600s was tight on slow uplinks and killed the run mid-download
    # (fixed 2026-10-31). 20 min per asset set, verify keeps its own.
    dl_timeout = 1200 * max(1, len(urls))
    try:
        if is_win:
            # CLEAN staged install: extract into a fresh staging dir,
            # verify, then swap it in (old bin kept as .bak-<ts>).
            ps = ["$ErrorActionPreference='Stop'",
                  '$root="$env:USERPROFILE\\llama.cpp"',
                  '$stage=Join-Path $root (".bin.staging-"+[guid]::NewGuid().ToString("N"))',
                  '$backup=Join-Path $root ("bin.bak-"+$(Get-Date -Format "yyyyMMdd-HHmmss"))',
                  'Remove-Item -Recurse -Force $stage -ErrorAction SilentlyContinue',
                  'New-Item -ItemType Directory -Force -Path $stage | Out-Null']
            for i, u in enumerate(urls):
                ps.append(
                    f"Invoke-WebRequest -Uri '{u}' -OutFile \"$env:TEMP\\llamabin{i}.zip\"")
                ps.append(
                    f"Expand-Archive -Force \"$env:TEMP\\llamabin{i}.zip\" $stage")
                ps.append(f"Remove-Item \"$env:TEMP\\llamabin{i}.zip\"")
            ps += [
                '& (Join-Path $stage "llama-server.exe") --version | Out-Null',
                'if ($LASTEXITCODE -ne 0) { Remove-Item -Recurse -Force $stage -ErrorAction SilentlyContinue; throw "staged llama-server verify failed" }',
                'if (Test-Path (Join-Path $root "bin")) { Move-Item (Join-Path $root "bin") $backup -Force }',
                'try { Move-Item $stage (Join-Path $root "bin") -Force } catch { if (Test-Path $backup) { Move-Item $backup (Join-Path $root "bin") -Force }; throw }',
            ]
            code, out = run("; ".join(ps), timeout=dl_timeout)
        else:
            # CLEAN staged install: extracting in place left stale backend
            # plugins from a previous build (an old libggml-vulkan.so
            # crashing the new core). Stage -> verify -> swap -> keep old.
            cmds = [
                "set -e",
                'root="$HOME/llama.cpp"',
                'stage="$root/.bin.staging.$$"',
                'backup="$root/bin.bak-$(date +%Y%m%d-%H%M%S)"',
                'trap \'rm -rf "$stage"\' EXIT',
                'rm -rf "$stage"; mkdir -p "$stage"',
            ]
            for i, u in enumerate(urls):
                cmds.append(
                    f"curl -fSL --retry 3 -o /tmp/llamabin{i}.tgz '{u}'")
                cmds.append(
                    f'tar -xzf /tmp/llamabin{i}.tgz -C "$stage" '
                    "--strip-components=1")
                cmds.append(f"rm -f /tmp/llamabin{i}.tgz")
            cmds += [
                'chmod +x "$stage/"* 2>/dev/null || true',
                'test -x "$stage/llama-server"',
                '"$stage/llama-server" --version >/dev/null',
                'if [ -x "$stage/ggml-rpc-server" ]; then '
                '"$stage/ggml-rpc-server" --version >/dev/null 2>&1 || true; fi',
                'if [ -e "$root/bin" ]; then mv "$root/bin" "$backup"; fi',
                'if ! mv "$stage" "$root/bin"; then '
                'if [ -e "$backup" ]; then mv "$backup" "$root/bin"; fi; '
                'exit 1; fi',
                'trap - EXIT',
            ]
            code, out = run("\n".join(cmds), timeout=dl_timeout)
        if code != 0:
            return {"ok": False, "machine": name,
                    "detail": f"install failed (exit {code}): {out[-500:]}"}
    except (RuntimeError, OSError, subprocess.TimeoutExpired) as e:
        return {"ok": False, "machine": name, "detail": f"transfer failed: {e}"}

    # post-install verify: the server binary must at least start — a
    # partial extract (disk full, zip cut short) used to report success
    # and only blow up on first launch.
    if is_win:
        vcmd = '& "$env:USERPROFILE\\llama.cpp\\bin\\llama-server.exe" --version'
    else:
        vcmd = '"$HOME/llama.cpp/bin/llama-server" --version'
    try:
        vcode, vout = run(vcmd, timeout=120)
    except (RuntimeError, OSError, subprocess.TimeoutExpired) as e:
        vcode, vout = 1, f"verify crashed: {e}"
    if vcode != 0:
        return {"ok": False, "machine": name,
                "detail": f"installed but llama-server verify failed "
                          f"(exit {vcode}): {(vout or '')[-300:]}"}
    bin_dir = DEST_WIN if is_win else DEST_POSIX
    rpc_name = "ggml-rpc-server.exe" if is_win else "ggml-rpc-server"
    if is_win:
        rpc_cmd = f'if exist "{bin_dir}\\{rpc_name}" (exit 0) else (exit 1)'
    else:
        rpc_cmd = f'test -x "{bin_dir}/{rpc_name}"'
    try:
        rcode, rout = run(rpc_cmd, timeout=60)
    except (RuntimeError, OSError, subprocess.TimeoutExpired) as e:
        rcode, rout = 1, f"verify crashed: {e}"
    if rcode != 0:
        return {"ok": False, "machine": name,
                "detail": f"installed but llama RPC verify failed "
                          f"({rpc_name}): {(rout or '')[-300:]}"}
    bin_path = (f"{bin_dir}\\llama-server.exe" if is_win
                else f"{bin_dir}/llama-server")
    # Windows: the upgrade just (re)wrote the exes — re-ensure the
    # firewall rules match the CURRENT exe path before the machine is
    # trusted to serve again (see the WIN_FW block comment above).
    fw_note = _ensure_win_firewall(run, log) if is_win else ""
    if is_win:
        # PATH export: the app launches through the machine's llama_bin
        # pointer, but a manual terminal should find llama-server too.
        # Idempotent USER-scope append (no admin needed); non-fatal —
        # failure only costs manual PATH convenience, never serving.
        path_ps = ("$p=[Environment]::GetEnvironmentVariable('Path','User'); "
                   "if($p -notlike '*llama.cpp\\bin*'){ "
                   "[Environment]::SetEnvironmentVariable('Path', "
                   "($p.TrimEnd(';')+';'+\"$env:USERPROFILE\\llama.cpp\\bin\"), 'User') }")
        try:
            run(path_ps, timeout=60)
        except Exception:
            pass
    try:
        cfg.machines.update(name, {"llama_bin": bin_dir})
    except Exception:
        pass
    dev_note, dev_warn = "", ""
    try:
        dev_cmd = (f'& "{bin_dir}\\llama-server.exe" --list-devices'
                   if is_win else
                   f'"{bin_dir}/llama-server" --list-devices')
        dcode, dout = run(dev_cmd, timeout=60)
        dev_note = _device_summary(dout)
        dev_warn = _device_warning(machine, dout)
    except (RuntimeError, OSError, subprocess.TimeoutExpired):
        pass
    detail = (f"installed {', '.join(plan_info['assets'])} "
              f"({plan_info['tag']}) to {bin_dir} (llama-server + "
              f"ggml-rpc-server)")
    if fw_note:
        detail += f"; {fw_note}"
    if dev_note:
        detail += f"; devices: {dev_note}"
    if dev_warn:
        detail += f"; WARNING: {dev_warn}"
    return {"ok": True, "machine": name, "bin": bin_path,
            "backend": plan_info["backend"], "devices": dev_note,
            "warning": dev_warn, "detail": detail}


# ---------- dynamic setup classification ----------
# Never pattern-match the update path off a path string: probe the
# machine itself. A machine added five minutes ago must classify the
# same as one that's been here a year, and the flags it was built with
# (CUDA/ROCm/Vulkan, arch list, RPC/FA/NCCL) decide how it updates —
# observed 2026-10-24: a stock Vulkan prebuilt installed over aibox1's
# custom CUDA build broke every --device CUDA* launch on the farm.

def _machine_run(machine):
    """(run, is_local) for a machine — bash -c locally. Remote source
    work goes through the peer agent and never reaches this helper."""
    import local_node
    is_local = local_node.is_local(machine)

    def run_local(cmd, timeout=7200):
        import subprocess
        proc = subprocess.run(["bash", "-c", cmd], capture_output=True,
                              text=True, timeout=timeout, shell=False)
        return proc.returncode, proc.stdout + proc.stderr

    def run_remote(cmd, timeout=7200):
        raise RuntimeError("remote install is agent-only; use the peer agent")

    return (run_local if is_local else run_remote), is_local


def source_build_args(machine):
    """CMake args for a fresh source build, derived from real specs:
    NVIDIA 10-series+ (compute caps), AMD/HIP, Intel/SYCL, else CPU.
    RPC is always on; CUDA flash-attention on for NVIDIA."""
    vendor = _vendor_of(machine)
    caps = sorted(set(machine_cuda_archs(machine)))
    args = ["-DGGML_RPC=ON", "-DGGML_NATIVE=ON"]
    if vendor == "nvidia":
        args += ["-DGGML_CUDA=ON", "-DGGML_CUDA_FA=ON"]
        if caps:
            args.append("-DCMAKE_CUDA_ARCHITECTURES="
                        + ";".join(str(c) for c in caps))
    elif vendor == "amd":
        args += ["-DGGML_HIP=ON"]
    elif vendor == "intel":
        args += ["-DGGML_SYCL=ON"]
    return args


def _avail_ram_mb():
    try:
        with open("/proc/meminfo", encoding="utf-8") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) // 1024
    except OSError:
        pass
    return 0


def bootstrap_source_on(machine, log=lambda m: None):
    """No compatible prebuilt AND no source tree: clone + configure +
    build one from source using the machine's detected specs."""
    import subprocess
    import shlex
    import local_node
    name = machine.get("name", "?")
    if machine.get("os", "linux") not in ("linux", "macos"):
        return {"ok": False, "machine": name,
                "detail": "no compatible prebuilt exists and source "
                          "bootstrap is POSIX-only on this machine"}
    try:
        tag, _assets, _base = _fetch_latest_assets()
    except Exception as e:
        return {"ok": False, "machine": name,
                "detail": f"cannot determine the release tag: {e}"}
    if not tag:
        return {"ok": False, "machine": name,
                "detail": "no llama.cpp release tag found"}
    args = source_build_args(machine)
    vendor = _vendor_of(machine)
    log(f"{name}: no prebuilt/no source tree → bootstrapping {tag} "
        f"({vendor}; {' '.join(args)})")
    is_local = local_node.is_local(machine)
    if not is_local:
        try:
            import agent_client
            res = agent_client.build_bootstrap(
                machine, "~/llama.cpp", "~/llama.cpp/build",
                "~/llama.cpp/build/bin/llama-server", tag, args)
        except Exception as e:
            return {"ok": False, "machine": name,
                    "detail": f"source bootstrap failed: {e}"}
        if not res.get("ok"):
            return {"ok": False, "machine": name,
                    "detail": res.get("error") or "source bootstrap failed"}
        bin_dir = res.get("llama_bin") or "~/llama.cpp/build/bin"
        try:
            import config_loader as _cfg
            _cfg.machines.update(name, {"llama_bin": bin_dir})
        except Exception:
            pass
        log(f"{name} devices: "
            + " | ".join((res.get("devices") or "").splitlines()[:6]))
        return {"ok": True, "machine": name,
                "bin": f"{bin_dir}/llama-server",
                "backend": "source", "devices": res.get("devices") or "",
                "detail": f"source build {res.get('version', '')} ({tag})"}
    jobs = max(1, min(os.cpu_count() or 2, (_avail_ram_mb() or 4096) // 2048))
    targets = list(SOURCE_BUILD_TARGETS)
    src = os.path.expanduser("~/llama.cpp")
    build = os.path.join(src, "build")
    tool_check = {
        "nvidia": 'command -v nvcc >/dev/null || [ -x /usr/local/cuda/bin/nvcc ]',
        "amd": "command -v hipcc >/dev/null",
        "intel": "command -v icpx >/dev/null || command -v icx >/dev/null",
    }.get(vendor, "true")
    qargs = " ".join(shlex.quote(a) for a in args)
    qtargets = " ".join(shlex.quote(t) for t in targets)
    script = f"""
set -e
{tool_check} || {{ echo "missing toolchain for {vendor} (nvcc/hipcc/icpx)" >&2; exit 2; }}
src="$HOME/llama.cpp"; build="$src/build"
if [ ! -d "$src/.git" ]; then
  if [ -e "$src" ] && [ -n "$(ls -A "$src" 2>/dev/null)" ]; then
    echo "$src exists but is not a git checkout" >&2; exit 3
  fi
  git clone --depth 1 --branch {shlex.quote(tag)} https://github.com/ggml-org/llama.cpp "$src"
else
  git -C "$src" fetch --depth 1 origin tag {shlex.quote(tag)} || git -C "$src" fetch --tags origin
  git -C "$src" checkout --detach -f {shlex.quote(tag)}
fi
mkdir -p "$build"
if [ ! -f "$build/CMakeCache.txt" ]; then
  cmake -S "$src" -B "$build" -DCMAKE_BUILD_TYPE=Release {qargs}
fi
cmake --build "$build" --target {qtargets} -j {jobs}
test -x "$build/bin/llama-server"
test -x "$build/bin/ggml-rpc-server"
"$build/bin/llama-server" --version
"$build/bin/llama-server" --list-devices
"""
    log(f"{name}: building source with -j{jobs} …")
    try:
        proc = subprocess.run(["bash", "-c", script], capture_output=True,
                              text=True, timeout=21600)
    except (OSError, subprocess.SubprocessError) as e:
        return {"ok": False, "machine": name,
                "detail": f"source bootstrap failed: {e}"}
    out = (proc.stdout or "") + (proc.stderr or "")
    if proc.returncode != 0:
        return {"ok": False, "machine": name,
                "detail": f"source bootstrap failed (exit {proc.returncode}): "
                          + out[-800:]}
    try:
        import config_loader as _cfg
        _cfg.machines.update(name, {"llama_bin": build + "/bin"})
    except Exception:
        pass
    lines = [l for l in out.splitlines() if l.strip()]
    version = next((l for l in lines if l.startswith("version:")), "")
    log(f"{name}: source build ready ({version or 'ok'})")
    return {"ok": True, "machine": name,
            "bin": os.path.join(build, "bin", "llama-server"),
            "backend": "source", "devices": out[-800:],
            "detail": f"source build {version or ''} ({tag})"}


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
    if (lb and not local_node.is_local(machine)
            and agent_client.should_use_agent(machine)):
        try:
            res = agent_client.build_probe(machine, lb)
            res["machine"] = machine.get("name", "?")
            return res
        except Exception:
            out["machine"] = machine.get("name", "?")
            return out
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
            # A source build can be <root>/build/bin (the installer
            # prebuilt may sit beside it in <root>/bin). Prefer the
            # directory that actually has a CMakeCache.
            f'for c in "$d" "$d/build"; do '
            f'  if [ -f "$c/CMakeCache.txt" ]; then '
            f'    echo "D=$c"; echo CACHE; '
            f'    [ -f "$c/bin/llama-server" ] && echo BIN || echo NOBIN; '
            f'    exit 0; fi; done; '
            f'[ -d "$d" ] || exit 3; echo "D=$d"; echo NOCACHE; '
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
               f'GGML_SYCL:BOOL|GGML_METAL:BOOL|GGML_OPENCL:BOOL|'
               f'GGML_RPC:BOOL|GGML_CUDA_FA:BOOL|GGML_CUDA_NCCL:BOOL|'
               f'GGML_BLAS:BOOL|GGML_NATIVE:BOOL|LLAMA_CURL:BOOL|'
               f'CMAKE_BUILD_TYPE:STRING|CMAKE_CUDA_ARCHITECTURES:|'
               f'CMAKE_CUDA_COMPILER:FILEPATH|'
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
        # Record the current binaries so bootstrap/manifest can report
        # them without re-probing. Both are required: llama-server serves
        # models, ggml-rpc-server is the peer's RPC backend.
        try:
            _c, vout = run(f'"{out["llama_bin"]}/llama-server" --version 2>&1 | head -1')
            out["version"] = (vout or "").strip()
        except Exception:
            out["version"] = ""
        try:
            _c, rout = run(f'test -x "{out["llama_bin"]}/ggml-rpc-server" '
                           '&& echo RPC || echo NORPC')
            out["rpc"] = "RPC" in (rout or "")
        except Exception:
            out["rpc"] = False
        out["kind"] = "source"
        return out
    except Exception as e:
        log(f"probe failed on {machine.get('name', '?')}: {e}")
        return out          # unreachable -> prebuilt attempt is harmless


def _version_matches_tag(version_line, tag):
    """True when a --version line reports the build number in a `bNNNN` tag."""
    import re
    m = re.search(r"build (\d+)", version_line or "")
    t = (tag or "").strip().lower()
    if not m or not t:
        return False
    if not re.fullmatch(r"b\d+", t):
        return True  # unknown tag shape — don't block on it
    return m.group(1) == t[1:]


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
    try:
        target_tag, _assets, _base = _fetch_latest_assets()
    except Exception:
        target_tag = None
    log(f"{name}: source build target = {target_tag or 'origin/master'}")
    if (not local_node.is_local(machine)
            and agent_client.should_use_agent(machine)):
        bin_path = (f"{probe.get('llama_bin') or build_dir + '/bin'}"
                    f"/llama-server")
        last_err = None
        # A just-updated agent can close the first long request while it is
        # still settling after restart. Retry once after confirming health.
        for attempt in (1, 2):
            try:
                res = agent_client.build_update(
                    machine, src_root, build_dir, bin_path,
                    ref=target_tag)
                if res.get("ok"):
                    ver = str(res.get("version") or "")
                    if target_tag and not _version_matches_tag(
                            ver, target_tag):
                        return {"ok": False, "machine": name,
                                "detail": f"built {ver or 'unknown'}, "
                                          f"expected {target_tag}"}
                    try:
                        import config_loader as _cfg
                        _cfg.machines.update(
                            name,
                            {"llama_bin": probe.get("llama_bin")
                             or (build_dir + "/bin")})
                    except Exception:
                        pass
                    log(f"{name} devices: "
                        + " | ".join((res.get("devices") or "")
                                      .splitlines()[:6]))
                    return {"ok": True, "machine": name, "bin": bin_path,
                            "detail": f"source build updated to {ver}",
                            "devices": res.get("devices") or ""}
                return {"ok": False, "machine": name,
                        "detail": res.get("error") or "source update failed"}
            except Exception as e:
                last_err = e
                if attempt == 1:
                    import time as _t
                    _t.sleep(5)
                    try:
                        agent_client.health(machine, timeout=5)
                    except Exception:
                        pass
                    continue
        return {"ok": False, "machine": name,
                "detail": f"agent source update failed: {last_err}"}
    run, _is_local = _machine_run(machine)

    try:
        code, out = run(f"cd '{src_root}' && git status --porcelain "
                        "| grep -v '^??' | head -5")
        if (out or "").strip():
            return {"ok": False, "machine": name,
                    "detail": "tracked files modified: "
                              + (out or "").strip()[:200]}
        if target_tag:
            code, out = run(f"cd '{src_root}' && git fetch origin tag "
                            f"'{target_tag}' 2>&1 | tail -1", timeout=900)
            if code != 0:
                code, out = run(f"cd '{src_root}' && git fetch --tags "
                                "origin 2>&1 | tail -1", timeout=900)
            code, out = run(f"cd '{src_root}' && git rev-parse --verify "
                            f"'{target_tag}^{{commit}}' 2>&1 | tail -1")
            if code != 0:
                return {"ok": False, "machine": name,
                        "detail": f"release tag {target_tag} not found "
                                  f"upstream: {(out or '')[-200:]}"}
            code, out = run(f"cd '{src_root}' && git checkout --detach "
                            f"'{target_tag}' 2>&1 | tail -1")
        else:
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
            code, out = run(f"cd '{src_root}' && git checkout origin/master "
                            "2>&1 | tail -1")
        if code != 0:
            return {"ok": False, "machine": name,
                    "detail": f"git checkout failed: {(out or '')[-300:]}"}
        fl = probe.get("flags") or {}
        kept = ", ".join(f"{k}={v}" for k, v in sorted(fl.items())[:6]) or "cache defaults"
        log(f"{name}: checked out {target_tag or 'origin/master'} — rebuilding "
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

        # A stale CMAKE_COMMAND in the cache (snap/venv path deleted)
        # makes gmake fail before compiling. Reconfigure with the cmake on
        # PATH; cached CMake options are preserved.
        stale_cmake = ""
        try:
            with open(os.path.join(build_dir, "CMakeCache.txt"),
                      encoding="utf-8", errors="replace") as _cf:
                for _line in _cf:
                    if _line.startswith("CMAKE_COMMAND:"):
                        stale_cmake = _line.split("=", 1)[-1].strip()
                        break
        except OSError:
            pass
        if stale_cmake and not os.path.isfile(stale_cmake):
            log(f"{name}: CMake cache points at missing {stale_cmake} — "
                "reconfiguring with cmake on PATH")
            code, out = run(f"cmake -S '{src_root}' -B '{build_dir}' "
                            "2>&1 | tail -3", timeout=900)
            if code != 0:
                return {"ok": False, "machine": name,
                        "detail": f"cmake reconfigure failed: "
                                  f"{(out or '')[-300:]}"}
        bin_dir = probe.get("llama_bin") or (build_dir + "/bin")
        bin_path = f"{bin_dir}/llama-server"
        rpc_path = f"{bin_dir}/ggml-rpc-server"
        # Force the link step. Updating libs without relinking the server
        # is exactly what produced aibox3's mixed-version SIGSEGV.
        run(f"rm -f '{bin_path}' '{bin_dir}/libllama-server-impl.so' "
            f"'{rpc_path}'")
        _tgt = " ".join(SOURCE_BUILD_TARGETS)
        build_cmd = (
            f"cd '{src_root}' && "
            f"cmake --build '{build_dir}' --target {_tgt} -j {jobs} "
            f"> /tmp/llamacpp-build.log 2>&1; _ec=$?; "
            "if [ $_ec -ne 0 ]; then "
            f"cmake --build '{build_dir}' --target llama-server "
            f"ggml-rpc-server -j {jobs} >> /tmp/llamacpp-build.log 2>&1; "
            "_ec=$?; fi; tail -3 /tmp/llamacpp-build.log; exit $_ec")
        code, out = run(build_cmd, timeout=7200)
        if code != 0:
            return {"ok": False, "machine": name,
                    "detail": f"build failed (exit {code}): "
                              f"{(out or '')[-400:]}"}

        code, out = run(f"'{bin_path}' --version 2>&1 | head -1")
        version = (out or "").strip()
        if code != 0 or not version:
            return {"ok": False, "machine": name,
                    "detail": f"built llama-server does not run "
                              f"(exit {code}); mixed/incomplete libs?"}
        if target_tag and not _version_matches_tag(version, target_tag):
            return {"ok": False, "machine": name,
                    "detail": f"built {version}, expected {target_tag}"}
        code, _out = run(f"test -x '{rpc_path}'")
        if code != 0:
            return {"ok": False, "machine": name,
                    "detail": "built llama-server but ggml-rpc-server is "
                              "missing (GGML_RPC off or target failed) — "
                              "RPC peers could not start"}
        code, out = run(f"test -x '{rpc_path}' && echo RPC || echo NORPC")
        if "RPC" not in (out or ""):
            return {"ok": False, "machine": name,
                    "detail": "source build did not produce ggml-rpc-server "
                              "(enable GGML_RPC=ON in the build cache)"}
        code, devs = run(f"'{bin_path}' --list-devices 2>&1 | head -8")
        try:
            import config_loader as _cfg
            _cfg.machines.update(name, {"llama_bin": bin_dir})
        except Exception:
            pass
        log(f"{name} now: {version} (llama-server + ggml-rpc-server)")
        log(f"{name} devices: " + " | ".join((devs or "").splitlines()[:6]))
        return {"ok": True, "machine": name, "bin": bin_path,
                "detail": f"source build updated to {version} "
                          f"(llama-server + ggml-rpc-server)",
                "devices": devs or ""}
    except Exception as e:
        return {"ok": False, "machine": name,
                "detail": f"source update failed: {e}"}
