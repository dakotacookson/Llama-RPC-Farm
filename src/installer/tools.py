# tools.py — per-tool executors for the installer execution layer.
#
# Each function installs ONE tool from a plan step (InstallTool/RepairTool).
# They are deliberately dumb: the DECISION (skip/upgrade/repair/refuse) was
# already made by probe.py; a tool executor only knows HOW.
#
# Injections keep this hermetic:
#   downloader(url, dest)  — writes the URL to dest (prod: HTTPS; tests: stub)
#   run(cmd) -> RunResult  — RunCmd interface (prod: shell; tests: recording stub)
#
# Convergence rules honored here (design §5):
#   - downloads land in a temp path first, then ATOMIC os.replace into place
#   - fixed per-tool destinations (~/llama.cpp/bin, ~/.local/bin)
#   - never installs GPU drivers (design §4 hard rule): executors touch only
#     the artifacts named in their plan step
#   - tag-NUMBER asset naming for llama-swap (v256 -> llama-swap_256_...),
#     llama.cpp asset names carry the bNNNNN tag verbatim (verified asset list)

from __future__ import annotations

import os
import re
import shlex

# ---------------------------------------------------------------------------
# Asset maps (design §2, mirrored from llamacpp_installer / swap_installer —
# the hard-won knowledge: glibc 2.38 CUDA/vulkan fallback, tag-NUMBER naming,
# cudart companions). Kept as data so tests can assert them hermetically.
# ---------------------------------------------------------------------------

SWAP_OS_MAP = {"linux": "linux", "macos": "darwin", "windows": "windows"}
SWAP_BASE = "https://github.com/mostlygeek/llama-swap/releases/download"

LLCPP_BASE = "https://github.com/ggml-org/llama.cpp/releases/download"
# per llama.cpp asset naming: backend key -> (asset template, needs cudart)
LLCPP_ASSETS = {
    "cuda-13.3": ("llama-{tag}-bin-ubuntu-cuda-13.3-{arch}.tar.gz",
                  "cudart-llama-{tag}-bin-ubuntu-cuda-13.3-{arch}.tar.gz"),
    "rocm-10.0": ("llama-{tag}-bin-ubuntu-rocm-10.0-{arch}.tar.gz", None),
    "cpu":       ("llama-{tag}-bin-ubuntu-{arch}.tar.gz", None),
    "vulkan":    ("llama-{tag}-bin-ubuntu-vulkan-{arch}.tar.gz", None),
    "sycl":      ("llama-{tag}-bin-ubuntu-sycl-fp16-{arch}.tar.gz", None),
    "metal":     ("llama-{tag}-bin-macos-{arch}.tar.gz", None),
    "win-cuda":  ("llama-{tag}-bin-win-cuda-13.3-{arch}.zip",
                  "cudart-llama-bin-win-cuda-13.3-{arch}.zip"),
    "win-cpu":   ("llama-{tag}-bin-win-cpu-{arch}.zip", None),
}

UV_INSTALL_URL = "https://astral.sh/uv/install.sh"   # versioned fetch upstream


def _arch_kind(arch):
    return "arm64" if arch in ("arm64", "aarch64") else "x64"


def _swap_arch(arch):
    return "arm64" if arch in ("arm64", "aarch64") else "amd64"


def _tag_number(tag):
    """llama-swap releases carry the tag NUMBER in asset names (v255 -> 255)."""
    return (tag or "").lstrip("v")


def swap_asset_names(os_name, arch, version):
    """Tag-NUMBER asset naming per design §2 (verified against v255/v256)."""
    kind = _swap_arch(arch)
    ext = ".zip" if os_name == "windows" else ".tar.gz"
    num = _tag_number(version)
    return [f"llama-swap_{num}_{SWAP_OS_MAP[os_name]}_{kind}{ext}"]


def llcpp_asset_names(backend, arch, version):
    """Asset(s) for a llama.cpp backend: main tarball + cudart companion.
    (glibc<2.38 vulkan fallback is applied by llcpp_assets_for, which has
    the SystemInfo glibc hint; verified asset names per design §2.)"""
    main, cudart = LLCPP_ASSETS.get(backend, LLCPP_ASSETS["cpu"])
    arch_k = _arch_kind(arch)
    names = [main.format(tag=version, arch=arch_k)]
    if cudart:
        names.append(cudart.format(tag=version, arch=arch_k))
    return names


def llcpp_assets_for(step, info=None):
    """Resolve the llama.cpp asset list for a plan step, honoring the
    vulkan fallback when the step carries the glibc hint in extra."""
    backend = step.backend or "cpu"
    glibc = (step.extra or {}).get("glibc") or (info.glibc if info else None)
    if backend == "cuda-13.3" and glibc and tuple(glibc) < (2, 38):
        backend = "vulkan"
    return llcpp_asset_names(backend, (step.extra or {}).get("arch", "x86_64"),
                             step.version)


# ---------------------------------------------------------------------------
# llama-swap: single Go binary -> temp -> atomic rename into ~/.local/bin
# ---------------------------------------------------------------------------

def install_swap(step, downloader, run, log, os_name, arch):
    extra = step.extra or {}
    explicit = list(extra.get("assets") or [])
    if explicit:
        names = explicit
        base = (extra.get("base") or "").rstrip("/")
        url = (f"{base}/{names[0]}" if base
               else f"{SWAP_BASE}/v{_tag_number(step.version)}/{names[0]}")
    else:
        names = swap_asset_names(os_name, arch, step.version)
        url = f"{SWAP_BASE}/v{_tag_number(step.version)}/{names[0]}"
    tmp = "/tmp/llama-swap-installer"
    bin_dest = "llama-swap.exe" if os_name == "windows" else "llama-swap"
    dest = os.path.join(step.dest, bin_dest)
    log(f"swap: downloading {names[0]}")
    downloader(url, tmp + ".pkg")
    if os_name == "windows":
        r = run(f'powershell -NoProfile -Command "Expand-Archive -Force '
                f"'{tmp}.pkg' '{tmp}'\"")
        code = r.exit_code
        src = os.path.join(tmp, "llama-swap.exe")
    else:
        r = run(f"mkdir -p {shlex.quote(tmp)} && "
                f"tar -xzf {shlex.quote(tmp + '.pkg')} -C {shlex.quote(tmp)} "
                f"llama-swap")
        code = r.exit_code
        src = os.path.join(tmp, "llama-swap")
    if code != 0:
        raise RuntimeError(f"llama-swap tarball extraction failed (exit {code})")
    # atomic convergence: temp file + replace (design §5)
    dest = os.path.expanduser(dest)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    os.replace(os.path.expanduser(src), dest)
    run(f"chmod +x {shlex.quote(dest)}")
    log(f"swap: installed {step.version} -> {dest}")
    return {"version": step.version, "method": "prebuilt", "dest": dest}


# ---------------------------------------------------------------------------
# llama.cpp: prebuilt tarball(s) + cudart companion -> fixed dest
# ---------------------------------------------------------------------------

def install_llcpp(step, downloader, run, log, os_name, arch):
    extra = step.extra or {}
    explicit = list(extra.get("assets") or [])
    if explicit:
        names = explicit
        base = (extra.get("base") or "").rstrip("/")
        urls = ([f"{base}/{n}" for n in names] if base
                else [f"{LLCPP_BASE.rsplit('/download/', 1)[0]}"
                      f"/download/{step.version}/{n}" for n in names])
    else:
        names = llcpp_assets_for(step)
        base = LLCPP_BASE.rsplit("/download/", 1)[0]
        urls = [f"{base}/download/{step.version}/{n}" for n in names]
    # Bootstrap runs on the target host: expand ~ (POSIX) / %VAR% (Windows)
    # before any shell command. Tilde inside quotes never expands.
    dest = os.path.expanduser(os.path.expandvars(step.dest))
    log(f"llama.cpp: {', '.join(names)} (backend={step.backend})")
    for i, (name, url) in enumerate(zip(names, urls)):
        pkg = f"/tmp/llamabin{i}.pkg"
        downloader(url, pkg)
        if os_name == "windows":
            r = run(f"powershell -NoProfile -Command \"Expand-Archive -Force "
                    f"'{pkg}' '{dest}'\"")
        else:
            # POSIX prebuilts nest in llama-<tag>/ -> --strip-components=1
            # lands binaries flat in dest (app expects bin/llama-server).
            r = run(f"mkdir -p {shlex.quote(dest)} && "
                    f"tar -xzf {shlex.quote(pkg)} -C {shlex.quote(dest)} "
                    f"--strip-components=1")
        if r.exit_code != 0:
            raise RuntimeError(f"llama.cpp extraction failed for {name}: "
                               f"{(r.stderr or r.stdout)[-300:]}")
        run(f"rm -f {shlex.quote(pkg)}")
    run(f"chmod +x {shlex.quote(dest)}/* 2>/dev/null; true")

    # Both binaries are part of an install: llama-server serves models and
    # ggml-rpc-server is what peers expose to the cluster. A partial extract
    # used to "succeed" and only fail on first launch.
    server = os.path.join(dest, "llama-server.exe" if os_name == "windows"
                          else "llama-server")
    rpc = os.path.join(dest, "ggml-rpc-server.exe" if os_name == "windows"
                       else "ggml-rpc-server")
    if os_name == "windows":
        r = run(f'if exist "{server}" (exit 0) else (exit 1)')
        if r.exit_code != 0:
            raise RuntimeError(f"llama.cpp install incomplete: {server} missing")
        r = run(f'if exist "{rpc}" (exit 0) else (exit 1)')
        if r.exit_code != 0:
            raise RuntimeError(f"llama.cpp install incomplete: {rpc} missing")
    else:
        r = run(f'test -x "{server}" && test -x "{rpc}"')
        if r.exit_code != 0:
            raise RuntimeError(
                f"llama.cpp install incomplete: need {server} and {rpc}")
    r = run(f'"{server}" --version 2>&1')
    if r.exit_code != 0:
        raise RuntimeError(f"llama-server --version failed: "
                           f"{(r.stdout or r.stderr or '')[-300:]}")
    if os_name == "windows":
        try:
            from llamacpp_installer import _ensure_win_firewall

            def fw_run(cmd, timeout=30):
                rr = run(cmd, timeout)
                return rr.exit_code, (rr.stdout or "") + (rr.stderr or "")
            _ensure_win_firewall(fw_run, log)
        except Exception as e:
            log(f"llama.cpp: firewall rules not updated: {e}")
    log(f"llama.cpp: installed {step.version} ({step.backend}) -> {dest}")
    return {"version": step.version, "method": "prebuilt",
            "backend": step.backend, "dest": dest}


def install_llcpp_source(step, run, log):
    """Rebuild an existing llama.cpp source checkout in place.

    Preserves the CMakeCache (backend, CUDA arch list, GGML_RPC/FA/NCCL,
    compiler) and builds BOTH llama-server and ggml-rpc-server. Refuses on
    local-only commits or dirty tracked files instead of silently changing
    the build configuration.
    """
    extra = step.extra or {}
    src = os.path.expanduser(os.path.expandvars(extra.get("src_root") or ""))
    build = os.path.expanduser(os.path.expandvars(extra.get("build_dir") or ""))
    dest = os.path.expanduser(os.path.expandvars(step.dest))
    if not src or not build:
        raise RuntimeError("source step missing src_root/build_dir")

    def pair(cmd, timeout=60):
        r = run(cmd, timeout)
        return r.exit_code, (r.stdout or "") + (r.stderr or "")

    code, out = pair(f"cd {shlex.quote(src)} && "
                     "git fetch origin master 2>&1 | tail -1", 300)
    if code != 0:
        raise RuntimeError(f"git fetch failed: {out[-300:]}")
    code, out = pair(f"cd {shlex.quote(src)} && "
                     "(git merge-base --is-ancestor HEAD origin/master "
                     "&& echo CLEAN-ANCESTOR || echo LOCAL-ONLY)")
    if "CLEAN-ANCESTOR" not in out:
        raise RuntimeError("source tree has local-only commits — "
                           "update refused, merge manually")
    code, out = pair(f"cd {shlex.quote(src)} && git status --porcelain "
                     "| grep -v '^??' | head -5")
    if out.strip():
        raise RuntimeError(f"tracked files modified: {out.strip()[:200]}")
    code, out = pair(f"cd {shlex.quote(src)} && "
                     "git checkout origin/master 2>&1 | tail -1", 300)
    if code != 0:
        raise RuntimeError(f"git checkout failed: {out[-300:]}")

    code, out = pair("nproc; free -m | awk '/^Mem:/{print $7}'")
    parts = out.split()
    try:
        cores = int(parts[0])
    except (ValueError, IndexError):
        cores = 2
    try:
        ram_mb = int(parts[1])
    except (ValueError, IndexError):
        ram_mb = 4096
    jobs = max(1, min(cores, ram_mb // 2048))
    log(f"llama.cpp source: rebuilding {build} with -j{jobs} "
        f"(CMakeCache preserved)")

    code, out = pair(
        f"cd {shlex.quote(src)} && cmake --build {shlex.quote(build)} "
        f"-j {jobs} > /tmp/llamacpp-build.log 2>&1; "
        "_ec=$?; tail -3 /tmp/llamacpp-build.log; exit $_ec", 7200)
    if code != 0:
        raise RuntimeError(f"source build failed (exit {code}): {out[-400:]}")

    server = os.path.join(dest, "llama-server.exe" if os.name == "nt"
                          else "llama-server")
    rpc = os.path.join(dest, "ggml-rpc-server.exe" if os.name == "nt"
                       else "ggml-rpc-server")
    if os.name == "nt":
        ok = run(f'if exist "{server}" (exit 0) else (exit 1)').exit_code == 0
        ok_rpc = run(f'if exist "{rpc}" (exit 0) else (exit 1)').exit_code == 0
    else:
        ok = run(f'test -x "{server}"').exit_code == 0
        ok_rpc = run(f'test -x "{rpc}"').exit_code == 0
    if not ok or not ok_rpc:
        raise RuntimeError(
            "source build did not produce both binaries; enable "
            "GGML_RPC=ON in the build cache (need llama-server + "
            "ggml-rpc-server)")
    code, out = pair(f'"{server}" --version 2>&1 | head -1', 120)
    if code != 0:
        raise RuntimeError(f"llama-server --version failed: {out[-300:]}")
    version = out.strip()
    log(f"llama.cpp source: installed {version} -> {dest}")
    return {"version": version, "method": "source",
            "backend": step.backend, "dest": dest}


def _has_uv(run):
    return run("command -v uv >/dev/null 2>&1 && echo yes || echo no").stdout \
        .strip() == "yes"


def ensure_uv(run, log, downloader):
    """python/uv bootstrap per design §1: probe uv, report if missing.
    Installing uv itself is the ENTRY script's job (curl|sh / irm|iex);
    this only gates with a documented one-liner (never sudo)."""
    if _has_uv(run):
        log("uv: present")
        return {"present": True}
    log(f"uv: MISSING — run: curl -LsSf {UV_INSTALL_URL} | sh")
    return {"present": False}


# ---------------------------------------------------------------------------
# dispatcher
# ---------------------------------------------------------------------------

def install_tool(step, downloader, run, log, info, offline=False):
    """Dispatch an InstallTool/RepairTool step to its executor.
    Returns the installer-state manifest entry."""
    name = step.name
    if name == "llama-swap":
        return install_swap(step, downloader, run, log, info.os, info.arch)
    if name == "llama.cpp":
        if (step.source or "") == "source":
            return install_llcpp_source(step, run, log)
        return install_llcpp(step, downloader, run, log, info.os, info.arch)
    if name == "uv":
        return ensure_uv(run, log, downloader)
    raise ValueError(f"no executor for tool: {name}")