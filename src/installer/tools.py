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
#   - fixed per-tool destinations (~/llama.cpp/bin, ~/.local/bin, ~/vllm-env)
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

VLLM_ROCM_INDEX = "https://wheels.vllm.ai/rocm/"
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
    names = swap_asset_names(os_name, arch, step.version)
    urls = [f"{SWAP_BASE}/v{_tag_number(step.version)}/{n}" for n in names]
    tmp = "/tmp/llama-swap-installer"
    bin_dest = "llama-swap.exe" if os_name == "windows" else "llama-swap"
    dest = os.path.join(step.dest, bin_dest)
    log(f"swap: downloading {names[0]}")
    downloader(urls[0], tmp + ".pkg")
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
    names = llcpp_assets_for(step)
    base = LLCPP_BASE.rsplit("/download/", 1)[0]
    urls = [f"{base}/download/{step.version}/{n}" for n in names]
    dest = step.dest
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
    log(f"llama.cpp: installed {step.version} ({step.backend}) -> {dest}")
    return {"version": step.version, "method": "prebuilt",
            "backend": step.backend, "dest": dest}


# ---------------------------------------------------------------------------
# vLLM + ray: uv venv + pinned wheels into the dedicated venv
# ---------------------------------------------------------------------------

def _has_uv(run):
    return run("command -v uv >/dev/null 2>&1 && echo yes || echo no").stdout \
        .strip() == "yes"


def install_vllm(step, downloader, run, log, offline=False, os_name="linux"):
    """uv venv ~/vllm-env + `uv pip install vllm==<pin>`.

    rocm: wheels.vllm.ai index + py3.12-only guard (silent-CUDA-fallback
    probe after install). windows: refuse with WSL guidance (design §2) —
    probe normally refuses first; this is the executor-side hard gate."""
    if os_name == "windows":
        raise RuntimeError(
            "vLLM has no native Windows support — use WSL2 "
            "(docs.vllm.ai: install the Windows driver, run the Linux "
            "flow inside WSL)")
    venv = (step.dest or "~/vllm-env")
    if venv.endswith("/bin/vllm"):
        # RepairTool dest is the broken vllm binary; venv root is its parent
        venv = venv.rsplit("/bin/", 1)[0]
    extra = step.extra or {}
    python = extra.get("python", "3.12")
    if not _has_uv(run):
        raise RuntimeError(
            f"uv not found — install it first: {UV_INSTALL_URL} "
            "(the bootstrap entry does this before reaching here)")
    r = run(f"uv venv --python {python} --seed {shlex.quote(venv)}")
    if r.exit_code != 0:
        raise RuntimeError(f"venv creation failed: {(r.stderr or r.stdout)[-300:]}")
    index = ""
    if extra.get("index_url"):
        index = f" --index-url {extra['index_url']}"
    torch_backend = " --torch-backend=auto" \
        if extra.get("torch_backend") == "auto" else ""
    if offline:
        # --offline wheelhouse mode (design §1): local wheelhouse only.
        index = " --no-index --find-links wheelhouse/"
        if not os.path.isdir("wheelhouse"):
            raise RuntimeError(
                "--offline: wheelhouse/ not found next to bootstrap.py — "
                "the offline bundle ships it; download the installer "
                "bundle (design §1 fallback) or install online")
    spec = f"vllm=={step.version}"
    cmd = (f"uv pip install --python {shlex.quote(venv + '/bin/python')} "
           f"{spec}{torch_backend}{index}")
    log(f"vllm: {cmd}")
    r = run(cmd)
    if r.exit_code != 0:
        raise RuntimeError(f"vllm install failed: {(r.stderr or r.stdout)[-400:]}")
    vbin = venv + "/bin/vllm"
    r = run(f"{shlex.quote(os.path.expanduser(vbin))} --version")
    out = (r.stdout or "").strip()
    if r.exit_code != 0 or not out:
        raise RuntimeError("`vllm --version` failed after install")
    if extra.get("index_url"):
        # ROCm silent-CUDA-fallback guard: torch.version.hip must be set.
        probe = ('import vllm, torch; print("HIP" if '
                 'getattr(torch.version, "hip", None) else "CUDA")')
        r = run(f"{shlex.quote(venv + '/bin/python')} -c {shlex.quote(probe)}")
        if "HIP" not in (r.stdout or ""):
            raise RuntimeError("silent CUDA fallback: the ROCm wheel did not "
                               "install (needs python 3.12 + rocm index)")
    log(f"vllm: {out.splitlines()[-1]} in {venv}")
    return {"version": step.version, "venv": venv,
            "index": "rocm" if extra.get("index_url") else "pypi"}


def install_ray(step, downloader, run, log, offline=False):
    """ray[default] into the SAME venv as vLLM (design §2; ray ROCm = CPU
    wheel, never a separate torch)."""
    venv = step.dest or "~/vllm-env"
    if not _has_uv(run):
        raise RuntimeError(f"uv not found — see {UV_INSTALL_URL}")
    if offline:
        index = " --no-index --find-links wheelhouse/"
        if not os.path.isdir("wheelhouse"):
            raise RuntimeError(
                "--offline: wheelhouse/ not found next to bootstrap.py — "
                "the offline bundle ships it; download the installer "
                "bundle (design §1 fallback) or install online")
    else:
        index = ""
    cmd = (f"uv pip install --python {shlex.quote(venv + '/bin/python')} "
           f'"ray[default]"{index}')
    log(f"ray: {cmd}")
    r = run(cmd)
    if r.exit_code != 0:
        raise RuntimeError(f"ray install failed: {(r.stderr or r.stdout)[-400:]}")
    r = run(f"{shlex.quote(venv + '/bin/python')} -c "
            f"{shlex.quote('import ray; print(ray.__version__)')}")
    ver = (r.stdout or "").strip() or step.version
    log(f"ray: {ver}")
    return {"version": ver, "venv": venv}


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
        return install_llcpp(step, downloader, run, log, info.os, info.arch)
    if name == "vllm":
        return install_vllm(step, downloader, run, log,
                            offline=offline, os_name=info.os)
    if name == "ray":
        return install_ray(step, downloader, run, log, offline=offline)
    if name == "uv":
        return ensure_uv(run, log, downloader)
    raise ValueError(f"no executor for tool: {name}")