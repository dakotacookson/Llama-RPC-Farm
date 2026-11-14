"""Board/platform identity and GPU-usability rules shared by the agent.

Kept dependency-free on purpose: the peer agent ships as a universal wheel
and must run on old distros (Ubuntu 18.04 / Jetson) as well as current ones.
"""
import os
import re
import shutil
import subprocess
import sys

BOARD_MODEL_PATHS = ("/proc/device-tree/model",
                     "/sys/firmware/devicetree/base/model")


def board_model() -> str:
    """SBC/board model from the device tree (Raspberry Pi, Jetson, ...)."""
    if not sys.platform.startswith("linux"):
        return ""
    for path in BOARD_MODEL_PATHS:
        try:
            with open(path, "rb") as f:
                return f.read().replace(b"\x00", b"").decode(
                    "utf-8", "replace").strip()
        except OSError:
            continue
    return ""


def cuda_version():
    """(major, minor) of the installed CUDA toolkit, or None.

    Checks nvcc first (Jetson puts it in /usr/local/cuda/bin, often not on
    PATH), then the version file. No CUDA -> None.
    """
    candidates = []
    which = shutil.which("nvcc")
    if which:
        candidates.append(which)
    candidates.append("/usr/local/cuda/bin/nvcc")
    for nvcc in candidates:
        if not os.path.isfile(nvcc):
            continue
        try:
            proc = subprocess.run([nvcc, "--version"], capture_output=True,
                                  text=True, timeout=10)
        except (OSError, subprocess.SubprocessError):
            continue
        m = re.search(r"release (\d+)\.(\d+)", (proc.stdout or "") + (proc.stderr or ""))
        if m:
            return int(m.group(1)), int(m.group(2))
    for path in ("/usr/local/cuda/version.txt",):
        try:
            with open(path, encoding="utf-8", errors="replace") as f:
                m = re.search(r"CUDA Version (\d+)\.(\d+)", f.read())
            if m:
                return int(m.group(1)), int(m.group(2))
        except OSError:
            pass
    return None


def tegra_gpu_unsupported_reason() -> str:
    """Non-empty when this Tegra GPU cannot be used by current llama.cpp.

    Legacy Jetson boards (Nano/TX1/TX2 and Xavier on JetPack 4) ship CUDA
    10.2 / L4T R32. Current llama.cpp CUDA needs CUDA >= 11 (cuda_bf16.h)
    and C++17 nvcc support, so the GPU is unusable there. Orin/JetPack 5+
    reports CUDA >= 11 and is left alone.
    """
    board = board_model().lower()
    tegra = ("jetson" in board or "tegra" in board)
    if not tegra:
        try:
            tegra = os.path.exists("/etc/nv_tegra_release")
        except OSError:
            tegra = False
    if not tegra:
        return ""
    cuda = cuda_version()
    if cuda is None:
        return ("Jetson/Tegra: no CUDA toolkit found — CPU-only "
                "(install JetPack CUDA to change this)")
    if cuda[0] < 11:
        return (f"Jetson/Tegra GPU unsupported: CUDA {cuda[0]}.{cuda[1]} "
                f"is too old for current llama.cpp (needs CUDA 11+) — "
                f"CPU-only")
    return ""
