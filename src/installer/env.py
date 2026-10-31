# env.py — injected environment: SystemInfo value object + interfaces + real impls.
#
# probe.py consumes ONLY this. Everything the real world can answer is captured
# here: OS/arch/GPU facts, run(cmd) results, filesystem existence/version checks.
# Hermetic tests swap the two protocols for stubs (pyfakefs for FS, dict-backed
# stub for RunCmd) — see tests/installer/.

from __future__ import annotations

import os
import platform
import subprocess
import shlex
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Protocol, Tuple


@dataclass
class SystemInfo:
    """Static facts about the target machine (design §2/§4 detection output)."""
    os: str                      # 'linux' | 'macos' | 'windows'
    arch: str                    # 'x86_64' | 'arm64'
    gpu_vendor: str              # 'nvidia' | 'amd' | 'apple' | 'intel' | 'none'
    glibc: Optional[Tuple[int, int]] = None   # (major, minor) or None on windows
    driver_version: Optional[str] = None      # NVIDIA driver as reported
    vram_gb: List[int] = field(default_factory=list)
    wsl: bool = False            # Linux side of WSL2


@dataclass
class RunResult:
    """Outcome of one captured command probe."""
    cmd: str
    exit_code: int
    stdout: str
    stderr: str

    @property
    def ok(self) -> bool:
        return self.exit_code == 0


class RunCmd(Protocol):
    """Command-probe interface. prod: RealRunCmd. tests: dict-backed stub."""
    def run(self, cmd: str, timeout: float = 30.0) -> RunResult: ...


class FS(Protocol):
    """Filesystem interface (existence + binary smoke). prod: RealFS."""
    def exists(self, path: str) -> bool: ...
    def isfile(self, path: str) -> bool: ...
    def expanduser(self, path: str) -> str: ...
    def read_file(self, path: str) -> str: ...
    def file_exists_and_runs(self, binary: str, version_arg: str = "--version") -> "RunResult":
        """Smoke probe: does the binary exist AND answer --version cleanly.
        Returns a RunResult; exit_code != 0 or empty stdout => broken/partial."""
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Real implementations (prod path; never imported by probe.py)
# ---------------------------------------------------------------------------

class RealRunCmd:
    def run(self, cmd: str, timeout: float = 30.0) -> RunResult:
        try:
            p = subprocess.run(
                cmd, shell=isinstance(cmd, str),
                capture_output=True, text=True, timeout=timeout,
            )
            return RunResult(str(cmd), p.returncode, p.stdout or "", p.stderr or "")
        except FileNotFoundError:
            return RunResult(str(cmd), 127, "", "not found")
        except subprocess.TimeoutExpired:
            return RunResult(str(cmd), 124, "", "timeout")


class RealFS:
    def exists(self, path: str) -> bool:
        return os.path.exists(self.expanduser(path))

    def isfile(self, path: str) -> bool:
        return os.path.isfile(self.expanduser(path))

    def expanduser(self, path: str) -> str:
        return os.path.expanduser(os.path.expandvars(path))

    def read_file(self, path: str) -> str:
        with open(self.expanduser(path), "r", encoding="utf-8") as f:
            return f.read()

    def file_exists_and_runs(self, binary, version_arg="--version") -> RunResult:
        b = self.expanduser(binary)
        if not os.path.isfile(b):
            return RunResult(binary, 127, "", "missing")
        p = subprocess.run(shlex.split(f"{shlex.quote(b)} {version_arg}"),
                           capture_output=True, text=True, timeout=30)
        return RunResult(f"{binary} {version_arg}", p.returncode,
                         p.stdout or "", p.stderr or "")


# Convenience: the platform's real environment snapshot.
def detect_system_info(run: RunCmd) -> SystemInfo:
    """Populate SystemInfo via the design §4 probe ladder (cheap, read-only)."""
    osname = platform.system().lower()          # 'linux' | 'darwin' | 'windows'
    osname = {"darwin": "macos"}.get(osname, osname)
    machine = platform.machine().lower()
    arch = "arm64" if machine in ("arm64", "aarch64") else "x86_64"

    glibc = None
    if osname == "linux":
        r = run.run("ldd --version | head -1")
        if r.ok and r.stdout:
            try:
                ver = r.stdout.split()[-1]
                major, minor = ver.split(".")[:2]
                glibc = (int(major), int(minor))
            except (ValueError, IndexError):
                pass

    vendor, driver, vram = "none", None, []
    r = run.run("nvidia-smi --query-gpu=name,memory.total,driver_version "
                "--format=csv,noheader")
    if r.ok and r.stdout.strip():
        vendor = "nvidia"
        first = r.stdout.strip().splitlines()[0]
        parts = [p.strip() for p in first.split(",")]
        if len(parts) >= 3:
            driver = parts[2]
            try:
                vram = [int(parts[1].split()[0])]
            except (ValueError, IndexError):
                vram = []
    else:
        r = run.run("rocm-smi --showproductname")
        if r.ok and r.stdout.strip():
            vendor = "amd"
        elif osname == "macos":
            vendor = "apple"
        elif osname == "linux":
            r = run.run("vulkaninfo --summary")
            if r.ok:
                vendor = "intel"

    wsl = False
    if osname == "linux" and os.path.exists("/usr/lib/wsl/lib/nvidia-smi"):
        wsl = True

    return SystemInfo(os=osname, arch=arch, gpu_vendor=vendor,
                      glibc=glibc, driver_version=driver, vram_gb=vram, wsl=wsl)