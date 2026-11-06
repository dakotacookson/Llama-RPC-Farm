"""Typed llama.cpp source probe/update."""
import os
import re
import subprocess
from pathlib import Path

from . import files
from .config import AgentConfig

_FLAG_RE = re.compile(
    r"^(GGML_CUDA:BOOL|GGML_HIP:BOOL|GGML_VULKAN:BOOL|GGML_RPC:BOOL|"
    r"GGML_CUDA_FA:BOOL|GGML_CUDA_NCCL:BOOL|CMAKE_CUDA_ARCHITECTURES:|"
    r"CMAKE_CUDA_COMPILER:FILEPATH|GGML_CUDA_COMPRESSION_MODE:STRING)")


class AgentBuildError(RuntimeError):
    pass


def _run(args, timeout=120, cwd=None):
    try:
        return subprocess.run(args, capture_output=True, text=True,
                              timeout=timeout, cwd=cwd)
    except (OSError, subprocess.SubprocessError) as e:
        raise AgentBuildError(str(e)) from e


def probe(config: AgentConfig, llama_bin) -> dict:
    lb = str(llama_bin or "").strip()
    if not lb:
        return {"kind": "missing", "llama_bin": None}
    p = files._path(config, lb)
    build = p.parent.parent if p.parent.name == "bin" else p.parent
    out = {"kind": "prebuilt", "llama_bin": lb,
           "build_dir": None, "src_root": None, "flags": {}}
    if not build.is_dir():
        out["kind"] = "missing"
        return out
    out["build_dir"] = str(build)
    if (build / "bin" / "llama-server").is_file():
        out["llama_bin"] = str(build / "bin")
    if not (build / "CMakeCache.txt").is_file():
        return out
    proc = _run(["git", "-C", str(build), "rev-parse", "--show-toplevel"],
                timeout=20)
    root = (proc.stdout or "").strip()
    if proc.returncode != 0 or not root.startswith("/"):
        return out
    out["src_root"] = root
    out["kind"] = "source"
    try:
        for line in (build / "CMakeCache.txt").read_text(
                encoding="utf-8", errors="replace").splitlines():
            if not _FLAG_RE.match(line) or "=" not in line:
                continue
            k, _, v = line.partition(":")
            v = v.split("=", 1)[1].strip() if "=" in v else v.strip()
            out["flags"][k.strip()] = v
    except OSError:
        pass
    return out


def update(config: AgentConfig, src_root, build_dir, bin_path,
           jobs=None) -> dict:
    src = files._path(config, src_root)
    build = files._path(config, build_dir)
    binp = files._path(config, bin_path)
    if not src.is_dir() or not build.is_dir():
        raise AgentBuildError("source or build dir not found")
    if not (build / "CMakeCache.txt").is_file():
        raise AgentBuildError("build dir has no CMakeCache.txt")
    proc = _run(["git", "-C", str(src), "fetch", "origin", "master"],
                timeout=600)
    if proc.returncode != 0:
        raise AgentBuildError("git fetch failed: " + (proc.stderr or "")[-300:])
    proc = _run(["git", "-C", str(src), "merge-base", "--is-ancestor",
                 "HEAD", "origin/master"], timeout=60)
    if proc.returncode != 0:
        raise AgentBuildError("source tree has local-only commits")
    proc = _run(["git", "-C", str(src), "status", "--porcelain"], timeout=60)
    dirty = [ln for ln in (proc.stdout or "").splitlines()
             if ln and not ln.startswith("??")]
    if dirty:
        raise AgentBuildError("tracked files modified: " + "; ".join(dirty[:5]))
    proc = _run(["git", "-C", str(src), "checkout", "origin/master"],
                timeout=120)
    if proc.returncode != 0:
        raise AgentBuildError("git checkout failed: " + (proc.stderr or "")[-300:])
    if jobs is None:
        try:
            mem = 0
            with open("/proc/meminfo", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("MemAvailable:"):
                        mem = int(line.split()[1]) // 1024
                        break
            jobs = max(1, min(os.cpu_count() or 2, mem // 2048))
        except (OSError, ValueError):
            jobs = max(1, os.cpu_count() or 2)
    proc = _run(["cmake", "--build", str(build), "-j", str(int(jobs))],
                timeout=7200)
    output = ((proc.stdout or "") + (proc.stderr or ""))[-1000:]
    if proc.returncode != 0:
        return {"ok": False, "error": f"build failed: {output}"}
    v = _run([str(binp), "--version"], timeout=120)
    version = ((v.stdout or "") + (v.stderr or "")).strip().splitlines()
    return {"ok": True, "version": version[0] if version else "",
            "output": output}
