"""Typed llama.cpp source probe/update."""
import os
import platform
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

from . import files
from .config import AgentConfig

_FLAG_RE = re.compile(
    r"^(GGML_CUDA:BOOL|GGML_HIP:BOOL|GGML_VULKAN:BOOL|GGML_SYCL:BOOL|"
    r"GGML_METAL:BOOL|GGML_OPENCL:BOOL|GGML_RPC:BOOL|GGML_CUDA_FA:BOOL|"
    r"GGML_CUDA_NCCL:BOOL|GGML_BLAS:BOOL|GGML_NATIVE:BOOL|LLAMA_CURL:BOOL|"
    r"CMAKE_BUILD_TYPE:STRING|CMAKE_CUDA_ARCHITECTURES:|"
    r"CMAKE_CUDA_COMPILER:FILEPATH|GGML_CUDA_COMPRESSION_MODE:STRING)")


class AgentBuildError(RuntimeError):
    pass


def _run(args, timeout=120, cwd=None, env=None):
    try:
        return subprocess.run(args, capture_output=True, text=True,
                              timeout=timeout, cwd=cwd, env=env)
    except (OSError, subprocess.SubprocessError) as e:
        raise AgentBuildError(str(e)) from e


def _tool_major(binary):
    proc = _run([binary, "-dumpversion"], timeout=10)
    m = re.search(r"(\d+)", (proc.stdout or "") + (proc.stderr or ""))
    return int(m.group(1)) if m else 0


def _pick_compiler():
    """(cc, cxx, extra_cmake_args) for a C++17 <filesystem> compiler.

    Old distros (Ubuntu 18.04 / Jetson) ship g++ 7.5, which has no
    `<filesystem>`. Prefer g++-8..12 when the default is too old; g++ 8
    needs -lstdc++fs at link time."""
    if os.name == "nt":
        return "", "", []
    candidates = []
    for name in (os.environ.get("CXX"), "g++", "g++-8", "g++-9",
                 "g++-10", "g++-11", "g++-12"):
        if not name:
            continue
        path = shutil.which(name) or (name if os.path.isabs(name)
                                       and os.path.isfile(name) else None)
        if path and path not in candidates:
            candidates.append(path)
    if sys.platform == "darwin":
        for name in ("clang++", "c++"):
            p = shutil.which(name)
            if p:
                candidates.insert(0, p)
                break
    chosen = ""
    for cxx in candidates:
        if sys.platform == "darwin" or _tool_major(cxx) >= 8:
            chosen = cxx
            break
    if not chosen:
        raise AgentBuildError(
            "no C++17 <filesystem> compiler found (need g++ >= 8). "
            "On Ubuntu 18.04: sudo apt install g++-8")
    suffix = ""
    m = re.search(r"-(\d+(?:\.\d+)?)$", chosen)
    if m:
        suffix = "-" + m.group(1)
    cc = shutil.which("gcc" + suffix) or shutil.which("gcc") or ""
    if sys.platform == "darwin":
        cc = shutil.which("clang") or shutil.which("cc") or cc
    extra = []
    if sys.platform.startswith("linux") and _tool_major(chosen) == 8:
        extra.append("-DCMAKE_CXX_STANDARD_LIBRARIES=-lstdc++fs")
    if cc:
        extra.append(f"-DCMAKE_C_COMPILER={cc}")
    extra.append(f"-DCMAKE_CXX_COMPILER={chosen}")
    return cc, chosen, extra


_CMAKE_PORTABLE = "3.28.3"


def _cmake_ok(binary):
    proc = _run([binary, "--version"], timeout=10)
    m = re.search(r"(\d+)\.(\d+)", (proc.stdout or "") + (proc.stderr or ""))
    if not m:
        # Probe failed (mocked runner, odd wrapper): prefer the binary we
        # already have over attempting a network download.
        return True
    try:
        return (int(m.group(1)), int(m.group(2))) >= (3, 14)
    except (TypeError, ValueError):
        return True


def _ensure_cmake():
    """A cmake >= 3.14, downloading a portable build when needed.

    Ubuntu 18.04/Jetson ships cmake 3.10, which cannot configure current
    llama.cpp. Downloading Kitware's static-ish aarch64 build keeps the
    first install automatic without sudo."""
    system = shutil.which("cmake")
    if system and _cmake_ok(system):
        return system
    local = (Path.home() / ".llama-peer-agent" / ".tools" / "cmake"
             / "bin" / "cmake")
    if local.is_file() and _cmake_ok(str(local)):
        return str(local)
    arch = {"x86_64": "x86_64", "amd64": "x86_64", "aarch64": "aarch64",
            "arm64": "aarch64"}.get(platform.machine().lower())
    if not arch or os.name == "nt" or sys.platform == "darwin":
        raise AgentBuildError(
            "cmake >= 3.14 is required; install a newer CMake for this platform")
    dest = Path.home() / ".llama-peer-agent" / ".tools" / "cmake"
    dest.mkdir(parents=True, exist_ok=True)
    url = (f"https://github.com/Kitware/CMake/releases/download/"
           f"v{_CMAKE_PORTABLE}/cmake-{_CMAKE_PORTABLE}-linux-{arch}.tar.gz")
    tmp = Path(tempfile.mkdtemp(prefix="farm-cmake-"))
    try:
        tgz = tmp / "cmake.tar.gz"
        with urllib.request.urlopen(url, timeout=180) as r, open(tgz, "wb") as f:
            shutil.copyfileobj(r, f)
        with tarfile.open(tgz) as tf:
            members = []
            for m in tf.getmembers():
                parts = m.name.split("/", 1)
                if len(parts) == 2:
                    m.name = parts[1]
                    members.append(m)
            try:
                tf.extractall(dest, members=members, filter="data")
            except TypeError:
                tf.extractall(dest, members=members)
        local = dest / "bin" / "cmake"
        os.chmod(local, 0o755)
        if not _cmake_ok(str(local)):
            raise AgentBuildError("downloaded cmake is not runnable")
        return str(local)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _bin_meta(llama_bin):
    """(version line, ggml-rpc-server present?) for a llama.cpp bin dir.

    Accepts a directory OR the llama-server executable itself, expands
    ~, and uses the platform suffix — a bare "llama-server" path made
    Windows probes return an empty version and rpc=False."""
    base = Path(os.path.expanduser(str(llama_bin or "")))
    exe = ".exe" if os.name == "nt" else ""
    if base.name.lower() in ("llama-server", "llama-server.exe"):
        base = base.parent  # caller passed the executable, not the dir
    server = base / f"llama-server{exe}"
    rpc = base / f"ggml-rpc-server{exe}"
    version = ""
    if server.is_file():
        v = _run([str(server), "--version"], timeout=60)
        lines = ((v.stdout or "") + (v.stderr or "")).strip().splitlines()
        version = lines[0] if lines else ""
    return version, rpc.is_file()


def probe(config: AgentConfig, llama_bin) -> dict:
    lb = str(llama_bin or "").strip()
    if not lb:
        return {"kind": "missing", "llama_bin": None, "version": "",
                "rpc": False, "build_dir": None, "src_root": None,
                "flags": {}}
    p = files._path(config, lb)
    # Two common layouts:
    #   <src>/build/bin/llama-server  (a source build; llama_bin points at build/bin)
    #   <src>/bin/llama-server        (installer prebuilt; the source build,
    #                                  if any, is still <src>/build)
    # Never assume the first "bin" parent is the build dir — aibox3's
    # Vulkan prebuilt sat in ~/llama.cpp/bin while its CUDA source build
    # lived in ~/llama.cpp/build, so the update button kept reinstalling
    # a prebuilt and never rebuilt the actual engine.
    if p.parent.name == "bin":
        base = p.parent.parent
        candidates = [base, base / "build"]
    else:
        base = p.parent
        candidates = [base, base / "build"]
    build = next((c for c in candidates
                  if (c / "CMakeCache.txt").is_file()), base)
    out = {"kind": "prebuilt", "llama_bin": lb, "version": "",
           "rpc": False, "build_dir": None, "src_root": None, "flags": {}}
    if not build.is_dir():
        out["kind"] = "missing"
        return out
    out["build_dir"] = str(build)
    build_bin = build / "bin"
    exe = ".exe" if os.name == "nt" else ""
    server = build_bin / f"llama-server{exe}"
    if server.is_file():
        out["llama_bin"] = str(build_bin)
    out["version"], out["rpc"] = _bin_meta(out["llama_bin"])
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
           jobs=None, ref=None) -> dict:
    src = files._path(config, src_root)
    build = files._path(config, build_dir)
    binp = files._path(config, bin_path)
    if not src.is_dir() or not build.is_dir():
        raise AgentBuildError("source or build dir not found")
    if not (build / "CMakeCache.txt").is_file():
        raise AgentBuildError("build dir has no CMakeCache.txt")
    proc = _run(["git", "-C", str(src), "status", "--porcelain"], timeout=60)
    dirty = [ln for ln in (proc.stdout or "").splitlines()
             if ln and not ln.startswith("??")]
    if dirty:
        raise AgentBuildError("tracked files modified: " + "; ".join(dirty[:5]))
    target = (ref or "").strip()
    if target:
        # Build the exact release the farm advertises (e.g. b12224), not
        # whatever origin/master happens to be. A source build is required
        # on glibc < 2.38 + NVIDIA because upstream's CUDA prebuilts need
        # glibc 2.38; this is how those machines stay modern and CUDA.
        _run(["git", "-C", str(src), "fetch", "origin", "tag", target],
             timeout=900)
        proc = _run(["git", "-C", str(src), "rev-parse", "--verify",
                     target + "^{commit}"], timeout=60)
        if proc.returncode != 0:
            _run(["git", "-C", str(src), "fetch", "--tags", "origin"],
                 timeout=900)
            proc = _run(["git", "-C", str(src), "rev-parse", "--verify",
                         target + "^{commit}"], timeout=60)
        if proc.returncode != 0:
            raise AgentBuildError(f"release tag {target} not found upstream")
        proc = _run(["git", "-C", str(src), "checkout", "--detach", target],
                    timeout=120)
    else:
        proc = _run(["git", "-C", str(src), "fetch", "origin", "master"],
                    timeout=600)
        if proc.returncode != 0:
            raise AgentBuildError("git fetch failed: "
                                  + (proc.stderr or "")[-300:])
        proc = _run(["git", "-C", str(src), "merge-base", "--is-ancestor",
                     "HEAD", "origin/master"], timeout=60)
        if proc.returncode != 0:
            raise AgentBuildError("source tree has local-only commits")
        proc = _run(["git", "-C", str(src), "checkout", "origin/master"],
                    timeout=120)
    if proc.returncode != 0:
        raise AgentBuildError("git checkout failed: "
                              + (proc.stderr or "")[-300:])
    # A CMakeCache records the absolute CMAKE_COMMAND used when it was
    # configured. Snap/venv/upgraded CMake leaves that path behind, and
    # gmake then dies with `CMAKE_COMMAND: No such file or directory`
    # before building anything. Reconfigure once with the cmake on PATH;
    # CMake preserves every cached option while rewriting the Makefiles
    # and CMAKE_COMMAND.
    stale_cmake = ""
    try:
        cache = (build / "CMakeCache.txt").read_text(
            encoding="utf-8", errors="replace")
        m = re.search(r"^CMAKE_COMMAND:[^=]*=(.+)$", cache, re.M)
        stale_cmake = m.group(1).strip() if m else ""
    except OSError:
        pass
    if stale_cmake and not Path(stale_cmake).is_file():
        home = str(src)
        try:
            h = re.search(r"^CMAKE_HOME_DIRECTORY:[^=]*=(.+)$",
                          cache, re.M)
            if h and Path(h.group(1).strip()).is_dir():
                home = h.group(1).strip()
        except Exception:
            pass
        proc = _run([_ensure_cmake(), "-S", home, "-B", str(build)],
                    timeout=900)
        if proc.returncode != 0:
            out = ((proc.stdout or "") + (proc.stderr or ""))[-1000:]
            return {"ok": False,
                    "error": f"cmake reconfigure failed after stale cache "
                             f"({stale_cmake}): {out}"}
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
    # Remove the server/rpc executables before building. A previous
    # installer that updated libraries without relinking left an old
    # llama-server beside new libggml*.so (aibox3: Oct-24 binary + Nov-8
    # libs → SIGSEGV with an empty log). Forcing the link step is what
    # makes the update produce a consistent, runnable set.
    for name in ("llama-server", "llama-server.exe",
                 "libllama-server-impl.so", "llama-server.dll",
                 "ggml-rpc-server", "ggml-rpc-server.exe"):
        try:
            (binp.parent / name).unlink()
        except OSError:
            pass
    # Build only what the farm uses, not tests/examples/benches (much
    # faster). If a target is absent in an older tree, fall back to the
    # minimal server+rpc pair.
    targets = ["llama-server", "ggml-rpc-server", "llama-cli",
               "llama-completion", "llama-quantize", "llama-fit-params",
               "llama-bench", "llama-tokenize"]
    cmake = _ensure_cmake()
    proc = _run([cmake, "--build", str(build), "--target", *targets,
                 "-j", str(int(jobs))], timeout=7200)
    if proc.returncode != 0:
        proc = _run([cmake, "--build", str(build), "--target",
                     "llama-server", "ggml-rpc-server",
                     "-j", str(int(jobs))], timeout=7200)
    output = ((proc.stdout or "") + (proc.stderr or ""))[-1500:]
    if proc.returncode != 0:
        return {"ok": False, "error": f"build failed: {output}"}
    rpc = binp.parent / ("ggml-rpc-server.exe" if os.name == "nt"
                         else "ggml-rpc-server")
    if not rpc.is_file():
        return {"ok": False,
                "error": "build did not produce ggml-rpc-server "
                         "(enable GGML_RPC=ON in the build cache)"}
    # A build that exits 0 but leaves a crashing/mixed binary must not be
    # reported as success. Run it and capture the backend list too.
    v = _run([str(binp), "--version"], timeout=120)
    version = ((v.stdout or "") + (v.stderr or "")).strip().splitlines()
    if v.returncode != 0 or not version:
        return {"ok": False,
                "error": f"built llama-server does not run "
                         f"(exit {v.returncode}); mixed/incomplete libs?"}
    dev = _run([str(binp), "--list-devices"], timeout=120)
    devices = ((dev.stdout or "") + (dev.stderr or "")).strip()[:1200]
    rpc_v = _run([str(rpc), "--version"], timeout=120)
    return {"ok": True, "version": version[0] if version else "",
            "devices": devices, "rpc": str(rpc),
            "rpc_ok": rpc_v.returncode == 0, "ref": target,
            "output": output}


_CMAKE_ARG_OK = re.compile(
    r"^-D(GGML_(CUDA|CUDA_FA|CUDA_NCCL|HIP|SYCL|RPC|NATIVE|BLAS|METAL|"
    r"VULKAN)|CMAKE_BUILD_TYPE|CMAKE_CUDA_ARCHITECTURES|AMDGPU_TARGETS)"
    r"=[A-Za-z0-9_.;,+-]*$")


def _have_tool(name):
    if shutil.which(name):
        return True
    for pat in (f"/usr/local/cuda*/bin/{name}", f"/opt/rocm*/bin/{name}",
                f"/opt/intel/oneapi/compiler/*/bin/{name}"):
        if any(Path(x).is_file() for x in __import__("glob").glob(pat)):
            return True
    return False


def _missing_tools(cmake_args):
    missing = [t for t in ("git", "cmake") if not shutil.which(t)]
    blob = " ".join(cmake_args)
    if "GGML_CUDA=ON" in blob and not _have_tool("nvcc"):
        missing.append("nvcc (CUDA toolkit)")
    if "GGML_HIP=ON" in blob and not _have_tool("hipcc"):
        missing.append("hipcc (ROCm)")
    if "GGML_SYCL=ON" in blob and not (_have_tool("icpx")
                                       or _have_tool("icx")):
        missing.append("icpx (oneAPI)")
    return missing


def preflight(config: AgentConfig, cmake_args=None) -> dict:
    """Toolchain check for a source bootstrap (no clone/build)."""
    args = [str(a) for a in (cmake_args or [])]
    for a in args:
        if "\x00" in a or not _CMAKE_ARG_OK.match(a):
            raise AgentBuildError(f"disallowed cmake argument: {a!r}")
    return {"missing": _missing_tools(args)}


def bootstrap(config: AgentConfig, src_root, build_dir, bin_path, tag,
              cmake_args=None, jobs=None) -> dict:
    """Fresh source provision: clone the release tag, configure with the
    machine's specs, build the farm targets, verify. POSIX only."""
    if os.name == "nt":
        raise AgentBuildError("source bootstrap is POSIX-only")
    tag = str(tag or "").strip()
    if not re.match(r"^[A-Za-z0-9._-]+$", tag):
        raise AgentBuildError("invalid release ref")
    args = [str(a) for a in (cmake_args or [])]
    for a in args:
        if "\x00" in a or not _CMAKE_ARG_OK.match(a):
            raise AgentBuildError(f"disallowed cmake argument: {a!r}")
    src = files._path(config, src_root, write=True)
    build = files._path(config, build_dir, write=True)
    files._path(config, bin_path, write=True)  # validate under roots
    cmake = _ensure_cmake()
    cc, cxx, compiler_args = _pick_compiler()
    args = args + compiler_args
    build_env = dict(os.environ)
    if cc:
        build_env["CC"] = cc
    if cxx:
        build_env["CXX"] = cxx
    missing = _missing_tools(args)
    if missing:
        raise AgentBuildError("missing build tools: " + ", ".join(missing))
    src.parent.mkdir(parents=True, exist_ok=True)
    if not (src / ".git").is_dir():
        if src.exists() and any(src.iterdir()):
            raise AgentBuildError(f"{src} exists but is not a git checkout")
        proc = _run(["git", "clone", "--depth", "1", "--branch", tag,
                     "https://github.com/ggml-org/llama.cpp", str(src)],
                    timeout=1800)
        if proc.returncode != 0:
            raise AgentBuildError("git clone failed: "
                                  + (proc.stderr or "")[-400:])
    else:
        proc = _run(["git", "-C", str(src), "fetch", "--depth", "1",
                     "origin", "tag", tag], timeout=1200)
        if proc.returncode != 0:
            _run(["git", "-C", str(src), "fetch", "--tags", "origin"],
                 timeout=1800)
        proc = _run(["git", "-C", str(src), "checkout", "--detach", "-f",
                     tag], timeout=180)
        if proc.returncode != 0:
            raise AgentBuildError("git checkout failed: "
                                  + (proc.stderr or "")[-300:])
    build.mkdir(parents=True, exist_ok=True)
    if not (build / "CMakeCache.txt").is_file():
        proc = _run([cmake, "-S", str(src), "-B", str(build),
                     "-DCMAKE_BUILD_TYPE=Release", *args],
                    timeout=1800, env=build_env)
        if proc.returncode != 0:
            out = ((proc.stdout or "") + (proc.stderr or ""))[-1200:]
            raise AgentBuildError("cmake configure failed: " + out)
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
    targets = ["llama-server", "ggml-rpc-server", "llama-cli",
               "llama-completion", "llama-quantize", "llama-fit-params",
               "llama-bench", "llama-tokenize"]
    proc = _run([cmake, "--build", str(build), "--target", *targets,
                 "-j", str(int(jobs))], timeout=21600, env=build_env)
    if proc.returncode != 0:
        proc = _run([cmake, "--build", str(build), "--target",
                     "llama-server", "ggml-rpc-server",
                     "-j", str(int(jobs))], timeout=21600, env=build_env)
    if proc.returncode != 0:
        raise AgentBuildError("build failed: "
                              + (((proc.stdout or "") + (proc.stderr or ""))[-1500:]))
    bin_dir = build / "bin"
    binp = bin_dir / "llama-server"
    rpc = bin_dir / "ggml-rpc-server"
    if not rpc.is_file():
        raise AgentBuildError("build did not produce ggml-rpc-server "
                              "(GGML_RPC=ON required)")
    v = _run([str(binp), "--version"], timeout=120)
    lines = ((v.stdout or "") + (v.stderr or "")).strip().splitlines()
    if v.returncode != 0 or not lines:
        raise AgentBuildError("built llama-server does not run")
    dev = _run([str(binp), "--list-devices"], timeout=120)
    return {"ok": True, "version": lines[0], "ref": tag,
            "llama_bin": str(bin_dir), "src_root": str(src),
            "build_dir": str(build), "rpc": str(rpc),
            "devices": ((dev.stdout or "") + (dev.stderr or "")).strip()[:1200]}
