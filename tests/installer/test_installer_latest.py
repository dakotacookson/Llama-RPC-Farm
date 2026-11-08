"""Bootstrap resolves the newest installable release, like the UI cards.

probe() stays hermetic (no network); bootstrap passes the resolved plan in.
These tests assert the resolved version/assets flow through probe and that
the executors download the explicit per-release asset URLs.
"""
import os
import sys

SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from installer.env import RunResult, SystemInfo          # noqa: E402
from installer.probe import InstallTool, probe           # noqa: E402
from installer import tools as tools_mod                  # noqa: E402


class Run:
    def __init__(self):
        self.cmds = []

    def __call__(self, cmd, timeout=30.0):
        return self.run(cmd, timeout)

    def run(self, cmd, timeout=30.0):
        self.cmds.append(str(cmd))
        return RunResult(str(cmd), 0, "", "")


class SwapRun(Run):
    """Creates the extracted llama-swap file so the atomic move works."""
    def run(self, cmd, timeout=30.0):
        res = super().run(cmd, timeout)
        if "tar -xzf" in str(cmd) and "/tmp/llama-swap-installer" in str(cmd):
            os.makedirs("/tmp/llama-swap-installer", exist_ok=True)
            with open("/tmp/llama-swap-installer/llama-swap", "wb") as f:
                f.write(b"bin")
        return res


class FS:
    """Minimal FS stub: no binaries installed."""
    def __init__(self):
        self.files = {}

    def exists(self, path):
        return self.isfile(path)

    def isfile(self, path):
        return path in self.files

    def expanduser(self, path):
        return os.path.expanduser(os.path.expandvars(path))

    def read_file(self, path):
        return self.files.get(path, "")

    def file_exists_and_runs(self, binary, version_arg="--version"):
        return RunResult(binary, 127, "", "missing")


class DL:
    def __init__(self):
        self.urls = []

    def __call__(self, url, dest):
        self.urls.append(url)
        os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
        with open(dest, "wb") as f:
            f.write(b"bytes")


def info():
    return SystemInfo(os="linux", arch="x86_64", gpu_vendor="nvidia",
                      glibc=(2, 39))


LC_NAMES = ["llama-b20000-bin-ubuntu-cuda-13.4-x64.tar.gz",
            "cudart-llama-b20000-bin-ubuntu-cuda-13.4-x64.tar.gz"]
LC_BASE = "https://github.com/ggml-org/llama.cpp/releases/download/b20000"
SW_NAME = "llama-swap_300_linux_amd64.tar.gz"
SW_BASE = "https://github.com/mostlygeek/llama-swap/releases/download/v300"


def test_probe_uses_resolved_latest_plan():
    resolved = {
        "llama.cpp": {"version": "b20000", "assets": LC_NAMES,
                      "base": LC_BASE, "dest": "~/llama.cpp/bin",
                      "backend": "cuda", "note": "CUDA 13.4"},
        "llama-swap": {"version": "300", "assets": [SW_NAME],
                       "base": SW_BASE, "dest": "~/.local/bin"},
    }
    plan = probe(info(), FS(), Run(), role="main",
                 pinned={"llama.cpp": {"version": "b1"},
                         "llama-swap": {"version": "1"}},
                 resolved=resolved)
    steps = {s.name: s for s in plan if isinstance(s, InstallTool)}
    assert steps["llama.cpp"].version == "b20000"
    assert steps["llama.cpp"].extra["assets"] == LC_NAMES
    assert steps["llama.cpp"].extra["base"] == LC_BASE
    assert steps["llama.cpp"].backend == "cuda"
    assert steps["llama-swap"].version == "300"
    assert steps["llama-swap"].extra["assets"] == [SW_NAME]


def test_llcpp_executor_uses_explicit_release_urls(tmp_path):
    dl, run = DL(), Run()
    step = InstallTool("llama.cpp", "prebuilt", "b20000", backend="cuda",
                       dest=str(tmp_path / "bin"),
                       extra={"assets": LC_NAMES, "base": LC_BASE})
    tools_mod.install_llcpp(step, dl, run.run, lambda m: None,
                            "linux", "x86_64")
    assert dl.urls == [f"{LC_BASE}/{n}" for n in LC_NAMES]


def test_swap_executor_uses_explicit_release_url(tmp_path):
    dl, run = DL(), SwapRun()
    dest = tmp_path / "bin"
    dest.mkdir()
    step = InstallTool("llama-swap", "prebuilt", "300", dest=str(dest),
                       extra={"assets": [SW_NAME], "base": SW_BASE})
    tools_mod.install_swap(step, dl, run.run, lambda m: None,
                           "linux", "x86_64")
    assert dl.urls == [f"{SW_BASE}/{SW_NAME}"]
    assert (dest / "llama-swap").exists()


def test_bootstrap_resolves_latest_via_ui_selectors(monkeypatch):
    import bootstrap
    import llamacpp_installer as lcpp
    import swap_installer as swp

    monkeypatch.setattr(lcpp, "_fetch_latest_assets",
                        lambda limit=30: ("b20000", LC_NAMES, LC_BASE))
    monkeypatch.setattr(lcpp, "_pick_assets",
                        lambda machine, tag, assets, glibc=None:
                        (LC_NAMES, "~/llama.cpp/bin", "CUDA 13.4", False))
    monkeypatch.setattr(swp, "_fetch_latest_assets",
                        lambda limit=10: ("v300", [SW_NAME], SW_BASE))
    monkeypatch.setattr(swp, "_pick_asset",
                        lambda machine, tag, assets:
                        (SW_NAME, "~/.local/bin", "300"))

    resolved = bootstrap._resolve_latest(info(), "main", lambda m: None)
    assert resolved["llama.cpp"]["version"] == "b20000"
    assert resolved["llama.cpp"]["assets"] == LC_NAMES
    assert resolved["llama.cpp"]["backend"] == "cuda"
    assert resolved["llama-swap"]["version"] == "300"
    assert resolved["llama-swap"]["base"] == SW_BASE


# ---------------------------------------------------------------------------
# Source builds: preserve the CMakeCache flags and build BOTH binaries
# ---------------------------------------------------------------------------

SOURCE_PROBE = {
    "kind": "source",
    "src_root": "/src/llama.cpp",
    "build_dir": "/src/llama.cpp/build",
    "llama_bin": "/src/llama.cpp/build/bin",
    "flags": {
        "GGML_CUDA": "ON",
        "CMAKE_CUDA_ARCHITECTURES": "61;75;86;89",
        "GGML_RPC": "ON",
    },
    "version": "version: b12209",
    "rpc": True,
}


def test_probe_plans_source_update_when_build_is_detected():
    import pytest  # noqa: F401  (kept local to avoid changing module imports)
    plan = probe(info(), FS(), Run(), role="main",
                 source_probe=SOURCE_PROBE)
    step = next(s for s in plan
                if isinstance(s, InstallTool) and s.name == "llama.cpp")
    assert step.source == "source"
    assert step.dest == "/src/llama.cpp/build/bin"
    assert step.extra["src_root"] == "/src/llama.cpp"
    assert step.extra["build_dir"] == "/src/llama.cpp/build"
    assert step.extra["flags"]["GGML_RPC"] == "ON"
    assert step.extra["flags"]["CMAKE_CUDA_ARCHITECTURES"] == "61;75;86;89"
    manifest = next(s for s in plan
                    if getattr(s, "tool", None) == "llama.cpp"
                    and hasattr(s, "entry"))
    assert manifest.entry["method"] == "source"


class SourceRun(Run):
    """Simulates git fetch/checkout, cmake build, and both binary probes."""
    def run(self, cmd, timeout=30.0):
        self.cmds.append(str(cmd))
        c = str(cmd)
        if "git fetch" in c:
            return RunResult(c, 0, "", "")
        if "merge-base --is-ancestor" in c:
            return RunResult(c, 0, "CLEAN-ANCESTOR", "")
        if "git status --porcelain" in c:
            return RunResult(c, 0, "", "")
        if "git checkout origin/master" in c:
            return RunResult(c, 0, "", "")
        if c.startswith("nproc;"):
            return RunResult(c, 0, "8\n16384\n", "")
        if "cmake --build" in c:
            return RunResult(c, 0, "built", "")
        if c.startswith("test -x"):
            return RunResult(c, 0, "", "")
        if "--version" in c:
            return RunResult(c, 0, "version: b12209\n", "")
        return RunResult(c, 0, "", "")


def test_source_executor_rebuilds_in_place_and_keeps_flags():
    run = SourceRun()
    step = InstallTool(
        "llama.cpp", "source", "b12209", backend="cuda",
        dest="/src/llama.cpp/build/bin",
        extra={"src_root": "/src/llama.cpp",
               "build_dir": "/src/llama.cpp/build",
               "flags": SOURCE_PROBE["flags"],
               "rpc_required": True})
    entry = tools_mod.install_llcpp_source(step, run, lambda m: None)
    assert entry["method"] == "source"
    assert "b12209" in entry["version"]
    assert any("cmake --build" in c for c in run.cmds)
    # Never reconfigure: the existing CMakeCache is what preserves the flags.
    assert not any("cmake -S" in c or "cmake -B" in c for c in run.cmds)
    assert any("ggml-rpc-server" in c for c in run.cmds)


class MissingRpcRun(Run):
    def run(self, cmd, timeout=30.0):
        c = str(cmd)
        if c.startswith("test -x") and "ggml-rpc-server" in c:
            return RunResult(c, 1, "", "missing")
        return super().run(cmd, timeout)


def test_prebuilt_install_requires_llama_rpc():
    import pytest
    step = InstallTool("llama.cpp", "prebuilt", "b20000", backend="cuda",
                       dest="/tmp/bin",
                       extra={"assets": LC_NAMES, "base": LC_BASE})
    with pytest.raises(RuntimeError, match="ggml-rpc-server"):
        tools_mod.install_llcpp(step, DL(), MissingRpcRun().run,
                                lambda m: None, "linux", "x86_64")
