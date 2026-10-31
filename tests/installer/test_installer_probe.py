# tests/installer/ — hermetic installer suite (pyfakefs + stubbed run).
# No network, no real tools, no root; offline and fast by construction.

import json
import os
import sys
from dataclasses import dataclass

import pytest

SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from installer.env import RunResult, SystemInfo  # noqa: E402
from installer import probe as probe_mod         # noqa: E402
from installer import execute as execute_mod     # noqa: E402
from installer.probe import probe                # noqa: E402
from installer.execute import Executor, failure_envelope  # noqa: E402

HOME = os.path.expanduser("~")


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------

class StubFS:
    """Filesystem answers backed by a dict of paths -> text. Runs on pyfakefs."""

    def __init__(self, files=None, bad_bins=None):
        self.files = files or {}          # path -> content
        self.bad_bins = set(bad_bins or set())   # binaries that exist but --version fails

    def add_file(self, path, content=""):
        """Register a virtual file (works identically on real disk and pyfakefs)."""
        self.files[path] = content

    def exists(self, path):
        return self.isfile(path) or os.path.isdir(path)

    def isfile(self, path):
        return path in self.files or os.path.isfile(path)

    def expanduser(self, path):
        return os.path.expanduser(os.path.expandvars(path))

    def read_file(self, path):
        return self.files.get(path) or open(path, "r", encoding="utf-8").read()

    def file_exists_and_runs(self, binary, version_arg="--version"):
        if not self.isfile(binary) and not self.isfile(self.expanduser(binary)):
            return RunResult(binary, 127, "", "missing")
        if binary in self.bad_bins or self.expanduser(binary) in self.bad_bins:
            return RunResult(binary, 1, "", "segfault")
        ver = self.files.get("__ver__:" + binary,
                             self.files.get("__ver__:" + self.expanduser(binary),
                                            "1.2.3"))
        return RunResult(binary, 0, ver, "")


@dataclass
class StubRun:
    """dict-backed RunCmd: cmd prefix -> RunResult."""
    answers: dict

    def run(self, cmd, timeout=30.0):
        for prefix, res in self.answers.items():
            if str(cmd).startswith(prefix):
                return res
        return RunResult(cmd, 127, "", "no stub")


@pytest.fixture
def stub_fs(fs):
    """Hermetic FS: pyfakefs on disk + StubFS dict overlay in front.
    The fixture `fs` comes from the pyfakefs pytest plugin (dev dep)."""
    return StubFS()


def sysinfo(os="linux", arch="x86_64", gpu="nvidia", glibc=(2, 39),
            driver="580.65", vram=None, wsl=False):
    return SystemInfo(os=os, arch=arch, gpu_vendor=gpu, glibc=glibc,
                      driver_version=driver, vram_gb=vram or [], wsl=wsl)


def settings_yaml(text="model_dir: store://llm-models\ntheme_family: arctic\n"):
    return {"~/llama-rpc-farm/config/settings.yaml": text,
            "/home/user/llama-rpc-farm/config/settings.yaml": text}


# ---------------------------------------------------------------------------
# Fresh machine -> full install plan
# ---------------------------------------------------------------------------

class TestFreshMachine:

    def test_full_install_plan_linux_cuda(self, stub_fs):
        plan = probe(sysinfo(), stub_fs, StubRun({}))
        names = [type(s).__name__ for s in plan]
        assert "InstallTool" in names
        tools = [s.name for s in plan if type(s).__name__ in
                 ("InstallTool", "SkipTool", "RepairTool")]
        # engine parity: BOTH engines planned on CUDA
        assert "llama.cpp" in tools
        assert "vllm" in tools
        vllm_steps = [s for s in plan if getattr(s, "name", None) == "vllm"
                      and isinstance(s, probe_mod.InstallTool)]
        assert vllm_steps[0].backend == "pypi"

    def test_llcpp_backend_per_gpu(self):
        # CUDA / ROCm / CPU / Metal asset backends
        assert probe_mod.LLCPP_BACKEND[("linux", "nvidia")] == "cuda-13.3"
        assert probe_mod.LLCPP_BACKEND[("linux", "amd")] == "rocm-10.0"
        assert probe_mod.LLCPP_BACKEND[("linux", "none")] == "cpu"
        assert probe_mod.LLCPP_BACKEND[("macos", "apple")] == "metal"

    def test_windows_vllm_refused_with_note(self, stub_fs):
        plan = probe(sysinfo(os="windows", gpu="nvidia"), stub_fs, StubRun({}))
        notes = [s for s in plan if isinstance(s, probe_mod.PlanNote)]
        assert any(n.tool == "vllm" and n.kind == "refused" for n in notes)

    def test_macos_vllm_metal_optin(self, stub_fs):
        info = sysinfo(os="macos", gpu="apple")
        p1 = probe(info, stub_fs, StubRun({}))           # default: no opt-in
        p2 = probe(info, StubFS(), StubRun({}), with_vllm_metal=True)
        assert not any(isinstance(s, probe_mod.InstallTool) and s.name == "vllm"
                       for s in p1)
        assert any(isinstance(s, probe_mod.InstallTool) and s.name == "vllm"
                   and s.backend == "vllm-metal" for s in p2)

    def test_rocm_vllm_index(self, stub_fs):
        plan = probe(sysinfo(gpu="amd"), stub_fs, StubRun({}))
        vllm_steps = [s for s in plan if isinstance(s, probe_mod.InstallTool)
                      and s.name == "vllm"]
        assert vllm_steps[0].extra.get("index_url") == "https://wheels.vllm.ai/rocm/"
        assert vllm_steps[0].extra.get("python") == "3.12"


# ---------------------------------------------------------------------------
# Present + current -> all-skip with discovered paths
# ---------------------------------------------------------------------------

class TestPresentCurrent:

    def _setup(self, stub_fs):
        home = HOME
        good = home + "/llama.cpp/bin/llama-server"
        stub_fs.add_file(good, "#!/bin/sh\necho llama.cpp\n")
        stub_fs.add_file("__ver__:" + good, "b10934 (test)")
        swap = home + "/.local/bin/llama-swap"
        stub_fs.add_file(swap, "#!/bin/sh\n")
        stub_fs.add_file("__ver__:" + swap, "255")
        vllm = home + "/vllm-env/bin/vllm"
        stub_fs.add_file(vllm, "#!/bin/sh\n")
        stub_fs.add_file("__ver__:" + vllm, "0.30.0")
        return good, swap, vllm

    def test_all_skip_with_paths(self, stub_fs):
        good, swap, vllm = self._setup(stub_fs)
        plan = probe(sysinfo(), stub_fs, StubRun({}))
        skips = {s.name: s for s in plan
                 if isinstance(s, probe_mod.SkipTool)
                 and s.reason == "present-current"}
        assert set(skips) >= {"llama.cpp", "llama-swap", "vllm"}
        assert skips["llama.cpp"].found_path.endswith("llama-server")
        assert skips["vllm"].version == "0.30.0"

    def test_settings_prefill_discovered_path(self, stub_fs):
        good, swap, vllm = self._setup(stub_fs)
        plan = probe(sysinfo(), stub_fs, StubRun({}))
        writes = [s for s in plan if isinstance(s, probe_mod.WriteSettings)]
        keys = {s.key for s in writes}
        assert "llama_bin" in keys
        assert "vllm.venv_bin" in keys
        assert "binary" in keys      # swap.yaml

    def test_user_value_kept(self, stub_fs):
        good, swap, vllm = self._setup(stub_fs)
        # User value must point at a VALID existing binary to be kept (§6 b/c:
        # a non-existent path is stale and gets repaired to the discovered one).
        stub_fs.add_file("config/settings.yaml",
                         "model_dir: store://llm-models\nllama_bin: %s\n" % swap)
        plan = probe(sysinfo(), stub_fs, StubRun({}))
        writes = [s for s in plan if isinstance(s, probe_mod.WriteSettings)]
        assert not any(w.key == "llama_bin" for w in writes)


# ---------------------------------------------------------------------------
# Broken --version -> repair
# ---------------------------------------------------------------------------

class TestRepair:

    def test_broken_llcpp_repaired(self, stub_fs):
        home = HOME
        good = home + "/llama.cpp/bin/llama-server"
        stub_fs.add_file(good, "garbage")           # exists, --version segfaults
        stub_fs.bad_bins.add(good)
        plan = probe(sysinfo(), stub_fs, StubRun({}))
        repairs = [s for s in plan if isinstance(s, probe_mod.RepairTool)]
        assert len(repairs) == 1
        assert repairs[0].name == "llama.cpp"
        assert repairs[0].dest == "~/llama.cpp/bin/llama-server"
        warns = [s for s in plan if isinstance(s, probe_mod.PlanNote)
                 and s.kind == "warn"]
        assert warns


# ---------------------------------------------------------------------------
# Settings-merge rules (§6)
# ---------------------------------------------------------------------------

class TestSettingsMerge:

    def _probe_with(self, stub_fs, text):
        home = HOME
        good = home + "/llama.cpp/bin/llama-server"
        stub_fs.add_file(good, "#!/bin/sh\n")
        stub_fs.add_file("__ver__:" + good, "b10934")
        stub_fs.add_file("config/settings.yaml", text)
        return probe(sysinfo(), stub_fs, StubRun({}))

    def test_empty_key_written(self, stub_fs):
        plan = self._probe_with(stub_fs, "model_dir: store://x\n")
        w = [s for s in plan if isinstance(s, probe_mod.WriteSettings)
             and s.key == "llama_bin"]
        assert w and w[0].merge_reason == "empty"

    def test_stale_path_repaired(self, stub_fs):
        plan = self._probe_with(
            stub_fs, "model_dir: store://x\nllama_bin: /gone/dead-llama\n")
        w = [s for s in plan if isinstance(s, probe_mod.WriteSettings)
             and s.key == "llama_bin"]
        assert w and w[0].merge_reason == "stale"
        assert w[0].value.endswith("llama-server")

    def test_valid_user_value_untouched(self, stub_fs):
        # points at a real existing binary -> kept
        home = HOME
        valid = home + "/llama.cpp/bin/llama-server"
        self._probe_with(stub_fs,
                         "model_dir: store://x\nllama_bin: %s\n" % valid)
        # _probe_with adds the binary too; assert no llama_bin write
        plan = self._probe_with(stub_fs,
                                "model_dir: store://x\nllama_bin: %s\n" % valid)
        assert not [s for s in plan if isinstance(s, probe_mod.WriteSettings)
                    and s.key == "llama_bin"]

    def test_unrelated_keys_untouched(self, stub_fs):
        text = "model_dir: store://llm-models\ntheme_family: arctic\n"
        plan = self._probe_with(stub_fs, text)
        for s in plan:
            if isinstance(s, probe_mod.WriteSettings):
                assert s.key in ("llama_bin", "vllm.venv_bin", "binary")


# ---------------------------------------------------------------------------
# Golden idempotency: apply plan -> probe again -> all-skip
# ---------------------------------------------------------------------------

class TestIdempotency:

    def test_second_probe_all_skip(self, stub_fs):
        # Simulate executing plan A's skips only (installs are stubbed), then
        # probe again on the resulting state — must converge to all-skip.
        home = HOME
        good = home + "/llama.cpp/bin/llama-server"
        stub_fs.add_file(good, "#!/bin/sh\n")
        stub_fs.add_file("__ver__:" + good, "b10934")
        swap = home + "/.local/bin/llama-swap"
        stub_fs.add_file(swap, "#!/bin/sh\n")
        stub_fs.add_file("__ver__:" + swap, "255")
        vllm = home + "/vllm-env/bin/vllm"
        stub_fs.add_file(vllm, "#!/bin/sh\n")
        stub_fs.add_file("__ver__:" + vllm, "0.30.0")

        plan = probe(sysinfo(), stub_fs, StubRun({}))
        ex = Executor(log=lambda m: None, dry_run=True)
        summary = ex.execute(plan)
        assert summary["failures"] == []

        plan2 = probe(sysinfo(), stub_fs, StubRun({}))
        actions = [s for s in plan2 if isinstance(
            s, (probe_mod.InstallTool, probe_mod.RepairTool))]
        assert actions == []
        skips = [s for s in plan2 if isinstance(s, probe_mod.SkipTool)]
        assert {s.name for s in skips} >= {"llama.cpp", "llama-swap", "vllm"}
        assert all(s.reason == "present-current" for s in skips)

    def test_executor_dry_run_golden(self, stub_fs):
        plan = probe(sysinfo(), stub_fs, StubRun({}))
        lines = []
        ex = Executor(log=lines.append, dry_run=True)
        summary = ex.execute(plan)
        assert any("dry-run" in l for l in lines)
        assert summary["failures"] == []


# ---------------------------------------------------------------------------
# Failure envelope (req #4): schema + redaction
# ---------------------------------------------------------------------------

class TestFailureEnvelope:

    def test_schema(self):
        env = failure_envelope("installer", "boom",
                               context={"tool": "vllm", "log_tail": "traceback"})
        assert env["schema"] == "llama-rpc-farm-failure-envelope/1"
        assert env["source"] == "installer"
        assert env["context"]["tool"] == "vllm"
        assert "captured_at" in env
        json.dumps(env)     # must be serializable

    def test_redaction(self):
        env = failure_envelope("installer",
                               "export SSH_PASS=hunter2 && hf_token=abc123")
        assert "hunter2" not in env["error"]
        assert "abc123" not in env["error"]
        assert "<redacted>" in env["error"]

    def test_executor_captures_failure(self):
        class BoomExec(Executor):
            def _apply(self, step, summary):
                raise RuntimeError("disk full")
        plan = [probe_mod.SkipTool("llama.cpp", "present-current")]
        summary = BoomExec(log=lambda m: None).execute(plan)
        assert len(summary["failures"]) == 1
        assert summary["failures"][0]["schema"].endswith("/1")


# ---------------------------------------------------------------------------
# Engine defaults per backend (operator clarification #3)
# ---------------------------------------------------------------------------

class TestEngineDefaults:

    def test_cuda_env_vars(self):
        info = sysinfo(gpu="nvidia")
        assert probe_mod.tool_defaults("llama.cpp", info)["device_env"] == \
            "CUDA_VISIBLE_DEVICES"
        assert probe_mod.tool_defaults("vllm", info)["device_env"] == \
            "CUDA_VISIBLE_DEVICES"

    def test_rocm_env_vars(self):
        info = sysinfo(gpu="amd")
        assert probe_mod.tool_defaults("vllm", info)["device_env"] == \
            "HIP_VISIBLE_DEVICES"
        assert probe_mod.tool_defaults("llama.cpp", info)["device_env"] == \
            "HIP_VISIBLE_DEVICES"

    def test_metal_no_device_env(self):
        info = sysinfo(os="macos", gpu="apple")
        assert probe_mod.tool_defaults("vllm", info)["device_env"] is None
        assert probe_mod.tool_defaults("llama.cpp", info)["device_env"] is None


# ---------------------------------------------------------------------------
# Slow integration test (quarantined): real network, downloads the llama-swap
# release binary ONLY into a throwaway dest. Skipped by default; run with
#   pytest tests/installer -m slow
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_real_swap_download_roundtrip(tmp_path):
    from installer.probe import InstallTool
    from installer.execute import Executor
    from installer.env import SystemInfo
    # The hermetic suite's ExtractingRun/StubDownloader stubs poison the
    # shared /tmp staging paths (/tmp/llama-swap-installer{,.pkg}) with
    # non-gzip "fake-bytes"; the real tar here must start clean. Clear
    # both before running (never delete the dir — just the files).
    for stale in ("/tmp/llama-swap-installer.pkg",
                  "/tmp/llama-swap-installer/llama-swap"):
        try:
            os.remove(stale)
        except FileNotFoundError:
            pass
        except Exception:
            pass
    step = InstallTool("llama-swap", "prebuilt", "255",
                       dest=str(tmp_path / "local" / "bin"))
    info = SystemInfo(os="linux", arch="x86_64", gpu_vendor="none")
    ex = Executor(log=lambda m: None, info=info,
                  manifest_path=str(tmp_path / "state.yaml"))  # real shell+net
    summary = ex.execute([step])
    assert summary["failures"] == [], summary["failures"]
    assert summary["installed"] == ["llama-swap"]
    bin_path = tmp_path / "local" / "bin" / "llama-swap"
    assert bin_path.is_file() and bin_path.stat().st_size > 1_000_000