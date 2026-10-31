# tests/installer/test_installer_execute.py — execution layer, hermetic.
# All downloads via stub downloader (no network); shell run via recording stub.
# Covers: asset maps (design §2), llama-swap/llama.cpp/vLLM/ray executors,
# vulkan glibc fallback, manifest write (incl. skip-only), offline stub,
# settings persister round-trip.

import os
import sys

import pytest

SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from installer.env import RunResult, SystemInfo          # noqa: E402
from installer import probe as probe_mod                  # noqa: E402
from installer import tools as tools_mod                  # noqa: E402
from installer.execute import Executor, RealDownloader    # noqa: E402

HOME = os.path.expanduser("~")


def sysinfo(os_name="linux", gpu="nvidia", glibc=(2, 39), arch="x86_64"):
    return SystemInfo(os=os_name, arch=arch, gpu_vendor=gpu,
                      glibc=glibc if os_name != "windows" else None)


class RecordingRun:
    """RunCmd stub that records commands; tar/expand steps succeed."""

    def __init__(self):
        self.cmds = []

    def __call__(self, cmd, timeout=30.0):
        return self.run(cmd, timeout=timeout)

    def run(self, cmd, timeout=30.0):
        self.cmds.append(str(cmd))
        c = str(cmd)
        # probe `_has_uv` in executors reads stdout: report uv present
        if "command -v uv" in c:
            return RunResult(c, 0, "yes", "")
        # `vllm --version` after a stubbed install: report the pinned version
        if c.endswith("/bin/vllm --version"):
            return RunResult(c, 0, "0.30.0", "")
        return RunResult(c, 0, "", "")


class ExtractingRun(RecordingRun):
    """Simulates tar extraction side effects so the swap executor's atomic
    os.replace has a real file to move (hermetic — no real tarball)."""

    def run(self, cmd, timeout=30.0):
        res = super().run(cmd, timeout=timeout)
        c = str(cmd)
        if "tar -xzf" in c and "/tmp/llama-swap-installer" in c:
            os.makedirs("/tmp/llama-swap-installer", exist_ok=True)
            with open("/tmp/llama-swap-installer/llama-swap", "wb") as f:
                f.write(b"bin")
        return res


class StubDownloader:
    """url -> dest; records every URL. No network."""

    def __init__(self, fail_on=None):
        self.urls = []
        self.wrote = []
        self.fail_on = fail_on or set()

    def __call__(self, url, dest):
        if any(f in url for f in self.fail_on):
            raise RuntimeError(f"unreachable: {url}")
        self.urls.append(url)
        with open(dest, "wb") as f:
            f.write(b"fake-bytes")
        self.wrote.append((url, dest))


def swap_step(version="255", dest="~/.local/bin"):
    return probe_mod.InstallTool("llama-swap", "prebuilt", version, dest=dest)


def llcpp_step(backend="cuda-13.3", version="b10934",
               dest="~/llama.cpp/bin", extra=None):
    return probe_mod.InstallTool("llama.cpp", "prebuilt", version,
                                 backend=backend, dest=dest,
                                 extra=extra or {})


def vllm_step(pin="0.30.0", extra=None, dest="~/vllm-env"):
    return probe_mod.InstallTool("vllm", "venv:pypi", pin, backend="pypi",
                                 dest=dest, extra=extra or {})


class TestAssetMaps:
    def test_swap_tag_number_naming(self):
        names = tools_mod.swap_asset_names("linux", "x86_64", "255")
        assert names == ["llama-swap_255_linux_amd64.tar.gz"]
        assert tools_mod.swap_asset_names("macos", "arm64", "v256") == \
            ["llama-swap_256_darwin_arm64.tar.gz"]
        assert tools_mod.swap_asset_names("windows", "x86_64", "255") == \
            ["llama-swap_255_windows_amd64.zip"]

    def test_llcpp_assets_cuda_with_cudart(self):
        names = tools_mod.llcpp_assets_for(
            llcpp_step(backend="cuda-13.3", extra={"arch": "x64"}))
        assert names == [
            "llama-b10934-bin-ubuntu-cuda-13.3-x64.tar.gz",
            "cudart-llama-b10934-bin-ubuntu-cuda-13.3-x64.tar.gz"]

    def test_llcpp_rocm_cpu_metal(self):
        assert tools_mod.llcpp_assets_for(llcpp_step("rocm-10.0",
                                                     extra={"arch": "x64"}))\
            == ["llama-b10934-bin-ubuntu-rocm-10.0-x64.tar.gz"]
        assert tools_mod.llcpp_assets_for(llcpp_step("cpu",
                                                     extra={"arch": "arm64"}))\
            == ["llama-b10934-bin-ubuntu-arm64.tar.gz"]
        assert tools_mod.llcpp_assets_for(llcpp_step("metal",
                                                     extra={"arch": "arm64"}))\
            == ["llama-b10934-bin-macos-arm64.tar.gz"]

    def test_llcpp_vulkan_fallback_glibc_below_238(self):
        step = llcpp_step(backend="cuda-13.3", extra={"arch": "x64",
                                                      "glibc": (2, 35)})
        names = tools_mod.llcpp_assets_for(step)
        assert names == ["llama-b10934-bin-ubuntu-vulkan-x64.tar.gz"]

    def test_llcpp_cuda_kept_on_new_glibc(self):
        step = llcpp_step(backend="cuda-13.3", extra={"arch": "x64",
                                                      "glibc": (2, 39)})
        assert "cuda-13.3" in tools_mod.llcpp_assets_for(step)[0]

    def test_win_cuda_pair(self):
        names = tools_mod.llcpp_assets_for(
            llcpp_step(backend="win-cuda", extra={"arch": "x64"}))
        assert names[0] == "llama-b10934-bin-win-cuda-13.3-x64.zip"
        assert names[1] == "cudart-llama-bin-win-cuda-13.3-x64.zip"


class TestSwapExecutor:
    def test_install_temp_then_atomic(self, tmp_path):
        dest = str(tmp_path / "bin")
        step = swap_step(dest=dest)
        dl = StubDownloader()
        run = ExtractingRun()
        entry = tools_mod.install_swap(step, dl, run, lambda m: None,
                                       "linux", "x86_64")
        # asset fetched via tag-NUMBER url
        assert dl.urls == ["https://github.com/mostlygeek/llama-swap/"
                           "releases/download/v255/"
                           "llama-swap_255_linux_amd64.tar.gz"]
        # atomic convergence: extraction to temp, replace into dest
        assert any("tar -xzf" in c and "/tmp/llama-swap-installer" in c
                   for c in run.cmds)
        assert entry["dest"] == dest + "/llama-swap"
        assert entry["version"] == "255"

    def test_download_failure_raises(self, tmp_path):
        dl = StubDownloader(fail_on={"llama-swap"})
        with pytest.raises(RuntimeError):
            tools_mod.install_swap(swap_step(dest=str(tmp_path)),
                                   dl, RecordingRun(), lambda m: None,
                                   "linux", "x86_64")


class TestLlcppExecutor:
    def test_extract_flat_strip_components(self, tmp_path):
        dest = str(tmp_path / "llama.cpp" / "bin")
        dl, run = StubDownloader(), RecordingRun()
        entry = tools_mod.install_llcpp(
            llcpp_step(backend="cpu", dest=dest, extra={"arch": "x64"}),
            dl, run, lambda m: None, "linux", "x86_64")
        assert len(dl.urls) == 1
        assert any("--strip-components=1" in c for c in run.cmds)
        assert entry["backend"] == "cpu"

    def test_cudart_companion_downloaded(self, tmp_path):
        dl, run = StubDownloader(), RecordingRun()
        tools_mod.install_llcpp(
            llcpp_step(backend="cuda-13.3", dest=str(tmp_path),
                       extra={"arch": "x64"}),
            dl, run, lambda m: None, "linux", "x86_64")
        assert len(dl.urls) == 2 and "cudart-" in dl.urls[1]

    def test_extraction_failure_raises(self, tmp_path):
        class FailRun(RecordingRun):
            def run(self, cmd, timeout=30.0):
                self.cmds.append(str(cmd))
                if "tar" in str(cmd):
                    return RunResult(str(cmd), 2, "", "tar: error")
                return RunResult(str(cmd), 0, "", "")
        with pytest.raises(RuntimeError, match="extraction failed"):
            tools_mod.install_llcpp(
                llcpp_step(backend="cpu", dest=str(tmp_path),
                           extra={"arch": "x64"}),
                StubDownloader(), FailRun(), lambda m: None, "linux", "x86_64")


class TestVllmExecutor:
    def test_venv_pypi_happy(self, tmp_path):
        venv = str(tmp_path / "vllm-env")
        run = RecordingRun()
        entry = tools_mod.install_vllm(vllm_step(dest=venv),
                                       StubDownloader(), run,
                                       lambda m: None)
        assert any("uv venv --python 3.12" in c for c in run.cmds)
        assert any("uv pip install --python" in c and "vllm==0.30.0" in c
                   for c in run.cmds)
        assert entry["index"] == "pypi"

    def test_rocm_index_and_backend_probe(self, tmp_path):
        venv = str(tmp_path / "vllm-env")
        run = RecordingRun()
        step = vllm_step(dest=venv, extra={
            "index_url": "https://wheels.vllm.ai/rocm/", "python": "3.12"})
        # Stub runner that answers uv probe, version smoke, and HIP backend
        hip_run = RecordingRun()
        hip_run.cmds = []

        def _run_hip(cmd, timeout=30.0):
            hip_run.cmds.append(str(cmd))
            if "torch.version" in str(cmd):
                return RunResult(str(cmd), 0, "HIP", "")
            if str(cmd).endswith("/bin/vllm --version"):
                return RunResult(str(cmd), 0, "0.30.0", "")
            return RunResult(str(cmd), 0, "yes" if "command -v uv"
                             in str(cmd) else "0.30.0", "")
        hip_run.run = _run_hip
        entry = tools_mod.install_vllm(step, StubDownloader(), hip_run,
                                       lambda m: None)
        assert any("wheels.vllm.ai/rocm/" in c for c in hip_run.cmds)
        assert entry["index"] == "rocm"

    def test_rocm_silent_cuda_fallback_detected(self, tmp_path):
        class CudaFallbackRun(RecordingRun):
            def run(self, cmd, timeout=30.0):
                self.cmds.append(str(cmd))
                if "torch.version" in str(cmd):
                    return RunResult(str(cmd), 0, "CUDA", "")
                if "command -v uv" in str(cmd):
                    return RunResult(str(cmd), 0, "yes", "")
                if str(cmd).endswith("/bin/vllm --version"):
                    return RunResult(str(cmd), 0, "0.30.0", "")
                return RunResult(str(cmd), 0, "0.30.0", "")
        step = vllm_step(dest=str(tmp_path), extra={
            "index_url": "https://wheels.vllm.ai/rocm/", "python": "3.12"})
        with pytest.raises(RuntimeError, match="silent CUDA fallback"):
            tools_mod.install_vllm(step, StubDownloader(), CudaFallbackRun(),
                                   lambda m: None)

    def test_windows_refused_with_wsl_guidance(self, tmp_path):
        with pytest.raises(RuntimeError, match="WSL"):
            tools_mod.install_vllm(vllm_step(dest=str(tmp_path)),
                                   StubDownloader(), RecordingRun(),
                                   lambda m: None, os_name="windows")

    def test_missing_uv_fails_with_doc_url(self, tmp_path):
        class NoUvRun(RecordingRun):
            def run(self, cmd, timeout=30.0):
                self.cmds.append(str(cmd))
                if "command -v uv" in str(cmd):
                    return RunResult(str(cmd), 1, "no", "")
                return RunResult(str(cmd), 0, "", "")
        with pytest.raises(RuntimeError, match="astral.sh/uv"):
            tools_mod.install_vllm(vllm_step(dest=str(tmp_path)),
                                   StubDownloader(), NoUvRun(),
                                   lambda m: None)

    def test_offline_wheelhouse(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)          # no wheelhouse/ in cwd: the
        # executor-level offline mode is guarded at the bootstrap layer;
        # this tests the pip-index substitution only.
        import installer.tools as t
        orig = t.os.path.isdir
        t.os.path.isdir = lambda p: True if p == "wheelhouse" else orig(p)
        try:
            run = RecordingRun()
            tools_mod.install_vllm(vllm_step(dest=str(tmp_path)),
                                   StubDownloader(), run, lambda m: None,
                                   offline=True)
            assert any("--no-index --find-links wheelhouse/" in c
                       for c in run.cmds)
        finally:
            t.os.path.isdir = orig

    def test_offline_wheelhouse_missing_raises_clearly(self, tmp_path, monkeypatch):
        # §1 stub: a missing wheelhouse/ is a CLEAR not-implemented-style
        # error, not a pip wall of text.
        monkeypatch.chdir(tmp_path)          # no wheelhouse/ here
        with pytest.raises(RuntimeError, match="--offline: wheelhouse/"):
            tools_mod.install_vllm(vllm_step(dest=str(tmp_path)),
                                   StubDownloader(), RecordingRun(),
                                   lambda m: None, offline=True)
        with pytest.raises(RuntimeError, match="--offline: wheelhouse/"):
            tools_mod.install_ray(probe_mod.InstallTool(
                "ray", "venv:pypi", "2.58.0", dest=str(tmp_path)),
                StubDownloader(), RecordingRun(), lambda m: None,
                offline=True)

    def test_offline_wheelhouse_present_ok(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "wheelhouse").mkdir()
        run = RecordingRun()
        tools_mod.install_vllm(vllm_step(dest=str(tmp_path)),
                               StubDownloader(), run, lambda m: None,
                               offline=True)
        assert any("--find-links wheelhouse/" in c for c in run.cmds)


class TestRayExecutor:
    def test_ray_into_same_venv(self, tmp_path):
        venv = str(tmp_path / "vllm-env")
        run = RecordingRun()
        step = probe_mod.InstallTool("ray", "venv:pypi", "2.58.0", dest=venv)
        entry = tools_mod.install_ray(step, StubDownloader(), run,
                                      lambda m: None)
        assert any('"ray[default]"' in c and venv in c for c in run.cmds)


class TestEnsureUv:
    def test_present_and_missing(self):
        assert tools_mod.ensure_uv(RecordingRun(), lambda m: None, None) == \
            {"present": True}
        class NoUvRun(RecordingRun):
            def run(self, cmd, timeout=30.0):
                if "command -v uv" in str(cmd):
                    return RunResult(str(cmd), 1, "no", "")
                return RunResult(str(cmd), 0, "", "")
        assert tools_mod.ensure_uv(NoUvRun(), lambda m: None, None) == \
            {"present": False}


class TestExecutorWiring:
    def test_real_plan_dispatches_to_tools(self, tmp_path, monkeypatch):
        # Full wiring: probe's install plan runs through the real executors
        # with stub downloader/run — steps land in summary, no failures.
        # Destinations redirected under tmp_path (never touch the real home).
        # WriteSettings steps use repo-relative config paths; redirect the
        # persistor under tmp_path so the repo's real config/*.yaml are
        # never mutated regardless of the cwd pytest runs from.
        real_persist = Executor.__dict__["_persist_setting"]

        def sandboxed_persist(path, key, value):
            cfg = tmp_path / path
            cfg.parent.mkdir(parents=True, exist_ok=True)
            cfg.touch()   # real persistor assumes an existing config file
            return real_persist(str(cfg), key, value)

        monkeypatch.setattr(Executor, "_persist_setting",
                            staticmethod(sandboxed_persist))
        from installer.probe import probe
        class DictFS:
            def __init__(self):
                self.files = {}
            def exists(self, p):
                return os.path.isfile(os.path.expanduser(p))
            def isfile(self, p):
                return os.path.isfile(os.path.expanduser(p))
            def expanduser(self, p):
                return os.path.expanduser(p)
            def read_file(self, p):
                return self.files.get(os.path.expanduser(p), "")
            def file_exists_and_runs(self, binary, version_arg="--version"):
                return RunResult(binary, 127, "", "missing")
        plan = probe(sysinfo(), DictFS(), RecordingRun(), role="main")
        # point every fixed dest inside tmp_path (incl. §6 settings-prefill
        # writes — their config/ paths are CWD-relative and would otherwise
        # hit the real repo config)
        for s in plan:
            d = getattr(s, "dest", None)
            if d and d.startswith("~"):
                s.dest = str(tmp_path / d.lstrip("~/"))
            if isinstance(s, probe_mod.WriteSettings):
                # §6 prefill steps carry repo-relative settings paths —
                # redirect under tmp_path so the executor's persist write
                # never touches the repo's real config/*.yaml.
                f = str(tmp_path / s.file)
                os.makedirs(os.path.dirname(f), exist_ok=True)
                s.file = f
        dl = StubDownloader()
        # seed the redirected settings files so _persist_setting can read them
        (tmp_path / "config").mkdir(exist_ok=True)
        (tmp_path / "config" / "settings.yaml").write_text(
            "model_dir: store://llm-models\n")
        (tmp_path / "config" / "swap.yaml").write_text(
            "listen_port: 8082\n")
        ex = Executor(log=lambda m: None, downloader=dl, info=sysinfo(),
                      manifest_path=str(tmp_path / "state.yaml"),
                      runner=ExtractingRun())
        summary = ex.execute(plan)
        assert summary["failures"] == []
        # probe plans RepairTool for missing-but-repairable binaries; both
        # step kinds run the same real executors
        done = set(summary["installed"]) | set(summary["repaired"])
        assert done >= {"llama.cpp", "llama-swap", "vllm"}
        # manifest accumulated per tool
        assert set(summary["manifest"]["tools"]) >= \
            {"llama.cpp", "llama-swap", "vllm"}
        # URLs hit the real asset maps (stubbed, no network)
        assert any("llama-swap_255_linux_amd64.tar.gz" in u for u in dl.urls)
        assert any("bin-ubuntu-cuda-13.3" in u for u in dl.urls)
        assert any("cudart-llama-b10934" in u or "cudart-llama-b1" in u
                   for u in dl.urls)

    def test_manifest_written_on_skip_only(self, tmp_path):
        # §5: manifest at end of EVERY run, incl. skip-only.
        class DictFS(TestExecutorWiring._present_fs()):
            pass
        plan = [probe_mod.SkipTool("llama.cpp", "present-current",
                                   HOME + "/llama.cpp/bin/llama-server",
                                   "b10934"),
                probe_mod.ManifestUpdate("llama.cpp", {"version": "b10934"})]
        ex = Executor(log=lambda m: None,
                      manifest_path=str(tmp_path / "state.yaml"))
        summary = ex.execute(plan)
        path = ex.write_manifest(summary)
        import yaml
        with open(path) as f:
            doc = yaml.safe_load(f)
        assert doc["installer_version"] == 1
        tools = doc["tools"]
        assert tools["llama.cpp"]["version"] == "b10934"
        assert "installed_at" in tools["llama.cpp"]

    @staticmethod
    def _present_fs():
        class _FS:
            def __init__(self):
                self.files = {}
            def exists(self, p):
                return os.path.isfile(os.path.expanduser(p))
            def isfile(self, p):
                return os.path.isfile(os.path.expanduser(p))
            def expanduser(self, p):
                return os.path.expanduser(p)
            def read_file(self, p):
                return ""
            def file_exists_and_runs(self, binary, version_arg="--version"):
                return RunResult(binary, 0, "ok", "")
        return _FS

    def test_manifest_written_after_failure_run(self, tmp_path):
        # §5: even a failing run records what DID land (advisory manifest;
        # probe wins on re-run). Steps before the failure are persisted.
        # A probe plan raising mid-iteration: execute() records the
        # failure envelope and returns; steps applied so far stay in the
        # summary (manifest write happens after execute in bootstrap).
        class BoomPlan(list):
            def __iter__(self):
                yield probe_mod.ManifestUpdate("llama-swap",
                                               {"version": "255"})
                raise RuntimeError("llama.cpp extraction failed")
        ex = Executor(log=lambda m: None,
                      manifest_path=str(tmp_path / "state.yaml"))
        summary = ex.execute(BoomPlan())
        assert summary["failures"]
        path = ex.write_manifest(summary)
        import yaml
        with open(path) as f:
            doc = yaml.safe_load(f)
        assert doc["tools"]["llama-swap"]["version"] == "255"
        # no secret material ever lands in the manifest
        assert "ssh_pass" not in str(doc) and "hf_token" not in str(doc)

    def test_failure_envelope_redacts_secrets(self):
        from installer.execute import failure_envelope
        env = failure_envelope(
            "installer", "cmd failed: export SSH_PASS=hunter2 && "
            "HF_TOKEN=abc123 curl -H 'Authorization: Bearer ***' ...")
        blob = str(env)
        assert "hunter2" not in blob and "abc123" not in blob \
            and "Bearer tok" not in blob
        assert "redacted" in blob

    def test_redact_authorization_value_dropped(self):
        # Regression: the old pattern redacted the KEY but left the secret
        # VALUE ('Bearer tok') in the envelope. The value is the secret.
        from installer.execute import _redact
        out = _redact("curl -H 'Authorization: Bearer sk-live-tok' -x y")
        assert "sk-live-tok" not in out and "Authorization:" in out
        assert "<redacted>" in out

    def test_downloader_retries_then_raises(self, monkeypatch):
        import urllib.request
        calls = []
        def boom(req, timeout=None):
            calls.append(1)
            raise OSError("no net")
        monkeypatch.setattr(urllib.request, "urlopen", boom)
        with pytest.raises(RuntimeError, match="download failed"):
            RealDownloader(retries=3)("http://example.invalid/x", "/tmp/x")
        assert len(calls) == 3


class TestSettingsPersisterRoundTrip:
    def test_comments_preserved(self, tmp_path):
        cfg = tmp_path / "settings.yaml"
        cfg.write_text("# my settings\nmodel_dir: store://x\n"
                       "llama_bin:\n# keep me\n")
        Executor._persist_setting(str(cfg), "llama_bin",
                                  HOME + "/llama.cpp/bin/llama-server")
        text = cfg.read_text()
        assert "# my settings" in text and "# keep me" in text
        assert "llama_bin: " in text


class TestWriteSettingsMergeRules:
    """§6 merge rules enforced by the WriteSettings executor step (t_7a9c63f8):
    comment-preserving via settings_persister, idempotent, never clobbers a
    user-set value pointing at a valid binary."""

    @staticmethod
    def _ws(tmp_path, key, value, reason, cfg="settings.yaml"):
        from installer.probe import WriteSettings
        p = tmp_path / cfg
        p.parent.mkdir(parents=True, exist_ok=True)
        p.touch()
        return WriteSettings(key=key, value=str(tmp_path / "bin" / "tool"),
                             file=str(p), merge_reason=reason)

    def _exec(self):
        ex = Executor(log=lambda m: None)
        return ex

    def test_empty_value_written_and_comments_kept(self, tmp_path):
        cfg = tmp_path / "settings.yaml"
        cfg.write_text("# top comment\nllama_bin:\n# trailing\nunrelated: 1\n")
        ws = self._ws(tmp_path, "llama_bin", "", "empty")
        ws.value = HOME + "/llama.cpp/bin/llama-server"
        self._exec()._write_setting(ws)
        text = cfg.read_text()
        assert "# top comment" in text and "# trailing" in text
        assert "llama_bin: " + HOME + "/llama.cpp/bin/llama-server" in text
        assert "unrelated: 1" in text

    def test_valid_user_value_never_clobbered(self, tmp_path):
        cfg = tmp_path / "settings.yaml"
        good = tmp_path / "my" / "llama-server"
        good.parent.mkdir(parents=True)
        good.write_text("#!/bin/sh\n")
        cfg.write_text(f"llama_bin: {good}\n")
        ws = self._ws(tmp_path, "llama_bin", "", "empty")
        ws.value = HOME + "/llama.cpp/bin/llama-server"
        ex = self._exec()
        ex._write_setting(ws)          # merge_reason empty, but value valid
        import yaml
        data = yaml.safe_load(cfg.read_text())
        assert data["llama_bin"] == str(good)

    def test_stale_value_repaired(self, tmp_path):
        cfg = tmp_path / "settings.yaml"
        gone = tmp_path / "old-place" / "llama-server"   # does not exist
        cfg.write_text(f"llama_bin: {gone}\n")
        ws = self._ws(tmp_path, "llama_bin", "", "stale")
        ws.value = HOME + "/llama.cpp/bin/llama-server"
        self._exec()._write_setting(ws)
        import yaml
        data = yaml.safe_load(cfg.read_text())
        assert data["llama_bin"] == HOME + "/llama.cpp/bin/llama-server"

    def test_idempotent_double_apply(self, tmp_path):
        cfg = tmp_path / "swap.yaml"
        cfg.write_text("binary:\nlisten_port: 8082\n")
        ws = self._ws(tmp_path, "binary", "", "empty", cfg="swap.yaml")
        ws.value = HOME + "/.local/bin/llama-swap"
        ex = self._exec()
        ex._write_setting(ws)
        first = cfg.read_text()
        ex._write_setting(ws)          # second run: value now valid → kept
        assert cfg.read_text() == first

    def test_dotted_key_vllm_venv_bin(self, tmp_path):
        cfg = tmp_path / "settings.yaml"
        cfg.write_text("vllm:\n  venv_bin:\n  # why comment\n")
        ws = self._ws(tmp_path, "vllm.venv_bin", "", "empty")
        ws.value = HOME + "/vllm-env/bin/vllm"
        self._exec()._write_setting(ws)
        text = cfg.read_text()
        assert "  venv_bin: " + HOME + "/vllm-env/bin/vllm" in text
        assert "# why comment" in text