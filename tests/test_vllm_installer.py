"""Tests for src/vllm_installer.py (ARD B6): gates, delivery paths, and
report rollup. Pure logic — every run() call goes through a scripted
fake runner, no network, no real subprocesses.

Covers the B6 checkpoint:
- OS gating table: windows refusal (WSL2 guidance), macOS pointer,
  linux allowed
- ROCm: python 3.12-only gate + post-install HIP backend probe (silent
  CUDA fallback guard)
- skip-if-present (venv `vllm --version` ok / docker image present)
- venv happy path + docker happy path (step names + settings update)
- report rollup (_finish): ok/failed/detail from step semantics
"""
import os

import pytest
import yaml

import vllm_installer
from vllm_installer import (
    VLLM_VERSION_PIN,
    _detect_present,
    _gate_docker,
    _gate_gpu,
    _gate_os,
    _gate_python_312,
    _finish,
    install,
    install_on,
    plan,
)

SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ---------------------------------------------------------------- helpers

class FakeRunner:
    """run(cmd, timeout) -> (code, out) driven by a script: a list of
    (predicate, code, out) checked in order; the first predicate that
    matches wins. Unmatched commands return (0, "")."""

    def __init__(self, script=None):
        self.calls = []
        self.script = list(script or [])

    def run(self, cmd, timeout=60):
        self.calls.append((cmd, timeout))
        for entry in self.script:
            if callable(entry):
                # stateful entry: returns (code, out) or None to pass
                res = entry(cmd)
                if res is not None:
                    return res
                continue
            pred, code, out = entry
            if pred(cmd):
                return (code, out)
        return (0, "")

    def cmds(self):
        return [c for c, _ in self.calls]


def nvidia_machine(**over):
    m = {"name": "box1", "os": "linux", "gpu_vendor": "nvidia"}
    m.update(over)
    return m


def amd_machine(**over):
    m = {"name": "rocm1", "os": "linux", "gpu_vendor": "amd"}
    m.update(over)
    return m


def make_install(monkeypatch, runner):
    """install() with _make_runner patched to the fake runner — mirrors
    the house monkeypatch style; the production signature is unchanged."""
    monkeypatch.setattr(vllm_installer, "_make_runner",
                        lambda machine: runner.run)
    return lambda machine, settings_path: install(
        machine, settings_path=settings_path)


def write_settings(tmp_path, vllm_block):
    path = str(tmp_path / "settings.yaml")
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump({"engine": "vllm", "vllm": vllm_block}, f,
                       sort_keys=False)
    return path


def step(report, name):
    return next((s for s in report["steps"] if s["step"] == name), {})


def version_probe():
    """Stateful FakeRunner entry for the identical `... --version`
    commands: the skip-if-present probe (1st call) fails, the
    post-install verify:version (2nd call) succeeds."""
    seen = {"n": 0}

    def entry(cmd):
        if cmd.endswith("--version"):
            seen["n"] += 1
            return (0, "0.18.0") if seen["n"] > 1 else (1, "no")
        return None

    return entry


def docker_inspect():
    """Stateful FakeRunner entry for the identical `docker image inspect`
    commands: skip probe (1st call) -> image missing, post-pull verify
    (2nd call) -> image present."""
    seen = {"n": 0}

    def entry(cmd):
        if "docker image inspect" in cmd:
            seen["n"] += 1
            return (0, "[]") if seen["n"] > 1 else (1, "no such image")
        return None

    return entry


# ---------------- OS gating table ----------------

def test_windows_refused_with_wsl2_guidance():
    r = _gate_os({"os": "windows"})
    assert r.get("failed")
    assert "WSL2" in r["detail"]


def test_macos_refused_with_vllm_metal_pointer():
    r = _gate_os({"os": "macos"})
    assert r.get("failed")
    assert "vLLM-Metal" in r["detail"] or "Metal" in r["detail"]


def test_linux_allowed():
    assert not _gate_os({"os": "linux"}).get("failed")


def test_missing_os_defaults_to_linux():
    assert not _gate_os({}).get("failed")


# ---------------- GPU gates ----------------

def test_nvidia_gate_requires_nvidia_smi():
    run = FakeRunner([(lambda c: "nvidia-smi" in c, 1, "not found")])
    r = _gate_gpu(nvidia_machine(), run.run)
    assert r.get("failed")
    assert "nvidia-smi" in r["detail"] or "driver" in r["detail"]


def test_nvidia_gate_ok_when_driver_answers():
    run = FakeRunner([(lambda c: "nvidia-smi" in c, 0,
                       "550.54.15, NVIDIA GeForce RTX 3090")])
    r = _gate_gpu(nvidia_machine(), run.run)
    assert not r.get("failed")
    assert "550.54.15" in r["detail"]


def test_amd_gate_requires_rocm_tooling():
    run = FakeRunner([(lambda c: "rocm-smi" in c or "amd-smi" in c,
                       1, "command not found")])
    r = _gate_gpu(amd_machine(), run.run)
    assert r.get("failed")


def test_amd_gate_ok_with_rocm_smi():
    run = FakeRunner([(lambda c: "rocm-smi" in c or "amd-smi" in c,
                       0, "/usr/bin/rocm-smi")])
    assert not _gate_gpu(amd_machine(), run.run).get("failed")


def test_intel_and_unset_refused():
    r = _gate_gpu(nvidia_machine(gpu_vendor="intel"), FakeRunner().run)
    assert r.get("failed")
    assert "Intel" in r["detail"]
    r = _gate_gpu(nvidia_machine(gpu_vendor=""), FakeRunner().run)
    assert r.get("failed")


# ---------------- ROCm python 3.12 gate ----------------

def test_rocm_python_312_required():
    run = FakeRunner([(lambda c: "python3 --version" in c, 0,
                       "Python 3.10.12")])
    r = _gate_python_312(run.run)
    assert r.get("failed")
    assert "3.12" in r["detail"]


def test_rocm_python_312_ok():
    run = FakeRunner([(lambda c: "python3 --version" in c, 0,
                       "Python 3.12.3")])
    assert not _gate_python_312(run.run).get("failed")


# ---------------- skip-if-present ----------------

def test_venv_skip_when_vllm_version_answers():
    run = FakeRunner([(lambda c: "--version" in c, 0, "0.18.0")])
    r = _detect_present({"venv_bin": "~/vllm-env/bin/vllm"}, run.run)
    assert r.get("skipped")
    assert "already installed" in r["detail"]


def test_venv_not_present_when_binary_missing():
    run = FakeRunner([(lambda c: "--version" in c, 127, "")])
    r = _detect_present({"venv_bin": "~/vllm-env/bin/vllm"}, run.run)
    assert not r.get("skipped")


def test_docker_delivery_not_probed_instead_of_skipped():
    """venv-only wave (USER DIRECTIVE 2026-09-24): delivery: docker is
    dead code — the skip probe must NOT report an existing image as a
    usable install (skipped=False), it reports the refusal detail."""
    run = FakeRunner([(lambda c: "docker image inspect" in c, 0, "")])
    r = _detect_present({"delivery": "docker"}, run.run)
    assert not r.get("skipped")
    assert "not supported" in r["detail"]
    # no probe command ran at all
    assert run.cmds() == []


# ---------------- venv install flow (mock runner) ----------------

def test_install_venv_happy_path_updates_settings(tmp_path, monkeypatch):
    """Full venv delivery: gates pass, venv created, pinned wheel
    installed, version smokes, settings vllm.venv_bin pointed at the
    installed binary."""
    sp = write_settings(tmp_path, {"delivery": "venv"})
    run = FakeRunner([
        (lambda c: "nvidia-smi" in c, 0, "550.54.15, RTX 3090"),
        version_probe(),
        (lambda c: "echo yes" in c, 0, "no"),
        (lambda c: "python3 -m venv" in c, 0, "done"),
        (lambda c: "-m pip install" in c and "vllm==" in c, 0, "ok"),
    ])
    do_install = make_install(monkeypatch, run)
    report = do_install(nvidia_machine(), sp)
    assert report["ok"], report
    assert report["delivery"] == "venv"
    assert not report.get("skipped")
    assert report["target"].endswith("/bin/vllm")
    names = [s["step"] for s in report["steps"]]
    assert names == ["gate:os", "verify:existing", "gate:gpu",
                     "venv:create", "venv:install", "verify:version",
                     "settings:update"]
    # pinned version on the command line
    assert any(f"vllm=={VLLM_VERSION_PIN}" in c for c in run.cmds())
    # settings updated to the installed binary
    with open(sp, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    assert data["vllm"]["venv_bin"] == report["target"]


def test_install_skips_when_already_present(tmp_path, monkeypatch):
    sp = write_settings(tmp_path, {"delivery": "venv"})
    run = FakeRunner([
        (lambda c: "--version" in c, 0, "0.18.0"),
    ])
    do_install = make_install(monkeypatch, run)
    report = do_install(nvidia_machine(), sp)
    assert report["ok"]
    assert report.get("skipped") is True
    # no install commands ran (no venv creation, no pip)
    assert not any("python3 -m venv" in c for c in run.cmds())
    assert not any("pip install" in c for c in run.cmds())


def test_install_windows_refused_before_any_command(tmp_path, monkeypatch):
    sp = write_settings(tmp_path, {"delivery": "venv"})
    run = FakeRunner()
    do_install = make_install(monkeypatch, run)
    report = do_install(nvidia_machine(os="windows"), sp)
    assert not report["ok"]
    assert "WSL2" in report["detail"]
    # refused before any command ran
    assert run.cmds() == []


def test_install_rocm_happy_path_runs_backend_probe(tmp_path, monkeypatch):
    """AMD host: python 3.12 gate + ROCm wheel index + the post-install
    `import vllm` backend probe asserting HIP (silent-CUDA-fallback
    guard)."""
    sp = write_settings(tmp_path, {"delivery": "venv"})
    run = FakeRunner([
        (lambda c: "rocm-smi" in c or "amd-smi" in c, 0, "/usr/bin/rocm-smi"),
        (lambda c: "python3 --version" in c, 0, "Python 3.12.3"),
        version_probe(),
        (lambda c: "echo yes" in c, 0, "yes"),
        (lambda c: "uv venv" in c, 0, "done"),
        (lambda c: "uv pip install" in c, 0, "ok"),
        (lambda c: "-c" in c and "import vllm" in c, 0,
         "vllm 0.18.0 backend HIP"),
    ])
    do_install = make_install(monkeypatch, run)
    report = do_install(amd_machine(), sp)
    assert report["ok"], report
    names = [s["step"] for s in report["steps"]]
    assert "gate:python" in names
    assert "verify:backend" in names
    installs = [c for c in run.cmds() if "install" in c and "vllm==" in c]
    assert installs and "wheels.vllm.ai/rocm" in installs[0]
    assert not any("--torch-backend=auto" in c for c in installs)


def test_install_rocm_silent_cuda_fallback_fails(tmp_path, monkeypatch):
    """The CUDA wheel imports fine but reports backend CUDA — must fail
    the install, not pass silently."""
    sp = write_settings(tmp_path, {"delivery": "venv"})
    run = FakeRunner([
        (lambda c: "rocm-smi" in c or "amd-smi" in c, 0, "/usr/bin/rocm-smi"),
        (lambda c: "python3 --version" in c, 0, "Python 3.12.3"),
        version_probe(),
        (lambda c: "echo yes" in c, 0, "yes"),
        (lambda c: "uv venv" in c, 0, "done"),
        (lambda c: "uv pip install" in c, 0, "ok"),
        (lambda c: "-c" in c and "import vllm" in c, 0,
         "vllm 0.18.0 backend CUDA"),
    ])
    do_install = make_install(monkeypatch, run)
    report = do_install(amd_machine(), sp)
    assert not report["ok"]
    assert "silent CUDA fallback" in report["detail"]


# ---------------- docker delivery refused (venv-only wave) ----------------

def test_install_docker_refused_before_any_command(tmp_path, monkeypatch):
    """USER DIRECTIVE (2026-09-24): only the PyPI venv path is supported
    this wave. delivery: docker refuses cleanly (ok=False, the fixed
    refusal message) BEFORE any command runs — no gates, no pull."""
    sp = write_settings(tmp_path, {"delivery": "docker",
                                   "docker_image": "vllm/vllm-openai:v0.9"})
    run = FakeRunner()
    do_install = make_install(monkeypatch, run)
    report = do_install(nvidia_machine(), sp)
    assert not report["ok"]
    assert report["delivery"] == "docker"
    assert report["detail"] == ("docker delivery not supported yet — use "
                                "venv (PyPI)")
    names = [s["step"] for s in report["steps"]]
    assert names == ["gate:delivery"]
    assert run.cmds() == []  # refused before any command ran


def test_plan_docker_refused(monkeypatch):
    """plan() refuses delivery: docker with the same fixed message —
    no OS/GPU gates consulted, no commands possible (plan never runs
    any, but the refusal short-circuits before the gate read)."""
    monkeypatch.setattr(vllm_installer, "_vllm_block",
                        lambda settings_path=None: {"delivery": "docker"})
    p = plan(nvidia_machine())
    assert not p["ok"]
    assert p["delivery"] == "docker"
    assert p["error"] == ("docker delivery not supported yet — use "
                          "venv (PyPI)")


# ---------------- plan() ----------------

def test_plan_venv_and_os_gate():
    # pin settings_path to a nonexistent file — without it plan() reads the
    # repo's REAL config/settings.yaml, which concurrent sibling runs
    # rewrite at runtime (cross-worktree interference flake, UX-3 finding).
    p = plan(nvidia_machine(), settings_path="/tmp/nonexistent-ux3-settings.yaml")
    assert p["ok"]
    assert p["delivery"] == "venv"
    assert p["version"] == VLLM_VERSION_PIN
    assert p["target"] == "~/vllm-env/bin/vllm"
    assert "opt-in" in p["gguf_plugin"]


def test_plan_windows_reports_error():
    p = plan(nvidia_machine(os="windows"))
    assert not p["ok"]
    assert "WSL2" in p["error"]


# ---------------- report rollup ----------------

def test_finish_marks_first_failure_detail():
    report = {"steps": [{"step": "gate:os"},
                        {"step": "gate:gpu", "failed": True,
                         "detail": "nvidia-smi missing"}],
              "failed": True}
    out = _finish(report)
    assert out["ok"] is False
    assert out["detail"] == "nvidia-smi missing"


def test_finish_ok_report():
    out = _finish({"steps": [{"step": "gate:os"},
                             {"step": "venv:create"}], "failed": False})
    assert out["ok"] is True
    assert out["detail"] == "install complete"


def test_finish_skipped_report():
    out = _finish({"steps": [{"step": "verify:existing", "skipped": True}],
                   "failed": False, "skipped": True})
    assert out["ok"] is True
    assert "already present" in out["detail"]


# ---------------- contract ----------------

def test_install_on_alias_matches_llamacpp_contract():
    assert install_on is install

# ---------------- B13: per-machine delivery method matching -----------------
# Fit matrix: plan(machine) is the pre-flight readout the UI shows in the
# Install-vLLM tooltip, so every row is asserted via plan() on fixtures —
# REAL asserts, no fakes beyond the scripted runner / settings block.

def test_b13_nvidia_default_venv_ok_with_reason():
    """nvidia linux + no override -> venv (global default), ok, reason
    names the global default. settings_path pinned to a nonexistent file
    so the repo's live settings.yaml (rewritten by concurrent sibling
    runs) can never flip the venv_bin assertion."""
    p = plan(nvidia_machine(),
             settings_path="/tmp/nonexistent-ux3-settings.yaml")
    assert p["ok"] is True
    assert p["delivery"] == "venv"
    assert p["method_reason"] == "global default (venv)"
    assert p["target"] == "~/vllm-env/bin/vllm"
    assert p["version"] == VLLM_VERSION_PIN


def test_b13_nvidia_docker_override_accepted_but_refused():
    """nvidia linux + vllm_delivery: docker -> the override IS read and
    routed (plan reports delivery docker + the machine-override reason)
    but REFUSED with the directive message — venv-only wave."""
    p = plan(nvidia_machine(vllm_delivery="docker"))
    assert p["ok"] is False
    assert p["delivery"] == "docker"
    assert p["method_reason"] == "machine override (vllm_delivery: docker)"
    assert p["error"] == ("docker delivery not supported yet — use venv "
                          "(PyPI)")
    assert p["target"] is None


def test_b13_rocm_py311_venv_refused_not_silently_cuda_wheeled():
    """amd (ROCm) host reporting python 3.11 -> the existing py3.12 gate
    still REFUSES the venv install (never silently CUDA-wheeled). The
    plan's method itself is venv (no override), so the refusal must come
    from install()'s gate:python step."""
    monkey = pytest.MonkeyPatch()
    try:
        monkeypatch_311 = FakeRunner(script=[
            (lambda cmd: "python3 --version" in cmd,
             0, "Python 3.11.8"),
            (lambda cmd: "rocm-smi" in cmd or "amd-smi" in cmd,
             0, "6.2"),
        ])
        monkey.setattr(vllm_installer, "_make_runner",
                       lambda machine: monkeypatch_311.run)
        report = install(amd_machine(), settings_path=None)
    finally:
        monkey.undo()
    assert report["ok"] is False
    assert report["delivery"] == "venv"
    assert report["method_reason"] == "global default (venv)"
    assert any(s["step"] == "gate:python" and s.get("failed")
               for s in report["steps"])
    assert "Python 3.12-only" in report["detail"]
    # refused before touching the venv
    assert not any(s["step"] == "venv:install" for s in report["steps"])


def test_b13_rocm_py311_docker_override_routed_to_refusal():
    """amd (ROCm) + py311 + vllm_delivery: docker -> the override wins
    over the global (method_reason proves it was READ) but docker is
    refused in this wave — the refusal, not the py3.12 gate, fires."""
    p = plan(amd_machine(vllm_delivery="docker"))
    assert p["ok"] is False
    assert p["delivery"] == "docker"
    assert p["method_reason"] == "machine override (vllm_delivery: docker)"
    assert p["error"] == ("docker delivery not supported yet — use venv "
                          "(PyPI)")


def test_b13_cpu_only_refused():
    """cpu-only vendor -> venv plan still carries the OS ok but the
    capability gate is the install's; plan() itself reports the venv
    method while install() refuses via gate:gpu (no nvidia-smi/rocm-smi).
    The machine-fit answer must be a clean refusal, not a CUDA wheel."""
    monkey = pytest.MonkeyPatch()
    try:
        cpu_runner = FakeRunner(script=[
            (lambda cmd: "nvidia-smi" in cmd, 127, ""),
            (lambda cmd: "rocm-smi" in cmd or "amd-smi" in cmd, 127, ""),
        ])
        monkey.setattr(vllm_installer, "_make_runner",
                       lambda machine: cpu_runner.run)
        report = install(nvidia_machine(gpu_vendor="cpu"),
                         settings_path=None)
    finally:
        monkey.undo()
    assert report["ok"] is False
    assert report["delivery"] == "venv"
    assert any(s["step"] == "gate:gpu" and s.get("failed")
               for s in report["steps"])
    assert not any(s["step"] == "venv:install" for s in report["steps"])


def test_b13_windows_refused_even_with_venv_override():
    """windows -> refused at gate:os regardless of delivery; the plan
    reports the WSL2 guidance and ok=False."""
    p = plan(nvidia_machine(os="windows"))
    assert p["ok"] is False
    assert p["delivery"] == "venv"
    assert "WSL2" in p["error"]


def test_b13_venv_override_wins_over_global_docker():
    """The reverse pin: settings global docker (which would refuse) but
    the machine override venv -> the OVERRIDE wins and the machine gets
    a working venv plan."""
    monkey = pytest.MonkeyPatch()
    try:
        monkey.setattr(vllm_installer, "_vllm_block",
                       lambda settings_path=None: {"delivery": "docker"})
        p = plan(nvidia_machine(vllm_delivery="venv"))
    finally:
        monkey.undo()
    assert p["ok"] is True
    assert p["delivery"] == "venv"
    assert p["method_reason"] == "machine override (vllm_delivery: venv)"


def test_b13_bad_override_value_fails_plan():
    """A typo'd override ('poddman') must FAIL the plan loudly — never a
    silent fallback to the global delivery."""
    p = plan(nvidia_machine(vllm_delivery="poddman"))
    assert p["ok"] is False
    assert p["delivery"] is None
    assert "invalid machine vllm_delivery" in p["error"]


def test_b13_install_routes_override_docker_to_refusal(monkeypatch):
    """install() with a docker override refuses before ANY command — the
    override is routed to the refusal, not silently ignored."""
    run = FakeRunner()
    do_install = make_install(monkeypatch, run)
    report = do_install(nvidia_machine(vllm_delivery="docker"), None)
    assert report["ok"] is False
    assert report["delivery"] == "docker"
    assert report["method_reason"] == ("machine override "
                                       "(vllm_delivery: docker)")
    assert report["detail"] == ("docker delivery not supported yet — use "
                                "venv (PyPI)")
    assert [s["step"] for s in report["steps"]] == ["gate:delivery"]
    assert run.cmds() == []


def test_b13_install_reports_method_reason_on_venv_path(monkeypatch,
                                                        tmp_path):
    """install() happy path: the report carries delivery + method_reason
    so the caller sees WHY venv was chosen."""
    sp = write_settings(tmp_path, {"venv_bin": "~/vllm-env/bin/vllm"})
    run = FakeRunner(script=[
        (lambda cmd: "nvidia-smi" in cmd, 0, "550.54.15, RTX 3090"),
        version_probe(),
        (lambda cmd: "python3 -m venv" in cmd, 0, "done"),
        (lambda cmd: "-m pip install" in cmd and "vllm==" in cmd, 0, "ok"),
    ])
    do_install = make_install(monkeypatch, run)
    report = do_install(nvidia_machine(), sp)
    assert report["ok"] is True
    assert report["delivery"] == "venv"
    assert report["method_reason"] == "global default (venv)"
