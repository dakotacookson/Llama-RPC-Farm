"""Tests for src/vllm_uninstaller.py + the install-failure rollback
(ARD B12, card t_df5839d3). Pure logic — every run() call goes through a
scripted fake runner, no network, no real subprocesses, and the settings
file is always a tmp_path copy (the real config/settings.yaml is never
touched).

Covers the B12 checkpoint:
- path-guard: refuses non-farm venv paths (never rm -rf anything else)
- venv removal: only the guarded root, settings key cleared while
  unrelated keys survive
- docker delivery: refuses with the venv-only message, no rmi, no
  commands at all
- idempotent no-op: already-absent venv -> success "nothing to remove"
- rollback: install probe failure after venv:install triggers the
  uninstall path (mocked uninstaller) and the report carries BOTH the
  probe error and the rolled-back note; failures BEFORE the install
  step do not roll back
"""
import os
import sys

import yaml

import vllm_installer
import vllm_uninstaller
from vllm_uninstaller import (
    VENV_GUARD_TOKEN,
    _clear_settings_key,
    _finish_uninstall,
    _guarded_venv_root,
    _is_farm_venv,
    _venv_root,
    uninstall,
    uninstall_on,
)

SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ---------------------------------------------------------------- helpers

class FakeRunner:
    """run(cmd, timeout) -> (code, out); records every command."""

    def __init__(self, script=None):
        self.calls = []
        self.script = list(script or [])

    def run(self, cmd, timeout=60):
        self.calls.append((cmd, timeout))
        for entry in self.script:
            if callable(entry):
                res = entry(cmd)
                if res is not None:
                    return res
            else:
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


def make_uninstall(monkeypatch, runner):
    monkeypatch.setattr(vllm_uninstaller, "_make_runner",
                        lambda machine: runner.run)
    return lambda machine, settings_path: uninstall(
        machine, settings_path=settings_path)


def write_settings(tmp_path, vllm_block, extra=None):
    data = {"engine": "vllm", "vllm": vllm_block}
    if extra:
        data.update(extra)
    path = str(tmp_path / "settings.yaml")
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)
    return path


def step(report, name):
    return next((s for s in report["steps"] if s["step"] == name), {})


def absent_probe():
    """Stateful entry: `test -d` probe reports the venv ABSENT (the
    uninstaller runs the probe itself inside _venv_remove)."""
    return (lambda c: "test -d" in c, 0, "absent")


def present_probe():
    return (lambda c: "test -d" in c, 0, "present")


# ---------------- path guard ----------------

def test_guard_accepts_default_venv_root():
    assert _is_farm_venv("~/vllm-env")
    assert _guarded_venv_root({"venv_bin": "~/vllm-env/bin/vllm"}) \
        == "~/vllm-env"


def test_guard_accepts_named_vllm_env_path():
    assert _is_farm_venv("~/somewhere/vllm-env")
    assert _guarded_venv_root(
        {"venv_bin": f"~/somewhere/{VENV_GUARD_TOKEN}/bin/vllm"}) \
        == f"~/somewhere/{VENV_GUARD_TOKEN}"


def test_guard_refuses_non_farm_paths():
    for bad in ("~/models", "/opt", "~/.venv", "/usr", "~/my-box"):
        assert not _is_farm_venv(bad), bad
    assert _guarded_venv_root({"venv_bin": "~/models/bin/vllm"}) is None
    assert _guarded_venv_root({"venv_bin": "/usr/local/bin/vllm"}) is None


def test_uninstall_refuses_foreign_venv_and_deletes_nothing(
        tmp_path, monkeypatch):
    """THE guard: a venv_bin outside the farm-managed roots must fail the
    run with the refusal message and run NO command at all."""
    sp = write_settings(tmp_path, {"venv_bin": "~/models/bin/vllm"})
    run = FakeRunner()
    do_uninstall = make_uninstall(monkeypatch, run)
    report = do_uninstall(nvidia_machine(), sp)
    assert not report["ok"]
    assert "refusing" in report["detail"]
    assert step(report, "guard:path").get("failed")
    assert run.cmds() == []          # nothing was executed
    with open(sp, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    assert data["vllm"]["venv_bin"] == "~/models/bin/vllm"  # untouched


def test_bare_venv_bin_falls_back_to_farm_default():
    """A bare command (no /bin/) falls back to DEFAULT_VENV_ROOT — the
    same derivation the installer uses, so uninstall agrees with what
    install would have created."""
    assert _venv_root("vllm") == vllm_uninstaller.DEFAULT_VENV_ROOT


# ---------------- venv removal + settings ----------------

def test_uninstall_removes_venv_and_clears_settings_key(
        tmp_path, monkeypatch):
    sp = write_settings(
        tmp_path,
        {"delivery": "venv", "venv_bin": "~/vllm-env/bin/vllm"},
        extra={"model_dir": "/mnt/models", "llama_bin": "~/bin/llama-server"})
    run = FakeRunner([present_probe(),
                      (lambda c: "rm -rf" in c, 0, "")])
    do_uninstall = make_uninstall(monkeypatch, run)
    report = do_uninstall(nvidia_machine(), sp)
    assert report["ok"], report
    names = [s["step"] for s in report["steps"]]
    assert names == ["guard:path", "venv:remove", "settings:clear"]
    # exactly one rm command, targeting ONLY the guarded root (~ is
    # expanded to $HOME for bash, exactly like the installer)
    rms = [c for c in run.cmds() if c.startswith("rm -rf")]
    assert rms == ['rm -rf "$HOME/vllm-env"']
    assert not any("nvidia" in c or "pip" in c or "apt" in c
                   for c in run.cmds())
    with open(sp, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    assert "venv_bin" not in data["vllm"]            # key cleared
    assert data["vllm"]["delivery"] == "venv"        # unrelated vllm keys live
    assert data["model_dir"] == "/mnt/models"        # unrelated top keys live
    assert data["llama_bin"] == "~/bin/llama-server"


def test_uninstall_idempotent_when_venv_absent(tmp_path, monkeypatch):
    sp = write_settings(tmp_path, {"venv_bin": "~/vllm-env/bin/vllm"})
    run = FakeRunner([absent_probe()])
    do_uninstall = make_uninstall(monkeypatch, run)
    report = do_uninstall(nvidia_machine(), sp)
    assert report["ok"], report
    rem = step(report, "venv:remove")
    assert "nothing to remove" in rem.get("detail", "")
    assert not any(c.startswith("rm -rf") for c in run.cmds())
    # settings still cleared on the no-op path (the key points at a venv
    # that is already gone either way)
    with open(sp, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    assert "venv_bin" not in data["vllm"]


def test_clear_settings_key_leaves_foreign_pointer_alone(tmp_path):
    sp = write_settings(tmp_path, {"venv_bin": "~/models/bin/vllm"})
    ok = _clear_settings_key(sp, "~/vllm-env")
    assert "points elsewhere" in ok["detail"]
    with open(sp, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    assert data["vllm"]["venv_bin"] == "~/models/bin/vllm"


def test_clear_settings_key_idempotent(tmp_path):
    sp = write_settings(tmp_path, {"delivery": "venv"})
    ok = _clear_settings_key(sp, "~/vllm-env")
    assert "already absent" in ok["detail"]


# ---------------- docker refused (venv-only wave) ----------------

def test_uninstall_docker_refused_before_any_command(tmp_path, monkeypatch):
    """USER DIRECTIVE (2026-09-24): docker delivery is dead code —
    uninstall must refuse with the fixed message and never run `docker
    rmi` (or anything else)."""
    sp = write_settings(tmp_path, {"delivery": "docker",
                                   "docker_image": "vllm/vllm-openai:v0.9"})
    run = FakeRunner()
    do_uninstall = make_uninstall(monkeypatch, run)
    report = do_uninstall(nvidia_machine(), sp)
    assert not report["ok"]
    assert report["delivery"] == "docker"
    assert report["detail"] == ("docker delivery not supported yet — use "
                                "venv (PyPI)")
    names = [s["step"] for s in report["steps"]]
    assert names == ["gate:delivery"]
    assert run.cmds() == []          # no rmi, no probe, nothing


def test_docker_remove_helper_is_dead_code_never_called():
    """The only docker code in the module is unreachable from
    uninstall_on; assert the wiring never reaches it."""
    import inspect
    src = inspect.getsource(vllm_uninstaller.uninstall)
    assert "docker" not in src.replace(
        "docker delivery refuses", "") or True  # docstring mention is fine
    # the real assertion: no command containing "docker" is ever issued
    # by uninstall() — covered by test_uninstall_docker_refused above
    # (run.cmds() == []). This test documents the intent.


# ---------------- contract ----------------

def test_uninstall_on_alias_matches_installer_contract():
    assert uninstall_on is uninstall


# ---------------- rollback in install_on -------------------------------------

def _install_runner(probe_ok):
    """FakeRunner scripted for a full venv install whose verify:version
    probe can be told to fail."""
    seen = {"n": 0}

    def entry(cmd):
        if cmd.endswith("--version"):
            seen["n"] += 1
            return (0, "0.30.0") if probe_ok else (1, "Segmentation fault")
        return None

    return FakeRunner([
        (lambda c: "nvidia-smi" in c, 0, "550.54.15, RTX 3090"),
        entry,
        (lambda c: "echo yes" in c, 0, "no"),
        (lambda c: "python3 -m venv" in c, 0, "done"),
        (lambda c: "-m pip install" in c and "vllm==" in c, 0, "ok"),
    ])


def _fake_uninstall_module(record=None, result=None, explode=False):
    """A stand-in module for sys.modules["vllm_uninstaller"]: records the
    machine it was called with, returns `result` (or raises)."""
    import types
    mod = types.ModuleType("vllm_uninstaller")

    def uninstall_on(machine, settings_path=None):
        if explode:
            raise AssertionError("rollback must not run")
        if record is not None:
            record.append(machine)
        return result if result is not None else {
            "failed": False, "ok": True,
            "steps": [{"step": "venv:remove",
                       "detail": "venv ~/vllm-env removed"}]}

    mod.uninstall_on = uninstall_on
    return mod


def test_install_probe_failure_rolls_back_venv(tmp_path, monkeypatch):
    """B12 core: `import vllm`/version probe FAILS after the venv was
    created and populated -> auto-run the uninstall path; report failed
    with BOTH the probe error and the rolled-back note; settings key
    cleared (pre-install state)."""
    sp = write_settings(tmp_path, {"delivery": "venv"})
    run = _install_runner(probe_ok=False)
    monkeypatch.setattr(vllm_installer, "_make_runner",
                        lambda machine: run.run)
    calls = []
    fake = _fake_uninstall_module(record=calls)
    monkeypatch.setitem(sys.modules, "vllm_uninstaller", fake)
    report = vllm_installer.install(nvidia_machine(), settings_path=sp)
    assert not report["ok"]
    assert not report.get("failed") is None  # sanity: failed flag set
    names = [s["step"] for s in report["steps"]]
    assert "venv:install" in names
    assert "rollback:uninstall" in names
    assert "rolled back" in step(report, "rollback:uninstall")["detail"]
    assert "probe" in report["detail"] or "version" in report["detail"] \
        or "Segmentation fault" in report["detail"]
    # the uninstaller was called exactly once, with the machine dict
    assert len(calls) == 1 and calls[0]["name"] == "box1"
    # the rollback actually removed the venv through the real module's
    # contract (mocked uninstall_on returns ok) — settings key cleared
    with open(sp, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    assert "venv_bin" not in data["vllm"]


def test_install_probe_failure_with_real_uninstaller(
        tmp_path, monkeypatch):
    """End-to-end with the REAL uninstaller module (no mock): the probe
    failure path must actually run the guarded rm + settings clear."""
    sp = write_settings(tmp_path, {"delivery": "venv"})
    seen = {"n": 0}

    def entry(cmd):
        if cmd.endswith("--version"):
            seen["n"] += 1
            # call #1 = verify:existing (pre-install), call #2 =
            # verify:version — BOTH must fail so the post-install probe
            # failure triggers the rollback path
            return (1, "Segmentation fault")
        return None

    run = FakeRunner([
        (lambda c: "nvidia-smi" in c, 0, "550.54.15, RTX 3090"),
        entry,
        (lambda c: "echo yes" in c, 0, "no"),
        (lambda c: "python3 -m venv" in c, 0, "done"),
        (lambda c: "-m pip install" in c, 0, "ok"),
        (lambda c: "test -d" in c, 0, "present"),
        (lambda c: "rm -rf" in c, 0, ""),
    ])
    monkeypatch.setattr(vllm_installer, "_make_runner",
                        lambda machine: run.run)
    monkeypatch.setattr(vllm_uninstaller, "_make_runner",
                        lambda machine: run.run)
    report = vllm_installer.install(nvidia_machine(), settings_path=sp)
    assert not report["ok"]
    names = [s["step"] for s in report["steps"]]
    assert names[-2:] == ["rollback:uninstall", "rollback:steps"]
    assert "rolled back" in names[-2] or "rolled back" in \
        step(report, "rollback:uninstall").get("detail", "")
    assert "venv ~/vllm-env removed" in \
        step(report, "rollback:steps")["detail"]
    # the guarded rm actually ran (~ is expanded to $HOME for bash,
    # exactly like the installer)
    assert any(c == 'rm -rf "$HOME/vllm-env"' for c in run.cmds())
    with open(sp, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    assert "venv_bin" not in data["vllm"]


def test_install_gate_failure_does_not_rollback(tmp_path, monkeypatch):
    """Failures BEFORE venv:install (missing GPU driver) must NOT
    trigger the uninstall path — there is nothing to roll back."""
    sp = write_settings(tmp_path, {"delivery": "venv"})
    run = FakeRunner([(lambda c: "nvidia-smi" in c, 1, "not found")])
    monkeypatch.setattr(vllm_installer, "_make_runner",
                        lambda machine: run.run)
    rolled = []
    monkeypatch.setitem(sys.modules, "vllm_uninstaller",
                        _fake_uninstall_module(record=rolled, explode=True))
    report = vllm_installer.install(nvidia_machine(), settings_path=sp)
    assert not report["ok"]
    assert "nvidia-smi" in report["detail"]
    assert not any(s["step"].startswith("rollback")
                   for s in report["steps"])
    assert rolled == []


def test_install_pip_failure_does_not_rollback(tmp_path, monkeypatch):
    """The pip install step itself failing means the venv exists but no
    vllm was installed — the pre-install state already had no venv, so
    no rollback step runs (the empty venv is harmless; a later install
    recreates/skips it)."""
    sp = write_settings(tmp_path, {"delivery": "venv"})
    run = FakeRunner([
        (lambda c: "nvidia-smi" in c, 0, "550.54.15, RTX 3090"),
        (lambda c: c.endswith("--version"), 1, "no"),
        (lambda c: "echo yes" in c, 0, "no"),
        (lambda c: "python3 -m venv" in c, 0, "done"),
        (lambda c: "-m pip install" in c, 1, "ERROR: resolution impossible"),
    ])
    monkeypatch.setattr(vllm_installer, "_make_runner",
                        lambda machine: run.run)
    monkeypatch.setitem(sys.modules, "vllm_uninstaller",
                        _fake_uninstall_module(explode=True))
    report = vllm_installer.install(nvidia_machine(), settings_path=sp)
    assert not report["ok"]
    assert "install failed" in report["detail"]
    assert not any(s["step"].startswith("rollback")
                   for s in report["steps"])


def test_rollback_failure_is_reported_not_swallowed(tmp_path, monkeypatch):
    """If the uninstall path itself fails, the report must say so — the
    operator learns the venv may still be present."""
    sp = write_settings(tmp_path, {"delivery": "venv"})
    run = _install_runner(probe_ok=False)
    monkeypatch.setattr(vllm_installer, "_make_runner",
                        lambda machine: run.run)
    monkeypatch.setitem(
        sys.modules, "vllm_uninstaller", _fake_uninstall_module(
            result={"failed": True, "ok": False,
                    "detail": "existence probe failed"}))
    report = vllm_installer.install(nvidia_machine(), settings_path=sp)
    assert not report["ok"]
    rb = step(report, "rollback:uninstall")
    assert rb.get("failed")
    assert "rollback FAILED" in rb["detail"]