"""Tests for src/llamacpp_uninstaller.py (DEC5 / T5-1, B12 parity).
Pure logic — every run() call goes through a scripted fake runner, no
network, no real subprocesses, no machine records touched outside tmp.

Covers:
- path-guard: refuses foreign llama_bin paths (never rm anything else)
- guarded removal: only the installer's bin dir, llama_bin cleared
- idempotent no-op: already-absent bin dir -> success "nothing to remove"
- windows path shape (PowerShell commands)
- endpoint wiring is additive (route exists, machine 404)
"""
import os
import sys

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import llamacpp_uninstaller as un
from llamacpp_uninstaller import (
    _configured_bin_dir,
    _is_farm_bin_dir,
    uninstall,
    uninstall_on,
)


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


def machine(**over):
    m = {"name": "box1", "os": "linux", "gpu_vendor": "nvidia",
         "llama_bin": "~/llama.cpp/bin"}
    m.update(over)
    return m


def make_uninstall(monkeypatch, runner):
    monkeypatch.setattr(un, "_make_runner", lambda m: runner.run)
    return lambda machine: uninstall(machine)


def step(report, name):
    return next((s for s in report["steps"] if s["step"] == name), {})


def present_file():
    return (lambda c: "test -f" in c, 0, "present")


def absent_file():
    return (lambda c: "test -f" in c, 0, "absent")


def present_dir():
    return (lambda c: "test -d" in c, 0, "present")


def absent_dir():
    return (lambda c: "test -d" in c, 0, "absent")


# ---------------- path guard ----------------

def test_guard_accepts_installer_dest():
    assert _is_farm_bin_dir("~/llama.cpp/bin", machine())
    assert _is_farm_bin_dir(_configured_bin_dir(machine()), machine())


def test_guard_refuses_named_but_foreign_bin_dir():
    """The installer writes ONLY its own dest — an install in a foreign
    parent (even named llama.cpp) is refused, same as a source build."""
    for bad in ("~/somewhere/llama.cpp/bin",
                "~/llama.cpp/build/bin",
                "~/llama.cpp",
                "~/models", "/usr"):
        m = machine(llama_bin=bad)
        assert _is_farm_bin_dir(bad, m) is False, bad
        assert un._guarded_bin_dir(m) is None, bad


def test_guard_refuses_foreign_and_source_paths():
    for bad in ("~/models", "/usr", "~/.local",
                "~/llama.cpp/build/bin/llama-server",
                "~/src/llama.cpp/build/bin/llama-server",
                "~/somewhere/llama.cpp/bin/llama-server"):
        m = machine(llama_bin=bad)
        assert un._guarded_bin_dir(m) is None, bad


def test_uninstall_refuses_foreign_path_and_deletes_nothing(monkeypatch):
    """THE guard: a llama_bin outside the farm-managed install dir must
    fail the run with the refusal message and run NO command at all."""
    run = FakeRunner()
    do_uninstall = make_uninstall(monkeypatch, run)
    report = do_uninstall(machine(llama_bin="~/models/llama.cpp"))
    assert not report["ok"]
    assert "refusing" in report["detail"]
    assert step(report, "guard:path").get("failed")
    assert run.cmds() == []          # nothing was executed


def test_uninstall_refuses_no_pointer(monkeypatch):
    run = FakeRunner()
    do_uninstall = make_uninstall(monkeypatch, run)
    report = do_uninstall(machine(llama_bin=""))
    assert not report["ok"]
    assert run.cmds() == []


# ---------------- guarded removal + machines.clear ----------------

def stub_machines_update(monkeypatch, seen=None):
    """Stub the config layer's machines.update (no real config touched);
    `seen` records the (name, updates) call."""
    import config_loader as cfg
    seen = seen if seen is not None else {}

    def fake_update(name, updates):
        seen["name"] = name
        seen["updates"] = updates

    monkeypatch.setattr(cfg.machines, "update", fake_update)
    return seen


def test_uninstall_removes_bin_and_clears_llama_bin(tmp_path, monkeypatch):
    seen = stub_machines_update(monkeypatch)
    run = FakeRunner([present_file(),
                      (lambda c: "--version" in c, 0, "b1234"),
                      present_dir(),
                      (lambda c: c.startswith("rm -rf"), 0, "")])
    do_uninstall = make_uninstall(monkeypatch, run)
    report = do_uninstall(machine())
    assert report["ok"], report
    names = [s["step"] for s in report["steps"]]
    assert names == ["guard:path", "verify:version", "bins:remove",
                     "machines:clear"]
    rms = [c for c in run.cmds() if c.startswith("rm -rf")]
    assert rms == ['rm -rf "$HOME/llama.cpp/bin"']
    assert seen == {"name": "box1", "updates": {"llama_bin": ""}}
    assert not any("apt" in c or "pip" in c or "docker" in c
                   for c in run.cmds())


def test_uninstall_idempotent_when_bin_absent(monkeypatch):
    stub_machines_update(monkeypatch)
    run = FakeRunner([absent_file()])
    do_uninstall = make_uninstall(monkeypatch, run)
    report = do_uninstall(machine())
    assert report["ok"], report
    rem = step(report, "bins:remove")
    assert "nothing to remove" in rem.get("detail", "")
    assert not any(c.startswith("rm -rf") for c in run.cmds())


def test_uninstall_version_probe_failure_stops_before_removal(monkeypatch):
    run = FakeRunner([present_file(),
                      (lambda c: "--version" in c, 1, "Segmentation fault")])
    do_uninstall = make_uninstall(monkeypatch, run)
    report = do_uninstall(machine())
    assert not report["ok"]
    assert not any(c.startswith("rm -rf") for c in run.cmds())


# ---------------- windows shape ----------------

def test_windows_uninstall_uses_powershell(monkeypatch):
    stub_machines_update(monkeypatch)
    run = FakeRunner([
        (lambda c: "Test-Path" in c, 0, "present"),
        (lambda c: "--version" in c, 0, "b1234"),
        (lambda c: "Remove-Item" in c, 0, ""),
    ])
    do_uninstall = make_uninstall(monkeypatch, run)
    report = do_uninstall(machine(os="windows",
                                  llama_bin="%USERPROFILE%\\llama.cpp\\bin"))
    assert report["ok"], report
    assert any("Remove-Item" in c for c in run.cmds())
    assert not any(c.startswith("rm -rf") for c in run.cmds())


def test_windows_guard_accepts_installer_dest():
    m = machine(os="windows", llama_bin="%USERPROFILE%\\llama.cpp\\bin")
    assert un._guarded_bin_dir(m) == "%USERPROFILE%\\llama.cpp\\bin"


# ---------------- contract ----------------

def test_uninstall_on_alias_matches_installer_contract():
    assert uninstall_on is uninstall


# ---------------- endpoint wiring (additive route) ----------------

