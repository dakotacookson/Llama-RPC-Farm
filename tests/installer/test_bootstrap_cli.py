# Hermetic CLI tests for src/bootstrap.py (design §4: report-setup redaction,
# roles wiring, --verbose tee to install.log). No network, no real probing —
# the CLI's world is injected via monkeypatched env/probe/execute symbols.

import importlib.util
import os
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _load_bootstrap():
    path = os.path.join(REPO, "src", "bootstrap.py")
    spec = importlib.util.spec_from_file_location("bootstrap_under_test", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["bootstrap_under_test"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture()
def bs(monkeypatch):
    mod = _load_bootstrap()
    # Inert stubs: nothing real is run or touched unless a test injects it.
    monkeypatch.setattr(mod, "RealRunCmd", lambda: object())
    monkeypatch.setattr(mod, "RealFS", lambda: object())
    return mod


# --- report-setup: redaction (design §4 — no paths with usernames) ---------

def test_report_setup_redacts_home_prefix(bs, monkeypatch, capsys):
    info = _fake_info()
    monkeypatch.setattr(bs, "detect_system_info", lambda run: info)
    # Simulate a found binary under $HOME; the printed path must not leak it.
    home = os.path.expanduser("~")
    monkeypatch.setattr(os.path, "isfile", lambda p: p == os.path.join(
        home, ".local/bin/llama-swap"))
    rc = bs.main(["report-setup"])
    out = capsys.readouterr().out
    assert rc == 0
    assert home not in out
    assert "FOUND ~/.local/bin/llama-swap" in out
    assert "llama-rpc-farm setup report" in out


def test_report_setup_not_found(bs, monkeypatch, capsys):
    monkeypatch.setattr(bs, "detect_system_info", lambda run: _fake_info())
    monkeypatch.setattr(os.path, "isfile", lambda p: False)
    assert bs.main(["report-setup"]) == 0
    out = capsys.readouterr().out
    assert "llama-swap: not found" in out
    assert "vllm: not found" in out


def test_report_setup_no_redaction_outside_home(bs, monkeypatch, capsys):
    monkeypatch.setattr(bs, "detect_system_info", lambda run: _fake_info())
    outside = "/opt/lcpp/llama-server"
    monkeypatch.setattr(os.path, "isfile", lambda p: p == outside)
    assert bs.main(["report-setup"]) == 0
    # /opt paths are not candidates, so verify _redact itself keeps them whole.
    assert bs._redact(outside) == outside
    # a path under a DIFFERENT home is untouched; only the runtime user's
    # home prefix is rewritten (correct: report shows this machine only).
    assert bs._redact("/home/someone/x") == "/home/someone/x"
    out = capsys.readouterr().out
    assert os.path.expanduser("~") not in out


def _fake_info():
    from installer.env import SystemInfo
    return SystemInfo(os="linux", arch="x86_64", gpu_vendor="nvidia",
                      glibc=(2, 38), driver_version="580.65",
                      vram_gb=[24], wsl=False)


# --- install: role wiring + dry-run plan printing ---------------------------

def test_install_client_role_passes_role_to_probe(bs, monkeypatch, capsys):
    seen = {}

    def fake_probe(info, fs, run, role="main", pinned=None, with_vllm=True,
                   with_vllm_metal=False, force_settings=False, manifest=None):
        seen["role"] = role
        seen["with_vllm"] = with_vllm
        seen["manifest"] = manifest
        return []

    monkeypatch.setattr(bs, "detect_system_info", lambda run: _fake_info())
    monkeypatch.setattr(bs, "probe", fake_probe)
    monkeypatch.setattr(bs, "_read_manifest", lambda log: {"tools": {}})
    rc = bs.main(["install", "--role", "client", "--no-vllm", "--dry-run"])
    assert rc == 0
    assert seen["role"] == "client"
    assert seen["with_vllm"] is False
    assert seen["manifest"] == {"tools": {}}


def test_install_dry_run_does_not_execute(bs, monkeypatch, capsys):
    executed = []

    class FakeEx:
        def __init__(self, **kw):
            pass

        def execute(self, plan):
            executed.append(plan)
            return {"installed": [], "repaired": [], "skipped": [], "failures": []}

        def write_manifest(self, summary, detected=None):
            executed.append("manifest")

    monkeypatch.setattr(bs, "detect_system_info", lambda run: _fake_info())
    monkeypatch.setattr(bs, "probe", lambda *a, **k: [])
    monkeypatch.setattr(bs, "_read_manifest", lambda log: None)
    monkeypatch.setattr(bs, "Executor", FakeEx)
    assert bs.main(["install", "--dry-run"]) == 0
    assert executed == []


# --- --verbose tee to install.log (§4) --------------------------------------

def test_verbose_tee_writes_log_file(bs, monkeypatch, tmp_path):
    log_path = tmp_path / "install.log"
    monkeypatch.setattr(bs, "detect_system_info", lambda run: _fake_info())
    monkeypatch.setattr(bs, "probe", lambda *a, **k: [])
    monkeypatch.setattr(bs, "_read_manifest", lambda log: None)

    rc = bs.main(["install", "--verbose", "--dry-run",
                  "--log-file", str(log_path)])
    assert rc == 0
    text = log_path.read_text(encoding="utf-8")
    assert "DETECT os=linux" in text
    # every logged line is timestamped
    for line in text.splitlines():
        assert "T" in line.split(" ", 1)[0]


def test_non_verbose_still_logs_to_file(bs, monkeypatch, tmp_path):
    log_path = tmp_path / "install.log"
    monkeypatch.setattr(bs, "detect_system_info", lambda run: _fake_info())
    monkeypatch.setattr(bs, "probe", lambda *a, **k: [])
    monkeypatch.setattr(bs, "_read_manifest", lambda log: None)
    assert bs.main(["install", "--dry-run", "--log-file", str(log_path)]) == 0
    assert log_path.exists()


def test_no_subcommand_prints_help(bs, capsys):
    assert bs.main([]) == 2
    assert "usage:" in capsys.readouterr().out


def test_default_pins_match_design(bs):
    assert bs.DEFAULT_PINS["llama.cpp"]["version"].startswith("b")
    assert bs.DEFAULT_PINS["vllm"]["version"]
    assert bs.DEFAULT_PINS["llama-swap"]["version"]