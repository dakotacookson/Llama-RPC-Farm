"""Self-service install: unit/plist/task generation (no real services)."""
import os
import plistlib
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "peer_agent"))

import selfservice as app_ss  # noqa: E402
from llama_peer_agent import selfservice as agent_ss  # noqa: E402


def test_app_linux_unit(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    monkeypatch.setattr(app_ss.shutil, "which", lambda n: None)
    app_ss._linux_install()
    unit = (tmp_path / "systemd" / "user" / "llama-rpc-farm.service").read_text()
    assert "ExecStart=" in unit
    assert "Restart=on-failure" in unit


def test_agent_linux_unit(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    monkeypatch.setattr(agent_ss.shutil, "which", lambda n: None)
    agent_ss._linux_install()
    unit = (tmp_path / "systemd" / "user" / "llama-peer-agent.service").read_text()
    assert "--port" in unit and "8198" in unit
    assert "Restart=on-failure" in unit


def test_app_macos_plist(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    app_ss._mac_install()
    plist = tmp_path / "Library" / "LaunchAgents" / "com.llamarpc.farm.app.plist"
    data = plistlib.loads(plist.read_bytes())
    assert data["RunAtLoad"] is True
    assert data["ProgramArguments"]


def test_agent_macos_plist(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    agent_ss._mac_install()
    plist = (tmp_path / "Library" / "LaunchAgents"
             / "com.llamarpc.farm.agent.plist")
    data = plistlib.loads(plist.read_bytes())
    assert "--port" in data["ProgramArguments"]
    assert "8198" in data["ProgramArguments"]


def test_windows_firewall_and_task(monkeypatch):
    calls = []
    monkeypatch.setattr(app_ss.shutil, "which", lambda n: "/" + n)
    monkeypatch.setattr(
        app_ss, "_run",
        lambda args, timeout=30: calls.append(args) or
        type("R", (), {"returncode": 1 if "show" in args else 0,
                       "stdout": "", "stderr": ""})())
    app_ss._windows_install()
    flat = [" ".join(map(str, c)) for c in calls]
    assert any("localport=8199" in c for c in flat)
    assert any("/RL HIGHEST" in c for c in flat)
    assert any("LlamaRPCFarm" in c for c in flat)


def test_agent_windows_firewall_ports(monkeypatch):
    calls = []
    monkeypatch.setattr(agent_ss.shutil, "which", lambda n: "/" + n)
    monkeypatch.setattr(
        agent_ss, "_run",
        lambda args, timeout=30: calls.append(args) or
        type("R", (), {"returncode": 1 if "show" in args else 0,
                       "stdout": "", "stderr": ""})())
    agent_ss._windows_install()
    flat = [" ".join(map(str, c)) for c in calls]
    assert any("localport=8198" in c for c in flat)
    assert any("localport=8199" in c for c in flat)
    assert any("LlamaPeerAgent" in c for c in flat)


def test_maybe_auto_install_frozen_only(monkeypatch):
    called = []
    monkeypatch.setattr(app_ss, "install", lambda: called.append("install"))
    monkeypatch.setattr(app_ss.sys, "frozen", False, raising=False)
    app_ss.maybe_auto_install([])
    assert called == []
    monkeypatch.setattr(app_ss.sys, "frozen", True, raising=False)
    app_ss.maybe_auto_install([])
    assert called == ["install"]
