"""Unit tests for the port-targeted stop/start RPC lifecycle and the
"any rpc running" status probes, with the SSH layer stubbed so no real
host is contacted.

These assert the new contract: stop_rpc(port=N) kills only the instance
on port N (port-matched pkill/CIM filter), never a blanket pkill; and
machine_status/stats report rpc_running=True if ANY ggml-rpc-server is
up, not just the base port.
"""
import types

import pytest

import rpc_manager


class FakeSSH:
    """Records commands run and returns canned results. Mimics the
    .run(code, out, err) and .check_port()/is_reachable() surface that
    rpc_manager + stats_collector use."""
    def __init__(self, machine=None, port_open=False, reachable=True):
        self.machine = machine
        self.calls = []
        self._port_open = port_open
        self._reachable = reachable

    def run(self, cmd, timeout=20):
        self.calls.append(cmd)
        return 0, "", ""

    def check_port(self, port, timeout=3):
        self.calls.append(f"check_port:{port}")
        return self._port_open

    def is_reachable(self):
        return self._reachable

    def gpu_info(self, vendor=None):
        return []


# ---------------- stop_rpc: per-port targeting ----------------

def test_stop_rpc_per_port_linux_targets_port(monkeypatch):
    """stop_rpc(port=N) on linux pkills ONLY 'ggml-rpc-server.*-p N',
    not all ggml-rpc-server instances."""
    machine = {"name": "aibox2", "ip": "10.0.0.11", "os": "linux",
               "username": "zazi", "rpc_port": 50052}
    captured = []

    def fake_ssh_for(m):
        ssh = FakeSSH(m)
        def run(cmd, timeout=20):
            captured.append(cmd)
            # pgrep returns 1 = no matches = process dead (kill succeeded)
            return (1, "", "") if "pgrep" in cmd else (0, "", "")
        ssh.run = run
        return ssh

    monkeypatch.setattr(rpc_manager, "_ssh_for", fake_ssh_for)
    monkeypatch.setattr(rpc_manager, "local_node",
                        types.SimpleNamespace(is_local=lambda m: False))
    monkeypatch.setattr(rpc_manager, "cfg",
                        types.SimpleNamespace(machines=types.SimpleNamespace(
                            require=lambda n: machine)))

    res = rpc_manager.stop_rpc("aibox2", port=50157)
    assert res["killed"] is True
    assert any("ggml-[r]pc-server.*-p 50157" in c for c in captured), captured
    # must NOT issue a blanket pkill of all ggml-rpc-server
    assert not any("pkill -u zazi -f \"llama-[s]wap|llama-[s]erver|ggml-[r]pc-server\"" in c
                   for c in captured), captured


def test_stop_rpc_no_port_kills_all(monkeypatch):
    """stop_rpc(port=None) keeps the legacy 'stop everything' behavior
    (used by the global stop / stop-all buttons)."""
    machine = {"name": "aibox2", "ip": "10.0.0.11", "os": "linux",
               "username": "zazi", "rpc_port": 50052}
    captured = []

    def fake_ssh_for(m):
        ssh = FakeSSH(m)
        def run(cmd, timeout=20):
            captured.append(cmd)
            # pgrep returns 1 = no matches = process dead (kill succeeded)
            return (1, "", "") if "pgrep" in cmd else (0, "", "")
        ssh.run = run
        return ssh

    monkeypatch.setattr(rpc_manager, "_ssh_for", fake_ssh_for)
    monkeypatch.setattr(rpc_manager, "local_node",
                        types.SimpleNamespace(is_local=lambda m: False))
    monkeypatch.setattr(rpc_manager, "cfg",
                        types.SimpleNamespace(machines=types.SimpleNamespace(
                            require=lambda n: machine)))

    res = rpc_manager.stop_rpc("aibox2")
    assert res["killed"] is True
    # legacy blanket pkill over llama-swap|llama-server|ggml-rpc-server
    assert any("pkill -u zazi -f" in c and "ggml-[r]pc-server" in c
               for c in captured), captured


def test_stop_rpc_per_port_win_cim_filter(monkeypatch):
    """Windows per-port stop uses a CIM filter on '-p <port>', not
    taskkill /F /IM (which would kill all instances)."""
    machine = {"name": "winbox", "ip": "10.0.0.13", "os": "windows",
               "username": "zazit", "rpc_port": 50052}
    captured = []

    def fake_ssh_for(m):
        ssh = FakeSSH(m)
        def run(cmd, timeout=20):
            captured.append(cmd)
            return 0, "", ""
        ssh.run = run
        return ssh

    monkeypatch.setattr(rpc_manager, "_ssh_for", fake_ssh_for)
    monkeypatch.setattr(rpc_manager, "local_node",
                        types.SimpleNamespace(is_local=lambda m: False))
    monkeypatch.setattr(rpc_manager, "cfg",
                        types.SimpleNamespace(machines=types.SimpleNamespace(
                            require=lambda n: machine)))

    res = rpc_manager.stop_rpc("winbox", port=50157)
    assert res["killed"] is True
    assert any("-p 50157" in c and "Win32_Process" in c for c in captured), captured
    assert not any("taskkill /F /IM ggml-rpc-server" in c for c in captured), captured


# ---------------- machine_status: any rpc running ----------------

def test_machine_status_rpc_running_any_port(monkeypatch):
    """machine_status reports rpc_running=True if a ggml-rpc-server is
    running on ANY port (pgrep), even when the base port is closed."""
    machine = {"name": "aibox2", "ip": "10.0.0.11", "os": "linux",
               "username": "zazi", "rpc_port": 50052, "role": "peer"}

    monkeypatch.setattr(rpc_manager, "local_node",
                        types.SimpleNamespace(is_local=lambda m: False))
    monkeypatch.setattr(rpc_manager, "cfg",
                        types.SimpleNamespace(machines=types.SimpleNamespace(
                            require=lambda n: machine)))

    ssh = FakeSSH(machine, port_open=False, reachable=True)
    # pgrep returns code 0 (found a process) -> any rpc running True
    def run(cmd, timeout=20):
        if "pgrep" in cmd:
            return 0, "12345", ""  # found a pid
        return 0, "", ""
    ssh.run = run
    monkeypatch.setattr(rpc_manager, "_ssh_for", lambda m: ssh)
    monkeypatch.setattr(rpc_manager, "_local_gpu_info", lambda m: [])
    monkeypatch.setattr(rpc_manager, "_any_rpc_running_remote",
                        lambda m, s: True)

    st = rpc_manager.machine_status("aibox2")
    assert st["reachable"] is True
    assert st["rpc_running"] is True


def test_machine_status_rpc_down_when_no_process(monkeypatch):
    machine = {"name": "aibox2", "ip": "10.0.0.11", "os": "linux",
               "username": "zazi", "rpc_port": 50052, "role": "peer"}
    monkeypatch.setattr(rpc_manager, "local_node",
                        types.SimpleNamespace(is_local=lambda m: False))
    monkeypatch.setattr(rpc_manager, "cfg",
                        types.SimpleNamespace(machines=types.SimpleNamespace(
                            require=lambda n: machine)))
    monkeypatch.setattr(rpc_manager, "_ssh_for",
                        lambda m: FakeSSH(m, port_open=False, reachable=True))
    monkeypatch.setattr(rpc_manager, "_local_gpu_info", lambda m: [])
    monkeypatch.setattr(rpc_manager, "_any_rpc_running_remote",
                        lambda m, s: False)

    st = rpc_manager.machine_status("aibox2")
    assert st["rpc_running"] is False