"""Unit tests for MAC-based machine identity and DHCP IP-drift healing.

All network I/O is faked: SSH runs return canned MAC query output, and the
ARP table comes from a stubbed subprocess. These verify the pieces that
keep machines reachable across DHCP re-leases:

- MAC normalization (the identity that never changes)
- per-OS MAC query parsing (linux / windows / macos)
- arp_table parsing (ip neigh + arp -a formats)
- resolve_ip_by_mac: MAC -> current IP
- _sync_machine_ip: stale IP healed from a stored MAC, config persisted
- _MacCapturingSSH: MAC captured once on first successful run, stored
"""
import subprocess
import types

import pytest

import discovery
import rpc_manager


# ---------------- MAC normalization ----------------

def test_normalize_mac_colon_form():
    assert discovery._normalize_mac("DE:AD:BE:EF:12:34") == "de:ad:be:ef:12:34"

def test_normalize_mac_dash_form():
    assert discovery._normalize_mac("DE-AD-BE-EF-12-34") == "de:ad:be:ef:12:34"

def test_normalize_mac_bare_hex():
    assert discovery._normalize_mac("de:ad:be:ef:12:34") == "de:ad:be:ef:12:34"

def test_normalize_mac_rejects_garbage():
    assert discovery._normalize_mac("not a mac") == ""
    assert discovery._normalize_mac("") == ""
    assert discovery._normalize_mac(None) == ""
    assert discovery._normalize_mac("d8:bb:c1") == ""  # too few octets


# ---------------- per-OS MAC query parsing ----------------

class _FakeSSH:
    def __init__(self, out=""):
        self._out = out
    def run(self, cmd, timeout=15):
        return 0, self._out, ""

def test_query_machine_mac_linux(monkeypatch):
    out = "10.0.0.98 bc:24:11:de:ad:be\n10.0.0.99 00:15:5d:aa:bb:cc\n"
    ip, mac = discovery.query_machine_mac(_FakeSSH(out), "linux")
    # virtual NICs (Hyper-V 00:15:5d) skipped; physical wins
    assert ip == "10.0.0.98"
    assert mac == "bc:24:11:de:ad:be"

def test_query_machine_mac_windows(monkeypatch):
    out = "10.0.0.124 DE-AD-BE-EF-12-34\n"
    ip, mac = discovery.query_machine_mac(_FakeSSH(out), "windows")
    assert ip == "10.0.0.124"
    assert mac == "de:ad:be:ef:12:34"  # dashes -> colons, lowered

def test_query_machine_mac_macos():
    out = "10.0.0.5 a4:83:e7:11:22:33\n"
    ip, mac = discovery.query_machine_mac(_FakeSSH(out), "macos")
    assert ip == "10.0.0.5"
    assert mac == "a4:83:e7:11:22:33"

def test_query_machine_mac_no_output():
    ip, mac = discovery.query_machine_mac(_FakeSSH(""), "linux")
    assert ip == "" and mac == ""

def test_query_machine_mac_ssh_failure():
    class _DeadSSH:
        def run(self, cmd, timeout=15):
            raise discovery.SSHError("unreachable")
    ip, mac = discovery.query_machine_mac(_DeadSSH(), "linux")
    assert ip == "" and mac == ""


# ---------------- ARP table parsing ----------------

def test_arp_table_from_arp_a(monkeypatch):
    out = "? (10.0.0.124) at de:ad:be:ef:12:34 [ether] on enp6s18\n" \
          "? (10.0.0.98) at bc:24:11:de:ad:be [ether] on enp6s18\n"
    monkeypatch.setattr(subprocess, "run",
                        lambda *a, **k: types.SimpleNamespace(stdout=out))
    table = discovery.arp_table()
    assert table["de:ad:be:ef:12:34"] == "10.0.0.124"
    assert table["bc:24:11:de:ad:be"] == "10.0.0.98"

def test_arp_table_from_ip_neigh(monkeypatch):
    out = ("10.0.0.124 dev enp6s18 lladdr de:ad:be:ef:12:34 REACHABLE\n"
           "10.0.0.98 dev enp6s18 lladdr bc:24:11:de:ad:be STALE\n")
    monkeypatch.setattr(subprocess, "run",
                        lambda *a, **k: types.SimpleNamespace(stdout=out))
    table = discovery.arp_table()
    assert table["de:ad:be:ef:12:34"] == "10.0.0.124"
    assert table["bc:24:11:de:ad:be"] == "10.0.0.98"


def test_resolve_ip_by_mac(monkeypatch):
    monkeypatch.setattr(discovery, "arp_table",
                        lambda: {"de:ad:be:ef:12:34": "10.0.0.124"})
    assert discovery.resolve_ip_by_mac("de:ad:be:ef:12:34") == "10.0.0.124"
    # dash form normalizes to the same lookup
    assert discovery.resolve_ip_by_mac("DE-AD-BE-EF-12-34") == "10.0.0.124"
    # unknown mac
    assert discovery.resolve_ip_by_mac("aa:bb:cc:dd:ee:ff") is None
    # empty/invalid mac: no lookup at all
    assert discovery.resolve_ip_by_mac("") is None


# ---------------- IP-drift healing (_sync_machine_ip) ----------------

def _machine(ip="10.0.0.13", mac="de:ad:be:ef:12:34", os="windows"):
    return {"name": "winbox", "id": "winbox", "ip": ip, "mac": mac,
            "os": os, "port": 22, "username": "zazit", "password": "pw",
            "role": "peer", "detected": {}}


def test_sync_heals_stale_ip(monkeypatch):
    """Stored MAC resolves to a new IP with SSH open -> config updated."""
    m = _machine(ip="10.0.0.13")  # stale
    monkeypatch.setattr(discovery, "resolve_ip_by_mac",
                        lambda mac: "10.0.0.124")
    updates = {}
    monkeypatch.setattr(rpc_manager.cfg.machines, "update",
                        lambda name, upd: updates.update(upd) or types.SimpleNamespace(**upd))
    # socket connect to the healed IP must succeed (fake it)
    import socket as _sock
    class _FakeSock:
        def __enter__(self): return self
        def __exit__(self, *a): return False
    monkeypatch.setattr(_sock, "create_connection",
                        lambda addr, timeout=None: _FakeSock())
    out = rpc_manager._sync_machine_ip(m)
    assert out["ip"] == "10.0.0.124"
    assert updates["ip"] == "10.0.0.124"
    assert updates["mac"] == "de:ad:be:ef:12:34"


def test_sync_no_mac_is_noop(monkeypatch):
    """Machines without a stored MAC are never touched."""
    called = []
    monkeypatch.setattr(discovery, "resolve_ip_by_mac",
                        lambda mac: called.append(mac))
    m = _machine(mac="")
    out = rpc_manager._sync_machine_ip(m)
    assert out["ip"] == "10.0.0.13"
    assert called == []  # no resolution attempted

def test_sync_no_change_is_noop(monkeypatch):
    """MAC resolves to the SAME ip -> nothing written."""
    m = _machine(ip="10.0.0.124")  # already current
    monkeypatch.setattr(discovery, "resolve_ip_by_mac",
                        lambda mac: "10.0.0.124")
    updates = []
    monkeypatch.setattr(rpc_manager.cfg.machines, "update",
                        lambda name, upd: updates.append(upd))
    out = rpc_manager._sync_machine_ip(m)
    assert out["ip"] == "10.0.0.124"
    assert updates == []

def test_sync_skips_local_machine(monkeypatch):
    """The machine running the app never needs healing."""
    monkeypatch.setattr(rpc_manager.local_node, "is_local", lambda m: True)
    m = _machine()
    assert rpc_manager._sync_machine_ip(m) is m

def test_sync_new_ip_without_ssh_is_rejected(monkeypatch):
    """A MAC hit whose IP doesn't speak SSH is NOT adopted (safety)."""
    m = _machine(ip="10.0.0.13")
    monkeypatch.setattr(discovery, "resolve_ip_by_mac",
                        lambda mac: "10.0.0.124")
    import socket as _sock
    class _Refused(OSError):
        pass
    def _fail(addr, timeout=None):
        raise _Refused("refused")
    monkeypatch.setattr(_sock, "create_connection", _fail)
    updates = []
    monkeypatch.setattr(rpc_manager.cfg.machines, "update",
                        lambda name, upd: updates.append(upd))
    out = rpc_manager._sync_machine_ip(m)
    assert out["ip"] == "10.0.0.13"  # unchanged
    assert updates == []


# ---------------- MAC capture on successful connect ----------------

def test_mac_captured_on_first_successful_run(monkeypatch):
    """A successful SSH command triggers a one-time MAC capture, stored
    on the machine config."""
    m = _machine(mac="")  # no MAC known yet
    monkeypatch.setattr(discovery, "query_machine_mac",
                        lambda ssh, os_type: ("10.0.0.124", "de:ad:be:ef:12:34"))
    updates = {}
    monkeypatch.setattr(rpc_manager.cfg.machines, "update",
                        lambda name, upd: updates.update(upd))
    # the underlying SSHManager.run: pretend success without the network
    monkeypatch.setattr(rpc_manager.SSHManager, "run",
                        lambda self, cmd, timeout=60: (0, "ok", ""))
    ssh = rpc_manager._MacCapturingSSH(m)
    code, out, err = ssh.run("echo ok")
    assert code == 0
    assert m["mac"] == "de:ad:be:ef:12:34"
    assert updates["mac"] == "de:ad:be:ef:12:34"

def test_mac_capture_skipped_when_stored(monkeypatch):
    """Already-known MAC: no capture query is ever issued."""
    m = _machine(mac="de:ad:be:ef:12:34")
    queried = []
    monkeypatch.setattr(discovery, "query_machine_mac",
                        lambda ssh, os_type: queried.append(1) or ("", ""))
    monkeypatch.setattr(rpc_manager.SSHManager, "run",
                        lambda self, cmd, timeout=60: (0, "ok", ""))
    ssh = rpc_manager._MacCapturingSSH(m)
    ssh.run("echo ok")
    assert queried == []  # capture is once per machine

def test_mac_capture_failure_never_breaks_command(monkeypatch):
    """A broken MAC capture must not fail the working SSH command."""
    m = _machine(mac="")
    def _boom(ssh, os_type):
        raise RuntimeError("capture exploded")
    monkeypatch.setattr(discovery, "query_machine_mac", _boom)
    monkeypatch.setattr(rpc_manager.SSHManager, "run",
                        lambda self, cmd, timeout=60: (0, "ok", ""))
    ssh = rpc_manager._MacCapturingSSH(m)
    code, out, err = ssh.run("echo ok")
    assert code == 0 and out == "ok"  # command result untouched