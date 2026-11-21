"""RPC mTLS enable/disable route (no real agents)."""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from fastapi import FastAPI
from fastapi.testclient import TestClient

import routes.rpc_security as rs


def _client(monkeypatch):
    monkeypatch.setattr(rs.access, "gate_open", lambda: True)
    monkeypatch.setattr(rs.access, "view_allow", lambda: None)
    monkeypatch.setattr(rs.access, "is_admin", lambda _r: True)
    app = FastAPI()
    app.include_router(rs.router)
    return TestClient(app)


class _M:
    def __init__(self, name, ip, local=False):
        self.m = {"name": name, "ip": ip}
        self.local = local

    def get(self, k, default=None):
        return self.m.get(k, default)


def _machines(monkeypatch):
    rows = [_M("aibox1", "127.0.0.1", True), _M("peer1", "10.0.0.2")]
    monkeypatch.setattr(rs.cfg, "machines", type("X", (), {
        "all": staticmethod(lambda: [r.m for r in rows])})())
    monkeypatch.setattr(rs.local_node, "is_local",
                        lambda m: m.get("name") == "aibox1")


def test_status(monkeypatch):
    c = _client(monkeypatch)
    _machines(monkeypatch)
    monkeypatch.setattr(rs.rpc_tls, "status",
                        lambda: {"enabled": False, "proxies": []})
    r = c.get("/api/rpc/security/status")
    assert r.status_code == 200
    assert r.json()["settings"]["enabled"] is False


def test_enable_installs_every_peer_then_turns_on(monkeypatch):
    c = _client(monkeypatch)
    _machines(monkeypatch)
    calls = {"enabled": None, "pushed": False, "installed": []}
    monkeypatch.setattr(rs.rpc_tls, "ensure_ca", lambda: None)
    monkeypatch.setattr(rs.rpc_tls, "issue_master_cert", lambda: None)
    monkeypatch.setattr(rs.rpc_tls, "cert_bundle",
                        lambda m: {"cert": "c", "key": "k", "ca": "a"})
    monkeypatch.setattr(rs.rpc_tls, "refresh_local_proxies",
                        lambda: {"enabled": True, "proxies": []})
    monkeypatch.setattr(rs.rpc_tls, "set_enabled",
                        lambda on: calls.update(enabled=on))
    monkeypatch.setattr(rs.agent_client, "tls_install",
                        lambda m, c, k, a: calls["installed"].append(m["name"]))
    monkeypatch.setattr(rs.agent_client, "tls_https_start",
                        lambda m: {"running": True, "port": 8197})
    monkeypatch.setattr(rs.agent_client, "tls_status",
                        lambda m: {"tunnel_port": 5399})
    monkeypatch.setattr(rs.swap_manager, "push_scripts",
                        lambda: calls.update(pushed=True))
    monkeypatch.setattr(rs, "_reload_swap", lambda: None)
    r = c.post("/api/rpc/security/enable", json={"reload": False})
    assert r.status_code == 200
    assert r.json()["ok"] is True and r.json()["enabled"] is True
    assert calls["installed"] == ["peer1"]
    assert calls["enabled"] is True and calls["pushed"] is True


def test_enable_stays_off_on_peer_failure(monkeypatch):
    c = _client(monkeypatch)
    _machines(monkeypatch)
    called = {"enabled": False}
    monkeypatch.setattr(rs.rpc_tls, "ensure_ca", lambda: None)
    monkeypatch.setattr(rs.rpc_tls, "issue_master_cert", lambda: None)
    monkeypatch.setattr(rs.rpc_tls, "cert_bundle",
                        lambda m: {"cert": "c", "key": "k", "ca": "a"})
    monkeypatch.setattr(rs.agent_client, "tls_install",
                        lambda *a: (_ for _ in ()).throw(
                            RuntimeError("agent offline")))
    monkeypatch.setattr(rs.rpc_tls, "set_enabled",
                        lambda on: called.update(enabled=on))
    r = c.post("/api/rpc/security/enable", json={"reload": False})
    assert r.status_code == 200
    assert r.json()["ok"] is False
    assert r.json()["enabled"] is False
    assert called["enabled"] is False


def test_disable_stops_proxies_and_agents(monkeypatch):
    c = _client(monkeypatch)
    _machines(monkeypatch)
    calls = {"off": False, "agent": False}
    monkeypatch.setattr(rs.rpc_tls, "set_enabled",
                        lambda on: calls.update(off=not on))
    monkeypatch.setattr(rs.rpc_tls, "stop_all_local_proxies", lambda: None)
    monkeypatch.setattr(rs.agent_client, "tls_stop",
                        lambda m: calls.update(agent=True))
    monkeypatch.setattr(rs.swap_manager, "push_scripts", lambda: None)
    monkeypatch.setattr(rs, "_reload_swap", lambda: None)
    r = c.post("/api/rpc/security/disable", json={"reload": False})
    assert r.status_code == 200
    assert r.json()["enabled"] is False
    assert calls["off"] is True and calls["agent"] is True
