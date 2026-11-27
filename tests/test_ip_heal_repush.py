"""IP-heal re-push: a healed peer IP must reach the generated swap files.

The start wrappers and remote-local `proxy:` URLs bake the peer IP, so
healing machines.yaml alone is not enough — the swap config/scripts have to
be regenerated and pushed (llama-swap is run with --watch-config, so no
daemon restart is needed).
"""
import threading

import pytest

import config_loader
import discovery
import rpc_manager
import swap_manager


def _models(monkeypatch, models):
    monkeypatch.setattr(swap_manager.cfg, "list_models", lambda: models)


@pytest.fixture(autouse=True)
def _enable_repush(monkeypatch):
    """conftest disables the automatic push suite-wide; opt back in here."""
    monkeypatch.delenv("FARM_IP_REPUSH_DISABLE", raising=False)


# ---------------- machine reference detection ----------------

def test_refs_via_servers():
    assert swap_manager._machine_refs(
        {"servers": ["hp-box"], "device_order": "hp-box-RAM"},
        {"hp-box"}) is True


def test_refs_via_device_order_hyphenated_machine():
    # machine names contain hyphens; split at the LAST dash
    assert swap_manager._machine_refs(
        {"device_order": "HP-Pavilion-onvertible-RAM"}, {
            "HP-Pavilion-onvertible"}) is True


def test_refs_via_expose_key():
    assert swap_manager._machine_refs(
        {"expose": {"jetson-nano": {"mode": "ram"}}},
        {"jetson-nano"}) is True


def test_refs_ignores_unrelated():
    assert swap_manager._machine_refs(
        {"servers": ["aibox1"],
         "device_order": "aibox1-CUDA0",
         "expose": {"aibox1": {}}}, {"hp-box"}) is False


def test_models_referencing_includes_variants(monkeypatch):
    _models(monkeypatch, [
        {"model_id": "base", "servers": ["aibox1"]},
        {"model_id": "with-variant", "servers": ["aibox1"],
         "variants": [{"servers": ["hp-box"],
                       "device_order": "hp-box-RAM"}]},
        {"model_id": "unrelated", "servers": ["aibox1"]},
    ])
    assert swap_manager.models_referencing_machines(["hp-box"]) == [
        "with-variant"]


# ---------------- skip conditions ----------------

def test_repush_disabled_by_env(monkeypatch):
    monkeypatch.setenv("FARM_IP_REPUSH_DISABLE", "1")
    res = swap_manager.repush_for_ip_change(["hp-box"], background=False)
    assert res["scheduled"] is False
    assert res["reason"] == "disabled"


def test_repush_skips_when_swap_not_running(monkeypatch):
    monkeypatch.setattr(swap_manager, "status", lambda: {"running": False})
    res = swap_manager.repush_for_ip_change(["hp-box"], background=False)
    assert res["scheduled"] is False
    assert res["reason"] == "swap not running"


def test_repush_skips_when_no_model_references(monkeypatch):
    monkeypatch.setattr(swap_manager, "status", lambda: {"running": True})
    _models(monkeypatch, [{"model_id": "m", "servers": ["aibox1"]}])
    res = swap_manager.repush_for_ip_change(["hp-box"], background=False)
    assert res["scheduled"] is False
    assert res["reason"] == "no models use those machines"


# ---------------- the push itself ----------------

def test_repush_runs_push_for_referenced_model(monkeypatch):
    monkeypatch.setattr(swap_manager, "status", lambda: {"running": True})
    _models(monkeypatch, [
        {"model_id": "laya", "servers": ["hp-box"],
         "device_order": "hp-box-RAM"}])
    calls = []
    monkeypatch.setattr(swap_manager, "push_scripts",
                        lambda: calls.append(1) or {"written": ["cfg"]})
    res = swap_manager.repush_for_ip_change(["hp-box"], background=False)
    assert res["scheduled"] is True
    assert res["models"] == ["laya"]
    assert calls == [1]
    assert swap_manager._IP_REPUSH_STATE["pending"] is False


def test_repush_coalesces_concurrent_triggers(monkeypatch):
    monkeypatch.setattr(swap_manager, "status", lambda: {"running": True})
    _models(monkeypatch, [{"model_id": "laya", "servers": ["hp-box"]}])
    started = threading.Event()
    release = threading.Event()
    calls = []

    def _push():
        calls.append(1)
        started.set()
        release.wait(3)
        return {"written": []}

    monkeypatch.setattr(swap_manager, "push_scripts", _push)
    first = swap_manager.repush_for_ip_change(["hp-box"], background=True)
    assert first["scheduled"] is True
    assert started.wait(3)
    second = swap_manager.repush_for_ip_change(["hp-box"], background=True)
    assert second["scheduled"] is False
    assert second["reason"] == "push already pending"
    release.set()
    swap_manager._IP_REPUSH_STATE["thread"].join(5)
    assert calls == [1]
    assert swap_manager._IP_REPUSH_STATE["pending"] is False


# ---------------- hooks ----------------

def test_reconcile_triggers_repush(monkeypatch):
    machines = [
        {"name": "hp-box", "id": "hp", "ip": "10.0.0.5",
         "mac": "de:ad:be:ef:12:34", "role": "peer"},
        {"name": "master", "id": "master", "ip": "127.0.0.1",
         "mac": "bc:24:11:de:ad:be", "role": "master"},
    ]

    class _Machines:
        def all(self):
            return machines

        def update(self, name, upd):
            for m in machines:
                if m["name"] == name or m.get("id") == name:
                    m.update(upd)

    monkeypatch.setattr(discovery, "discover_agents",
                        lambda **k: {"results": [
                            {"ip": "10.0.0.9",
                             "mac": "de:ad:be:ef:12:34"}]})
    monkeypatch.setattr(config_loader, "machines", _Machines())
    import local_node
    monkeypatch.setattr(local_node, "is_local",
                        lambda m: m.get("role") == "master")
    triggers = []
    monkeypatch.setattr(swap_manager, "repush_for_ip_change",
                        lambda names, reason="ip-heal", background=True:
                        triggers.append((list(names), reason)))
    changed = discovery.reconcile_machine_ips()
    assert changed == [("hp-box", "10.0.0.5", "10.0.0.9")]
    assert triggers == [(["hp-box"], "agent-discovery")]


def test_sync_machine_ip_triggers_repush(monkeypatch):
    m = {"name": "hp-box", "id": "hp", "ip": "10.0.0.5",
         "mac": "de:ad:be:ef:12:34", "os": "linux", "role": "peer"}
    monkeypatch.setattr(discovery, "resolve_ip_by_mac",
                        lambda mac: "10.0.0.9")
    monkeypatch.setattr(rpc_manager.cfg.machines, "update",
                        lambda name, upd: None)
    import socket as _sock

    class _FakeSock:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(_sock, "create_connection",
                        lambda addr, timeout=None: _FakeSock())
    triggers = []
    monkeypatch.setattr(swap_manager, "repush_for_ip_change",
                        lambda names, reason="ip-heal", background=True:
                        triggers.append((list(names), reason)))
    out = rpc_manager._sync_machine_ip(m)
    assert out["ip"] == "10.0.0.9"
    assert triggers == [(["hp-box"], "ip-heal")]
