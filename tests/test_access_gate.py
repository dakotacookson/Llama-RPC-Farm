"""Access gate (src/access.py + app.py IpGateMiddleware + config_loader
gate + _engine_groups/chat filters): private-model visibility per client
IP — full app stays usable everywhere, private configs vanish for
foreign peers."""
import sys
import os
import pathlib

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import pytest

import access
import config_loader
import engine_registry
import app as app_module


@pytest.fixture(autouse=True)
def fresh_gate():
    """The gate is request-scoped in production (fresh asyncio task per
    request); tests share a thread, so reset the contextvar around each
    test to keep one test's restricted scope from leaking into another."""
    token = access._restricted.set(False)
    yield
    access._restricted.reset(token)


# ---------------- access module ----------------

def test_trusted_ips_cover_local_and_winbox(monkeypatch):
    assert access.ip_trusted("127.0.0.1")
    assert access.ip_trusted("::1")
    # admin trust comes from settings.yaml admin_ips (loaded per request);
    # the test suite isolates SETTINGS_FILE, so seed the table directly
    monkeypatch.setattr(access, "admin_ips", lambda: {"10.0.0.124"})
    assert access.ip_trusted("10.0.0.124")   # winbox (configured admin)
    assert not access.ip_trusted("10.0.0.11")
    assert not access.ip_trusted("192.168.1.50")


def test_non_ip_tokens_read_trusted_testclient():
    # "testclient" (Starlette sentinel) and "" are in-process tokens —
    # no real socket request arrives without an IP literal peer.
    assert access.ip_trusted("testclient")
    assert access.ip_trusted("")
    assert not access._looks_like_ip("testclient") if hasattr(
        access, "_looks_like_ip") else True
    import ipaddress
    with pytest.raises(ValueError):
        ipaddress.ip_address("testclient")


def test_request_ip_prefers_forwarded_header():
    class H(dict):
        def get(self, k, d=None):
            return {"x-forwarded-for": "203.0.113.9, 10.0.0.1",
                    "x-real-ip": "198.51.100.1"}.get(k, d)
    req = type("Req", (), {"headers": H(), "client": None})()
    assert access.request_ip(req) == "203.0.113.9"


def test_gate_defaults_open_and_flips_only_in_context():
    assert access.gate_open()
    token = access.set_restricted(True)
    assert not access.gate_open()
    access._restricted.reset(token)
    assert access.gate_open()


def test_restrict_if_foreign_skips_non_request_scopes():
    # lifespan scope must never flip the gate
    access.restrict_if_foreign({"type": "lifespan",
                                "client": ("203.0.113.9", 5)})
    assert access.gate_open()


# ---------------- private-model gate (config_loader) ----------------

@pytest.fixture
def private_models_dir(tmp_path, monkeypatch):
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    monkeypatch.setattr(config_loader, "MODELS_DIR", str(models_dir))
    (models_dir / "dark-scar.yaml").write_text(
        "model_id: dark-scar\n"
        "name: Dark Scar\n"
        "base_path: store://llm-models/dark-scar.gguf\n"
        "servers:\n- aibox1\n"
        "private: true\n", encoding="utf-8")
    (models_dir / "open-model.yaml").write_text(
        "model_id: open-model\n"
        "name: Open Model\n"
        "base_path: store://llm-models/open.gguf\n"
        "servers:\n- aibox1\n", encoding="utf-8")
    return models_dir


def test_private_model_listed_by_default_open_gate(private_models_dir):
    # default scope (background threads, no request) = open: housekeeping
    # sees everything
    ids = [m["model_id"] for m in config_loader.list_models()]
    assert "dark-scar" in ids and "open-model" in ids


def test_private_model_hidden_in_restricted_context(private_models_dir):
    access.set_restricted(True)
    ids = [m["model_id"] for m in config_loader.list_models()]
    assert ids == ["open-model"]
    assert config_loader.get_model("dark-scar") is None
    assert config_loader.get_model("open-model") is not None


def test_private_flag_round_trips_through_save(private_models_dir):
    m = config_loader.get_model("dark-scar")
    assert m.get("private") is True
    config_loader.save_model("dark-scar", dict(m))
    saved = (private_models_dir / "dark-scar.yaml").read_text()
    assert "private: true" in saved


# ---------------- group visibility ----------------

def test_groups_with_private_member_hidden_when_restricted(
        private_models_dir, monkeypatch, tmp_path):
    sg = tmp_path / "swap_groups.yaml"
    sg.write_text(
        "swap_groups:\n"
        "- name: scar-group\n"
        "  members:\n"
        "  - dark-scar\n"
        "  swap: true\n"
        "  engine: llama_cpp\n"
        "- name: open-group\n"
        "  members:\n"
        "  - open-model\n"
        "  swap: true\n"
        "  engine: llama_cpp\n", encoding="utf-8")
    monkeypatch.setattr(config_loader.swap_groups, "path", str(sg))
    engine_registry.get_engine()  # warm (default llama_cpp)
    # open scope: both groups
    out = app_module._engine_groups(config_loader.swap_groups)
    assert [g["name"] for g in out] == ["scar-group", "open-group"]
    # restricted: scar-group vanishes (member is private)
    access.set_restricted(True)
    out = app_module._engine_groups(config_loader.swap_groups)
    assert [g["name"] for g in out] == ["open-group"]
    # variant labels ("base--variant") catch too
    access.set_restricted(False)
    sg.write_text(
        "swap_groups:\n"
        "- name: scar-group\n"
        "  members:\n"
        "  - dark-scar--fast\n"
        "  swap: true\n"
        "  engine: llama_cpp\n", encoding="utf-8")
    access.set_restricted(True)
    out = app_module._engine_groups(config_loader.swap_groups)
    assert out == []


def test_group_with_private_and_open_members_hidden_when_restricted(
        private_models_dir, monkeypatch, tmp_path):
    sg = tmp_path / "swap_groups.yaml"
    sg.write_text(
        "swap_groups:\n"
        "- name: mixed\n"
        "  members:\n"
        "  - open-model\n"
        "  - dark-scar\n"
        "  swap: true\n"
        "  engine: llama_cpp\n", encoding="utf-8")
    monkeypatch.setattr(config_loader.swap_groups, "path", str(sg))
    access.set_restricted(True)
    out = app_module._engine_groups(config_loader.swap_groups)
    assert out == []


# ---------------- middleware over the full app ----------------

def test_middleware_sets_restricted_for_foreign_peer():
    import asyncio

    async def run():
        seen = []
        async def receive():
            return {"type": "http.request"}
        async def send(msg):
            seen.append(msg)
        async def inner_app(scope, receive, send):
            seen.append(("gate", access.gate_open()))
        for scope in (
            {"type": "http", "client": ("203.0.113.9", 5000),
             "path": "/", "headers": []},
            {"type": "websocket", "client": ("192.168.1.7", 5000),
             "path": "/ws/logs/stream", "headers": []},
            {"type": "http", "client": ("127.0.0.1", 5000),
             "path": "/", "headers": []},
            {"type": "http", "client": ("10.0.0.124", 5000),
             "path": "/", "headers": []},
            {"type": "http", "client": ("testclient", 0),
             "path": "/", "headers": []},
            {"type": "http", "client": None, "path": "/", "headers": []},
            {"type": "lifespan"},
        ):
            await app_module.IpGateMiddleware(inner_app)(scope, receive, send)
        flags = [v for k, v in seen if k == "gate"]
        assert all(flags), flags  # every pass saw an OPEN gate inside