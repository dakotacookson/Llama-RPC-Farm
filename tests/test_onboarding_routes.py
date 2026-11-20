"""HTTP surface for the onboarding helpers."""
import os
import sys

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from fastapi import FastAPI
from fastapi.testclient import TestClient

import routes.onboarding as ob


def _client(monkeypatch):
    monkeypatch.setattr(ob.access, "gate_open", lambda: True)
    monkeypatch.setattr(ob.access, "view_allow", lambda: None)
    app = FastAPI()
    app.include_router(ob.router)
    return TestClient(app)


def test_recipes_endpoint(monkeypatch):
    c = _client(monkeypatch)
    r = c.get("/api/onboarding/recipes")
    assert r.status_code == 200
    data = r.json()
    assert data["recipes"]
    assert any(x["id"] == "demo-0.6b" for x in data["recipes"])


def test_clients_endpoint(monkeypatch):
    c = _client(monkeypatch)
    r = c.get("/api/onboarding/clients")
    assert r.status_code == 200
    ids = {x["id"] for x in r.json()["clients"]}
    assert {"openai-curl", "anthropic-curl", "claude-code"} <= ids


def test_clients_endpoint_model_dropdown_and_live_snippets(monkeypatch):
    c = _client(monkeypatch)
    monkeypatch.setattr(ob.cfg, "list_models", lambda: [
        {"model_id": "alpha", "name": "Alpha"},
        {"model_id": "beta", "name": "Beta"},
    ])
    r = c.get("/api/onboarding/clients")
    assert r.status_code == 200
    d = r.json()
    assert [m["id"] for m in d["models"]] == ["alpha", "beta"]
    assert d["model"] == "alpha"
    assert '"model":"alpha"' in d["clients"][0]["snippet"]

    r2 = c.get("/api/onboarding/clients?model=beta")
    assert r2.status_code == 200
    d2 = r2.json()
    assert d2["model"] == "beta"
    assert '"model":"beta"' in d2["clients"][0]["snippet"]


def test_scoped_user_sees_only_granted_models_in_connect(monkeypatch):
    c = _client(monkeypatch)
    monkeypatch.setattr(ob.access, "gate_open", lambda: False)
    monkeypatch.setattr(ob.access, "view_allow", lambda: {"m1"})
    monkeypatch.setattr(ob.access, "current_key_record",
                        lambda: {"name": "kid", "models": {"m1"}})
    monkeypatch.setattr(ob.access, "session_user", lambda: None)
    monkeypatch.setattr(ob.cfg, "list_models", lambda: [
        {"model_id": "m1", "name": "One"},
        {"model_id": "m2", "name": "Two"},
    ])
    r = c.get("/api/onboarding/clients")
    assert r.status_code == 200
    assert [m["id"] for m in r.json()["models"]] == ["m1"]


def test_keyless_foreign_connect_is_404(monkeypatch):
    c = _client(monkeypatch)
    monkeypatch.setattr(ob.access, "gate_open", lambda: False)
    monkeypatch.setattr(ob.access, "view_allow", lambda: None)
    monkeypatch.setattr(ob.access, "current_key_record", lambda: None)
    monkeypatch.setattr(ob.access, "session_user", lambda: None)
    r = c.get("/api/onboarding/clients")
    assert r.status_code == 404


def test_clients_endpoint_returns_model_kind(monkeypatch):
    c = _client(monkeypatch)
    monkeypatch.setattr(ob, "_model_options", lambda: [
        {"id": "nomic", "name": "Nomic", "kind": "embeddings"},
        {"id": "chat", "name": "Chat", "kind": "chat"},
    ])
    r = c.get("/api/onboarding/clients?model=nomic")
    assert r.status_code == 200
    d = r.json()
    assert d["kind"] == "embeddings"
    assert any("/v1/embeddings" in x["snippet"] for x in d["clients"])


def test_clients_default_prefers_chat_model(monkeypatch):
    c = _client(monkeypatch)
    monkeypatch.setattr(ob, "_model_options", lambda: [
        {"id": "laya", "name": "Laya", "kind": "decision",
         "kinds": ["decision"]},
        {"id": "chat", "name": "Chat", "kind": "chat", "kinds": ["chat"]},
    ])
    r = c.get("/api/onboarding/clients")
    assert r.status_code == 200
    assert r.json()["model"] == "chat"
    assert r.json()["kinds"] == ["chat"]


def test_test_endpoint_rejects_media_models(monkeypatch):
    c = _client(monkeypatch)
    monkeypatch.setattr(ob, "_model_options", lambda: [
        {"id": "asr", "name": "ASR", "kind": "audio", "kinds": ["audio"]},
    ])
    monkeypatch.setattr(ob.swap_manager, "status",
                        lambda: {"running": True})
    r = c.post("/api/onboarding/test", json={"model": "asr"})
    assert r.status_code == 400


def test_safety_endpoint(monkeypatch):
    c = _client(monkeypatch)
    monkeypatch.setattr(ob.onboarding, "safety_report",
                        lambda: {"checks": [], "ok": True,
                                 "bad": 0, "warn": 0})
    r = c.get("/api/onboarding/safety")
    assert r.status_code == 200
    assert r.json()["ok"] is True


def test_pairing_endpoint(monkeypatch):
    c = _client(monkeypatch)
    monkeypatch.setattr(ob.onboarding, "pairing_commands",
                        lambda: {"unix": "u", "windows": "w",
                                 "token": "t", "note": "n"})
    r = c.get("/api/onboarding/pairing-command")
    assert r.status_code == 200
    assert r.json()["unix"] == "u"


def test_access_key_endpoint(monkeypatch):
    c = _client(monkeypatch)
    monkeypatch.setattr(ob.onboarding, "create_access_key",
                        lambda name: "sk-test")
    r = c.post("/api/onboarding/access-key")
    assert r.status_code == 200
    assert r.json()["token"] == "sk-test"


def test_recipe_apply_and_job_endpoints(monkeypatch):
    c = _client(monkeypatch)
    monkeypatch.setattr(ob.onboarding, "apply_recipe",
                        lambda rid, machine, launch: "job1")
    monkeypatch.setattr(ob.onboarding, "_job",
                        lambda jid: {"id": jid, "phase": "done",
                                     "done": True})
    r = c.post("/api/onboarding/recipes/demo-0.6b/apply",
               json={"machine": "host", "launch": True})
    assert r.status_code == 200
    assert r.json()["job_id"] == "job1"
    j = c.get("/api/onboarding/recipes/jobs/job1")
    assert j.status_code == 200
    assert j.json()["done"] is True


def test_mark_and_metrics(monkeypatch):
    c = _client(monkeypatch)
    seen = {}

    def _mark(event, extra=None):
        seen["event"] = event
        return {"event": event}
    monkeypatch.setattr(ob.onboarding, "mark", _mark)
    monkeypatch.setattr(ob.onboarding, "metrics_summary",
                        lambda: {"events": {"setup_shown": 1}})
    r = c.post("/api/onboarding/mark", json={"event": "setup_shown"})
    assert r.status_code == 200
    assert seen["event"] == "setup_shown"
    m = c.get("/api/onboarding/metrics")
    assert m.status_code == 200
    assert m.json()["events"]["setup_shown"] == 1
