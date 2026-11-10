"""Activity tab request details: swap_captures + GET /api/swap/capture/{id}.

llama-swap drops its request captures within seconds, so the farm copies
the captures of FAILED requests; details are admin-only and credentials
are always redacted.
"""
import asyncio
import base64
import os
import sys

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import pytest
from fastapi import HTTPException

import access
import app as app_module
import routes.swap as swap_routes
import swap_captures as sc

# the real failed request seen live: Turkish text sent as cp1254, not UTF-8
BAD_BODY = ('{"model": "m", "input": ["sınıf öğretmenleri"]}').encode("cp1254")


def _cap(body=BAD_BODY):
    return {"id": 2, "req_path": "/v1/embeddings",
            "req_headers": {"Content-Type": "application/json",
                            "Authorization": "Bearer sk-secret",
                            "X-Api-Key": "k"},
            "req_body": base64.b64encode(body).decode(),
            "resp_headers": {"Server": "llama.cpp"}, "resp_body": None}


@pytest.fixture(autouse=True)
def clean(monkeypatch):
    monkeypatch.setattr(sc, "_store", sc.OrderedDict())
    yield


def test_summarize_redacts_and_flags_bad_utf8():
    s = sc.summarize(_cap(), {"capture_id": "c1", "resp_status_code": 400,
                              "model": "m"})
    assert s["req_headers"]["Authorization"] == "[redacted]"
    assert s["req_headers"]["X-Api-Key"] == "[redacted]"
    assert s["req_headers"]["Content-Type"] == "application/json"
    assert s["req_body_utf8"] is False and s["req_body_bytes"] == len(BAD_BODY)
    assert s["resp_body"] is None and s["status"] == 400


def test_poll_keeps_only_failed_requests():
    activity = {"data": [
        {"capture_id": "ok1", "has_capture": True, "resp_status_code": 200},
        {"capture_id": "bad1", "has_capture": True, "resp_status_code": 400,
         "model": "m"},
        {"capture_id": "gone", "has_capture": True, "resp_status_code": 500}]}
    caps = {"bad1": _cap(), "gone": {"error": {"message": "capture not found"}}}
    asked = []

    def fetch(path):
        asked.append(path)
        if path.startswith("/api/metrics/activity"):
            return activity
        return caps[path.rsplit("/", 1)[1]]
    sc.poll_once(fetch)
    assert sc.get("bad1")["req_body_utf8"] is False
    assert sc.get("ok1") is None and sc.get("gone") is None
    assert "/api/captures/ok1" not in asked           # successes not fetched
    n = len(asked)
    sc.poll_once(fetch)                               # already kept: no refetch
    assert "/api/captures/bad1" not in asked[n:]


def test_store_is_bounded(monkeypatch):
    monkeypatch.setattr(sc, "KEEP", 3)
    for i in range(5):
        sc.remember(f"c{i}", {"n": i})
    assert list(sc._store) == ["c2", "c3", "c4"]


def _admin(monkeypatch):
    monkeypatch.setattr(access, "gate_open", lambda: True)
    monkeypatch.setattr(access, "view_allow", lambda: None)


def test_endpoint_serves_kept_capture(monkeypatch):
    _admin(monkeypatch)
    sc.remember("bad1", sc.summarize(_cap(), {"capture_id": "bad1"}))
    r = asyncio.run(swap_routes.swap_capture("bad1"))
    assert r["source"] == "kept" and r["req_body_utf8"] is False


def test_endpoint_live_and_expired(monkeypatch):
    _admin(monkeypatch)
    monkeypatch.setattr(swap_routes.swap_manager, "status", lambda: {
        "running": True, "host": "h", "host_ip": "127.0.0.1", "port": 8080})
    live = {"live1": _cap('{"input": "ok"}'.encode())}
    monkeypatch.setattr(swap_routes, "_swap_api_get",
                        lambda st, path, timeout=5: live.get(
                            path.rsplit("/", 1)[1],
                            {"error": {"message": "capture not found"}}))
    r = asyncio.run(swap_routes.swap_capture("live1"))
    assert r["source"] == "live" and r["req_body_utf8"] is True
    assert r["req_headers"]["Authorization"] == "[redacted]"
    with pytest.raises(HTTPException) as e:
        asyncio.run(swap_routes.swap_capture("expired"))
    assert e.value.status_code == 404


def test_endpoint_is_admin_only(monkeypatch):
    monkeypatch.setattr(access, "gate_open", lambda: False)
    monkeypatch.setattr(access, "admin_sees_everything", lambda: False)
    monkeypatch.setattr(access, "raw_key_record",
                        lambda: {"name": "k", "models": {"m"}, "admin": False})
    sc.remember("bad1", sc.summarize(_cap(), {"capture_id": "bad1"}))
    with pytest.raises(HTTPException) as e:
        asyncio.run(swap_routes.swap_capture("bad1"))
    assert e.value.status_code == 403
