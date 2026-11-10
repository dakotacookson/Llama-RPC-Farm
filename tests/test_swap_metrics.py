"""Activity tab: /api/swap/activity.

It only reads llama-swap's own request log (/api/metrics/activity). Scoped keys must see only rows for their granted
models and never the client address; embedding rows get an input
tokens/s figure from the request duration (llama-server sends no
timings for embeddings).
"""
import asyncio
import os
import sys

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import pytest

import access
import app as app_module
import routes.swap as swap_routes


def _row(i, model, inp=1000, out=0, dur=2000, status=200, tps=-1):
    return {"id": i, "timestamp": "2026-11-09T02:55:23+03:00",
            "src": "ip:192.168.1.50", "model": model, "task": "embedding",
            "req_path": "/v1/embeddings", "resp_status_code": status,
            "duration_ms": dur,
            "tokens": {"input_tokens": inp, "output_tokens": out,
                       "cache_tokens": -1, "prompt_per_second": -1,
                       "tokens_per_second": tps}}


@pytest.fixture
def swap(monkeypatch):
    payloads = {
        "/api/metrics/activity": {"total": 3, "data": [
            _row(3, "emb", inp=1736, dur=1539),
            _row(2, "chat", inp=20, out=200, dur=4000, tps=50.0),
            _row(1, "emb", inp=10, dur=100, status=500)]},
    }
    monkeypatch.setattr(swap_routes.swap_manager, "status", lambda: {
        "running": True, "host": "h", "host_ip": "127.0.0.1", "port": 8080})
    monkeypatch.setattr(swap_routes, "_swap_api_get",
                        lambda st, path, timeout=5: payloads[path.split("?")[0]])
    return payloads


def test_activity_rows_and_per_model_totals(swap):
    a = asyncio.run(swap_routes.swap_activity(limit=50))
    assert a["running"] and not a["scoped"] and a["total"] == 3
    first = a["rows"][0]
    assert first["input_tps"] == round(1736 / 1.539, 1)
    assert first["gen_tps"] is None and first["src"] == "ip:192.168.1.50"
    emb = next(m for m in a["models"] if m["model"] == "emb")
    assert (emb["requests"], emb["errors"], emb["input_tokens"]) == (2, 1, 1746)
    chat = next(m for m in a["models"] if m["model"] == "chat")
    assert chat["gen_tps"] == 50.0 and chat["output_tokens"] == 200


def test_scoped_key_sees_only_granted_models_without_client(swap, monkeypatch):
    monkeypatch.setattr(access, "gate_open", lambda: False)
    monkeypatch.setattr(access, "admin_sees_everything", lambda: False)
    monkeypatch.setattr(access, "raw_key_record",
                        lambda: {"name": "k", "models": {"chat"}, "admin": False})
    a = asyncio.run(swap_routes.swap_activity(limit=50))
    assert a["scoped"]
    assert [r["model"] for r in a["rows"]] == ["chat"]
    assert all(r["src"] is None for r in a["rows"])
    assert [m["model"] for m in a["models"]] == ["chat"]


def test_unknown_caller_sees_nothing(swap, monkeypatch):
    monkeypatch.setattr(access, "gate_open", lambda: False)
    monkeypatch.setattr(access, "admin_sees_everything", lambda: False)
    monkeypatch.setattr(access, "raw_key_record", lambda: None)
    a = asyncio.run(swap_routes.swap_activity(limit=50))
    assert a["rows"] == [] and a["models"] == []


def test_swap_down(monkeypatch):
    monkeypatch.setattr(swap_routes.swap_manager, "status", lambda: {"running": False})
    assert asyncio.run(swap_routes.swap_activity())["running"] is False


def test_speed_ignores_cold_start_and_uses_wall_time():
    # a 60 s first request (model load, 2 tokens) then three 1 s requests
    # of 1000 tokens each, one per second, 10 s after the load ended
    def row(end, dur_ms, tokens, status=200):
        return swap_routes._activity_row({
            "timestamp": f"2026-11-09T03:00:{end:02d}+00:00", "model": "m",
            "resp_status_code": status, "duration_ms": dur_ms,
            "tokens": {"input_tokens": tokens, "output_tokens": 0}})
    rows = [row(0, 60000, 2), row(11, 1000, 1000), row(12, 1000, 1000),
            row(13, 1000, 1000), row(14, 1000, 999999, status=400)]
    sp = swap_routes._speed_stats(rows)
    assert sp["median_tps"] == 1000.0               # the load does not drag it
    assert sp["window_s"] == 73.0                    # 03:00:00-60 s .. 03:00:13
    assert sp["throughput_tps"] == round(3002 / 73.0, 1)   # failed row excluded
