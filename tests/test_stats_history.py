"""Performance tab: rolling per-machine history (stats_history) and
GET /api/stats/history."""
import asyncio
import os
import sys

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import pytest

import app as app_module
import routes.scanning as scanning_routes
import stats_history


def _stats(util, cpu=20.0, ram=8.0):
    return {"status": "online", "cpu_pct": cpu, "ram_used_gb": ram,
            "ram_total_gb": 16.0,
            "gpus": [{"index": 0, "name": "RTX 5060", "util": util, "temp": 60,
                      "mem_used_gb": 4.0, "mem_total_gb": 8.0, "uuid": "x"}]}


@pytest.fixture(autouse=True)
def clean(monkeypatch):
    monkeypatch.setattr(stats_history, "_history", {})
    yield


def test_record_trims_to_keep_window():
    t0 = 1_000_000.0
    stats_history.record("laptop", _stats(1), now=t0)
    stats_history.record("laptop", _stats(2), now=t0 + 1800)
    stats_history.record("laptop", _stats(3), now=t0 + stats_history.KEEP_SECONDS + 1)
    utils = [s["gpus"][0]["util"] for s in stats_history._history["laptop"]]
    assert utils == [2, 3]                       # the first one aged out


def test_sample_keeps_only_chart_fields():
    stats_history.record("laptop", _stats(5), now=10.0)
    s = stats_history._history["laptop"][0]
    assert set(s) == {"ts", "status", "cpu", "ram_used_gb", "ram_total_gb", "gpus"}
    assert "uuid" not in s["gpus"][0] and s["gpus"][0]["util"] == 5


def test_history_window_and_thinning():
    now = 50_000.0
    for i in range(100):                          # one sample per 6 s, 10 min
        stats_history.record("laptop", _stats(i), now=now - 600 + i * 6)
    h = stats_history.history(minutes=5, points=10, now=now)["laptop"]
    assert all(s["ts"] >= now - 300 for s in h)  # window cut
    assert len(h) <= 11                           # thinned (+ latest kept)
    assert h[-1]["gpus"][0]["util"] == 99         # latest always present


def test_collect_once_survives_a_failing_machine(monkeypatch):
    machines = [{"name": "master"}, {"name": "laptop"}]
    monkeypatch.setattr(stats_history.cfg.machines, "all", lambda: machines)

    def fake(m):
        if m["name"] == "laptop":
            raise RuntimeError("agent down")
        return _stats(7)
    monkeypatch.setattr(stats_history.stats_collector, "machine_stats", fake)
    stats_history.collect_once()
    assert list(stats_history._history) == ["master"]


def test_history_endpoint(monkeypatch):
    monkeypatch.setattr(app_module.cfg.machines, "all",
                        lambda: [{"name": "master", "role": "master"},
                                 {"name": "laptop", "role": "peer"}])
    stats_history.record("laptop", _stats(3))
    stats_history.record("laptop", _stats(4))
    r = asyncio.run(scanning_routes.get_stats_history(minutes=15, points=400))
    assert r["enabled"] == (stats_history.INTERVAL > 0)
    names = [m["name"] for m in r["machines"]]
    assert names == ["master", "laptop"]          # config order, even without samples
    laptop = r["machines"][1]
    assert laptop["role"] == "peer" and len(laptop["samples"]) == 2


def test_set_interval_is_clamped(monkeypatch):
    monkeypatch.setattr(stats_history, "INTERVAL", 10)
    monkeypatch.setattr(stats_history, "_started", True)      # no thread in tests
    assert stats_history.set_interval(1) == stats_history.MIN_INTERVAL
    assert stats_history.set_interval(999) == stats_history.MAX_INTERVAL
    assert stats_history.set_interval(30) == 30


def test_interval_endpoint_is_admin_only(monkeypatch):
    from fastapi import HTTPException
    import access
    monkeypatch.setattr(stats_history, "_started", True)
    monkeypatch.setattr(stats_history, "INTERVAL", 10)
    monkeypatch.setattr(access, "is_admin", lambda _=None: False)
    with pytest.raises(HTTPException) as e:
        asyncio.run(scanning_routes.set_stats_history_interval({"seconds": 5}))
    assert e.value.status_code == 403
    monkeypatch.setattr(access, "is_admin", lambda _=None: True)
    assert asyncio.run(scanning_routes.set_stats_history_interval({"seconds": 5})) == {"interval": 5}
    with pytest.raises(HTTPException) as e:
        asyncio.run(scanning_routes.set_stats_history_interval({"seconds": "x"}))
    assert e.value.status_code == 400
