"""Per-key request quotas: day/week/month windows and unlimited defaults."""
import os
import sys
import time

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import usage_collector as uc


def _row(ts):
    return {"ts": ts, "kind": "chat", "model": "m", "status": 200,
            "dur": 1.0, "tok_in": 1, "tok_out": 2}


def test_quota_unlimited_by_default(monkeypatch):
    monkeypatch.setattr(uc, "_iter_rows", lambda name: iter([]))
    uc.reset_quota_cache()
    state = uc.quota_state("user", {})
    assert state["windows"]["day"]["limit"] == 0
    assert state["windows"]["day"]["remaining"] is None
    assert uc.check_quota("user", {}) is None
    assert uc.check_quota("user", {"day": 0, "week": None}) is None


def test_quota_counts_current_windows_and_blocks(monkeypatch):
    now = time.time()
    rows = [_row(now - 10), _row(now - 20), _row(now - 8 * 86400)]
    monkeypatch.setattr(uc, "_iter_rows", lambda name: iter(rows))
    uc.reset_quota_cache()
    state = uc.quota_state("user", {"day": 2, "week": 5, "month": 10}, now=now)
    assert state["windows"]["day"]["used"] == 2
    assert state["windows"]["week"]["used"] >= 2
    assert state["windows"]["month"]["used"] == 3
    blocked = uc.check_quota("user", {"day": 2}, now=now)
    assert blocked and blocked["period"] == "day"
    assert blocked["reset_in"] > 0


def test_quota_cache_rolls_to_new_day(monkeypatch):
    now = time.time()
    rows = [_row(now - 10)]
    monkeypatch.setattr(uc, "_iter_rows", lambda name: iter(rows))
    uc.reset_quota_cache()
    state = uc.quota_state("user", {"day": 10}, now=now)
    assert state["windows"]["day"]["used"] == 1
    # Next local day: reload from rows (all rows are older than new day)
    tomorrow = now + 2 * 86400
    state2 = uc.quota_state("user", {"day": 10}, now=tomorrow)
    assert state2["windows"]["day"]["used"] == 0


def test_access_normalizes_limits():
    import access
    assert access.normalize_limits({"day": "3", "week": 0,
                                    "month": -1, "bad": 9}) == {"day": 3}
    assert access.normalize_limits(None) == {}
