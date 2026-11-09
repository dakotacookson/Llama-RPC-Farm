"""Rolling hardware history for every machine (Performance tab).

The dashboard only asks each machine for its CURRENT stats, so nothing
remembered how a peer's GPU, CPU or RAM moved over time. This sampler
collects the same per-machine stats (stats_collector.machine_stats: local
probes for the master, the peer agent for the others) every INTERVAL
seconds in a background thread and keeps the last KEEP_SECONDS in memory.

FARM_STATS_HISTORY_SEC sets the starting interval (default 5; 0 disables
the sampler); set_interval() changes it at runtime (Performance tab).
One round over all machines takes a few seconds (peers answer through
their agents), so the interval is kept at MIN_INTERVAL or above.
Nothing is written to disk: history starts when the app does.
"""
import os
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import config_loader as cfg
import stats_collector

MIN_INTERVAL, MAX_INTERVAL = 5, 60
INTERVAL = int(os.environ.get("FARM_STATS_HISTORY_SEC", "5") or 0)
if INTERVAL > 0:
    INTERVAL = max(MIN_INTERVAL, min(INTERVAL, MAX_INTERVAL))
KEEP_SECONDS = 3600

_lock = threading.Lock()
_history = {}            # machine name -> deque of samples
_started = False


def _sample(m):
    """Trim one machine_stats() result to what the charts need."""
    gpus = []
    for g in m.get("gpus") or []:
        gpus.append({
            "index": g.get("index"), "name": g.get("name"),
            "util": g.get("util"), "temp": g.get("temp"),
            "mem_used_gb": g.get("mem_used_gb"),
            "mem_total_gb": g.get("mem_total_gb"),
        })
    return {
        "ts": time.time(), "status": m.get("status"),
        "cpu": m.get("cpu_pct"),
        "ram_used_gb": m.get("ram_used_gb"),
        "ram_total_gb": m.get("ram_total_gb"),
        "gpus": gpus,
    }


def record(name, stats, now=None):
    """Store one machine's stats; drop samples older than KEEP_SECONDS."""
    s = _sample(stats)
    if now is not None:
        s["ts"] = now
    with _lock:
        q = _history.setdefault(name, deque())
        q.append(s)
        while q and s["ts"] - q[0]["ts"] > KEEP_SECONDS:
            q.popleft()


def collect_once():
    machines = cfg.machines.all()
    if not machines:
        return
    with ThreadPoolExecutor(max_workers=len(machines)) as pool:
        results = list(pool.map(_safe_stats, machines))
    for m, st in zip(machines, results):
        if st is not None:
            record(m.get("name"), st)


def _safe_stats(machine):
    try:
        return stats_collector.machine_stats(machine)
    except Exception:
        return None


def _loop():
    while True:
        t0 = time.time()
        collect_once()
        time.sleep(max(1.0, INTERVAL - (time.time() - t0)))


def set_interval(seconds):
    """Change the sampling interval (clamped to MIN..MAX); starts the
    sampler if it was disabled. Returns the interval now in effect."""
    global INTERVAL
    INTERVAL = max(MIN_INTERVAL, min(int(seconds), MAX_INTERVAL))
    start()
    return INTERVAL


def start():
    """Start the background sampler once (no-op when disabled)."""
    global _started
    if _started or INTERVAL <= 0:
        return
    _started = True
    threading.Thread(target=_loop, daemon=True, name="stats-history").start()


def history(minutes=15, points=180, now=None):
    """{name: [samples]} for the last `minutes`, thinned to `points`."""
    now = time.time() if now is None else now
    cut = now - minutes * 60
    out = {}
    with _lock:
        items = {k: list(v) for k, v in _history.items()}
    for name, rows in items.items():
        rows = [r for r in rows if r["ts"] >= cut]
        step = max(1, -(-len(rows) // points))     # ceil
        thin = rows[::step]
        if rows and thin[-1] is not rows[-1]:
            thin.append(rows[-1])                  # always keep the latest
        out[name] = thin
    return out
