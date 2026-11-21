"""Load-balanced virtual models over llama-swap deployments.

A model with `load_balance: true` treats its selected `device_order` as
per-machine replica groups: every machine gets a full model copy using all
of that machine's selected GPU/RAM chips. Requests to the public model id
are distributed across those replicas by the gateway.

llama-swap groups are NOT used here: they manage swap/co-residency, not
request distribution.
"""
from __future__ import annotations

import threading
import time

_LOCK = threading.RLock()
_RR = {}
_INFLIGHT = {}
_CATALOG = {"ts": 0.0, "data": {}}


def _safe_label(label):
    s = "".join(c if c.isalnum() or c in "-_" else "-"
                for c in str(label or "").strip())
    return s.strip("-") or "replica"


def replica_key(public_id, machine):
    return f"{public_id}__lb-{_safe_label(machine)}"


def groups_for_model(model):
    """Split a model's device_order into one full replica per machine.

    Returns [{"machine", "device_order", "key", "weight"}].
    Device tokens are `machine-RESOURCE`; machine names may contain hyphens,
    so split at the LAST dash.
    """
    public = (model.get("model_id") or "").strip()
    if not public:
        return []
    order = [t.strip() for t in
             str(model.get("device_order") or "").split(",") if t.strip()]
    by_machine = {}
    for tok in order:
        machine, sep, _res = tok.rpartition("-")
        if not sep or not machine:
            continue
        by_machine.setdefault(machine, []).append(tok)
    out = []
    for machine, toks in by_machine.items():
        out.append({
            "machine": machine,
            "device_order": ",".join(toks),
            "key": replica_key(public, machine),
            "weight": len(toks),
        })
    return out


def _catalog():
    now = time.time()
    if now - _CATALOG["ts"] < 3.0 and _CATALOG["data"]:
        return _CATALOG["data"]
    out = {}
    try:
        import config_loader as cfg
        for model in cfg.list_models():
            if not model.get("load_balance"):
                continue
            mid = (model.get("model_id") or "").strip()
            if not mid:
                continue
            groups = groups_for_model(model)
            if not groups:
                continue
            try:
                cap = max(1, int(model.get("lb_max_inflight") or 1))
            except (TypeError, ValueError):
                cap = 1
            reps = []
            for g in groups:
                reps.append({"key": g["key"], "public": mid,
                             "label": g["machine"],
                             "machine": g["machine"],
                             "weight": max(1, g["weight"]),
                             "max_inflight": cap})
            out[mid] = {"public": mid,
                        "strategy": (model.get("lb_strategy")
                                     or "least_inflight"),
                        "replicas": reps}
    except Exception:
        out = {}
    _CATALOG.update(ts=now, data=out)
    return out


def invalidate():
    with _LOCK:
        _CATALOG["ts"] = 0.0
        _CATALOG["data"] = {}


def replica_keys(public_id):
    ent = _catalog().get(public_id)
    return [r["key"] for r in ent["replicas"]] if ent else []


def public_id(internal_id):
    internal_id = str(internal_id or "")
    for public, ent in _catalog().items():
        for rep in ent["replicas"]:
            if rep["key"] == internal_id:
                return public
    return internal_id


def collapse(ids):
    out = []
    seen = set()
    for mid in ids or []:
        pid = public_id(mid)
        if pid not in seen:
            seen.add(pid)
            out.append(pid)
    return out


def choose(public_id):
    """Pick a replica and increment its in-flight counter."""
    ent = _catalog().get(public_id)
    if not ent or not ent["replicas"]:
        return None
    reps = ent["replicas"]
    with _LOCK:
        inf = _INFLIGHT.setdefault(public_id, {})
        below = [r for r in reps if inf.get(r["key"], 0) < r["max_inflight"]]
        pool = below or reps
        if (ent.get("strategy") or "least_inflight") == "round_robin":
            start = _RR.get(public_id, 0) % len(pool)
            chosen = pool[start]
            _RR[public_id] = (start + 1) % len(pool)
        else:
            chosen = min(pool, key=lambda r: (
                (inf.get(r["key"], 0) + 1) / max(1, r["weight"]),
                r["key"]))
        inf[chosen["key"]] = inf.get(chosen["key"], 0) + 1
        return {"public": public_id, "key": chosen["key"],
                "weight": chosen["weight"]}


def release(choice):
    if not choice:
        return
    with _LOCK:
        inf = _INFLIGHT.get(choice.get("public"))
        if not inf:
            return
        key = choice.get("key")
        n = inf.get(key, 0) - 1
        if n > 0:
            inf[key] = n
        else:
            inf.pop(key, None)


def inflight():
    with _LOCK:
        return {k: dict(v) for k, v in _INFLIGHT.items()}


def rewrite_model_list(parsed, allow=None, matcher=None):
    """Hide replica ids and expose one entry per load-balanced public model."""
    data = list(parsed.get("data") or [])
    lb_public = _catalog()
    by_id = {}
    internals = {}
    for item in data:
        mid = item.get("id") or ""
        pid = public_id(mid)
        if pid != mid:
            internals.setdefault(pid, []).append(item)
            continue
        by_id[mid] = item
    for public, items in internals.items():
        loaded = any(((i.get("status") or {}).get("value") == "loaded")
                     for i in items)
        if public not in by_id:
            base = dict(items[0])
            base["id"] = public
            if (base.get("status") or {}).get("value") != "loaded":
                base.setdefault("status", {})["value"] = (
                    "loaded" if loaded else "unloaded")
            by_id[public] = base
        elif loaded:
            by_id[public].setdefault("status", {})["value"] = "loaded"
    for public in lb_public:
        if public not in by_id:
            by_id[public] = {"id": public, "object": "model"}
    out = []
    for mid, item in by_id.items():
        if allow is not None and matcher and not matcher(mid, allow):
            continue
        out.append(item)
    parsed = dict(parsed)
    parsed["data"] = out
    return parsed
