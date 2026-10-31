"""Per-API-KEY usage tracking for the swap gateway (:8090).

STORE: one Fernet-encrypted JSONL per key NAME —
  config/usage/key.<name>.enc
Rows are sealed one-per-line Fernet tokens; the plaintext payload of
each row:
  {"ts": <unix float>, "kind": "chat|systemone|...", "model": <mid>,
   "status": <int>, "dur": <seconds>, "tok_in": <int>, "tok_out": <int>,
   "est": <bool>}          (tok fields absent on errors without tokens)

AGGREGATION slices the single store by timestamp buckets (day/month/
year/all) at read time — no daily files, no rotation windows: the per-
key file IS the long-term history. Files are small (a token per
request); reads decrypt line-by-line and bucket in memory.

PRIVACY: keyed by identity NAME (never the raw token), no prompt or
completion text is ever recorded — sizes, statuses, durations, tokens.
"""
import json
import os
import queue
import threading
import time

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
USAGE_DIR = os.path.join(BASE_DIR, "config", "usage")
USAGE_SETTINGS_KEY = "usage_tracking"   # settings.yaml gate (default on)

_q: "queue.Queue" = queue.Queue(maxsize=4096)
_qthread = None
_qlock = threading.Lock()


def enabled() -> bool:
    try:
        import yaml
        p = os.path.join(USAGE_DIR, "..", "settings.yaml")
        s = yaml.safe_load(open(p)) or {}
        return bool(s.get(USAGE_SETTINGS_KEY, True))
    except Exception:
        return True


def kind_for(path: str) -> str:
    p = "/" + (path or "").lstrip("/")
    for pref, kind in (
            ("v1/chat/completions", "chat"),
            ("v1/completions", "completions"),
            ("v1/embeddings", "embeddings"),
            ("v1/systemone", "systemone"),
            ("v1/responses", "responses"),
            ("v1/rerank", "rerank"),
            ("v1/audio/", "audio"),
            ("v1/images/", "images")):
        if p.lstrip('/').startswith(pref.lstrip('/')):
            return kind
    return "other"


def _keyfile(name):
    # lower(): key NAME case varies by source ("Dad" from the key record
    # vs "dad" from the login session) — same human, one file.
    safe = "".join(c if c.isalnum() or c in "-_." else "-"
                   for c in str(name).lower()).strip("-") or "unknown"
    return os.path.join(USAGE_DIR, f"key.{safe}.enc")


def record(name, path, model, status, duration_s, tokens=None,
           estimated=False):
    """Queue one request row (name = key NAME)."""
    _ensure_thread()
    if not enabled() or not name:
        return
    row = {"ts": round(time.time(), 3), "kind": kind_for(path),
           "name": name, "model": model or "",
           "status": int(status or 0), "dur": round(duration_s or 0, 3),
           "tok_in": ((tokens or {}).get("prompt_tokens")
                      or (tokens or {}).get("input_tokens") or 0),
           "tok_out": ((tokens or {}).get("completion_tokens")
                       or (tokens or {}).get("output_tokens") or 0)}
    if estimated:
        row["est"] = True
    try:
        _q.put_nowait(row)
    except queue.Full:
        pass  # telemetry never backpressures the proxy


def _writer_loop():
    os.makedirs(USAGE_DIR, exist_ok=True)
    import secrets_vault as _v
    while True:
        row = _q.get()
        if row is None:
            return
        try:
            tok = _v.encrypt_value(json.dumps(row, separators=(",", ":")))
            with open(_keyfile(row["name"]), "a", encoding="utf-8") as f:
                f.write(tok + "\n")
        except Exception:
            pass


def _ensure_thread():
    global _qthread
    with _qlock:
        if _qthread is None or not _qthread.is_alive():
            _qthread = threading.Thread(
                target=_writer_loop, name="usage-collector", daemon=True)
            _qthread.start()


def _iter_rows(name):
    """Yield decrypted row dicts from one key's file."""
    p = _keyfile(name)
    if not os.path.exists(p):
        return
    import secrets_vault as _v
    try:
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(_v.decrypt_value(line))
                    if isinstance(r, dict):
                        yield r
                except Exception:
                    continue
    except Exception:
        return


def _empty_bucket():
    return {"requests": 0, "errors": 0, "wall_s": 0.0, "hours": 0.0,
            "tok_in": 0, "tok_out": 0, "est": 0}


def _add(bucket, r):
    bucket["requests"] += 1
    # decode-throughput basis: ratio-of-sums (_Stok/_Sdur). Per-request
    # rates (tok_out/dur) must NEVER be arithmetically averaged — a
    # 1-token/0.01s blip would weigh the same as a 10k-token run and
    # the headline number would be fiction. Only EXACT rows count:
    # estimated token counts (streaming guesswork) are usage, not speed.
    dur = float(r.get("dur") or 0)
    tout = int(r.get("tok_out") or 0)
    if dur > 0 and tout > 0 and not r.get("est"):
        bucket["_spd_tok"] = bucket.get("_spd_tok", 0) + tout
        bucket["_spd_dur"] = round(bucket.get("_spd_dur", 0.0) + dur, 3)
    bucket["errors"] += 1 if int(r.get("status") or 0) >= 400 else 0
    bucket["wall_s"] = round(bucket["wall_s"] + float(r.get("dur") or 0), 3)
    bucket["tok_in"] += int(r.get("tok_in") or 0)
    bucket["tok_out"] += int(r.get("tok_out") or 0)
    bucket["est"] += 1 if r.get("est") else 0
    bucket["hours"] = round(bucket["wall_s"] / 3600.0, 4)


def _speed(b):
    """Decode throughput for a bucket: Σtok_out / Σdur over exact rows
    (ratio-of-sums). None when no exact decode volume exists."""
    t = b.get("_spd_tok") or 0
    d = b.get("_spd_dur") or 0
    if t > 0 and d > 0:
        return round(t / d, 2)
    return None


def aggregate(name, period="all", count=12):
    """Bucketed totals for ONE key name.
    period: all | day | month | year — day = last `count` days (rows
    bucketed by YYYY-MM-DD), month = last `count` months (YYYY-MM),
    year = last `count` years (YYYY); all = one grand-total bucket.
    Returns {"buckets": [{"label": ..., ...bucket}], "total": bucket,
             "per_model": {model: bucket}}."""
    _ensure_thread()
    import datetime as _dt
    now = time.time()
    fmt, labels = {
        "day": ("%Y-%m-%d", _dt.timedelta(days=1)),
        "month": ("%Y-%m", None),
        "year": ("%Y-%Y"[:4] + "%Y", None),
    }.get(period, (None, None))
    if period == "year":
        fmt = "%Y"
    if period == "all" or fmt is None:
        buckets = {"ALL": _empty_bucket()}
    else:
        # generate the last `count` labels so gaps show as zero-buckets
        buckets = {}
        if period == "day":
            base = _dt.date.today()
            for i in range(count - 1, -1, -1):
                buckets[(base - _dt.timedelta(days=i)).isoformat()] = \
                    _empty_bucket()
        elif period == "month":
            today = _dt.date.today()
            y, m = today.year, today.month
            for i in range(count - 1, -1, -1):
                mm = m - i
                yy = y
                while mm <= 0:
                    mm += 12
                    yy -= 1
                buckets[f"{yy:04d}-{mm:02d}"] = _empty_bucket()
        else:
            y = _dt.date.today().year
            for i in range(count - 1, -1, -1):
                buckets[str(y - i)] = _empty_bucket()
    per_model = {}
    total = _empty_bucket()
    for r in _iter_rows(name):
        # period slice
        if period == "day" and now - float(r.get("ts") or 0) > count * 86400:
            continue
        label = time.strftime(fmt, time.localtime(float(r.get("ts") or 0))) \
            if fmt else "ALL"
        b = buckets.get(label)
        if b is None:
            # row older than the window: fold into nothing for buckets,
            # but still count toward TOTALS (all-time truth)
            pass
        else:
            _add(b, r)
        _add(total, r)
        mb = per_model.setdefault(r.get("model") or "?", _empty_bucket())
        _add(mb, r)
    # decode-speed average per model: Σtok_out / Σdur (ratio-of-sums —
    # the same basis as the per-day cells below, so totals reconcile)
    per_model_speed = {m: _speed(b) for m, b in per_model.items()}
    def _clean(b):
        return {k: v for k, v in b.items()
                if not k.startswith("_")}
    return {"buckets": [{"label": k, **_clean(v)}
                        for k, v in sorted(buckets.items())],
            "total": _clean(total),
            "per_model": {m: _clean(v) for m, v in sorted(per_model.items())},
            "avg_tok_per_s": _speed(total),
            "tok_out_per_s": per_model_speed}


def by_model_daily(name, days=30):
    """Per-model × per-day matrix for the 'Usage by model' tab:
    {model: [{"label": date, "requests": n, "tok_out": n, "tok_in": n,
              "wall_s": s, "avg_spd": tok/s|None}]} sorted by date.
    Also carries per-model all-time totals (requests/tokens/hours/
    avg speed/errors/last_used)."""
    import datetime as _dt
    _ensure_thread()
    end = _dt.date.today()
    labels = [(end - _dt.timedelta(days=d)).isoformat()
              for d in range(days - 1, -1, -1)]
    per = {}
    for r in _iter_rows(name):
        model = r.get("model") or "?"
        d = time.strftime("%Y-%m-%d", time.localtime(r.get("ts") or 0))
        b = per.setdefault(model, {"days": {L: {"requests": 0, "tok_out": 0,
                                                 "tok_in": 0, "wall_s": 0.0}
                                    for L in labels},
                                   "total": _empty_bucket(),
                                   "last": 0})
        day = b["days"].get(d)
        if day is None:
            continue  # outside window
        day["requests"] += 1
        day["tok_out"] += int(r.get("tok_out") or 0)
        day["tok_in"] += int(r.get("tok_in") or 0)
        day["wall_s"] = round(day["wall_s"] + float(r.get("dur") or 0), 3)
        _add(b["total"], r)
        b["last"] = max(b["last"], float(r.get("ts") or 0))
    out = {}
    for model, b in per.items():
        days_out = []
        for L in labels:
            row = b["days"][L]
            spd = round(row["tok_out"] / row["wall_s"], 2) \
                if row["wall_s"] > 0 and row["tok_out"] > 0 else None
            days_out.append({"label": L, **row, "avg_spd": spd})
        out[model] = {
            "days": days_out,
            "total": {k: v for k, v in b["total"].items()
                      if not k.startswith("_")},
            "avg_spd": _speed(b["total"]),
            "last_used": b["last"],
        }
    return out


def key_files():
    """All tracked identities (file inventory → User page 'tracked users')."""
    os.makedirs(USAGE_DIR, exist_ok=True)
    out = []
    for f in sorted(os.listdir(USAGE_DIR)):
        if f.startswith("key.") and f.endswith(".enc"):
            out.append(f[4:-4])
    return out
