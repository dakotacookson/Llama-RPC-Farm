"""Keep llama-swap's request captures for failed requests (Activity tab).

llama-swap records each proxied request's headers and body as a
"capture" (/api/captures/<id>), but drops it within seconds (observed:
gone 15-30 s after the request). A background poller checks the recent
activity every POLL_SEC seconds while llama-swap runs and copies the
capture of every new failed request (status >= 400) into memory, so the
Activity tab can still show why it failed. Successful requests are not
kept; their capture is only fetched live, while llama-swap still has it.

Credentials never leave this module: Authorization / X-Api-Key / Cookie
style headers are redacted before anything is stored or returned.
"""
import base64
import threading
import time
from collections import OrderedDict

POLL_SEC = 5
KEEP = 100                 # failed-request captures kept in memory
BODY_LIMIT = 8000          # characters of a body returned to the UI

SECRET_HEADERS = {"authorization", "x-api-key", "cookie", "set-cookie",
                  "proxy-authorization"}

_lock = threading.Lock()
_store = OrderedDict()     # capture_id -> summary
_started = False


def redact(headers):
    return {k: ("[redacted]" if k.lower() in SECRET_HEADERS else v)
            for k, v in (headers or {}).items()}


def _body(raw, is_b64=True):
    """Decode a captured body. Returns (text, utf8_ok, size_bytes)."""
    if raw is None:
        return None, True, 0
    if is_b64:
        try:
            data = base64.b64decode(raw)
        except Exception:
            data = str(raw).encode("utf-8", "replace")
    else:
        data = str(raw).encode("utf-8", "replace")
    try:
        text, ok = data.decode("utf-8"), True
    except UnicodeDecodeError:
        text, ok = data.decode("utf-8", "replace"), False
    return text[:BODY_LIMIT], ok, len(data)


def summarize(capture, row=None):
    """The parts of a llama-swap capture the Activity tab shows."""
    req, req_ok, req_size = _body(capture.get("req_body"))
    resp, resp_ok, resp_size = _body(capture.get("resp_body"))
    row = row or {}
    return {
        "capture_id": row.get("capture_id"),
        "ts": row.get("timestamp"), "model": row.get("model"),
        "status": row.get("resp_status_code"),
        "path": capture.get("req_path") or row.get("req_path"),
        "req_headers": redact(capture.get("req_headers")),
        "req_body": req, "req_body_utf8": req_ok, "req_body_bytes": req_size,
        "resp_headers": redact(capture.get("resp_headers")),
        "resp_body": resp, "resp_body_utf8": resp_ok,
        "resp_body_bytes": resp_size,
    }


def remember(capture_id, summary):
    with _lock:
        _store[capture_id] = summary
        _store.move_to_end(capture_id)
        while len(_store) > KEEP:
            _store.popitem(last=False)


def get(capture_id):
    with _lock:
        return _store.get(capture_id)


def poll_once(fetch):
    """fetch(path) -> parsed JSON from llama-swap. Copies the capture of
    every new failed request still present in llama-swap."""
    data = fetch("/api/metrics/activity?limit=50&page=1")
    for row in data.get("data") or []:
        cid = row.get("capture_id")
        if (not cid or not row.get("has_capture")
                or (row.get("resp_status_code") or 0) < 400 or get(cid)):
            continue
        try:
            cap = fetch(f"/api/captures/{cid}")
        except Exception:
            continue
        if isinstance(cap, dict) and "error" not in cap:
            remember(cid, summarize(cap, row))


def start(status, fetch):
    """Run poll_once every POLL_SEC while status()['running'] is true."""
    global _started
    if _started:
        return
    _started = True

    def loop():
        while True:
            try:
                st = status()
                if st.get("running"):
                    poll_once(lambda path: fetch(st, path))
            except Exception:
                pass
            time.sleep(POLL_SEC)
    threading.Thread(target=loop, daemon=True, name="swap-captures").start()
