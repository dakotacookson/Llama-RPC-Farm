#!/usr/bin/env python3
"""Local-only Qwen3-0.6B load tests (not collected by pytest/CI).

Run against a live farm UI/gateway on this host:

    .venv/bin/python tests/local/load_tiny.py single
    .venv/bin/python tests/local/load_tiny.py multi

Single mode loads the existing qwen3-0-6b-demo model on the master.
Multi mode loads its saved `rpc` variant (qwen3-0-6b-demo--rpc) from the
same shared store; it does not create or delete model configs. The saved
variant must exist (`qwen3-0-6b-demo` with an `rpc` variant).
"""
import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request

DEFAULT_APP = "http://127.0.0.1:8199"
DEFAULT_GATEWAY = "http://127.0.0.1:8090"
MODEL = "qwen3-0-6b-demo"
MULTI_MODEL = "qwen3-0-6b-demo--rpc"


def _request(method, url, body=None, timeout=300):
    data = None
    headers = {"Accept": "application/json"}
    if body is not None:
        data = json.dumps(body, separators=(",", ":")).encode()
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers,
                                 method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read() or b"{}"
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "replace")[:1000]
        raise RuntimeError(f"HTTP {e.code} {method} {url}: {detail}") from e
    try:
        return json.loads(raw.decode())
    except Exception:
        return {"raw": raw.decode("utf-8", "replace")}


def log(msg):
    print(f"[tiny] {msg}", flush=True)


def machines(app):
    return _request("GET", f"{app}/api/machines", timeout=20)


def loaded_ids(app):
    d = _request("GET", f"{app}/api/swap/loaded", timeout=20)
    out = set()
    for m in d.get("loaded") or []:
        if isinstance(m, str):
            out.add(m)
        elif isinstance(m, dict):
            out.add(m.get("id") or m.get("model"))
    return out


def wait_loaded(app, model_id, timeout=900):
    end = time.time() + timeout
    while time.time() < end:
        try:
            if model_id in loaded_ids(app):
                return
        except Exception:
            pass
        time.sleep(2)
    raise RuntimeError(f"{model_id} did not appear in /api/swap/loaded")


def chat(gateway, model_id, timeout=300):
    payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": "Reply with exactly: OK"}],
        "max_tokens": 16,
        "temperature": 0,
        "stream": False,
    }
    out = _request("POST", f"{gateway}/v1/chat/completions",
                   payload, timeout=timeout)
    choices = out.get("choices") or []
    if not choices:
        raise RuntimeError(f"chat returned no choices: {out}")
    msg = choices[0].get("message") or {}
    log(f"chat ok model={model_id} content={msg.get('content')!r}")
    return out


def load_model(app, model_id, attempts=2):
    last = None
    for n in range(1, attempts + 1):
        try:
            return _request("POST", f"{app}/api/swap/load/{model_id}",
                            {}, timeout=900)
        except RuntimeError as e:
            last = e
            log(f"load attempt {n} failed: {e}")
            try:
                evict(app, model_id)
            except Exception:
                pass
            if n < attempts:
                time.sleep(2)
    raise last


def evict(app, model_id):
    try:
        _request("POST", f"{app}/api/swap/evict/{model_id}",
                 {}, timeout=120)
        log(f"evicted {model_id}")
    except Exception as e:
        log(f"evict {model_id} skipped: {e}")


def base_model(app, model_id):
    return _request("GET", f"{app}/api/models/{model_id}", timeout=30)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["single", "multi"])
    ap.add_argument("--app", default=DEFAULT_APP)
    ap.add_argument("--gateway", default=DEFAULT_GATEWAY)
    ap.add_argument("--model", default="",
                    help="model id (default: single mode qwen3-0-6b-demo, "
                         "multi mode qwen3-0-6b-demo--rpc)")
    ap.add_argument("--keep", action="store_true",
                    help="leave the loaded model running for debugging")
    args = ap.parse_args()
    app = args.app.rstrip("/")
    gateway = args.gateway.rstrip("/")

    rows = machines(app)
    master = next((m for m in rows if m.get("role") == "master"),
                  next((m for m in rows if m.get("ip") in
                        ("127.0.0.1", "::1")), None))
    if not master:
        raise SystemExit("no local master machine found")
    if args.mode == "single":
        model_id = args.model or MODEL
        was_loaded = model_id in loaded_ids(app)
        log(f"single: loading {model_id} on {master['name']}")
        try:
            load = load_model(app, model_id)
            log(f"load {json.dumps(load)[:500]}")
            wait_loaded(app, model_id)
            chat(gateway, model_id)
        finally:
            if not args.keep and not was_loaded:
                evict(app, model_id)
        return 0

    model_id = args.model or MULTI_MODEL
    base = base_model(app, MODEL)
    base_path = str(base.get("base_path") or "")
    if not base_path.startswith("store://"):
        raise RuntimeError(
            f"multi test requires a shared-store model path, got {base_path!r}")
    was_loaded = model_id in loaded_ids(app)
    log(f"multi: loading {model_id} from shared store {base_path}")
    try:
        load = load_model(app, model_id)
        log(f"load {json.dumps(load)[:500]}")
        wait_loaded(app, model_id)
        chat(gateway, model_id)
        preview = _request("GET", f"{app}/api/swap/preview", timeout=60)
        script = os.path.join(preview.get("scripts_dir") or "",
                              f"start-{model_id}.sh")
        if not os.path.isfile(script):
            raise RuntimeError(f"multi model script not found: {script}")
        text = open(script, "r", encoding="utf-8").read()
        rpc_line = next((ln for ln in text.splitlines()
                         if "--rpc" in ln), "")
        if not rpc_line:
            raise RuntimeError(f"multi model script has no --rpc: {script}")
        log(f"multi RPC script verified: {rpc_line.strip()}")
    finally:
        if not args.keep and not was_loaded:
            evict(app, model_id)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as e:
        print(f"[tiny] FAILED: {e}", file=sys.stderr)
        raise SystemExit(1)
