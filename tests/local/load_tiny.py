#!/usr/bin/env python3
"""Local-only Qwen3-0.6B load tests (not collected by pytest/CI).

Run against a live farm UI/gateway on this host:

    .venv/bin/python tests/local/load_tiny.py single
    .venv/bin/python tests/local/load_tiny.py multi --peer aibox3

Single mode loads the existing qwen3-0-6b-demo model on the master.
Multi mode creates a temporary two-machine RPC deployment, loads it,
checks a chat completion, then removes the temp model and evicts it.
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


def evict(app, model_id):
    try:
        _request("POST", f"{app}/api/swap/evict/{model_id}",
                 {}, timeout=120)
        log(f"evicted {model_id}")
    except Exception as e:
        log(f"evict {model_id} skipped: {e}")


def push(app):
    out = _request("POST", f"{app}/api/swap/push", {}, timeout=180)
    if not out.get("validated"):
        raise RuntimeError(f"swap push failed: {out}")
    log(f"swap push written={len(out.get('written') or [])}")
    return out


def base_model(app, model_id):
    return _request("GET", f"{app}/api/models/{model_id}", timeout=30)


def temp_multi_payload(base, master, peer, temp_id):
    mdev = (master.get("gpu") or "CUDA0").split(",")[0]
    pdev = (peer.get("gpu") or "CPU").split(",")[0]
    return {
        "model_id": temp_id,
        "name": "Local 0.6B multi-machine test",
        "base_path": base["base_path"],
        "tensor_split_mode": "fit",
        "tensor_split": "",
        "fit_margin": 1024,
        "fit_ram_allow": 1024,
        "llama_bin": base.get("llama_bin") or "",
        "servers": [master["name"], peer["name"]],
        "group": "",
        "pinned": False,
        "ttl": 0,
        "device_order": f"{master['name']}-{mdev},{peer['name']}-{pdev}",
        "device_order_policy": "as-selected",
        "launch_mode": "rpc",
        "expose": {},
        "flags": {
            "ctx-size": 2048,
            "parallel": 1,
            "jinja": True,
            "n-gpu-layers": 99,
        },
        "variants": [],
        "reset": False,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["single", "multi"])
    ap.add_argument("--app", default=DEFAULT_APP)
    ap.add_argument("--gateway", default=DEFAULT_GATEWAY)
    ap.add_argument("--model", default=MODEL,
                    help="model id for single mode")
    ap.add_argument("--peer", default="",
                    help="multi mode peer (default: first GPU peer)")
    ap.add_argument("--keep", action="store_true",
                    help="keep the temp model/loaded process for debugging")
    args = ap.parse_args()
    app = args.app.rstrip("/")
    gateway = args.gateway.rstrip("/")

    rows = machines(app)
    by_name = {m.get("name"): m for m in rows}
    master = next((m for m in rows if m.get("role") == "master"),
                  next((m for m in rows if m.get("ip") in
                        ("127.0.0.1", "::1")), None))
    if not master:
        raise SystemExit("no local master machine found")
    if args.mode == "single":
        model_id = args.model
        was_loaded = model_id in loaded_ids(app)
        log(f"single: loading {model_id} on {master['name']}")
        try:
            load = _request("POST", f"{app}/api/swap/load/{model_id}",
                            {}, timeout=900)
            log(f"load {json.dumps(load)[:500]}")
            wait_loaded(app, model_id)
            chat(gateway, model_id)
        finally:
            if not args.keep and not was_loaded:
                evict(app, model_id)
        return 0

    if args.peer:
        peer = by_name.get(args.peer)
        if not peer:
            raise SystemExit(f"unknown peer {args.peer!r}; choices: "
                             f"{', '.join(n for n in by_name if n != master['name'])}")
    else:
        peer = next((m for m in rows
                     if m.get("name") != master.get("name")
                     and m.get("gpu")), None)
        if not peer:
            raise SystemExit("no GPU peer available for multi-machine test")
    base = base_model(app, args.model)
    temp_id = f"local-tiny-multi-{os.getpid()}"
    payload = temp_multi_payload(base, master, peer, temp_id)
    created = False
    try:
        log(f"multi: creating {temp_id} on {master['name']} + {peer['name']}")
        _request("POST", f"{app}/api/models", payload, timeout=60)
        created = True
        push(app)
        load = _request("POST", f"{app}/api/swap/load/{temp_id}",
                        {}, timeout=900)
        log(f"load {json.dumps(load)[:500]}")
        wait_loaded(app, temp_id)
        chat(gateway, temp_id)
        preview = _request("GET", f"{app}/api/swap/preview", timeout=60)
        script = os.path.join(preview.get("scripts_dir") or "",
                              f"start-{temp_id}.sh")
        if os.path.isfile(script):
            text = open(script, "r", encoding="utf-8").read()
            if "--rpc" not in text:
                raise RuntimeError(f"multi model script has no --rpc: {script}")
            log(f"multi RPC script verified: {script}")
        else:
            log(f"warning: script not found at {script}")
    finally:
        if not args.keep:
            evict(app, temp_id)
            if created:
                try:
                    _request("DELETE", f"{app}/api/models/{temp_id}",
                             timeout=60)
                    log(f"deleted {temp_id}")
                except Exception as e:
                    log(f"delete {temp_id} skipped: {e}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as e:
        print(f"[tiny] FAILED: {e}", file=sys.stderr)
        raise SystemExit(1)
