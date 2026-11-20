#!/usr/bin/env python3
"""One-command solo quickstart: install engines, pick a recipe, launch, test.

Run after the app is up (scripts/quickstart.sh starts it):

    .venv/bin/python src/quickstart.py --recipe demo-0.6b

It prints the OpenAI and Anthropic endpoints plus a generated access key,
and finishes by making a real completion through the gateway.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request

APP = os.environ.get("FARM_URL") or "http://127.0.0.1:8199"
GATEWAY = os.environ.get("FARM_GATEWAY_URL") or "http://127.0.0.1:8090"


def log(msg):
    print(f"[quickstart] {msg}", flush=True)


def api(method, path, payload=None, timeout=120):
    data = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        data = json.dumps(payload).encode()
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(APP + path, data=data, headers=headers,
                                 method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read() or b"{}"
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "replace")[:1500]
        raise RuntimeError(f"HTTP {e.code} {method} {path}: {detail}") from e
    try:
        return json.loads(raw.decode())
    except Exception:
        return {"raw": raw.decode("utf-8", "replace")}


def wait_ui(timeout=120):
    end = time.time() + timeout
    last = None
    while time.time() < end:
        try:
            return api("GET", "/api/machines", timeout=5)
        except Exception as e:                          # noqa: BLE001
            last = e
            time.sleep(1)
    raise RuntimeError(f"farm UI did not come up at {APP}: {last}")


def local_machine():
    machines = api("GET", "/api/machines")
    if isinstance(machines, dict):
        machines = machines.get("machines") or []
    for m in machines:
        if m.get("is_local") or m.get("role") == "master":
            return m
    if machines:
        return machines[0]
    raise RuntimeError("no machine configured")


def mark(event, extra=None):
    try:
        api("POST", "/api/onboarding/mark",
            {"event": event, "extra": extra or {}})
    except Exception:
        pass


def install_engines(machine, timeout=1800):
    log(f"installing llama.cpp + llama-swap on {machine} (one time)…")
    res = api("POST", "/api/tools/install/sync", {
        "machine": machine, "role": "main",
        "tools": ["llama.cpp", "llama-swap"], "dry_run": False,
    }, timeout=timeout)
    if res.get("failures"):
        raise RuntimeError(f"engine install failed: {res['failures']}")
    log("engines ready")
    mark("engines_installed")
    return res


def apply_recipe(recipe, launch=True):
    log(f"installing recipe: {recipe}")
    r = api("POST", f"/api/onboarding/recipes/{recipe}/apply",
            {"launch": launch}, timeout=30)
    job = r.get("job_id")
    if not job:
        raise RuntimeError(f"recipe did not start: {r}")
    last = ""
    end = time.time() + 7200
    while time.time() < end:
        j = api("GET", f"/api/onboarding/recipes/jobs/{job}", timeout=30)
        phase = j.get("phase") or ""
        if phase != last:
            log(f"  {phase}")
            last = phase
        if j.get("done"):
            if j.get("error"):
                raise RuntimeError(j["error"])
            return j
        time.sleep(1)
    raise RuntimeError("recipe timed out")


def create_key():
    try:
        r = api("POST", "/api/onboarding/access-key", {}, timeout=30)
        return r.get("token") or ""
    except Exception as e:                              # noqa: BLE001
        log(f"could not create a key: {e}")
        return ""


def test_completion(model, key):
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": "Reply with OK."}],
        "max_tokens": 8,
    }).encode()
    req = urllib.request.Request(
        GATEWAY + "/v1/chat/completions", data=body, headers=headers,
        method="POST")
    with urllib.request.urlopen(req, timeout=300) as r:
        data = json.loads(r.read().decode())
    m = (data.get("choices") or [{}])[0].get("message") or {}
    msg = m.get("content") or m.get("reasoning_content") or ""
    return str(msg).strip()


def main():
    ap = argparse.ArgumentParser(description="Llama RPC Farm solo quickstart")
    ap.add_argument("--recipe", default="demo-0.6b",
                    help="starter recipe id (default: demo-0.6b)")
    ap.add_argument("--no-install", action="store_true",
                    help="skip engine install")
    ap.add_argument("--no-launch", action="store_true",
                    help="download + configure only")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    mark("quickstart_started")
    wait_ui()
    machine = local_machine()
    name = machine.get("name")
    log(f"farm UI ready; local machine: {name}")
    if not args.no_install:
        install_engines(name)
    job = apply_recipe(args.recipe, launch=not args.no_launch)
    model = ((job.get("result") or {}).get("model_id")
             or job.get("model_id") or args.recipe)
    key = create_key() if not args.no_launch else ""
    reply = ""
    if not args.no_launch:
        log("testing the endpoint through :8090…")
        reply = test_completion(model, key)
        mark("first_chat", {"model": model})
    out = {
        "ok": True,
        "ui": APP,
        "openai": GATEWAY + "/v1",
        "anthropic": GATEWAY,
        "model": model,
        "access_key": key,
        "reply": reply,
        "recipe": args.recipe,
    }
    if args.json:
        print(json.dumps(out, indent=2))
    else:
        print()
        print("=" * 64)
        print("  Llama RPC Farm is ready")
        print("=" * 64)
        print(f"  UI:         {out['ui']}")
        print(f"  OpenAI:     {out['openai']}")
        print(f"  Anthropic:  {out['anthropic']}")
        print(f"  Model:      {out['model']}")
        if key:
            print(f"  Access key: {key}")
            print("              (keep it private — it was just created)")
        if reply:
            print(f"  Test reply: {reply[:80]!r}")
        print()
        print("  Try:")
        print(f"    curl {out['openai']}/models -H 'Authorization: Bearer {key or '<key>'}'")
        print()
    mark("quickstart_finished", {"model": model})
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:                              # noqa: BLE001
        print(f"[quickstart] FAILED: {e}", file=sys.stderr, flush=True)
        sys.exit(1)
