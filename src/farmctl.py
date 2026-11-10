#!/usr/bin/env python3
"""farmctl — command-line client for the Llama RPC Farm HTTP API.

This is part of the main app (no separate daemon, no direct config access).
Every command talks to the running app at :8199 over HTTP, so it sees the
same in-memory state the UI sees and can reach every clustered machine the
app manages.

Commands:
    status                 cluster + llama-swap status (all machines)
    models                 configured models + variants
    start                  start the llama-swap daemon (no model warmup)
    stop                   stop the llama-swap daemon
    launch --model ID      load/warm ONE model through llama-swap
    unload ID              unload one model (free its VRAM/RAM)
    doctor                 setup/readiness checklist; nonzero on issues
    snapshot               write a redacted JSON snapshot for support

`start` and `launch` are deliberately separate: llama-swap loads models on
demand, so starting the daemon does not load every configured model. Use
`launch --model` for the model you actually want warm.

Auth: loopback is trusted by the app. From another host pass --key or set
FARMCTL_KEY / FARM_API_KEY. Base URL: --url or FARMCTL_URL (default
http://127.0.0.1:8199).

Examples:
    ./farmctl status
    ./farmctl --json status | jq .swap
    ./farmctl models
    ./farmctl start
    ./farmctl launch --model qwen3-8-27b-uncensored-q4-k-m-128k
    ./farmctl unload qwen3-8-27b-uncensored-q4-k-m-128k
    ./farmctl stop
    ./farmctl doctor
    ./farmctl snapshot --out /tmp/farm.json
"""
import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor

from secret_fields import is_secret_key

DEFAULT_URL = "http://127.0.0.1:8199"
_ENV_URLS = ("FARMCTL_URL", "FARM_URL", "LLAMA_FARM_URL")
_ENV_KEYS = ("FARMCTL_KEY", "FARM_API_KEY", "FARM_ACCESS_KEY")


def _opener():
    """Fresh opener per request: no shared mutable opener across threads."""
    return urllib.request.build_opener(urllib.request.ProxyHandler({}))


class FarmError(RuntimeError):
    """The app could not be reached or returned an unexpected response."""


class FarmHTTPError(FarmError):
    """The app answered with a non-2xx status. `.body` is decoded JSON/text."""

    def __init__(self, code, body, url):
        self.code = code
        self.body = body
        detail = body.get("detail") if isinstance(body, dict) else body
        super().__init__(f"HTTP {code} from {url}: {detail or 'request failed'}")


def _first_env(names, default=""):
    for name in names:
        value = (os.environ.get(name) or "").strip()
        if value:
            return value
    return default


def _decode(raw, content_type=""):
    if not raw:
        return {}
    text = raw.decode("utf-8", "replace")
    if "json" in (content_type or "") or text.lstrip().startswith(("{", "[")):
        try:
            return json.loads(text)
        except ValueError:
            pass
    return text


class Client:
    def __init__(self, base, key="", timeout=60.0):
        self.base = (base or DEFAULT_URL).rstrip("/")
        self.key = (key or "").strip()
        self.timeout = float(timeout)

    def request(self, method, path, body=None, timeout=None):
        url = self.base + path
        data = None
        headers = {"Accept": "application/json",
                   "User-Agent": "farmctl/1"}
        if self.key:
            headers["Authorization"] = f"Bearer {self.key}"
        if body is not None:
            data = json.dumps(body, separators=(",", ":")).encode("utf-8")
            headers["Content-Type"] = "application/json"
        req = urllib.request.Request(url, data=data, method=method,
                                     headers=headers)
        try:
            with _opener().open(req, timeout=timeout or self.timeout) as resp:
                return _decode(resp.read(8 * 1024 * 1024),
                               resp.headers.get("Content-Type", ""))
        except urllib.error.HTTPError as exc:
            raw = b""
            try:
                raw = exc.read(1024 * 1024)
            except Exception:
                pass
            body_out = _decode(raw, exc.headers.get("Content-Type", "")
                               if exc.headers else "")
            raise FarmHTTPError(exc.code, body_out, url) from exc
        except (urllib.error.URLError, OSError) as exc:
            reason = getattr(exc, "reason", exc)
            raise FarmError(
                f"cannot reach {self.base} ({reason}) — is the farm app "
                f"running? start it with ./run.sh") from exc

    def get(self, path, **kw):
        return self.request("GET", path, **kw)

    def post(self, path, body=None, **kw):
        return self.request("POST", path, {} if body is None else body, **kw)


# --------------------------------------------------------------------------
# formatting helpers
# --------------------------------------------------------------------------

def _safe_get(client, path, timeout=None):
    try:
        return client.get(path, timeout=timeout)
    except FarmError as exc:
        return {"error": str(exc)}


def _num(value, digits=1):
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "-"


def _short_gpu(name):
    name = str(name or "GPU")
    for prefix in ("NVIDIA GeForce ", "NVIDIA ", "AMD ", "Intel "):
        if name.startswith(prefix):
            name = name[len(prefix):]
            break
    return name


def _print_table(headers, rows):
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(str(cell)))
    line = "  ".join(h.ljust(widths[i]) for i, h in enumerate(headers))
    print(line)
    print("  ".join("-" * widths[i] for i in range(len(headers))))
    for row in rows:
        print("  ".join(str(cell).ljust(widths[i])
                        for i, cell in enumerate(row)))


def _print_json(data):
    print(json.dumps(data, indent=2, sort_keys=False, default=str))


def _swap_summary(swap, loaded):
    if not isinstance(swap, dict) or not swap.get("running"):
        err = swap.get("error") if isinstance(swap, dict) else None
        return f"llama-swap: DOWN{(' — ' + str(err)) if err else ''}"
    host = swap.get("host") or "?"
    port = swap.get("port") or "?"
    ids = []
    if isinstance(loaded, dict):
        ids = loaded.get("loaded") or []
    text = f"llama-swap: UP  {host}:{port}"
    if ids:
        text += f"  loaded: {', '.join(str(i) for i in ids)}"
    else:
        text += "  loaded: (none)"
    if isinstance(loaded, dict) and loaded.get("error"):
        text += f"  [loaded query: {loaded['error']}]"
    return text


def _machine_conf_map(machines):
    if isinstance(machines, list):
        return {m.get("name"): m for m in machines if isinstance(m, dict)}
    return {}


# --------------------------------------------------------------------------
# commands
# --------------------------------------------------------------------------

def cmd_status(args, client):
    with ThreadPoolExecutor(max_workers=4) as pool:
        f_machines = pool.submit(client.get, "/api/machines",
                                 timeout=args.timeout)
        f_stats = pool.submit(client.get, "/api/stats",
                              timeout=args.timeout)
        f_swap = pool.submit(_safe_get, client, "/api/swap/status", 15)
        f_loaded = pool.submit(_safe_get, client, "/api/swap/loaded", 15)
        machines = f_machines.result()
        stats = f_stats.result()
        swap = f_swap.result()
        loaded = f_loaded.result()
    conf = _machine_conf_map(machines)
    stats_rows = (stats or {}).get("data") or []
    by_name = {s.get("name"): s for s in stats_rows if isinstance(s, dict)}

    if args.json:
        _print_json({
            "url": client.base,
            "machines": machines,
            "stats": stats,
            "swap": swap,
            "loaded": loaded,
        })
        return 0

    print(f"Llama RPC Farm  {client.base}")
    print(_swap_summary(swap, loaded))
    print()

    headers = ["MACHINE", "ROLE", "OS", "STATE", "RPC", "GPU",
               "VRAM GiB", "TEMP", "RAM GiB", "CPU"]
    rows = []
    ordered = list(conf.values()) or stats_rows
    for m in ordered:
        name = m.get("name") or "?"
        s = by_name.get(name) or {}
        state = s.get("status") or "unknown"
        gpus = s.get("gpus") or []
        vram_used = sum(float(g.get("mem_used_gb") or 0) for g in gpus)
        vram_total = sum(float(g.get("mem_total_gb") or 0) for g in gpus)
        temps = [float(g.get("temp") or 0) for g in gpus
                 if float(g.get("temp") or 0) > 0]
        rows.append([
            name,
            m.get("role") or s.get("role") or "-",
            m.get("os") or s.get("os") or "-",
            state,
            "yes" if s.get("rpc_running") else "no",
            len(gpus),
            f"{vram_used:.1f}/{vram_total:.1f}" if gpus else "-",
            f"{max(temps):.0f}C" if temps else "-",
            f"{_num(s.get('ram_used_gb'), 0)}/{_num(s.get('ram_total_gb'), 0)}",
            f"{_num(s.get('cpu_pct'), 0)}%",
        ])
    _print_table(headers, rows)

    # Per-GPU detail keeps the main table narrow while still surfacing the
    # exact card that is hot/busy.
    for m in ordered:
        name = m.get("name") or "?"
        s = by_name.get(name) or {}
        gpus = s.get("gpus") or []
        if not gpus:
            continue
        print()
        print(f"{name}:")
        for g in gpus:
            print(f"  {str(g.get('index', '?')):>2} {_short_gpu(g.get('name')):<24} "
                  f"util {_num(g.get('util'), 0):>3}%  "
                  f"mem {_num(g.get('mem_used_gb'))}/{_num(g.get('mem_total_gb'))} GiB  "
                  f"{_num(g.get('temp'), 0):>3}C")

    totals = (stats or {}).get("totals") or {}
    if totals:
        vram = totals.get("vram") or {}
        ram = totals.get("ram") or {}
        cpu = totals.get("cpu") or {}
        print()
        print(f"Totals: VRAM {_num(vram.get('used'))}/{_num(vram.get('total'))} GiB "
              f"({_num(vram.get('pct'))}%)  "
              f"RAM {_num(ram.get('used'))}/{_num(ram.get('total'))} GiB  "
              f"CPU {_num(cpu.get('pct'))}%")
    return 0


def cmd_doctor(args, client):
    with ThreadPoolExecutor(max_workers=2) as pool:
        f_readiness = pool.submit(_safe_get, client, "/api/readiness",
                                  args.timeout)
        f_swap = pool.submit(_safe_get, client, "/api/swap/status", 15)
        readiness = f_readiness.result()
        swap = f_swap.result()
    readiness_error = ""
    if isinstance(readiness, dict) and readiness.get("error"):
        readiness_error = str(readiness["error"])
        readiness = {}

    if args.json:
        _print_json({"url": client.base, "readiness": readiness,
                     "readiness_error": readiness_error, "swap": swap})
        return 0 if (readiness.get("ready_all")
                     and isinstance(swap, dict) and swap.get("running")) else 1

    issues = 0
    print(f"Llama RPC Farm doctor  {client.base}")
    print()
    rows = (readiness or {}).get("machines") or {}
    if not rows:
        if readiness_error:
            print(f"FAIL  readiness: {readiness_error}")
        else:
            print("FAIL  readiness endpoint returned no machines")
        issues += 1
    for name, row in rows.items():
        ready = bool(row.get("ready"))
        if not ready:
            issues += 1
        tags = []
        if row.get("local"):
            tags.append("local")
        if row.get("is_swap_host"):
            tags.append("swap-host")
        suffix = f"  ({', '.join(tags)})" if tags else ""
        print(f"{'OK  ' if ready else 'FAIL'}  {name}{suffix}")

        agent = row.get("agent") or {}
        if not agent.get("ok"):
            print(f"        agent: {agent.get('error') or 'unreachable'}")
        else:
            print(f"        agent: ok"
                  f"{' v' + str(agent.get('version')) if agent.get('version') else ''}")

        cpp = row.get("llama_cpp") or {}
        print(f"        llama.cpp: {'ok  ' if cpp.get('installed') else 'MISS'}"
              f"  {cpp.get('version') or ''}"
              f"{'  ' + str(cpp.get('path')) if cpp.get('path') else ''}")
        if not cpp.get("installed") and cpp.get("error"):
            print(f"                   {cpp.get('error')}")

        rpc = row.get("llama_rpc") or {}
        print(f"        rpc:       {'ok  ' if rpc.get('installed') else 'MISS'}"
              f"{'  ' + str(rpc.get('path')) if rpc.get('path') else ''}")
        if not rpc.get("installed") and rpc.get("error"):
            print(f"                   {rpc.get('error')}")

        sw = row.get("llama_swap") or {}
        label = "required" if sw.get("required") else "optional"
        print(f"        llama-swap: {'ok  ' if sw.get('installed') else 'MISS'}"
              f"  ({label}){'  ' + str(sw.get('path')) if sw.get('path') else ''}")

    print()
    if isinstance(swap, dict) and swap.get("running"):
        print(f"OK    llama-swap daemon on {swap.get('host')}:{swap.get('port')}")
    else:
        err = swap.get("error") if isinstance(swap, dict) else None
        print(f"FAIL  llama-swap daemon is not running"
              f"{(' — ' + str(err)) if err else ''}")
        issues += 1

    print()
    print("RESULT: OK" if issues == 0 else f"RESULT: {issues} issue(s)")
    return 0 if issues == 0 else 1


def cmd_launch(args, client):
    """Load/warm exactly one model. Requires --model by construction."""
    path = "/api/swap/load/" + urllib.parse.quote(args.model, safe="")
    result = client.post(path, {}, timeout=args.load_timeout)
    if args.json:
        _print_json(result)
        return 0 if (result or {}).get("ok") else 1
    if isinstance(result, dict) and result.get("ok"):
        print(f"loaded {result.get('model')} in {result.get('elapsed_s')}s")
        preview = (result.get("preview") or "").strip()
        if preview:
            print(f"  preview: {preview}")
        return 0
    print(f"load failed: {result}")
    return 1


def cmd_start(args, client):
    """Start the llama-swap daemon. Models are NOT warmed here — llama-swap
    loads them on demand when a client first asks for one."""
    failed = False
    try:
        result = client.post("/api/swap/launch", {}, timeout=args.timeout)
    except FarmHTTPError as exc:
        # A failed start still returns the step report (HTTP 500).
        if isinstance(exc.body, dict) and exc.body.get("steps"):
            result = exc.body
            failed = True
        else:
            raise
    if args.json:
        _print_json(result)
        return 0 if not (failed or (result or {}).get("failed")) else 1
    if not isinstance(result, dict):
        print(result)
        return 1
    print(f"host: {result.get('host')}  port: {result.get('port')}"
          f"{' (rolled)' if result.get('port_rolled') else ''}")
    for step in result.get("steps") or []:
        name = step.get("step") or "?"
        if step.get("error"):
            print(f"  FAIL  {name}: {step['error']}")
            failed = True
        else:
            print(f"  ok    {name}")
    for warning in result.get("warnings") or []:
        print(f"  WARN  {warning}")
    failed = failed or bool(result.get("failed"))
    print("start: OK" if not failed else "start: FAILED")
    return 0 if not failed else 1


def cmd_stop(args, client):
    result = client.post("/api/swap/stop", {}, timeout=args.timeout)
    if args.json:
        _print_json(result)
        return 0 if (result or {}).get("stopped") else 1
    if not isinstance(result, dict):
        print(result)
        return 1
    stopped = bool(result.get("stopped"))
    print(f"llama-swap on {result.get('host')}: "
          f"{'stopped' if stopped else 'NOT stopped'}")
    if result.get("detail"):
        print(f"  {result['detail']}")
    for row in result.get("machines") or []:
        name = row.get("name") or "?"
        state = "stopped" if (row.get("killed") or row.get("stopped")) \
            else "left running"
        print(f"  {name}: {state}")
    return 0 if stopped else 1


def _model_row(label, flags, servers, split, variant_count):
    return [
        label,
        ",".join(servers or []) or "-",
        flags.get("ctx-size", flags.get("c", "-")),
        split or "auto",
        variant_count,
    ]


def cmd_models(args, client):
    models = client.get("/api/models", timeout=args.timeout)
    if not isinstance(models, list):
        if args.json:
            _print_json(models)
        else:
            print(models)
        return 1
    if args.json:
        _print_json(models)
        return 0
    rows = []
    for m in models:
        mid = m.get("model_id") or m.get("name") or "?"
        flags = m.get("flags") or {}
        variants = m.get("variants") or []
        rows.append(_model_row(
            mid, flags, m.get("servers"), m.get("tensor_split_mode"),
            len(variants)))
        for var in variants:
            label = (var.get("label") or "").strip()
            if not label:
                continue
            vflags = dict(flags)
            vflags.update(var.get("flags") or {})
            rows.append(_model_row(
                f"  {mid}--{label}", vflags,
                var.get("servers") or m.get("servers"),
                var.get("tensor_split_mode") or m.get("tensor_split_mode"),
                "-"))
    _print_table(["MODEL", "SERVERS", "CTX", "SPLIT", "VARIANTS"], rows)
    return 0


def cmd_unload(args, client):
    path = "/api/swap/evict/" + urllib.parse.quote(args.model, safe="")
    result = client.post(path, {}, timeout=args.timeout)
    ok = isinstance(result, dict) and bool(result.get("ok"))
    if args.json:
        _print_json(result)
        return 0 if ok else 1
    if ok:
        print(f"unloaded {args.model}")
        return 0
    err = result.get("error") if isinstance(result, dict) else result
    print(f"unload failed: {err}")
    return 1


def _redact(value):
    """Drop secret-named fields before a snapshot leaves the machine."""
    if isinstance(value, dict):
        return {key: _redact(item) for key, item in value.items()
                if not is_secret_key(key)}
    if isinstance(value, list):
        return [_redact(item) for item in value]
    return value


def cmd_snapshot(args, client):
    sections = [
        ("machines", "/api/machines"),
        ("models", "/api/models?full=1"),
        ("rpc_map", "/api/rpc-map"),
        ("swap_settings", "/api/swap/settings"),
        ("swap_status", "/api/swap/status"),
        ("swap_loaded", "/api/swap/loaded"),
        ("stats", "/api/stats"),
        ("settings", "/api/settings"),
        ("readiness", "/api/readiness"),
    ]
    if args.fast:
        sections = [s for s in sections if s[0] != "readiness"]

    def fetch(section):
        name, path = section
        try:
            return name, _redact(client.get(path, timeout=args.timeout)), ""
        except FarmError as exc:
            return name, {"error": str(exc)}, str(exc)

    with ThreadPoolExecutor(max_workers=len(sections) or 1) as pool:
        results = list(pool.map(fetch, sections))
    data = {name: value for name, value, _ in results}
    errors = {name: error for name, _, error in results if error}
    snapshot = {
        "farmctl_snapshot": 1,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "app_url": client.base,
        "data": data,
        "errors": errors,
    }
    text = json.dumps(snapshot, indent=2, sort_keys=False, default=str)
    if args.out == "-":
        print(text)
        return 0
    path = os.path.abspath(os.path.expanduser(
        args.out or time.strftime("farm-snapshot-%Y%m%d-%H%M%S.json")))
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(text + "\n")
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass
    if args.json:
        _print_json({"ok": True, "path": path, "errors": errors})
    else:
        print(f"snapshot written: {path}")
        if errors:
            print(f"  {len(errors)} section(s) failed: "
                  f"{', '.join(sorted(errors))}")
    return 0


# --------------------------------------------------------------------------
# argument parsing / entry point
# --------------------------------------------------------------------------

def _common_parser():
    # SUPPRESS keeps a flag accepted both before and after the subcommand
    # without the subparser overwriting a value already set by the parent.
    # Help text therefore must not interpolate %(default)s here: argparse
    # drops the default from the formatting params when it is SUPPRESS.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--url", default=argparse.SUPPRESS,
                        help="farm app base URL (FARMCTL_URL/FARM_URL)")
    common.add_argument("--key", default=argparse.SUPPRESS,
                        help="access key (or FARMCTL_KEY)")
    common.add_argument("--json", action="store_true",
                        default=argparse.SUPPRESS,
                        help="machine-readable JSON output")
    common.add_argument("--timeout", type=float, default=argparse.SUPPRESS,
                        help="request timeout in seconds (default: 120)")
    return common


def build_parser():
    common = _common_parser()
    parser = argparse.ArgumentParser(
        prog="farmctl",
        description="Command-line client for the Llama RPC Farm API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Examples:", 1)[-1].strip())
    parser.add_argument("--url", default=_first_env(_ENV_URLS, DEFAULT_URL),
                        help="farm app base URL (default: %(default)s)")
    parser.add_argument("--key", default=_first_env(_ENV_KEYS, ""),
                        help="access key (default: FARMCTL_KEY/FARM_API_KEY)")
    parser.add_argument("--json", action="store_true",
                        help="machine-readable JSON output")
    parser.add_argument("--timeout", type=float, default=120.0,
                        help="request timeout in seconds (default: %(default)s)")
    parser.add_argument("--version", action="version",
                        version="farmctl 1")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("status", parents=[common],
                       help="cluster + llama-swap status")
    p.set_defaults(func=cmd_status)

    p = sub.add_parser("start", parents=[common],
                       help="start the llama-swap daemon (no model warmup)")
    p.set_defaults(func=cmd_start)

    p = sub.add_parser("stop", parents=[common],
                       help="stop the llama-swap daemon")
    p.set_defaults(func=cmd_stop)

    p = sub.add_parser("launch", parents=[common],
                       help="load/warm ONE model through llama-swap")
    p.add_argument("--model", required=True, metavar="ID",
                   help="model id (or model--variant) to load")
    p.add_argument("--load-timeout", type=float, default=900.0,
                   help="timeout for the load in seconds (default: 900)")
    p.set_defaults(func=cmd_launch)

    p = sub.add_parser("unload", parents=[common],
                       help="unload one model (free its VRAM/RAM)")
    p.add_argument("model", metavar="ID",
                   help="model id (or model--variant) to unload")
    p.set_defaults(func=cmd_unload)

    p = sub.add_parser("models", parents=[common],
                       help="list configured models and variants")
    p.set_defaults(func=cmd_models)

    p = sub.add_parser("doctor", parents=[common],
                       help="setup/readiness checklist")
    p.set_defaults(func=cmd_doctor)

    p = sub.add_parser("snapshot", parents=[common],
                       help="write a redacted JSON support snapshot")
    p.add_argument("--out", default="", metavar="PATH",
                   help="output path ('-' = stdout; default: farm-snapshot-*.json)")
    p.add_argument("--fast", action="store_true",
                   help="skip the readiness probe (faster)")
    p.set_defaults(func=cmd_snapshot)
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    client = Client(args.url, getattr(args, "key", ""),
                    getattr(args, "timeout", 120.0))
    try:
        return int(args.func(args, client) or 0)
    except FarmHTTPError as exc:
        print(f"farmctl: {exc}", file=sys.stderr)
        return 1
    except FarmError as exc:
        print(f"farmctl: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
