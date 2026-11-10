"""FastAPI app: web UI + JSON API for Llama RPC Farm.

Run: python src/app.py  (or uvicorn app:app --port 8199)
"""
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

import config_loader as cfg
import hf_downloader
import local_node
import reaper
import stats_history
import swap_captures
import swap_manager
from machine_tokens import _ensure_farm_agent_token

from runtime_paths import asset_dir, data_dir, ensure_runtime_layout
BASE_DIR = data_dir()
ASSET_DIR = asset_dir()
templates = Jinja2Templates(directory=os.path.join(ASSET_DIR, "ui", "templates"))
templates.env.auto_reload = True  # pick up index.html edits without a restart

app = FastAPI(title="Llama RPC Farm")

from routes.auth import router as auth_router
from routes.chat import router as chat_router
from routes.flow import router as flow_router
from routes.scanning import router as scanning_router
from routes.files import router as files_router
from routes.hardware import router as hardware_router
from routes.inference import router as inference_router
from routes.launch import router as launch_router
from routes.machines import router as machines_router
from routes.models import router as models_router
from routes.settings import router as settings_router
from routes.install import router as install_router
from routes.tools import router as tools_router
from routes.tools_install import router as tools_install_router
from routes.views import router as views_router
from routes.hf import router as hf_router
from routes.readiness import router as readiness_router
from swap_client import _swap_api_get
from routes.swap import router as swap_router
from routes.ui_state import router as ui_state_router

for _router in (flow_router, hf_router, hardware_router, files_router,
                readiness_router, swap_router, ui_state_router,
                chat_router, inference_router, scanning_router,
                install_router, tools_router, views_router,
                tools_install_router, auth_router, machines_router,
                models_router, settings_router, launch_router):
    app.include_router(_router)


_GATEWAY_SERVER = None


def _port_open(port):
    try:
        import socket
        s = socket.socket()
        s.settimeout(0.3)
        try:
            return s.connect_ex(("127.0.0.1", int(port))) == 0
        finally:
            s.close()
    except Exception:
        return False


def _port_listener_pids(port):
    """Best-effort PIDs of processes LISTENing on a local TCP port."""
    port = int(port)
    if os.name == "nt":
        cmd = (
            f"(Get-NetTCPConnection -LocalPort {port} -State Listen "
            "-ErrorAction SilentlyContinue | "
            "Select-Object -ExpandProperty OwningProcess -Unique)"
        )
        try:
            out = subprocess.run(
                ["powershell", "-NoProfile", "-NonInteractive", "-Command",
                 cmd], capture_output=True, text=True, timeout=8)
            return sorted({int(x) for x in (out.stdout or "").split()
                           if x.strip().isdigit()})
        except Exception:
            return []
    pids = set()
    if shutil.which("lsof"):
        try:
            out = subprocess.run(
                ["lsof", "-nP", "-t", f"-iTCP:{port}", "-sTCP:LISTEN"],
                capture_output=True, text=True, timeout=8)
            pids.update(int(x) for x in (out.stdout or "").split()
                        if x.strip().isdigit())
        except Exception:
            pass
    if not pids and shutil.which("ss"):
        try:
            out = subprocess.run(["ss", "-ltnp"], capture_output=True,
                                 text=True, timeout=8)
            for line in (out.stdout or "").splitlines():
                if re.search(rf":{port}\s", line):
                    pids.update(int(m) for m in re.findall(r"pid=(\d+)", line))
        except Exception:
            pass
    if not pids and shutil.which("fuser"):
        try:
            out = subprocess.run(["fuser", "-n", "tcp", str(port)],
                                 capture_output=True, text=True, timeout=8)
            text = (out.stdout or "") + (out.stderr or "")
            pids.update(int(tok) for tok in re.split(r"\s+", text)
                        if tok.isdigit())
        except Exception:
            pass
    return sorted(pids)


def _pid_cmdline(pid):
    """Command line of a PID, or "" when unavailable."""
    pid = int(pid)
    if os.name == "nt":
        cmd = (f'(Get-CimInstance Win32_Process -Filter "ProcessId={pid}")'
               ".CommandLine")
        try:
            out = subprocess.run(
                ["powershell", "-NoProfile", "-NonInteractive", "-Command",
                 cmd], capture_output=True, text=True, timeout=8)
            return (out.stdout or "").strip()
        except Exception:
            return ""
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as f:
            return f.read().replace(b"\x00", b" ").decode(
                "utf-8", "replace").strip()
    except OSError:
        pass
    try:
        out = subprocess.run(["ps", "-p", str(pid), "-o", "command="],
                             capture_output=True, text=True, timeout=8)
        return (out.stdout or "").strip()
    except Exception:
        return ""


def _terminate_pid(pid):
    """SIGTERM a PID, escalating to SIGKILL after a short grace period."""
    pid = int(pid)
    if os.name == "nt":
        try:
            subprocess.run(["taskkill", "/PID", str(pid), "/F"],
                           capture_output=True, timeout=10)
        except Exception:
            pass
        return
    import signal
    try:
        os.kill(pid, signal.SIGTERM)
    except OSError:
        return
    deadline = time.time() + 2.0
    while time.time() < deadline:
        try:
            os.kill(pid, 0)
        except OSError:
            return
        time.sleep(0.1)
    try:
        os.kill(pid, signal.SIGKILL)
    except OSError:
        pass


def _stop_gateway_units():
    """Stop known legacy gateway systemd user units (best-effort)."""
    if os.name == "nt" or not shutil.which("systemctl"):
        return
    for unit in ("swap-gateway.service", "llama-rpc-farm-gateway.service"):
        try:
            res = subprocess.run(
                ["systemctl", "--user", "stop", unit],
                capture_output=True, text=True, timeout=10)
            if res.returncode == 0:
                print(f"[gateway] stopped legacy unit {unit}")
        except Exception:
            pass


def _stop_stale_gateway(port):
    """Stop a stale standalone `swap_gateway.py` so the app can own the port.

    Only processes whose command line is recognizably this project's
    gateway are killed; an unrelated service on the port is left alone
    (and the embedded gateway stays off, exactly as before). Returns True
    once the port is free.
    """
    if not _port_open(port):
        return False
    pids = _port_listener_pids(port)
    if not pids:
        print(f"[gateway] :{port} is answering but its PID could not be "
              "identified — leaving it alone")
        return False
    stopped = False
    for pid in pids:
        cmd = _pid_cmdline(pid)
        low = (cmd or "").lower()
        if "swap_gateway.py" not in low and \
                "llama-rpc-farm-gateway" not in low:
            print(f"[gateway] :{port} held by unrelated process {pid} "
                  f"({cmd[:120] or 'unknown'}) — leaving it alone")
            continue
        print(f"[gateway] stale standalone gateway pid {pid} on :{port} — "
              "stopping it so the embedded gateway can start")
        _stop_gateway_units()
        _terminate_pid(pid)
        stopped = True
    if not stopped:
        return False
    for _ in range(25):
        if not _port_open(port):
            return True
        time.sleep(0.2)
    print(f"[gateway] :{port} still bound after stopping the stale gateway")
    return False


def start_swap_gateway():
    """Run the view/access-key gateway in-process on :8090.

    Packaged builds have no source checkout to launch src/swap_gateway.py
    as a separate service, and every client-facing path (README examples,
    UI Chat/Tests) expects :8090. Embedding it keeps one binary and makes
    the gateway lifecycle follow the app: on startup a stale standalone
    `swap_gateway.py` listener is stopped so it can never outlive the app
    and serve old code. An unrelated service on the port is untouched
    (set FARM_GATEWAY_DISABLE=1 to keep an external gateway deliberately).
    """
    global _GATEWAY_SERVER
    if _GATEWAY_SERVER is not None:
        return
    if (os.environ.get("FARM_GATEWAY_DISABLE") or "").lower() in (
            "1", "true", "yes", "on"):
        return
    # pytest spins the app up repeatedly; never bind a real port there.
    if os.environ.get("PYTEST_CURRENT_TEST"):
        return
    try:
        port = int(os.environ.get("GATEWAY_PORT") or 8090)
    except ValueError:
        port = 8090
    if _port_open(port):
        if not _stop_stale_gateway(port):
            print(f"[gateway] :{port} already answering — using external "
                  "gateway")
            return
    if _port_open(port):
        print(f"[gateway] :{port} still answering — using external gateway")
        return
    try:
        import threading
        import uvicorn
        import swap_gateway
        cfg = uvicorn.Config(
            swap_gateway.app,
            host=(os.environ.get("GATEWAY_HOST") or "0.0.0.0"),
            port=port,
            log_level="warning",
            access_log=False,
        )
        server = uvicorn.Server(cfg)
        thread = threading.Thread(target=server.run, daemon=True,
                                  name="swap-gateway")
        thread.start()
        _GATEWAY_SERVER = server
        print(f"[gateway] embedded swap gateway listening on "
              f"{cfg.host}:{cfg.port}")
    except Exception as e:
        # A gateway failure must never take down the farm UI; clients can
        # still target llama-swap :8080 directly.
        print(f"[gateway] failed to start embedded gateway: {e}")


@app.on_event("shutdown")
def stop_swap_gateway():
    global _GATEWAY_SERVER
    if _GATEWAY_SERVER is not None:
        _GATEWAY_SERVER.should_exit = True
        _GATEWAY_SERVER = None


@app.on_event("startup")
def seed_local_master():
    """The machine running this app is the default master node."""
    # Frozen/portable builds start with an empty data dir: create config/,
    # generated/, model/usage dirs and minimal defaults before anything
    # reads them. Best-effort, never blocks boot.
    ensure_runtime_layout()
    # Agent tokens/passwords belong in the vault, never in machines.yaml.
    # Trigger the write boundary BEFORE any startup write so a legacy value
    # is vaulted rather than dropped by an unrelated rewrite.
    try:
        cfg.migrate_machine_secrets()
    except Exception:
        pass
    local_node.ensure_local_master()
    try:
        _ensure_farm_agent_token()
    except Exception:
        pass
    start_swap_gateway()
    # DEC-17: one-time stamp of `engine` on legacy group entries (both
    # collections) so existing groups keep rendering under their engine.
    # Best-effort: a locked/read-only config dir never blocks boot.
    try:
        cfg.migrate_group_engines()
    except Exception:
        pass
    # Engine reconcile at boot: whatever engine settings.yaml names is the
    # source of truth, so the swap host's config file (per-engine, DEC-03)
    # must be regenerated from the CURRENT model configs before anyone
    # points a client at :8080 — otherwise a daemon left over from a
    # previous run keeps serving its stale entries until a manual Launch
    # All. Best-effort + backgrounded: agent/config trouble must never
    # block boot (same policy as migrate_group_engines above).
    import threading as _thr

    def _reconcile_swap():
        import swap_manager as _sm
        # access.py: background housekeeping is always trusted, so the
        # reconcile sees every model (including `private: true` ones)
        # and keeps their swap entries pushed while the HTTP gate is
        # what actually hides them from foreign viewers.
        import access as _access
        with _access.trust_current_thread():
            try:
                _sm.sync_service_unit()
                report = _sm.push_scripts()
                if report.get("error"):
                    print(f"[boot-reconcile] push failed: {report['error']}")
                    return
                # Only a daemon that is ALREADY running needs the reload; a
                # stopped one picks the fresh config up at its next launch
                # (do_stop_swap/do_start in launch()). reload_daemon_async
                # skips when a reload is already in progress.
                if _sm.status().get("running"):
                    res = _sm.reload_daemon_async()
                    if res.get("error"):
                        print(f"[boot-reconcile] daemon reload failed: "
                              f"{res['error']}")
                print("[boot-reconcile] swap config pushed for active engine")
            except Exception as e:
                print(f"[boot-reconcile] skipped: {e}")

    _thr.Thread(target=_reconcile_swap, daemon=True,
                name="boot-swap-reconcile").start()

    def _cleanup_hf_cache():
        try:
            res = hf_downloader.cleanup_stale_cache()
            if (res.get("removed_incomplete")
                    or res.get("removed_locks")):
                print(
                    "[cache-cleanup] removed "
                    f"{res['removed_incomplete']} incomplete, "
                    f"{res['removed_locks']} lock file(s), "
                    f"{res['removed_bytes'] / 1e9:.1f} GB; "
                    f"skipped {res.get('skipped_active', 0)} active")
            for err in (res.get("errors") or [])[:5]:
                print(f"[cache-cleanup] {err}")
        except Exception as e:
            print(f"[cache-cleanup] skipped: {e}")

    _thr.Thread(target=_cleanup_hf_cache, daemon=True,
                name="boot-cache-cleanup").start()

    # Conservative orphan RPC-server reaper (per-port kills only, protects
    # loaded/loading models + active --rpc endpoints + young processes).
    try:
        reaper.start()
    except Exception as e:
        print(f"[reaper] start skipped: {e}")

    # Rolling per-machine hardware history for the Performance tab
    # (FARM_STATS_HISTORY_SEC, 0 disables).
    try:
        stats_history.start()
    except Exception as e:
        print(f"[stats-history] start skipped: {e}")

    # Keep llama-swap's short-lived captures of failed requests so the
    # Activity tab can show why they failed.
    try:
        swap_captures.start(swap_manager.status, _swap_api_get)
    except Exception as e:
        print(f"[swap-captures] start skipped: {e}")


class NoCacheStaticFiles(StaticFiles):
    """Static assets must always revalidate — a stale cached app.js made
    every button/comet look broken after UI fixes."""
    def file_response(self, *args, **kwargs):
        resp = super().file_response(*args, **kwargs)
        resp.headers["Cache-Control"] = "no-cache, must-revalidate"
        return resp

app.mount("/static", NoCacheStaticFiles(directory=os.path.join(ASSET_DIR, "ui", "static")), name="static")


# ---------------- client-IP view gate ----------------
# access.py decision, wired here: the app stays fully usable from every
# IP — what changes per peer is VISIBILITY of `private: true` configs.
# Raw-ASGI wrapper (not BaseHTTPMiddleware) so one pass covers http +
# websocket scopes. Starlette runs each request in its own asyncio task,
# so the contextvar the wrapper flips is scoped to that request alone —
# concurrent requests and background tasks never see each other's gate.
import access

class IpGateMiddleware:
    def __init__(self, asgi_app):
        self.app = asgi_app

    async def __call__(self, scope, receive, send):
        access.restrict_if_foreign(scope)
        try:
            await self.app(scope, receive, send)
        finally:
            # request-local hygiene: in production each request is a
            # fresh task (contextvar dies with it), but a reset keeps
            # the gate clean even when callers drive scopes manually
            # (tests, in-process probes).
            access.set_view_allow(None)
            access.set_restricted(False)

app.add_middleware(IpGateMiddleware)


# ---------------- pages ----------------
@app.middleware("http")
async def _no_cache_static(request: Request, call_next):
    """Static assets are versioned by ?v=mtime+boot, but Firefox can still
    serve a stale app.js after an upgrade and keep using old login logic.
    Force revalidation for every static file."""
    resp = await call_next(request)
    if (request.url.path.startswith("/static/")
            or request.url.path.startswith("/api/")):
        # API responses are live probes (agent/engine versions, jobs,
        # stats). Firefox will otherwise heuristically cache GETs and pin
        # buttons to stale "on latest"/disabled states after a release.
        resp.headers["Cache-Control"] = "no-store, max-age=0"
    return resp


import time as _time_mod
_DAEMON_BOOT = str(int(_time_mod.time()))  # per-boot unique

def _asset_version():
    """Cache buster: file mtimes + daemon boot time. The boot component
    guarantees a fresh URL after every daemon restart, so Firefox can
    never serve a stale cached app.js against a new index.html."""
    try:
        paths = [os.path.join(ASSET_DIR, "ui", "static", f)
                 for f in ("app.js", "style.css")]
        mt = str(int(max(os.path.getmtime(p) for p in paths)))
    except OSError:
        mt = "0"
    return f"{mt}-{_DAEMON_BOOT}"


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    # Never cache the shell page: a stale cached index.html with mismatched
    # element IDs used to silently break every button below the mismatch.
    # (app.js itself is versioned via ?v=mtime.)
    resp = templates.TemplateResponse(
        request, "index.html", {"asset_v": _asset_version()})
    resp.headers["Cache-Control"] = "no-store, max-age=0"
    return resp


# ---------------- cluster ops ----------------

def _ensure_stdio():
    """Give a missing sys.stdout / sys.stderr a sink.

    A detached start (Restart App on Windows, pythonw, a service) can run
    with no standard streams; uvicorn's log formatter calls
    sys.stdout.isatty() and the app died with "'NoneType' object has no
    attribute 'isatty'" / "Unable to configure formatter 'default'".
    Missing streams append to the same llama-web.log the restart paths
    use, falling back to os.devnull."""
    for name in ("stdout", "stderr"):
        if getattr(sys, name) is not None:
            continue
        try:
            stream = open(os.path.join(tempfile.gettempdir(), "llama-web.log"),
                          "a", encoding="utf-8", buffering=1)
        except OSError:
            stream = open(os.devnull, "w", encoding="utf-8")
        setattr(sys, name, stream)


if __name__ == "__main__":
    _ensure_stdio()
    # SINGLETON GUARD: every restart path (/api/restart, engine save) spawns
    # this module detached while the old process was supposed to die — and
    # sometimes it doesn't (observed: 4 Oct-1 orphans lived for 3 days next
    # to the live instance). An orphan is not idle: its boot-reconcile
    # re-pushes a stale swap config BEHIND the live app's back, silently
    # un-deploying variants the operator saved (observed live: --dad
    # vanished from llama-swap between two pushes). Guard: any instance
    # that fails to BIND :8199 kills its own swap-write capability and
    # exits — only the port owner ever survives. The pre-import sys.path
    # setup above is shared, so this block is intentionally at the bottom
    # (after all definitions) yet still before uvicorn starts.
    import socket as _socket
    # Short bind-wait first: /api/restart kills the old port owner THEN
    # spawns us — the dying process may still hold the socket for a
    # moment. Retrying up to ~10s keeps the legal restart path working;
    # a process STILL holding :8199 after that is a long-lived orphan,
    # and that exact shape is what must never boot.
    _deadline = time.time() + 10.0
    _bound = False
    while True:
        _probe = _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM)
        # SO_REUSEADDR: after fuser -k the old app's CLOSED connections sit
        # in TIME_WAIT up to 60s (observed: bind blocked 20s+, /proc/net/tcp
        # full of state-06 :2007 rows) — plain bind() fails the whole wait.
        # uvicorn's own bind (config.py) sets REUSEADDR and succeeds, so the
        # guard must probe the same way or it self-aborts on healthy boots.
        # A LIVE listener still conflicts even with REUSEADDR (same-addr
        # rebinding of an active LISTEN socket is EADDRINUSE), so the
        # orphan-detection semantics are unchanged.
        _probe.setsockopt(_socket.SOL_SOCKET, _socket.SO_REUSEADDR, 1)
        try:
            _probe.bind(("0.0.0.0", 8199))
            _bound = True
        except OSError:
            _probe.close()
            if time.time() >= _deadline:
                break
            time.sleep(0.5)
            continue
        _probe.close()
        break
    if not _bound:
        print("[singleton] :8199 already owned by another instance — "
              "exiting (no orphan reconciles, no stale swap pushes)",
              file=sys.stderr, flush=True)
        raise SystemExit(2)
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8199, reload=False)
