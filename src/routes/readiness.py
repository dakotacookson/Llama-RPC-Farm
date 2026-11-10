"""Per-machine setup/readiness probes."""
import os
import time
from concurrent.futures import ThreadPoolExecutor

from fastapi import APIRouter
from fastapi.concurrency import run_in_threadpool

import agent_client
import config_loader as cfg
import local_node
from machine_tokens import _ensure_agent_ready
import rpc_manager

router = APIRouter()

# Readiness probes every machine (agent + binary checks): cache briefly so
# dashboard repeats and farmctl doctor don't hammer the fleet.
_READINESS_CACHE = {}
_READINESS_TTL = 10.0


def _ready_cache_get(key):
    hit = _READINESS_CACHE.get(key)
    if hit and time.time() - hit[0] < _READINESS_TTL:
        return hit[1]
    return None


def _ready_cache_put(key, value):
    _READINESS_CACHE[key] = (time.time(), value)


# ---------------- setup readiness (per machine) ---------------------------

def _rpc_verify_remote(machine, path):
    """RPC binaries disagree on --version; --help is the reliable probe.
    Returns (ok, error)."""
    last = None
    for args in (["--help"], ["--version"]):
        try:
            r = agent_client.install_verify(
                machine, path, args, timeout=30) or {}
            if r.get("returncode") == 0:
                return True, None
            last = str(r.get("output") or "verify failed")
        except Exception as e:
            last = str(e)
            if "not allowed" in last:
                # Older agents only allow llama-server/swap in /install/verify.
                # Fall back to presence for those agents; installs after the
                # agent update use the real run check above.
                try:
                    st = agent_client.file_stat(machine, path) or {}
                    if st.get("exists") and st.get("is_file"):
                        return True, None
                except Exception:
                    pass
                break
    return False, (last or "RPC verify failed")[-300:]


def _local_engine_readiness(machine):
    """llama.cpp + ggml-rpc-server on this host, using the effective bin."""
    from installer.env import RealFS
    from installer.probe import _version_from_output

    fs = RealFS()
    is_win = (machine.get("os") or ("windows" if os.name == "nt" else "linux")) == "windows"
    bin_dir = ((machine.get("llama_bin") or "").strip()
               or rpc_manager._effective_bin_dir(machine) or "")
    bin_dir = os.path.expanduser(os.path.expandvars(bin_dir))
    server_name = "llama-server.exe" if is_win else "llama-server"
    rpc_name = "ggml-rpc-server.exe" if is_win else "ggml-rpc-server"
    if bin_dir.rstrip("/\\").endswith(server_name):
        bin_dir = os.path.dirname(bin_dir)
    server = os.path.join(bin_dir, server_name)
    rpc = os.path.join(bin_dir, rpc_name)
    version = None
    cpp_error = None
    rpc_error = None
    cpp_ok = False
    rpc_ok = False
    if fs.isfile(server):
        try:
            r = fs.file_exists_and_runs(server)
            cpp_ok = bool(r.ok)
            if cpp_ok:
                version = _version_from_output(r.stdout + " " + r.stderr)
            else:
                cpp_error = (r.stderr or r.stdout or "llama-server verify failed").strip()[-300:]
        except Exception as e:
            cpp_error = str(e)
    if fs.isfile(rpc):
        # --version is not universal for ggml-rpc-server; try --help too.
        for arg in ("--help", "--version"):
            try:
                rr = fs.file_exists_and_runs(rpc, arg)
                if rr.ok:
                    rpc_ok = True
                    break
                rpc_error = (rr.stderr or rr.stdout
                             or "ggml-rpc-server verify failed").strip()[-300:]
            except Exception as e:
                rpc_error = str(e)
    return {
        "llama_cpp": {"installed": cpp_ok,
                      "path": server if fs.isfile(server) else None,
                      "version": version, "error": cpp_error},
        "llama_rpc": {"installed": rpc_ok,
                      "path": rpc if fs.isfile(rpc) else None,
                      "error": rpc_error},
    }


def _readiness_for(machine, swap_host_name):
    """One machine's setup checklist: agent, llama.cpp, RPC, llama-swap."""
    name = machine.get("name") or "?"
    is_local = local_node.is_local(machine)
    out = {"machine": name, "local": is_local,
           "is_swap_host": name == swap_host_name,
           "agent": {"ok": is_local, "local": is_local,
                     "version": None, "error": None},
           "llama_cpp": {"installed": False, "path": None, "version": None},
           "llama_rpc": {"installed": False, "path": None},
           "llama_swap": {"installed": False, "path": None,
                          "required": name == swap_host_name}}
    if is_local:
        out.update(_local_engine_readiness(machine))
    else:
        caps, err = _ensure_agent_ready(machine)
        if err:
            out["agent"] = {"ok": False, "local": False,
                            "version": None, "error": err}
        else:
            try:
                h = agent_client.health(machine, timeout=4)
            except Exception:
                h = {}
            out["agent"] = {"ok": True, "local": False,
                            "version": h.get("agent_version"),
                            "protocol": h.get("protocol_version"),
                            "error": None}
            lp = (caps or {}).get("llama_binary")
            rp = (caps or {}).get("rpc_binary")
            # The machine's configured llama_bin is the launch path; the
            # agent's capability search can point at a different (stale)
            # build tree. Prefer the configured dir, then fall back to the
            # agent-discovered path.
            os_type = (machine.get("os") or "linux").lower()
            server_name = "llama-server.exe" if os_type == "windows" else "llama-server"
            rpc_name = "ggml-rpc-server.exe" if os_type == "windows" else "ggml-rpc-server"
            sep = "\\" if os_type == "windows" else "/"
            cfg_cpp = cfg_rpc = None
            lb = (machine.get("llama_bin") or "").strip()
            if lb:
                base = lb.rstrip("/\\")
                if base.endswith(server_name):
                    cfg_cpp = base
                    base = base.rsplit(sep, 1)[0]
                else:
                    cfg_cpp = base + sep + server_name
                cfg_rpc = base + sep + rpc_name
            last_err = None
            for cand in dict.fromkeys([c for c in (cfg_cpp, lp) if c]):
                try:
                    v = agent_client.install_verify(
                        machine, cand, ["--version"], timeout=35) or {}
                    if v.get("returncode") == 0:
                        out["llama_cpp"]["installed"] = True
                        out["llama_cpp"]["path"] = cand
                        from installer.probe import _version_from_output
                        out["llama_cpp"]["version"] = _version_from_output(
                            str(v.get("output") or ""))
                        break
                    last_err = (str(v.get("output") or "verify failed"))[-300:]
                except Exception as e:
                    last_err = str(e)
            out["llama_cpp"]["error"] = None if out["llama_cpp"]["installed"] else last_err
            last_rpc_err = None
            for cand in dict.fromkeys([c for c in (cfg_rpc, rp) if c]):
                rpc_ok, rpc_err = _rpc_verify_remote(machine, cand)
                if rpc_ok:
                    out["llama_rpc"]["installed"] = True
                    out["llama_rpc"]["path"] = cand
                    break
                last_rpc_err = rpc_err
            out["llama_rpc"]["error"] = None if out["llama_rpc"]["installed"] else last_rpc_err
    try:
        import swap_installer
        found, path = swap_installer.detect_on(machine)
        out["llama_swap"] = {"installed": bool(found), "path": path,
                             "required": name == swap_host_name}
    except Exception:
        pass
    out["ready"] = bool(
        out["agent"]["ok"]
        and out["llama_cpp"]["installed"]
        and out["llama_rpc"]["installed"]
        and (not out["is_swap_host"] or out["llama_swap"]["installed"]))
    return out


@router.get("/api/readiness")
async def readiness_all():
    """Per-machine setup checklist for the dashboard/machine page.

    Answers the new-user question: which peers still need llama.cpp +
    ggml-rpc-server, and is llama-swap installed on the swap host?
    """
    cached = _ready_cache_get("all")
    if cached is not None:
        return cached
    machines = cfg.machines.all()
    try:
        import swap_manager
        host_name = (swap_manager.get_host() or {}).get("name")
    except Exception:
        host_name = None
    with ThreadPoolExecutor(max_workers=len(machines) or 1) as pool:
        rows = list(pool.map(lambda m: _readiness_for(m, host_name), machines))
    result = {"machines": {r["machine"]: r for r in rows},
              "ready_all": all(r["ready"] for r in rows)}
    _ready_cache_put("all", result)
    return result


@router.get("/api/machines/{name}/readiness")
async def readiness_machine(name: str):
    cached = _ready_cache_get(("machine", name))
    if cached is not None:
        return cached
    machine = cfg.machines.require(name)
    try:
        import swap_manager
        host_name = (swap_manager.get_host() or {}).get("name")
    except Exception:
        host_name = None
    result = await run_in_threadpool(_readiness_for, machine, host_name)
    _ready_cache_put(("machine", name), result)
    return result
