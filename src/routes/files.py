"""Cross-platform file explorer endpoints (peer-agent backed)."""
import os
import sys

from fastapi import APIRouter, HTTPException
from fastapi.concurrency import run_in_threadpool

import agent_client
import config_loader as cfg
import path_resolver

router = APIRouter()


# ---------------- file explorer (cross-platform) ----------------

def _fs_list_local(raw_path, show_hidden=False):
    import pathlib
    # handle store:// alias for model dir; resolve to real path but remember alias
    disp_path = raw_path
    if raw_path and raw_path.startswith(path_resolver.PREFIX):
        try:
            resolved = path_resolver.resolve(raw_path)
            raw_path = resolved
        except Exception:
            pass
    # expand ~ and env, normalize
    if not raw_path or raw_path.strip() == "":
        raw_path = os.path.expanduser("~")
    else:
        raw_path = os.path.expanduser(os.path.expandvars(raw_path))
    raw_path = os.path.abspath(raw_path) if not raw_path.startswith("/") else os.path.normpath(raw_path)
    # if path is file, show its dir
    if os.path.isfile(raw_path):
        raw_path = os.path.dirname(raw_path)
    exists = os.path.exists(raw_path)
    is_dir = os.path.isdir(raw_path) if exists else False
    parent = os.path.dirname(raw_path) if raw_path != "/" else "/"
    entries = []
    if is_dir:
        try:
            with os.scandir(raw_path) as it:
                for entry in it:
                    if not show_hidden and entry.name.startswith("."):
                        continue
                    try:
                        is_d = entry.is_dir(follow_symlinks=False)
                    except OSError:
                        is_d = False
                    entries.append({"name": entry.name, "path": os.path.join(raw_path, entry.name), "is_dir": is_d})
        except PermissionError as e:
            raise HTTPException(status_code=403, detail=f"permission denied: {raw_path}")
        # dirs first, then alpha
        entries.sort(key=lambda e: (not e["is_dir"], e["name"].lower()))
    # alias for portability (e.g. /mnt/... -> store://llm-models/...)
    try:
        alias = path_resolver.to_alias(raw_path)
    except Exception:
        alias = raw_path
    return {"path": raw_path, "alias": alias, "display_path": disp_path, "parent": parent, "exists": exists, "is_dir": is_dir, "os": "windows" if os.name == "nt" else "macos" if sys.platform == "darwin" else "linux", "entries": entries}


def _fs_list_agent(machine, raw_path, show_hidden=False):
    import ntpath
    is_win = machine.get("os", "linux") == "windows"
    join = ntpath.join if is_win else os.path.join
    dirname = ntpath.dirname if is_win else os.path.dirname
    if not raw_path or raw_path.strip() == "":
        raw_path = "~"
    if raw_path.startswith(path_resolver.PREFIX):
        try:
            raw_path = path_resolver.resolve_for(raw_path, machine.get("os", "linux"))
        except Exception:
            pass
    stat = agent_client.file_stat(machine, raw_path)
    if not stat.get("exists"):
        raise HTTPException(status_code=404,
                            detail=f"path not found or unreachable: {raw_path}")
    real = stat["path"]
    is_dir = bool(stat.get("is_dir"))
    parent = dirname(real)
    if not is_dir:
        real = parent
        is_dir = True
    listing = agent_client.file_list(machine, real, show_hidden)
    entries = [{"name": e["name"], "path": join(real, e["name"]),
                "is_dir": bool(e.get("is_dir"))}
               for e in listing.get("entries", [])]
    entries.sort(key=lambda e: (not e["is_dir"], e["name"].lower()))
    try:
        alias = path_resolver.to_alias(real)
    except Exception:
        alias = real
    return {"path": real, "alias": alias, "parent": parent,
            "exists": True, "is_dir": is_dir,
            "os": "windows" if is_win else
                  "macos" if machine.get("os") == "macos" else "linux",
            "entries": entries}


def _fs_list_remote(machine, raw_path, show_hidden=False):
    try:
        return _fs_list_agent(machine, raw_path, show_hidden)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"agent browse failed: {e}")

@router.get("/api/fs/browse")
async def fs_browse(path: str = "", machine: str = "", show_hidden: bool = False):
    """Browse filesystem — local host if machine=, else remote via the agent.
    Works on both Linux and Windows (machine.os decides). Query: path, machine (name), show_hidden."""
    if machine:
        m = cfg.machines.get(machine)
        if not m:
            raise HTTPException(status_code=404, detail=f"machine '{machine}' not found")
        # remote
        try:
            return await run_in_threadpool(_fs_list_remote, m, path, show_hidden)
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    else:
        try:
            return await run_in_threadpool(_fs_list_local, path, show_hidden)
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
