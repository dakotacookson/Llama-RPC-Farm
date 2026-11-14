"""Copy missing model files between farm machines.

The shared-store model is still the default: every config keeps a
portable `store://` path. This module adds an opt-in fallback for boxes
that cannot reach that store (a Pi on a different mount, a Jetson with
an old CIFS client, ...):

  * a machine's `model_dir` overrides the store root for that machine;
  * settings `storage_mode: sync` (`model_sync.enabled` for old configs)
    + `model_sync.default_dir` gives every machine without an override a
    local default space (`~/llama-models` works on Linux/macOS/Windows
    once expanded on the target); `storage_mode: shared` keeps the
    NAS/CIFS mount and ignores per-machine model dirs;
  * when enabled, a deployment whose model file is missing on the box
    that will run llama-server is copied from any machine that has it.

Copies always go through the typed peer agent (new `/files/read` and
`/files/write_bin` endpoints), never a shell. The master orchestrates:
it reads locally, pulls from a peer, or relays peer -> peer.
"""
import os
import shutil
import tempfile
import threading
import time
import uuid

import agent_client
import config_loader as cfg
import local_node
import path_resolver
import settings_store
from runtime_paths import data_dir

CHUNK = 1024 * 1024
_LOG_EVERY = 128 * 1024 * 1024
_HOME_CACHE = {}
_JOBS = {}
_JOBS_LOCK = threading.Lock()
_RUNNING = {"all": False}


# ---------------------------------------------------------------- settings
def _load_settings():
    try:
        import yaml
        path = settings_store.SETTINGS_FILE
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
    except Exception:
        pass
    return {}


def sync_config(settings=None):
    """Storage mode + local-dir settings.

    `storage_mode` is the switch: "shared" keeps the store:// mount (NAS/
    CIFS); "sync" uses a per-machine model space and copies missing files.
    A legacy `model_sync.enabled` is honored when storage_mode is absent."""
    s = settings if isinstance(settings, dict) else _load_settings()
    ms = s.get("model_sync") or {}
    if not isinstance(ms, dict):
        ms = {}
    mode = str(s.get("storage_mode") or "").strip().lower()
    if mode not in ("shared", "sync"):
        mode = "sync" if bool(ms.get("enabled")) else "shared"
    return {"mode": mode, "enabled": mode == "sync",
            "default_dir": str(ms.get("default_dir") or "").strip()}


def enabled(settings=None):
    return sync_config(settings)["enabled"]


# ------------------------------------------------------------- path helpers
def local_machine():
    """The machine this app process runs on (downloads/scans land here)."""
    try:
        for m in cfg.machines.all():
            if local_node.is_local(m):
                return m
    except Exception:
        pass
    return {"name": "local", "os": path_resolver.LOCAL_PLATFORM,
            "model_dir": ""}


def resolve_store_root(store, settings=None):
    """Physical root for a store alias on the app host.

    shared mode -> the NAS/CIFS mount; sync mode -> the app host's local
    model dir. Both modes keep the `store://` alias in model configs."""
    name = str(store or "").strip()
    alias = (name if name.startswith(path_resolver.PREFIX)
             else f"{path_resolver.PREFIX}{name}")
    return resolve_model_path(local_machine(), alias, settings)


def alias_under(path, root, alias):
    """Map an absolute path under `root` back to `alias/<relative>`.

    Used by model scanning when the scan root is a local model dir: the
    saved config must keep the portable store:// alias, not the local
    absolute path."""
    if not path or not root or not alias:
        return path
    try:
        p = os.path.normpath(os.path.expanduser(str(path)))
        r = os.path.normpath(os.path.expanduser(str(root)))
    except Exception:
        return path
    if p == r:
        return str(alias).rstrip("/\\")
    try:
        rel = os.path.relpath(p, r)
    except ValueError:
        return path
    if rel == ".." or rel.startswith(".." + os.sep):
        return path
    return str(alias).rstrip("/\\") + "/" + rel.replace(os.sep, "/")


def _home(machine):
    name = (machine or {}).get("name") or (machine or {}).get("ip") or "?"
    if name in _HOME_CACHE:
        return _HOME_CACHE[name]
    if local_node.is_local(machine):
        home = os.path.expanduser("~")
    else:
        try:
            home = str((agent_client.info(machine) or {}).get("home") or "")
        except Exception:
            home = ""
    _HOME_CACHE[name] = home
    return home


def _expand_dir(machine, raw):
    """Expand ~ / %USERPROFILE% for the TARGET machine's home."""
    d = str(raw or "").strip()
    if not d:
        return ""
    osname = str((machine or {}).get("os") or "linux").lower()
    home = ""
    token = ""
    if d.startswith("~"):
        token = "~"
        rest = d[2:] if d.startswith("~/") or d.startswith("~\\") else d[1:]
    elif d.upper().startswith("%USERPROFILE%"):
        token = "%USERPROFILE%"
        rest = d[len(token):]
    elif d.lower().startswith("$env:userprofile"):
        token = d[:len("$env:userprofile")]
        rest = d[len(token):]
    if token:
        home = _home(machine)
        if not home:
            return ""
        rest = rest.lstrip("/\\")
        if osname == "windows":
            d = home.rstrip("\\/") + (("\\" + rest.replace("/", "\\"))
                                      if rest else "")
        else:
            d = home.rstrip("/") + (("/" + rest.replace("\\", "/"))
                                    if rest else "")
    d = os.path.expandvars(d) if local_node.is_local(machine) else d
    return d.rstrip("/\\")


def effective_model_dir(machine, settings=None):
    """Absolute local model root for this machine, or "" for shared store.

    Shared NAS/CIFS mode ignores per-machine model_dir entirely — the
    switch is authoritative so flipping it always returns to the store."""
    ms = sync_config(settings)
    if ms["mode"] != "sync":
        return ""
    md = str((machine or {}).get("model_dir") or "").strip()
    if not md:
        md = ms["default_dir"]
    return _expand_dir(machine, md)


def _join(root, rest, osname):
    if not rest:
        return root
    if osname == "windows":
        return root.rstrip("\\/") + "\\" + rest.replace("/", "\\")
    return root.rstrip("/") + "/" + rest


def resolve_model_path(machine, base_path, settings=None):
    """The path the target machine will actually read the model from."""
    if not base_path:
        return base_path
    base_path = str(base_path)
    if base_path.startswith(path_resolver.PREFIX):
        root = effective_model_dir(machine, settings)
        if root:
            rest = base_path[len(path_resolver.PREFIX):].partition("/")[2]
            return _join(root, rest,
                         str((machine or {}).get("os") or "linux").lower())
    return path_resolver.resolve_for(
        base_path, (machine or {}).get("os") or "linux")


def candidate_paths(machine, base_path, settings=None):
    """Both layouts that can hold this model on one machine.

    The effective (model_dir/default) path is what launch uses; the plain
    store path is still a valid copy SOURCE when that box can see it."""
    out = []
    p = resolve_model_path(machine, base_path, settings)
    if p:
        out.append(p)
    p2 = path_resolver.resolve_for(
        base_path, (machine or {}).get("os") or "linux")
    if p2 and p2 not in out:
        out.append(p2)
    return out


# ------------------------------------------------------------ file presence
def _local_stat(path):
    try:
        p = os.path.expanduser(str(path))
        st = os.stat(p)
        return {"exists": True, "is_file": os.path.isfile(p),
                "size": st.st_size}
    except OSError:
        return {"exists": False, "is_file": False, "size": 0}


def file_stat(machine, path):
    if not path:
        return {"exists": False, "is_file": False, "size": 0}
    if local_node.is_local(machine):
        return _local_stat(path)
    try:
        return agent_client.file_stat(machine, path)
    except Exception:
        return {"exists": False, "is_file": False, "size": 0}


def file_present(machine, path, size=None):
    st = file_stat(machine, path)
    if not st.get("exists") or not st.get("is_file"):
        return False
    if size is not None:
        try:
            if int(st.get("size") or 0) != int(size):
                return False
        except (TypeError, ValueError):
            return False
    return True


# -------------------------------------------------------------- deployments
def _deployments(model_id, variant=None):
    """(label, merged_config) for the base model and matching variants."""
    model = cfg.get_model(model_id)
    if not model:
        return []
    base = dict(model)
    base.pop("variants", None)
    out = [("", base)]
    want = (variant or "").strip().lower()
    for v in (model.get("variants") or []):
        label = str(v.get("label") or "").strip()
        if want and label.lower() != want:
            continue
        merged = dict(base)
        for k in ("servers", "group", "device_order",
                  "device_order_policy", "tensor_split_mode",
                  "tensor_split", "fit_margin", "fit_ram_allow",
                  "expose", "pinned", "ttl", "launch_mode",
                  "launch_mode_local_port"):
            if k in v and v[k] is not None:
                merged[k] = v[k]
        if v.get("llama_bin"):
            merged["llama_bin"] = v["llama_bin"]
        if isinstance(v.get("flags"), dict) and v["flags"]:
            merged["flags"] = {**(base.get("flags") or {}), **v["flags"]}
        out.append((label, merged))
    return out


def target_machine(model_id, label, merged):
    """Machine that runs llama-server for this deployment, or None."""
    runtime = str(merged.get("runtime") or "llama").lower()
    if runtime in ("custom", "mlx"):
        return None
    if not (merged.get("servers") or []):
        return None
    try:
        import swap_manager
        host = swap_manager.get_host()
        dec = swap_manager.deployment_launch_mode(model_id, label or None)
        box = dec.get("box")
        if dec.get("mode") == "local" and box:
            if isinstance(box, dict):
                return cfg.machines.get(box.get("name")) or box
            return cfg.machines.get(str(box))
        return host
    except Exception:
        servers = merged.get("servers") or []
        launch = str(merged.get("launch_mode") or "auto").lower()
        if launch == "local" and servers:
            return cfg.machines.get(servers[0])
        return next((m for m in cfg.machines.all() if local_node.is_local(m)),
                    None)


def _machines_ordered():
    machines = list(cfg.machines.all())

    def key(m):
        if local_node.is_local(m):
            return 0
        return 1 if m.get("role") == "master" else 2
    machines.sort(key=key)
    return machines


# ----------------------------------------------------------------- copying
class _ProgressReader:
    def __init__(self, obj, total, log, label):
        self.obj = obj
        self.total = int(total or 0)
        self.log = log
        self.label = label
        self.done = 0
        self._next = _LOG_EVERY

    def read(self, n=-1):
        data = self.obj.read(n)
        if data:
            self.done += len(data)
            if self.done >= self._next:
                self._next += _LOG_EVERY
                pct = (100.0 * self.done / self.total) if self.total else 0
                self.log(f"  {self.label}: {self.done/1024**2:.0f} MiB "
                         f"({pct:.0f}%)")
        return data


def _local_parent(path):
    parent = os.path.dirname(os.path.expanduser(str(path)))
    if parent:
        os.makedirs(parent, exist_ok=True)
    return parent


def _copy_local_to_local(src, dst, size, log, label):
    parent = _local_parent(dst)
    tmp = os.path.join(parent, f".{os.path.basename(dst)}.part")
    with open(os.path.expanduser(src), "rb") as fsrc, \
            open(tmp, "wb") as fdst:
        reader = _ProgressReader(fsrc, size, log, label)
        while True:
            chunk = reader.read(CHUNK)
            if not chunk:
                break
            fdst.write(chunk)
        fdst.flush()
        try:
            os.fsync(fdst.fileno())
        except OSError:
            pass
    os.replace(tmp, os.path.expanduser(dst))
    try:
        os.chmod(os.path.expanduser(dst), 0o644)
    except OSError:
        pass


def _copy_local_to_remote(src, dst_machine, dst, size, log, label):
    parent = _parent_for(dst_machine, dst)
    if parent:
        try:
            agent_client.file_mkdir(dst_machine, parent)
        except Exception:
            pass
    with open(os.path.expanduser(src), "rb") as fsrc:
        reader = _ProgressReader(fsrc, size, log, label)
        agent_client.file_write_stream(dst_machine, dst, reader, size,
                                       mode=0o644, create_parents=True)


def _copy_remote_to_local(src_machine, src, dst, size, log, label):
    parent = _local_parent(dst)
    tmp = os.path.join(parent, f".{os.path.basename(dst)}.part")
    resp = agent_client.file_read_stream(src_machine, src)
    try:
        reader = _ProgressReader(resp, size, log, label)
        with open(tmp, "wb") as fdst:
            while True:
                chunk = reader.read(CHUNK)
                if not chunk:
                    break
                fdst.write(chunk)
            fdst.flush()
            try:
                os.fsync(fdst.fileno())
            except OSError:
                pass
    finally:
        try:
            resp.close()
        except Exception:
            pass
    os.replace(tmp, os.path.expanduser(dst))
    try:
        os.chmod(os.path.expanduser(dst), 0o644)
    except OSError:
        pass


def _copy_remote_to_remote(src_machine, src, dst_machine, dst, size,
                           log, label):
    parent = _parent_for(dst_machine, dst)
    if parent:
        try:
            agent_client.file_mkdir(dst_machine, parent)
        except Exception:
            pass
    resp = agent_client.file_read_stream(src_machine, src)
    try:
        reader = _ProgressReader(resp, size, log, label)
        agent_client.file_write_stream(dst_machine, dst, reader, size,
                                       mode=0o644, create_parents=True)
    finally:
        try:
            resp.close()
        except Exception:
            pass


def _parent_for(machine, path):
    osname = str((machine or {}).get("os") or "linux").lower()
    p = str(path or "")
    if osname == "windows":
        return p.rsplit("\\", 1)[0] if "\\" in p else ""
    return os.path.dirname(p)


def copy_between(src_machine, src_path, dst_machine, dst_path, size, log):
    label = f"{src_machine.get('name')} -> {dst_machine.get('name')}"
    src_local = local_node.is_local(src_machine)
    dst_local = local_node.is_local(dst_machine)
    log(f"copy {size/1024**2:.1f} MiB  {src_path} -> {dst_path} ({label})")
    if src_local and dst_local:
        _copy_local_to_local(src_path, dst_path, size, log, label)
    elif src_local:
        _copy_local_to_remote(src_path, dst_machine, dst_path, size, log, label)
    elif dst_local:
        _copy_remote_to_local(src_machine, src_path, dst_path, size, log, label)
    else:
        _copy_remote_to_remote(src_machine, src_path, dst_machine, dst_path,
                               size, log, label)


# ------------------------------------------------------------------- ensure
def _source_for(base_path, target, target_path, settings):
    """Return (machine, path, size) for a file that can be copied.

    Prefers a same-machine alternate layout (shared store), then the
    master, then other peers."""
    tname = (target or {}).get("name")
    # same-machine alternate path (e.g. the shared store while model_dir is set)
    for p in candidate_paths(target, base_path, settings):
        if p == target_path:
            continue
        st = file_stat(target, p)
        if st.get("exists") and st.get("is_file"):
            return target, p, int(st.get("size") or 0)
    for m in _machines_ordered():
        if (m.get("name") or m.get("ip")) == tname:
            continue
        for p in candidate_paths(m, base_path, settings):
            st = file_stat(m, p)
            if st.get("exists") and st.get("is_file"):
                return m, p, int(st.get("size") or 0)
    return None


def ensure_model(model_id, variant=None, log=print, settings=None):
    """Ensure the model file exists on every deployment target."""
    settings = settings if isinstance(settings, dict) else _load_settings()
    if not enabled(settings):
        return {"enabled": False, "copied": [], "skipped": [], "errors": []}
    model = cfg.get_model(model_id)
    if not model:
        return {"enabled": True, "copied": [], "skipped": [],
                "errors": [f"{model_id}: not found"]}
    copied, skipped, errors = [], [], []
    for label, merged in _deployments(model_id, variant):
        target = target_machine(model_id, label, merged)
        if target is None:
            continue
        base_path = merged.get("base_path") or merged.get("path")
        if not base_path:
            continue
        target_path = resolve_model_path(target, base_path, settings)
        if not target_path:
            continue
        src = _source_for(base_path, target, target_path, settings)
        if src is None:
            if file_present(target, target_path):
                skipped.append(f"{model_id}--{label or 'base'}: present")
            else:
                errors.append(f"{model_id}--{label or 'base'}: no machine "
                              f"has {base_path}")
            continue
        src_machine, src_path, size = src
        if (src_machine.get("name") == target.get("name")
                and src_path == target_path):
            skipped.append(f"{model_id}--{label or 'base'}: present")
            continue
        if file_present(target, target_path, size):
            skipped.append(f"{model_id}--{label or 'base'}: present")
            continue
        try:
            copy_between(src_machine, src_path, target, target_path, size, log)
            copied.append(f"{model_id}--{label or 'base'}: "
                          f"{src_machine.get('name')} -> {target.get('name')}")
        except Exception as e:
            errors.append(f"{model_id}--{label or 'base'}: {e}")
            log(f"  copy failed: {e}")
    return {"enabled": True, "copied": copied, "skipped": skipped,
            "errors": errors}


def ensure_all(log=print, settings=None):
    settings = settings if isinstance(settings, dict) else _load_settings()
    if not enabled(settings):
        return {"enabled": False, "copied": [], "skipped": [], "errors": []}
    results = {"enabled": True, "copied": [], "skipped": [], "errors": []}
    for m in cfg.list_models():
        mid = m.get("model_id")
        if not mid:
            continue
        r = ensure_model(mid, log=log, settings=settings)
        for k in ("copied", "skipped", "errors"):
            results[k].extend(r.get(k) or [])
    return results


# -------------------------------------------------------------------- jobs
def _job(job_id):
    with _JOBS_LOCK:
        return _JOBS.get(job_id)


def _run_job(job_id, model_id=None):
    job = _job(job_id)
    if not job:
        return
    job["state"] = "running"

    def log(line):
        with _JOBS_LOCK:
            job["log"].append(str(line))
            job["log"] = job["log"][-400:]
            job["updated"] = time.time()
    try:
        if model_id:
            res = ensure_model(model_id, log=log)
        else:
            res = ensure_all(log=log)
        with _JOBS_LOCK:
            job["result"] = res
            job["state"] = "done"
            job["finished"] = time.time()
    except Exception as e:
        with _JOBS_LOCK:
            job["result"] = {"ok": False, "error": str(e)}
            job["state"] = "failed"
            job["finished"] = time.time()
            job["log"].append(f"error: {e}")
    finally:
        if not model_id:
            with _JOBS_LOCK:
                _RUNNING["all"] = False


def latest_job():
    with _JOBS_LOCK:
        if not _JOBS:
            return None
        jid = max(_JOBS.values(),
                  key=lambda j: j.get("started") or 0)["id"]
    return job_status(jid)


def start_job(model_id=None, log=print):
    """Start a background ensure job; coalesce ensure-all runs."""
    if not enabled():
        return {"ok": False, "error": "model sync is disabled"}
    with _JOBS_LOCK:
        if not model_id and _RUNNING["all"]:
            for j in _JOBS.values():
                if j.get("model_id") is None and j.get("state") in (
                        "queued", "running"):
                    return {"ok": True, "job_id": j["id"],
                            "coalesced": True}
        job_id = uuid.uuid4().hex[:12]
        _JOBS[job_id] = {"id": job_id, "machine": None,
                         "model_id": model_id, "state": "queued",
                         "log": [], "result": None,
                         "started": time.time(), "updated": time.time(),
                         "finished": None}
        if not model_id:
            _RUNNING["all"] = True
    threading.Thread(target=_run_job, args=(job_id, model_id),
                     daemon=True).start()
    return {"ok": True, "job_id": job_id}


def job_status(job_id):
    job = _job(job_id)
    if not job:
        return None
    out = dict(job)
    out["log"] = list(job.get("log") or [])[-200:]
    return out
