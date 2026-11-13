"""Machine/app install + update jobs and the unified tools installer."""
import json
import os
import platform as _platform
import re
import secrets
import sys
import shutil
import subprocess
import tempfile
import threading
import time

from fastapi import APIRouter, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse
from runtime_paths import asset_dir

import agent_client
from llama_version import _latest_llama_tag, _llama_build_no, _llama_version_payload, invalidate_llama_version
import config_loader as cfg
import local_node
import rpc_manager
import version as app_version
from update_state import _LLAMA_UPDATES_LOCK

ASSET_DIR = asset_dir()

router = APIRouter()


# ---------- per-machine llama.cpp UPDATE jobs (machine-page button) ----------
# Installs are slow multi-hundred-MB downloads (winbox's CUDA zip crawls
# at ~1.4MB/s — 15+ min), far past any sane synchronous request timeout,
# so the update button runs as a polled background job. One update at a
# time per machine; the registry is tiny and process-local like
# hf_downloader's jobs.
_LLAMA_UPDATES: dict = {}
@router.get("/api/llama-cpp-latest")
async def llama_cpp_latest():
    """Newest upstream llama.cpp release with binaries, for the update
    badges (compared against each machine's llama_version build number).
    Failures are 200 + latest=null + detail so a GitHub hiccup only
    blanks the badge, never errors the modal."""
    try:
        tag = await run_in_threadpool(_latest_llama_tag)
    except Exception as e:
        return {"latest": None, "latest_build": None,
                "detail": f"cannot reach GitHub releases: {e}"}
    return {"latest": tag, "latest_build": _llama_build_no(tag)}


@router.get("/api/machines/{name}/llama-cpp-version")
async def llama_cpp_version_machine(name: str):
    """Current llama-server build on one machine vs the latest upstream
    release — lets the UI swap the Update button for an 'on latest'
    chip. TTL-cached (agent probe + GitHub tag are both slow-ish), so the
    machine grid can re-ask on every render without hammering anything."""
    machine = cfg.machines.require(name)
    return await run_in_threadpool(_llama_version_payload, machine)



_AGENT_WHEEL_LOCK = threading.Lock()


def _agent_wheel_path():
    """Build/cache the current peer-agent wheel from this checkout."""
    import glob
    if getattr(sys, "frozen", False):
        wheels = sorted(glob.glob(os.path.join(
            ASSET_DIR, "agent-wheel", "llama_rpc_farm_agent-*.whl")),
            key=os.path.getmtime, reverse=True)
        if wheels:
            return wheels[0]
        raise RuntimeError("bundled agent wheel is missing from this build")
    src = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "peer_agent")
    out = os.path.join(rpc_manager.GENERATED_DIR, "agent-wheel")
    os.makedirs(out, exist_ok=True)
    with _AGENT_WHEEL_LOCK:
        wheels = sorted(glob.glob(os.path.join(out, "llama_rpc_farm_agent-*.whl")),
                        key=os.path.getmtime, reverse=True)
        newest_src = 0.0
        for root, _dirs, names in os.walk(src):
            if "build" in root.split(os.sep) or "__pycache__" in root:
                continue
            for n in names:
                try:
                    newest_src = max(newest_src, os.path.getmtime(os.path.join(root, n)))
                except OSError:
                    pass
        if wheels and os.path.getmtime(wheels[0]) >= newest_src:
            return wheels[0]
        proc = subprocess.run(
            [sys.executable, "-m", "pip", "wheel", "--no-deps", "-w", out, src],
            capture_output=True, text=True, timeout=180)
        if proc.returncode != 0:
            raise RuntimeError(f"could not build agent wheel: {(proc.stderr or proc.stdout)[-500:]}")
        wheels = sorted(glob.glob(os.path.join(out, "llama_rpc_farm_agent-*.whl")),
                        key=os.path.getmtime, reverse=True)
        if not wheels:
            raise RuntimeError("agent wheel build produced no wheel")
        return wheels[0]


_AGENT_REPO = os.environ.get("LLAMA_FARM_REPO",
                             "dakotacookson/Llama-RPC-Farm")

_AGENT_RELEASE_CACHE = {"ts": 0.0, "by_asset": {}}


def _agent_binary_asset(machine):
    """Release asset name for this peer's standalone agent, if published."""
    os_type = (machine.get("os") or "").lower()
    arch = ((machine.get("detected") or {}).get("arch") or "").lower()
    if not arch:
        try:
            arch = str((agent_client.info(machine) or {}).get("arch") or "").lower()
        except Exception:
            arch = ""
    if os_type == "windows":
        return "llama-peer-agent-windows-x64.exe"
    if os_type in ("macos", "darwin"):
        return ("llama-peer-agent-macos-arm64"
                if arch in ("arm64", "aarch64") else None)
    if os_type == "linux":
        return ("llama-peer-agent-linux-x86_64"
                if arch in ("x86_64", "amd64", "") else None)
    return None


_RELEASE_FEED_CACHE = {"ts": 0.0, "versions": [], "commits": {}}


def _release_feed_versions():
    """Newest release versions from the public releases Atom feed.

    The GitHub REST API is rate-limited (60/hr unauthenticated) and its
    array order changes when releases are recreated. The Atom feed is
    unauthenticated, ordered by publish time, and enough to construct the
    deterministic release asset URLs.
    """
    now = time.time()
    with _LLAMA_UPDATES_LOCK:
        if now - _RELEASE_FEED_CACHE["ts"] < 60:
            return list(_RELEASE_FEED_CACHE["versions"])
    import urllib.request
    try:
        req = urllib.request.Request(
            f"https://github.com/{_AGENT_REPO}/releases.atom",
            headers={"User-Agent": "llama-rpc-farm"})
        with urllib.request.urlopen(req, timeout=15) as r:
            body = r.read().decode("utf-8", "replace")
        vers = re.findall(r"/releases/tag/release-(\d+\.\d+\.\d+)", body)
        seen = []
        for v in vers:
            if v not in seen:
                seen.append(v)
        seen.sort(key=lambda v: tuple(int(p) for p in v.split(".")),
                  reverse=True)
        commits = {}
        for entry in re.findall(r"<entry>.*?</entry>", body, re.S):
            vm = re.search(r"release-(\d+\.\d+\.\d+)", entry)
            cm = re.search(r"/commit/([0-9a-f]{7,40})", entry)
            if vm and cm:
                commits[vm.group(1)] = cm.group(1)
    except Exception:
        seen = []
    with _LLAMA_UPDATES_LOCK:
        _RELEASE_FEED_CACHE.update(ts=now, versions=list(seen),
                                   commits=locals().get("commits") or {})
    return seen


def _release_commit(version):
    _release_feed_versions()
    with _LLAMA_UPDATES_LOCK:
        return _RELEASE_FEED_CACHE["commits"].get(version)


# Paths whose last-touching commit defines the app's build identity. Tests
# and docs are deliberately excluded: a docs-only merge must not prompt
# every app for an update. The agent identity is the last commit touching
# peer_agent/ (see the rolling workflow's prepare job).
APP_COMMIT_PATHS = (
    "src", "ui", "packaging", "scripts", "requirements.txt",
    "run.sh", "run.bat", "install.sh", "install.bat",
    "restart.sh", "restart-llama-swap.sh", ".github/workflows",
)


def _component_on_latest(current_commit, latest_commit, current_version,
                         latest_version, applied=""):
    """Update decision for one component.

    Commit fingerprints are exact: an app-only release keeps a peer agent's
    agent_commit unchanged, so the agent stays "on latest" even though the
    release tag moved. Fall back to tag/version comparison when either side
    has no fingerprint (older releases/agents)."""
    current_commit = (current_commit or "").strip()
    latest_commit = (latest_commit or "").strip()
    if latest_commit and current_commit:
        return current_commit == latest_commit
    return bool(latest_version and (current_version == latest_version
                                    or applied == latest_version))


def _repo_root():
    # This module lives at <repo>/src/routes/install.py — three levels up.
    return os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))


def _git_component_commit(paths):
    try:
        out = subprocess.run(["git", "-C", _repo_root(), "log", "-1",
                             "--format=%H", "--", *paths],
                            capture_output=True, text=True, timeout=10)
        return (out.stdout or "").strip() if out.returncode == 0 else ""
    except Exception:
        return ""


def _current_app_commit():
    try:
        import build_info
        commit = (getattr(build_info, "APP_COMMIT", "") or "").strip()
        if commit:
            return commit
    except Exception:
        pass
    return _git_component_commit(APP_COMMIT_PATHS)


_RELEASE_MANIFEST_CACHE = {"ts": 0.0, "version": None, "data": None}


def _release_manifest():
    """Newest release's component manifest (app/agent build commits).

    Published next to the release assets. Returns None for older releases,
    which makes callers fall back to the legacy tag comparison. Cached 5 min
    to stay well inside GitHub's public download limits.
    """
    versions = _release_feed_versions()
    if not versions:
        return None
    version = versions[0]
    now = time.time()
    with _LLAMA_UPDATES_LOCK:
        if (now - _RELEASE_MANIFEST_CACHE["ts"] < 300
                and _RELEASE_MANIFEST_CACHE["version"] == version):
            return _RELEASE_MANIFEST_CACHE["data"]
    import urllib.request
    data = None
    try:
        url = (f"https://github.com/{_AGENT_REPO}/releases/download/"
               f"release-{version}/release-manifest.json")
        req = urllib.request.Request(url, headers={"User-Agent":
                                                   "llama-rpc-farm"})
        with urllib.request.urlopen(req, timeout=15) as r:
            data = json.loads(r.read().decode("utf-8", "replace"))
        if not isinstance(data, dict):
            data = None
    except Exception:
        data = None
    with _LLAMA_UPDATES_LOCK:
        _RELEASE_MANIFEST_CACHE.update(ts=now, version=version, data=data)
    return data


def _manifest_commit(field):
    manifest = _release_manifest() or {}
    return (manifest.get(field) or "").strip()


def _agent_release_info(machine):
    """Latest release agent asset (release tag is the authoritative version)."""
    asset = _agent_binary_asset(machine)
    if not asset:
        return {"asset": None, "url": None, "version": None}
    versions = _release_feed_versions()
    if not versions:
        return {"asset": asset, "url": None, "version": None}
    version = versions[0]
    base = (f"https://github.com/{_AGENT_REPO}/releases/download/"
            f"release-{version}")
    wheel = f"llama_rpc_farm_agent-{version}-py3-none-any.whl"
    return {"asset": asset, "url": f"{base}/{asset}",
            "wheel_url": f"{base}/{wheel}", "version": version,
            "commit": _manifest_commit("agent_commit")}


_APP_RELEASE_CACHE = {"ts": 0.0, "by_asset": {}}


def _app_binary_asset():
    """Raw farm-app release asset for this host platform."""
    os_name = _platform.system().lower()
    arch = _platform.machine().lower()
    if os_name == "windows":
        return "llama-rpc-farm-windows-x64.exe"
    if os_name == "darwin":
        return ("llama-rpc-farm-macos-arm64"
                if arch in ("arm64", "aarch64") else None)
    if os_name == "linux":
        return ("llama-rpc-farm-linux-x86_64"
                if arch in ("x86_64", "amd64") else None)
    return None


def _app_release_info():
    """Latest release farm-app binary + universal source wheel."""
    asset = _app_binary_asset()
    versions = _release_feed_versions()
    if not versions:
        return {"asset": asset, "url": None, "wheel_url": None,
                "version": None}
    version = versions[0]
    base = (f"https://github.com/{_AGENT_REPO}/releases/download/"
            f"release-{version}")
    wheel = f"llama_rpc_farm-{version}-py3-none-any.whl"
    return {"asset": asset,
            "url": f"{base}/{asset}" if asset else None,
            "wheel_url": f"{base}/{wheel}",
            "version": version, "commit": _manifest_commit("app_commit")}


def _app_version_payload():
    current = getattr(app_version, "__version__", "")
    packaged = bool(getattr(sys, "frozen", False))
    own_commit = _current_app_commit()
    try:
        info = _app_release_info()
        latest = info.get("version")
        latest_commit = info.get("commit")
    except Exception:
        latest, latest_commit = None, None
    dirty = False
    if not packaged:
        repo = _repo_root()
        try:
            d = subprocess.run(["git", "-C", repo, "status", "--porcelain"],
                               capture_output=True, text=True, timeout=10)
            dirty = bool((d.stdout or "").strip())
        except Exception:
            pass
    return {"current": current, "latest": latest,
            "on_latest": _component_on_latest(
                own_commit, latest_commit, current, latest),
            "build_commit": own_commit, "latest_commit": latest_commit,
            "packaged": packaged, "source": not packaged,
            "dirty": dirty}


def _restart_app_binary():
    import time as _t
    _t.sleep(1.5)
    exe = os.path.abspath(sys.executable)
    args = [exe] + list(sys.argv[1:])
    try:
        if os.name == "nt":
            subprocess.Popen(args, close_fds=True)
            os._exit(0)
        os.execv(exe, args)
    except Exception:
        pass


def _update_app_binary():
    if not getattr(sys, "frozen", False):
        repo = _repo_root()
        git_repo = os.path.isdir(os.path.join(repo, ".git"))
        try:
            info = _app_release_info()
            version = info.get("version")
            if not version:
                return {"ok": False, "source": True,
                        "detail": "could not determine the latest release"}
            if git_repo:
                dirty = subprocess.run(
                    ["git", "-C", repo, "status", "--porcelain"],
                    capture_output=True, text=True, timeout=60)
                if (dirty.stdout or "").strip():
                    return {"ok": False, "source": True,
                            "detail": "source tree has local changes — commit "
                                      "or stash them before updating"}
                rel_sha = _release_commit(version)
                if rel_sha:
                    head = subprocess.run(
                        ["git", "-C", repo, "rev-parse", "HEAD"],
                        capture_output=True, text=True).stdout.strip()
                    if rel_sha != head:
                        chk = subprocess.run(
                            ["git", "-C", repo, "merge-base", "--is-ancestor",
                             rel_sha, "HEAD"], capture_output=True, text=True)
                        if chk.returncode == 0:
                            return {"ok": False, "source": True,
                                    "detail": f"source is newer than release "
                                              f"{version}; wait for the next "
                                              f"release"}
            # Source installs (git or plain copied trees) update from the
            # universal app wheel when the release publishes it, falling
            # back to the immutable source archive for older releases.
            import tarfile
            import urllib.request
            import zipfile
            with tempfile.TemporaryDirectory(prefix="farm-src-") as tmp:
                src = None
                wheel_url = info.get("wheel_url")
                if wheel_url:
                    try:
                        wpath = os.path.join(tmp, "app.whl")
                        with urllib.request.urlopen(wheel_url, timeout=180) as r, \
                                open(wpath, "wb") as out:
                            shutil.copyfileobj(r, out)
                        wroot = os.path.join(tmp, "wheel")
                        with zipfile.ZipFile(wpath) as zf:
                            for name in zf.namelist():
                                pth = os.path.normpath(name)
                                if pth.startswith("..") or os.path.isabs(pth):
                                    raise RuntimeError("unsafe wheel path")
                            zf.extractall(wroot)
                        if not (os.path.isfile(os.path.join(wroot, "src", "app.py"))
                                and os.path.isdir(os.path.join(wroot, "ui"))):
                            raise RuntimeError("wheel payload incomplete")
                        src = wroot
                    except Exception:
                        src = None
                if not src:
                    archive = (f"https://github.com/{_AGENT_REPO}/archive/"
                               f"refs/tags/release-{version}.tar.gz")
                    tarpath = os.path.join(tmp, "src.tar.gz")
                    with urllib.request.urlopen(archive, timeout=180) as r, \
                            open(tarpath, "wb") as out:
                        shutil.copyfileobj(r, out)
                    with tarfile.open(tarpath, "r:gz") as tar:
                        members = tar.getmembers()
                        for mem in members:
                            pth = os.path.normpath(mem.name)
                            if pth.startswith("..") or os.path.isabs(pth):
                                raise RuntimeError("unsafe archive path")
                        tar.extractall(tmp)
                    roots = [os.path.join(tmp, d) for d in os.listdir(tmp)
                             if os.path.isdir(os.path.join(tmp, d))]
                    if not roots:
                        raise RuntimeError("release archive has no source dir")
                    src = roots[0]
                for name in ("src", "ui", "peer_agent", "scripts",
                             "packaging"):
                    sdir = os.path.join(src, name)
                    ddir = os.path.join(repo, name)
                    if os.path.isdir(sdir):
                        shutil.copytree(sdir, ddir, dirs_exist_ok=True)
                for name in ("VERSION", "requirements.txt", "install.sh",
                             "install.bat", "run.sh", "run.bat",
                             "restart.sh", "restart-llama-swap.sh"):
                    sp = os.path.join(src, name)
                    if os.path.isfile(sp):
                        shutil.copy2(sp, os.path.join(repo, name))
            subprocess.run([sys.executable,
                            os.path.join(repo, "scripts", "stamp_version.py"),
                            version], capture_output=True, text=True,
                           timeout=120)
            subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                            "-r", os.path.join(repo, "requirements.txt")],
                           capture_output=True, text=True, timeout=600)
        except Exception as e:
            return {"ok": False, "source": True,
                    "detail": f"source release update failed: {e}"}

        def _restart_source():
            import time as _t
            _t.sleep(1.5)
            script = os.path.join(repo, "src", "app.py")
            try:
                os.execv(sys.executable, [sys.executable, script])
            except Exception:
                pass
        threading.Thread(target=_restart_source, daemon=True).start()
        return {"ok": True, "source": True, "version": version,
                "restarting": True,
                "detail": f"source updated to release {version}; restarting"}
    info = _app_release_info()
@router.get("/api/app-version")
async def app_version_info():
    return await run_in_threadpool(_app_version_payload)


@router.post("/api/app/update")
async def app_update():
    return await run_in_threadpool(_update_app_binary)


def _agent_version_payload(machine):
    if local_node.is_local(machine):
        return {"current": None, "latest": None, "on_latest": True,
                "local": True, "packaged": False}
    current = None
    build_commit = None
    packaged = False
    try:
        v = agent_client.version(machine, timeout=4) or {}
        current = v.get("agent_version")
        build_commit = v.get("build_commit")
        packaged = bool(v.get("packaged"))
    except Exception:
        current = None
    try:
        info = _agent_release_info(machine)
        latest = info.get("version")
        latest_commit = info.get("commit")
    except Exception:
        latest, latest_commit = None, None
    applied = str(machine.get("agent_release") or "").strip()
    return {"current": current, "latest": latest,
            "on_latest": _component_on_latest(
                build_commit, latest_commit, current, latest, applied),
            "build_commit": build_commit, "latest_commit": latest_commit,
            "packaged": packaged}


def _update_agent_remote(machine):
    """Install/update a standalone peer agent from the latest repo release."""
    name = machine.get("name") or machine.get("id") or "?"
    if local_node.is_local(machine):
        return {"ok": True, "machine": name, "skipped": True, "local": True,
                "detail": "local master uses the app directly; no peer agent"}
    av = _agent_version_payload(machine)
    prev_version = str(av.get("current") or "")
    try:
        info = _agent_release_info(machine)
    except Exception as e:
        return {"ok": False, "machine": name, "frozen": True,
                "detail": f"could not check GitHub releases: {e}"}
    version = info.get("version")
    if not av.get("packaged"):
        # Pip/source-installed agents update through the release wheel.
        wheel_url = info.get("wheel_url")
        if not wheel_url:
            return {"ok": False, "machine": name, "reinstall": True,
                    "detail": "no release agent wheel is published — "
                              "reinstall the peer agent package"}
        try:
            import urllib.request
            with urllib.request.urlopen(wheel_url, timeout=120) as r:
                blob = r.read()
            res = agent_client.agent_update(machine, blob, version or "")
        except Exception as e:
            return {"ok": False, "machine": name,
                    "detail": f"could not install release wheel: {e}"}
        if not res.get("ok"):
            return {"ok": False, "machine": name,
                    "detail": res.get("error") or "agent wheel update failed"}
        for _ in range(20):
            import time as _t
            _t.sleep(1)
            try:
                h = agent_client.health(machine, timeout=3)
                hver = h.get("agent_version")
                if hver and (hver == version or hver != prev_version):
                    if version:
                        try:
                            cfg.machines.update(name,
                                                {"agent_release": version})
                        except Exception:
                            pass
                    return {"ok": True, "machine": name,
                            "agent_version": hver,
                            "agent_release": version,
                            "method": "wheel"}
            except Exception:
                continue
        return {"ok": False, "machine": name,
                "detail": "agent installed the release wheel but did not "
                          "come back healthy"}
    asset = info.get("asset")
    url = info.get("url")
    if not asset:
        return {"ok": False, "machine": name, "frozen": True,
                "detail": "no standalone agent asset is published for this "
                          "platform — reinstall the peer agent package"}
    if not url:
        return {"ok": False, "machine": name, "frozen": True,
                "detail": f"no published standalone agent asset ({asset})"}
    try:
        res = agent_client.agent_update_binary(machine, url, version or "")
    except Exception as e:
        return {"ok": False, "machine": name, "frozen": True,
                "detail": f"could not download {asset}: {e}"}
    if not res.get("ok"):
        return {"ok": False, "machine": name, "frozen": True,
                "detail": res.get("error") or "standalone agent update failed"}
    # Success = healthy /version after the agent replaces and restarts itself.
    # Its internal build version may lag the release tag; record the release
    # we installed so the machine is not considered stale forever.
    for _ in range(20):
        import time as _t
        _t.sleep(1)
        try:
            h = agent_client.health(machine, timeout=3)
            hver = h.get("agent_version")
            if hver and (hver == version or hver != prev_version):
                if version:
                    try:
                        cfg.machines.update(name, {"agent_release": version})
                    except Exception:
                        pass
                return {"ok": True, "machine": name,
                        "agent_version": hver,
                        "agent_release": version,
                        "protocol_version": h.get("protocol_version"),
                        "method": "binary", "asset": asset}
        except Exception:
            continue
    return {"ok": False, "machine": name, "frozen": True,
            "detail": "agent downloaded the new binary but did not come "
                      "back healthy — reinstall the peer agent package"}


@router.get("/api/machines/{name}/agent-version")
async def agent_version_machine(name: str):
    machine = cfg.machines.require(name)
    return await run_in_threadpool(_agent_version_payload, machine)


@router.post("/api/machines/{name}/update-agent")
async def update_agent_machine(name: str):
    """Build current agent, install it through the agent bootstrap, restart it."""
    machine = cfg.machines.require(name)
    try:
        res = await run_in_threadpool(_update_agent_remote, machine)
    except Exception as e:
        raise HTTPException(status_code=502,
                            detail=f"agent update failed: {e}")
    if not res.get("ok"):
        raise HTTPException(status_code=502, detail=res.get("detail") or "agent update failed")
    return res


@router.post("/api/machines/{name}/update-llama-cpp")
async def update_llama_cpp_machine(name: str, force: bool = False):
    """One-click llama.cpp UPDATE for one machine (machine-page button).

    Background job that: (1) stops that machine's llama processes —
    Windows refuses to overwrite locked exes (observed live on winbox),
    and Linux keeps serving the old inode anyway; (2) installs the
    newest prebuilt via llamacpp_installer (auto-detects os / gpu_vendor
    / arch / glibc, uses the stored machine record); (3) resurrects
    llama-swap if this machine hosts it (stop_rpc's legacy stop-all
    kills it). Poll GET /api/llama-cpp-updates/{job_id} for progress."""
    machine = cfg.machines.require(name)
    with _LLAMA_UPDATES_LOCK:
        for j in _LLAMA_UPDATES.values():
            if j["machine"] == name and j["state"] in ("stopping", "installing", "updating"):
                return JSONResponse(
                    status_code=409,
                    content={"detail": f"update already running on {name}",
                             "job_id": j["id"],
                             "status_url": f"/api/llama-cpp-updates/{j['id']}"})
        job_id = secrets.token_hex(6)
        job = {"id": job_id, "machine": name, "state": "stopping",
               "log": [], "result": None,
               "started": time.time(), "finished": None}
        _LLAMA_UPDATES[job_id] = job

    def _run_update():
        import llamacpp_installer
        import rpc_manager
        import swap_manager
        import urllib.request
        was_swap_up = False
        try:
            # Install/update the peer agent first when it is older than the
            # app's wheel. Older agents cannot verify RPC and (critically)
            # drop .so symlinks when extracting prebuilt archives.
            if not local_node.is_local(machine):
                try:
                    av = _agent_version_payload(machine)
                    if (av.get("latest") and av.get("current")
                            and not av.get("on_latest")):
                        job["log"].append(
                            f"updating peer agent {av.get('current')} → "
                            f"{av.get('latest')} before engine install")
                        up = _update_agent_remote(machine)
                        if not up.get("ok"):
                            if up.get("reinstall"):
                                job["log"].append(
                                    "peer agent is not a standalone release "
                                    "build; continuing with the installed "
                                    "agent: " + str(up.get("detail") or ""))
                            else:
                                job["result"] = {
                                    "ok": False, "machine": name,
                                    "detail": "agent update failed before "
                                              "engine install: "
                                              + str(up.get("detail")
                                                    or "unknown")}
                                job["state"] = "failed"
                                return
                        else:
                            job["log"].append(
                                f"peer agent updated to "
                                f"{up.get('agent_version')}")
                except Exception as e:
                    job["log"].append(
                        f"agent pre-update check failed (continuing): {e}")
            # Already current? Skip the whole update (probe the running
            # binary's build vs the newest release tag) — a no-op
            # rebuild still costs a fetch + cmake pass, and the UI
            # shows 'on latest' for these machines anyway.
            vp = _llama_version_payload(machine)
            if force:
                job["log"].append(
                    f"{name}: forced clean reinstall of llama.cpp + RPC "
                    "(existing binaries will be overwritten)")
                try:
                    if name == (swap_manager.get_host() or {}).get("name"):
                        port = int(swap_manager.load_settings().get("listen_port") or 8080)
                        with urllib.request.urlopen(
                                f"http://127.0.0.1:{port}/health", timeout=3) as r:
                            was_swap_up = (r.status == 200)
                        job["log"].append(
                            f"llama-swap on {name}: "
                            + ("running — will restart after the install"
                               if was_swap_up else "not running"))
                except Exception:
                    was_swap_up = False
                job["log"].append(f"stopping llama processes on {name} …")
                try:
                    rep = rpc_manager.stop_rpc(name)
                    job["log"].append("stop_rpc: " + json.dumps(rep)[:400])
                except Exception as e:
                    job["log"].append(f"stop_rpc failed (continuing anyway): {e}")
                pref = (machine.get("engine_mode") or "auto").strip().lower()
                probe = None
                if pref != "prebuilt":
                    probe = llamacpp_installer.probe_setup(
                        machine, log=lambda m: job["log"].append(str(m)))
                if pref == "source" or (probe and probe.get("kind") == "source"):
                    if not probe or probe.get("kind") != "source":
                        job["result"] = {
                            "ok": False, "machine": name,
                            "detail": "engine_mode=source but no source "
                                      "build was detected (install one "
                                      "first, or set engine_mode=auto/"
                                      "prebuilt)"}
                        job["state"] = "failed"
                        return
                    job["state"] = "updating"
                    res = llamacpp_installer.update_source_on(
                        machine, probe,
                        log=lambda m: job["log"].append(str(m)))
                else:
                    job["state"] = "installing"
                    res = llamacpp_installer.install_on(
                        machine, log=lambda m: job["log"].append(str(m)))
                job["result"] = res
                job["state"] = "done" if res.get("ok") else "failed"
            else:
                # Fully dynamic routing: probe the machine itself (what
                # binary dir exists, is a git source tree + CMakeCache
                # behind it, which flags that cache carries) and pick the
                # path from the ANSWER — never from path-string guessing.
                # Probe FIRST: a machine can have an installer-managed
                # prebuilt in <root>/bin and a source build in
                # <root>/build/bin; the prebuilt may be current while the
                # source build (the one the agent actually launches) lags.
                job["state"] = "updating"
                probe = llamacpp_installer.probe_setup(
                    machine, log=lambda m: job["log"].append(str(m)))
                kind = probe.get("kind", "prebuilt")
                pref = (machine.get("engine_mode") or "auto").strip().lower()
                force_prebuilt = pref == "prebuilt"
                if pref == "source" and kind != "source":
                    job["result"] = {
                        "ok": False, "machine": name,
                        "detail": "engine_mode=source but no source build "
                                  "was detected (install one first, or set "
                                  "engine_mode=auto/prebuilt)"}
                    job["state"] = "failed"
                    return
                if force_prebuilt and kind == "source":
                    job["log"].append(
                        f"{name}: engine_mode=prebuilt → switching to the "
                        "prebuilt binary (source tree is kept)")
                    kind = "prebuilt"
                src_build = _llama_build_no(probe.get("version") or "")
                latest_build = vp.get("latest_build")
                source_current = bool(src_build and latest_build
                                      and src_build >= int(latest_build))
                if (kind != "source" and vp.get("on_latest")
                        and not force_prebuilt):
                    job["log"].append(
                        f"{name} already on the latest release "
                        f"({vp.get('latest')}) — nothing to do")
                    job["result"] = {"ok": True, "machine": name,
                                     "detail": f"already on the latest "
                                               f"release ({vp.get('latest')})"}
                    job["state"] = "done"
                    return
                if kind == "source" and source_current:
                    job["log"].append(
                        f"{name}: source build already on the latest "
                        f"release ({probe.get('version') or vp.get('latest')})"
                        " — nothing to do")
                    job["result"] = {"ok": True, "machine": name,
                                     "detail": "source build already current"}
                    job["state"] = "done"
                    return
                if kind == "source":
                    fl = ", ".join(f"{k}={v}" for k, v in
                                   sorted((probe.get("flags") or {}).items())[:6])
                    job["log"].append(
                        f"{name}: source build detected"
                        + (f" ({fl})" if fl else "")
                        + " → in-place rebuild, custom flags preserved")
                    # Rebuilding replaces the server binary the agent
                    # launches. Stop this machine's farm processes first
                    # (same rule as the prebuilt path) so no model starts
                    # while the target is being linked.
                    try:
                        if name == (swap_manager.get_host() or {}).get("name"):
                            port = int(swap_manager.load_settings().get("listen_port") or 8080)
                            with urllib.request.urlopen(
                                    f"http://127.0.0.1:{port}/health",
                                    timeout=3) as r:
                                was_swap_up = (r.status == 200)
                            job["log"].append(
                                f"llama-swap on {name}: "
                                + ("running — will restart after the rebuild"
                                   if was_swap_up else "not running"))
                    except Exception:
                        was_swap_up = False
                    job["log"].append(f"stopping llama processes on {name} …")
                    try:
                        rep = rpc_manager.stop_rpc(name)
                        job["log"].append("stop_rpc: "
                                          + json.dumps(rep)[:400])
                    except Exception as e:
                        job["log"].append(
                            f"stop_rpc failed (continuing anyway): {e}")
                    res = llamacpp_installer.update_source_on(
                        machine, probe, log=lambda m: job["log"].append(str(m)))
                else:
                    if kind == "missing":
                        job["log"].append(
                            f"{name}: no usable llama.cpp found → fresh "
                            "prebuilt install (os/vendor/arch/glibc-picked)")
                    else:
                        job["log"].append(
                            f"{name}: prebuilt install → stopping llama "
                            "processes first (Windows locks running exes)")
                        try:
                            if name == (swap_manager.get_host() or {}).get("name"):
                                port = int(swap_manager.load_settings().get("listen_port") or 8080)
                                with urllib.request.urlopen(
                                        f"http://127.0.0.1:{port}/health", timeout=3) as r:
                                    was_swap_up = (r.status == 200)
                                job["log"].append(
                                    f"llama-swap on {name}: "
                                    + ("running — will restart after the update"
                                       if was_swap_up else "not running"))
                        except Exception:
                            was_swap_up = False
                        job["log"].append(f"stopping llama processes on {name} …")
                        try:
                            rep = rpc_manager.stop_rpc(name)
                            job["log"].append("stop_rpc: " + json.dumps(rep)[:400])
                        except Exception as e:
                            job["log"].append(f"stop_rpc failed (continuing anyway): {e}")
                    job["state"] = "installing"
                    res = llamacpp_installer.install_on(
                        machine, log=lambda m: job["log"].append(str(m)))
                    if (not res.get("ok") and force_prebuilt
                            and probe.get("kind") == "source"):
                        job["log"].append(
                            "prebuilt is not viable on this host → falling "
                            "back to the source rebuild: "
                            + str(res.get("detail") or ""))
                        job["state"] = "updating"
                        res = llamacpp_installer.update_source_on(
                            machine, probe,
                            log=lambda m: job["log"].append(str(m)))
                job["result"] = res
                job["state"] = "done" if res.get("ok") else "failed"
        except Exception as e:
            job["state"] = "failed"
            job["result"] = {"ok": False, "detail": f"update crashed: {e}"}
        finally:
            if was_swap_up:
                try:
                    job["log"].append("restarting llama-swap on the swap host …")
                    rep = swap_manager.launch(None, None, None, None)
                    job["log"].append("llama-swap relaunch: "
                                      + json.dumps(rep)[:400])
                except Exception as e:
                    job["log"].append(
                        f"llama-swap relaunch failed: {e} "
                        "(restart it from the app)")
            # successful update means the version probe changed — drop
            # the cached one so the grid immediately shows 'on latest'
            invalidate_llama_version(name)
            job["finished"] = time.time()

    threading.Thread(target=_run_update, daemon=True).start()
    return {"job_id": job_id, "status_url": f"/api/llama-cpp-updates/{job_id}"}


@router.get("/api/llama-cpp-updates/{job_id}")
async def llama_cpp_update_status(job_id: str):
    """Poll a running/finished per-machine llama.cpp update job."""
    with _LLAMA_UPDATES_LOCK:
        job = _LLAMA_UPDATES.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="unknown update job id")
    out = dict(job)
    out["log"] = job["log"][-60:]
    return out


@router.post("/api/install-llama-swap")
async def install_llama_swap(name: str = ""):
    """Install the official llama-swap binary on a machine if not already
    present (local machines run locally — no shell transport), then point swap
    settings' binary at it. Mirrors the llama.cpp installer."""
    if not name:
        raise HTTPException(status_code=400, detail="machine name required")
    machine = cfg.machines.require(name)
    import swap_installer
    found, bin_path = await run_in_threadpool(swap_installer.detect_on, machine)
    if found:
        return {"ok": True, "machine": name, "bin": bin_path,
                "detail": f"already installed at {bin_path} (skipped)"}
    result = await run_in_threadpool(swap_installer.install_on, machine)
    return JSONResponse(status_code=200 if result.get("ok") else 502,
                        content=result)


# ---------------------------------------------------------------------------
# Unified installer endpoints (UX-1): probe/plan/execute layer in app
# ---------------------------------------------------------------------------
# Wraps src/installer/{probe,execute}.py. Remote machines run the probe
# ladder + executor; remote probing is agent-only (local RunCmd here).
# NO GPU drivers are ever installed (executor refuses);
# llama.cpp serving behavior is untouched (this only adds new endpoints).
