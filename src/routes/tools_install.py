"""Unified tools installer endpoints (probe/plan/execute + uninstall)."""
import json
import os

import pydantic
from fastapi import APIRouter, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse, StreamingResponse

import config_loader as cfg
from installer.env import RunResult
import local_node
import settings_store
from runtime_paths import data_dir

BASE_DIR = data_dir()

router = APIRouter()


class ToolsInstallIn(pydantic.BaseModel):
    machine: str
    tools: list[str] = []          # empty = all tools the plan offers
    role: str = "main"             # 'main' | 'client'
    dry_run: bool = False


def _machine_env(machine):
    """(SystemInfo, RunCmd) for a machine record: local machines probe
    locally; remote machines run the same read-only probe ladder over
    the agent. OS/gpu_vendor come from the machine record (config truth), the
    rest from the remote probes."""
    from installer.env import SystemInfo, RealFS, RunResult
    import gpu_vendor as _gv
    import local_node as _ln

    os_type = (machine.get("os") or _ln.local_os() or "linux").strip().lower()
    if os_type == "darwin":
        os_type = "macos"
    vendor = _gv.normalize_vendor(machine.get("gpu_vendor"))
    is_local = _ln.is_local(machine)

    if is_local:
        from installer.env import RealRunCmd as _RRC
        run = _RRC()
    else:
        raise HTTPException(
            status_code=400,
            detail="remote installer probes are agent-only; install via the "
                   "peer agent's /install/* endpoints or run the installer on "
                   "that machine")

    # detect the rest of SystemInfo with the shared probe ladder (cheap,
    # read-only; same code path bootstrap.py uses locally)
    from installer.env import detect_system_info
    info = detect_system_info(run)
    # record facts WIN over probed facts where the record is authoritative:
    # the machine card's os/gpu_vendor is what the operator declared.
    info.os = os_type
    info.gpu_vendor = vendor or info.gpu_vendor
    return info, run


def _resolve_install_machine(name):
    """Settings-modal machine id: '__local__'/'' = this host, else a name."""
    if name in ("", "__local__"):
        for m in cfg.machines.all():
            if local_node.is_local(m):
                return m
        raise HTTPException(status_code=400,
                            detail="no local machine configured")
    try:
        return cfg.machines.require(name)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e)) from None


def _install_remote_tools(machine, role, tools, dry_run, log):
    """Agent-backed install for a peer (the Settings modal's peer path).

    The peer agent already implements /install/artifact + /install/verify;
    this just routes the app's request there instead of trying to probe the
    remote host locally.
    """
    import llamacpp_installer
    import swap_installer

    name = machine.get("name") or "?"
    want = set(tools or [])
    if not want:
        want = ({"llama.cpp", "llama-swap"} if role == "main"
                else {"llama.cpp"})
    summary = {"installed": [], "skipped": [], "repaired": [],
               "settings": [], "notes": [], "failures": [],
               "manifest": {"tools": {}}}
    if dry_run:
        planned = []
        if "llama.cpp" in want:
            planned.append("llama.cpp")
        if role == "main" and "llama-swap" in want:
            planned.append("llama-swap")
        for tool in planned:
            log(f"[dry-run] would install {tool} on {name} through the "
                f"peer agent")
        summary["planned"] = planned
        summary["notes"].append("dry run — nothing executed")
        return summary

    def _finish(tool, res):
        for line in str(res.get("detail") or "").splitlines():
            if line.strip():
                log(line.strip())
        if res.get("ok"):
            summary["installed"].append(tool)
            summary["manifest"]["tools"][tool] = {
                "version": res.get("version") or "",
                "method": "agent",
            }
            summary["notes"].append(f"{tool}: {res.get('detail') or 'ok'}")
        else:
            summary["failures"].append({
                "error": res.get("detail") or f"{tool} install failed",
                "context": {"tool": tool},
            })

    if "llama.cpp" in want:
        log(f"installing llama.cpp + llama RPC on {name} via the peer agent")
        _finish("llama.cpp", llamacpp_installer.install_on(machine, log=log))
    if role == "main" and "llama-swap" in want:
        log(f"installing llama-swap on {name} via the peer agent")
        _finish("llama-swap", swap_installer.install_on(machine, log=log))
    return summary


def _tools_plan(machine, role, log=lambda m: None):
    """Build the SAME dynamic install plan the bootstrap uses: newest
    llama.cpp/llama-swap releases with matching assets, plus source-build
    detection, so the UI installer and bootstrap cannot drift."""
    from installer.env import RealFS
    from installer.probe import probe as installer_probe
    import bootstrap

    info, run = _machine_env(machine)
    fs = RealFS()
    resolved = bootstrap._resolve_latest(info, role, log)
    source_probe = bootstrap._detect_source_build(
        info, fs, log, settings_file=settings_store.SETTINGS_FILE)
    plan = installer_probe(info, fs, run, role=role, resolved=resolved,
                           source_probe=source_probe,
                           settings_root=BASE_DIR)
    return plan, info, run, fs


class _ToolsProgress:
    """Per-step install progress collector: Executor.log lines are
    appended as NDJSON events; the StreamingResponse consumer reads
    them as they land (same per-machine progress pattern as
    /api/stats/stream — per-EVENT, not per-machine)."""

    def __init__(self):
        import queue
        self.q = queue.Queue()
        self._done = object()

    def log(self, msg):
        self.q.put(("log", msg))

    def finish(self, summary):
        self.q.put(("done", summary))

    def stream(self):
        import queue
        while True:
            try:
                kind, payload = self.q.get(timeout=300)
            except queue.Empty:
                yield json.dumps({"event": "timeout"}) + "\n"
                return
            if kind == "log":
                yield json.dumps({"event": "log", "message": payload}) + "\n"
            else:
                yield json.dumps({"event": "done",
                                  "summary": payload}) + "\n"
                return


@router.post("/api/tools/install")
async def tools_install(p: ToolsInstallIn):
    """Unified install: plan for ONE machine (installer.probe), execute
    it (installer.execute), streaming per-step progress as NDJSON
    ({event: log|done}). The plan carries per-tool skip/refuse notes;
    llama.cpp behavior is untouched (this endpoint only ADDS surfaces)."""
    m = _resolve_install_machine(p.machine)
    role = p.role if p.role in ("main", "client") else "main"
    progress = _ToolsProgress()

    def run_install():
        try:
            if not local_node.is_local(m):
                summary = _install_remote_tools(
                    m, role, p.tools, p.dry_run, progress.log)
                progress.finish(summary)
                return
            from installer.execute import Executor

            plan, info, run, fs = _tools_plan(m, role, progress.log)
            if p.tools:
                keep = set(p.tools)
                # keep only the requested tools' steps; settings/manifest
                # steps for OTHER tools are dropped with their tool
                from installer.probe import (InstallTool, RepairTool,
                                             SkipTool, PlanNote)
                plan = [s for s in plan
                        if getattr(s, "name", None) in keep
                        or (type(s).__name__ not in
                            ("InstallTool", "RepairTool", "SkipTool",
                             "PlanNote"))]
            ex = Executor(log=progress.log, dry_run=p.dry_run,
                          info=info,
                          manifest_path=os.path.join(
                              os.path.expanduser("~"),
                              ".llama-rpc-farm", "installer-state.yaml"),
                          runner=lambda cmd, timeout=1800:
                              run.run(cmd, timeout=timeout))
            summary = ex.execute(plan)
            ex.write_manifest(summary, detected={})
            progress.finish(summary)
        except Exception as e:                          # noqa: BLE001
            from installer.execute import failure_envelope
            progress.finish({"failures": [failure_envelope("installer", e)],
                             "installed": [], "skipped": [],
                             "repaired": [], "settings": [], "notes": [],
                             "manifest": {}})

    import threading
    t = threading.Thread(target=run_install, daemon=True)
    t.start()
    return StreamingResponse(_tools_ndjson(progress),
                             media_type="application/x-ndjson")


async def _tools_ndjson(progress):
    """Drain the progress queue into NDJSON lines (async bridge so the
    event loop stays free while the install thread works)."""
    import queue as _q
    while True:
        try:
            kind, payload = await run_in_threadpool(
                lambda: progress.q.get(timeout=300))
        except _q.Empty:
            yield json.dumps({"event": "timeout"}) + "\n"
            return
        if kind == "log":
            yield json.dumps({"event": "log", "message": payload}) + "\n"
        else:
            yield json.dumps({"event": "done", "summary": payload}) + "\n"
            return


class _RemoteFS:
    """FS adapter for remote machines: existence/version checks through the agent.
    Only what probe.py touches."""

    def __init__(self, run, is_win=False):
        self._run = run
        self._win = is_win

    def exists(self, path):
        return self.isfile(path)

    def isfile(self, path):
        p = self.expanduser(path)
        if self._win:
            cmd = ('powershell -NoProfile -Command '
                   '"if (Test-Path \\"' + p + '\\") { echo Y } '
                   'else { exit 1 }"')
        else:
            cmd = '[ -f "' + p + '" ] && echo Y || exit 1'
        r = self._run.run(cmd)
        return r.ok

    def expanduser(self, path):
        # expand ~ on the REMOTE side so the path is correct there
        if not path.startswith("~"):
            return path
        r = self._run.run("echo " + path)
        out = (r.stdout or "").strip()
        return out if out.startswith(("/", "C:", "%")) else path

    def read_file(self, path):
        r = self._run.run('cat "' + self.expanduser(path) + '"')
        if not r.ok:
            raise FileNotFoundError(path)
        return r.stdout
    def file_exists_and_runs(self, binary, version_arg="--version"):
        p = self.expanduser(binary)
        cmd = (f'if [ -f "{p}" ]; then "{p}" {version_arg} 2>&1; '
               f'else echo __MISSING__; exit 127; fi')
        r = self._run.run(cmd)
        if "__MISSING__" in (r.stdout or ""):
            return RunResult(f"{binary} {version_arg}", 127, "", "missing")
        return RunResult(f"{binary} {version_arg}", r.exit_code,
                         r.stdout or "", r.stderr or "")


@router.post("/api/tools/install/sync")
async def tools_install_sync(p: ToolsInstallIn):
    """Non-streaming variant of /api/tools/install: same plan+execute,
    but the full summary is returned in one body (for tests + simple
    clients). Same engine gates, same refusal semantics."""
    m = _resolve_install_machine(p.machine)
    role = p.role if p.role in ("main", "client") else "main"
    if not local_node.is_local(m):
        summary = await run_in_threadpool(
            _install_remote_tools, m, role, p.tools, p.dry_run,
            lambda msg: print(msg, flush=True))
    else:
        plan, info, run, fs = await run_in_threadpool(
            _tools_plan, m, role, lambda msg: print(msg, flush=True))
        summary = await run_in_threadpool(
            _execute_plan, plan, info, p.dry_run, run)
    ok = not summary.get("failures")
    return JSONResponse(status_code=200 if ok else 502, content=summary)


def _execute_plan(plan, info, dry_run, runner=None):
    from installer.execute import Executor
    runner_fn = None
    if runner is not None:
        runner_fn = lambda cmd, timeout=1800: runner.run(cmd, timeout=timeout)
    ex = Executor(log=lambda m: None, dry_run=dry_run, info=info,
                  manifest_path=os.path.join(
                      os.path.expanduser("~"),
                      ".llama-rpc-farm", "installer-state.yaml"),
                  runner=runner_fn)
    summary = ex.execute(plan)
    ex.write_manifest(summary, detected={})
    return summary


@router.post("/api/machines/{name}/uninstall-llamacpp")
async def uninstall_llamacpp_machine(name: str = ""):
    """Uninstall llama.cpp from ONE machine via llamacpp_uninstaller
    (removes ONLY the farm-managed install dir the installer created —
    path-guarded, never a source build, never anything outside the
    guarded root) and clears the machine's llama_bin pointer.
    Returns the run_steps report (ok=true -> 200, failed -> 502).
    DEC5 (T5-1)."""
    if not name:
        raise HTTPException(status_code=400, detail="machine name required")
    try:
        machine = cfg.machines.require(name)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    import llamacpp_uninstaller
    result = await run_in_threadpool(llamacpp_uninstaller.uninstall_on,
                                     machine)
    # Passthrough: the caller gets the full run_steps report. A refusal
    # (foreign llama_bin path) is a FAILED uninstall, not a server
    # error: the report rides back with a 502 so the UI can show the
    # uninstaller's own message.
    return JSONResponse(status_code=200 if result.get("ok") else 502,
                        content=result)
