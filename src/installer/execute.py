# execute.py — dumb runner consuming plan steps (design §10). NO decisions.
#
# Each step type has exactly one handler. If a new decision is needed, it
# belongs in probe.py emitting a new/changed step — not here. The runner only
# knows HOW to perform a step, never WHETHER it should.
#
# Real implementations live in tools.py (asset maps + per-tool executors);
# this module owns the runner plumbing: the HTTPS downloader (prod), the
# manifest write (installer-state.yaml at end of EVERY run, incl. skip-only),
# and the settings persister (comment-preserving, ruamel round-trip,
# atomic tmp+replace).

from __future__ import annotations

import datetime
import os
import subprocess
import sys
import urllib.request

from .probe import (InstallTool, ManifestUpdate, PlanNote, RepairTool,
                    SkipTool, WriteSettings)
from . import tools as tool_exec

DEFAULT_MANIFEST_PATH = "~/.llama-rpc-farm/installer-state.yaml"


def _set_dotted(data, key, value):
    """Set a (possibly dotted) key in a mutable yaml mapping."""
    parts = key.split(".")
    cur = data
    for part in parts[:-1]:
        nxt = cur.get(part)
        if not isinstance(nxt, dict):
            nxt = {}
            cur[part] = nxt
        cur = nxt
    cur[parts[-1]] = value


def _redact(text):
    """Strip obvious secrets from a failure envelope (req #4).

    Two shapes: KEY=value / KEY: value (the value is the secret — drop it
    entirely) and `Authorization: Bearer <tok>` header lines (drop the
    token, keep the scheme word). The old pattern redacted the key but
    LEFT the secret value in the envelope — a real leak.
    """
    import re
    # KEY=secret / KEY: secret  →  KEY=<redacted>
    text = re.sub(
        r"((?:ssh_)?pass\w*|hf_token|password)\s*[=:]\s*\S+",
        r"\1=<redacted>", text, flags=re.I)
    # Authorization: Bearer <tok> (or any auth header value)
    text = re.sub(
        r"(Authorization\s*:\s*)(Bearer\s+)?\S+",
        r"\1<redacted>", text, flags=re.I)
    return text


def failure_envelope(source, error, context=None, tail_lines=50):
    """Machine-readable failure bundle (req #4 #4: ONE format for installs and
    launches). Pure function — returns the dict; caller persists/prints it."""
    return {
        "schema": "llama-rpc-farm-failure-envelope/1",
        "source": source,                      # 'installer' | 'launch' | ...
        "error": _redact(str(error)),
        "context": context or {},
        "log_tail": _redact((context or {}).get("log_tail", "")),
        "captured_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }


def _read_current_setting(path, key):
    """Read a (possibly dotted) key from a settings yaml file (real fs).
    Used by the WriteSettings handler to re-check §6 merge rules at write
    time; returns None when the file/key is missing or unparseable."""
    path = os.path.expanduser(path)
    if not os.path.isfile(path):
        return None
    try:
        import yaml
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except Exception:                                  # noqa: BLE001
        return None
    cur = data
    for part in key.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur if isinstance(cur, str) else None


class RealDownloader:
    """prod downloader: URL -> temp file (streamed, with retries).
    Tests inject a stub instead — no network in the hermetic suite."""

    def __init__(self, retries=3):
        self.retries = retries

    def __call__(self, url, dest):
        import urllib.request
        last = None
        for attempt in range(1, self.retries + 1):
            try:
                req = urllib.request.Request(
                    url, headers={"User-Agent": "llama-rpc-farm-installer"})
                with urllib.request.urlopen(req, timeout=120) as r, \
                        open(dest, "wb") as f:
                    while True:
                        chunk = r.read(1 << 16)
                        if not chunk:
                            break
                        f.write(chunk)
                return
            except Exception as e:                    # noqa: BLE001
                last = e
        raise RuntimeError(f"download failed after {self.retries} tries: "
                           f"{url}: {last}")


def _run(cmd, log):
    """Shell-out wrapper used by the tool executors (POSIX bash -c; the
    installer entry scripts own the Windows path)."""
    import shlex as _shlex
    p = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True,
                       timeout=1800)
    out = (p.stdout or "") + (p.stderr or "")
    if p.returncode != 0:
        log(f"cmd exit {p.returncode}: {cmd[:120]} :: {out[-200:]}")
    from .env import RunResult
    return RunResult(str(cmd), p.returncode, p.stdout or "", p.stderr or "")


def _stub_info_for(step):
    """SystemInfo fallback when the Executor was built without one (tests)."""
    from .env import SystemInfo
    return SystemInfo(os="linux", arch="x86_64", gpu_vendor="nvidia")


class Executor:
    """Applies a plan to the real (or test-faked) environment."""

    def __init__(self, log=print, dry_run=False, downloader=None,
                 offline=False, info=None, manifest_path=None, runner=None):
        self.log = log
        self.dry_run = dry_run
        self.downloader = downloader or RealDownloader()
        self.offline = offline          # --offline wheelhouse mode (§1)
        self.info = info                # SystemInfo for asset dispatch
        self.manifest_path = manifest_path or DEFAULT_MANIFEST_PATH
        # command runner: prod shells out (_run); tests inject a recording
        # stub so the hermetic suite never executes real commands
        self.runner = runner or (lambda cmd: _run(cmd, self.log))

    # ---- public -----------------------------------------------------------
    def execute(self, plan) -> dict:
        summary = {"installed": [], "skipped": [], "repaired": [],
                   "settings": [], "notes": [], "manifest": {},
                   "failures": []}
        # A plan may raise mid-iteration (probe/plan bug): the steps applied
        # so far stay in the summary and the failure is recorded like any
        # other — the manifest write (bootstrap) still runs afterwards.
        it = iter(plan)
        while True:
            try:
                step = next(it)
            except StopIteration:
                break
            except Exception as e:
                env = failure_envelope(
                    "installer", e, context={"step": "<plan iteration>",
                                             "tool": None})
                summary["failures"].append(env)
                self.log("FAIL " + env["error"])
                break
            try:
                self._apply(step, summary)
            except Exception as e:                      # req #4: AI-fixable bundle
                env = failure_envelope(
                    "installer", e,
                    context={"step": repr(step),
                             "tool": getattr(step, "name", None)})
                summary["failures"].append(env)
                self.log("FAIL " + env["error"])
        return summary

    def write_manifest(self, summary, detected=None, path=None):
        """installer-state.yaml at the END of every run, incl. skip-only
        (design §5: it records the *verified* state). Advisory-only: probe
        wins over manifest on any disagreement."""
        manifest = summary.get("manifest") or {}
        # summary["manifest"] already carries a "tools" sub-dict; the on-disk
        # doc nests tools directly (installer_version + tools).
        manifest = manifest.get("tools", manifest) if isinstance(manifest.get("tools"), dict) else manifest
        import yaml
        doc = {"installer_version": 1, "tools": manifest}
        if detected:
            doc["detected"] = detected
        target = path or self.manifest_path
        target = os.path.expanduser(target)
        os.makedirs(os.path.dirname(target), exist_ok=True)
        tmp = target + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            yaml.safe_dump(doc, f, default_flow_style=False, sort_keys=False)
        os.replace(tmp, target)
        summary["manifest_path"] = target
        return target

    def _apply(self, step, summary):
        if isinstance(step, SkipTool):
            self.log(f"SKIP {step.name} {step.version or ''} "
                     f"({step.reason}) @ {step.found_path or '-'}")
            summary["skipped"].append(step.name)
        elif isinstance(step, (InstallTool, RepairTool)):
            kind = "repair" if isinstance(step, RepairTool) else "install"
            self._install_tool(step, kind)
            (summary["repaired"] if kind == "repair"
             else summary["installed"]).append(step.name)
        elif isinstance(step, WriteSettings):
            self._write_setting(step)
            summary["settings"].append((step.file, step.key, step.value,
                                        step.merge_reason))
        elif isinstance(step, ManifestUpdate):
            self._manifest_update(step, summary["manifest"])
        elif isinstance(step, PlanNote):
            self.log(f"NOTE[{step.kind}] {step.tool or '-'}: {step.message}")
            summary["notes"].append((step.kind, step.tool, step.message))

    # ---- handlers ----------------------------------------------------------
    def _install_tool(self, step, kind):
        if self.dry_run:
            self.log(f"[dry-run] {kind} {step.name} {step.version} "
                     f"({step.backend or step.source}) -> {step.dest}")
            return
        info = self.info or _stub_info_for(step)
        entry = tool_exec.install_tool(
            step, self.downloader, self.runner, self.log,
            info, offline=self.offline)
        return entry

    def _write_setting(self, step: WriteSettings):
        if self.dry_run:
            self.log(f"[dry-run] settings {step.file}:{step.key} = {step.value} "
                     f"({step.merge_reason})")
            return
        # §6 merge rules enforced here too (idempotency): the probe already
        # decided, but a concurrent app write between plan and execute could
        # have landed a valid value — never clobber it.
        if step.merge_reason != "force":
            cur = _read_current_setting(step.file, step.key)
            if cur and os.path.exists(os.path.expanduser(cur)) and cur != step.value:
                self.log(f"settings already points at {cur} — kept "
                         f"({step.file}:{step.key})")
                return
        self._persist_setting(step.file, step.key, step.value)

    @staticmethod
    def _persist_setting(path, key, value):
        """Comment-preserving write via the app's shared persister
        (src/settings_persister.py — the SAME code path vllm_installer/
        vllm_uninstaller use). Never raw-dump YAML (would strip comments).
        Fresh installs may target a not-yet-existing file — the parent
        dir is created; the persister loads a missing file as {}."""
        repo_src = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        repo_src = os.path.dirname(repo_src)          # .../src
        if repo_src not in sys.path:
            sys.path.insert(0, repo_src)
        path = os.path.expanduser(path)
        if os.path.dirname(path):
            os.makedirs(os.path.dirname(path), exist_ok=True)
        try:
            import settings_persister
        except ImportError:                            # pragma: no cover
            raise
        data = settings_persister.load_yaml_preserving(
            os.path.expanduser(path))
        _set_dotted(data, key, value)
        settings_persister.dump_yaml_preserving(
            data, os.path.expanduser(path))

    @staticmethod
    def _manifest_update(step: ManifestUpdate, manifest):
        entry = dict(step.entry)
        entry["installed_at"] = datetime.datetime.now(
            datetime.timezone.utc).isoformat()
        manifest.setdefault("tools", {})[step.tool] = entry