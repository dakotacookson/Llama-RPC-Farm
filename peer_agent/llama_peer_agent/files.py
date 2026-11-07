"""Bounded file operations for the agent.

Only paths under configured roots (home + model roots + file roots) are
reachable. No shell, no glob, no symlink escape.
"""
import os
import shutil
import tempfile
import time
from pathlib import Path

from .config import AgentConfig


class AgentFileError(RuntimeError):
    pass


def _roots(config: AgentConfig) -> list[Path]:
    roots = [Path.home()] + [Path(p).expanduser() for p in config.model_roots]
    roots += [Path(p).expanduser() for p in config.file_roots]
    out = []
    for root in roots:
        try:
            out.append(root.resolve(strict=False))
        except OSError:
            continue
    return out


def _path(config: AgentConfig, raw, write=False) -> Path:
    if not isinstance(raw, str) or not raw.strip():
        raise AgentFileError("path is required")
    if "\x00" in raw:
        raise AgentFileError("invalid path")
    expanded = (raw.replace("%USERPROFILE%", str(Path.home()))
                .replace("$env:USERPROFILE", str(Path.home())))
    p = Path(os.path.expandvars(expanded)).expanduser()
    if not p.is_absolute():
        p = Path.home() / p
    try:
        p = p.resolve(strict=False)
    except OSError as e:
        raise AgentFileError(f"cannot resolve path: {e}") from e
    for root in _roots(config):
        if p == root or root in p.parents:
            return p
    raise AgentFileError("path outside allowed roots")


def _backup_name(path: Path, stamp=None) -> Path:
    stamp = stamp or time.strftime("%Y%m%d-%H%M%S")
    return path.with_name(path.name + f".bak-{stamp}")


def write(config: AgentConfig, path, text, mode=None, backup=False,
          create_parents=False) -> dict:
    p = _path(config, path, write=True)
    if not isinstance(text, str):
        raise AgentFileError("text must be a string")
    if create_parents:
        p.parent.mkdir(parents=True, exist_ok=True)
    if not p.parent.is_dir():
        raise AgentFileError(f"parent directory does not exist: {p.parent}")
    backup_path = None
    if backup and p.is_file():
        backup_path = _backup_name(p)
        shutil.copy2(p, backup_path)
    fd, tmp = tempfile.mkstemp(dir=str(p.parent), prefix=".tmp-agent-")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
        if mode is not None:
            os.chmod(tmp, int(mode))
        os.replace(tmp, p)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    if mode is not None:
        try:
            os.chmod(p, int(mode))
        except OSError:
            pass
    return {"path": str(p), "bytes": len(text.encode("utf-8")),
            "backup": str(backup_path) if backup_path else None}


def mkdir(config: AgentConfig, path) -> dict:
    p = _path(config, path, write=True)
    p.mkdir(parents=True, exist_ok=True)
    return {"path": str(p), "is_dir": p.is_dir()}


def chmod(config: AgentConfig, path, mode) -> dict:
    p = _path(config, path, write=True)
    if not p.exists():
        raise AgentFileError("path does not exist")
    os.chmod(p, int(mode))
    return {"path": str(p), "mode": int(mode)}


def backup(config: AgentConfig, path, stamp=None) -> dict:
    p = _path(config, path)
    if not p.is_file():
        return {"path": str(p), "backed_up": False,
                "note": "no existing file"}
    dest = _backup_name(p, stamp)
    shutil.copy2(p, dest)
    return {"path": str(p), "backup": str(dest), "backed_up": True}


def stat(config: AgentConfig, path) -> dict:
    p = _path(config, path)
    try:
        st = p.stat()
    except OSError:
        return {"path": str(p), "exists": False, "is_file": False,
                "is_dir": False}
    return {"path": str(p), "exists": True, "is_file": p.is_file(),
            "is_dir": p.is_dir(), "size": st.st_size,
            "mtime": st.st_mtime}


def list_dir(config: AgentConfig, path, hidden=False) -> dict:
    p = _path(config, path)
    if not p.is_dir():
        raise AgentFileError("not a directory")
    rows = []
    for entry in sorted(p.iterdir(), key=lambda e: e.name.lower()):
        if not hidden and entry.name.startswith("."):
            continue
        try:
            rows.append({"name": entry.name, "is_dir": entry.is_dir(),
                         "is_file": entry.is_file()})
        except OSError:
            continue
    return {"path": str(p), "entries": rows}
