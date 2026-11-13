"""Typed prebuilt artifact install + verification."""
import os
import shutil
import stat
import subprocess
import tarfile
import tempfile
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

from . import files
from .config import AgentConfig

_ALLOWED_HOSTS = {"github.com", "objects.githubusercontent.com",
                  "release-assets.githubusercontent.com"}
_ALLOWED_BINARIES = {"llama-server", "llama-swap",
                     "llama-server.exe", "llama-swap.exe",
                     "ggml-rpc-server", "ggml-rpc-server.exe"}


class AgentInstallError(RuntimeError):
    pass


def _url(raw) -> str:
    raw = str(raw or "").strip()
    parsed = urllib.parse.urlparse(raw)
    if parsed.scheme != "https" or parsed.hostname not in _ALLOWED_HOSTS:
        raise AgentInstallError("only GitHub HTTPS release URLs are allowed")
    return raw


def _member_parts(name, strip):
    parts = Path(name).parts
    if strip:
        parts = parts[strip:]
    return parts


def _checked_target(dest: Path, parts) -> Path:
    if not parts:
        raise AgentInstallError("archive member has no path")
    target = dest.joinpath(*parts)
    try:
        target.resolve(strict=False).relative_to(dest.resolve())
    except ValueError as e:
        raise AgentInstallError("archive path escapes destination") from e
    return target


def _safe_extract_tar(tar, dest: Path, strip: int = 0, member: str = ""):
    for item in tar.getmembers():
        if item.isdir():
            continue
        name = item.name
        if member and os.path.basename(name) != member:
            continue
        parts = _member_parts(name, strip)
        if not parts:
            continue
        target = _checked_target(dest, parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        if item.issym():
            # Prebuilt llama.cpp archives store libfoo.so.0 as a symlink
            # to libfoo.so.0.x.y. Skipping these breaks the binaries.
            link_parts = _member_parts(item.linkname, strip)
            if not link_parts:
                link_parts = (os.path.basename(item.linkname),)
            link_target = _checked_target(dest, link_parts)
            try:
                target.unlink()
            except OSError:
                pass
            try:
                os.symlink(os.path.relpath(link_target, target.parent),
                           target)
            except OSError:
                # Windows without symlink privilege: copy the real file.
                if link_target.is_file():
                    shutil.copy2(link_target, target)
                else:
                    raise
            continue
        if item.islnk():
            link_parts = _member_parts(item.linkname, strip)
            src = _checked_target(dest, link_parts)
            try:
                target.unlink()
            except OSError:
                pass
            if src.is_file():
                os.link(src, target)
            continue
        if not item.isfile():
            continue
        src = tar.extractfile(item)
        if src is None:
            continue
        with open(target, "wb") as f:
            shutil.copyfileobj(src, f)


def _safe_extract_zip(zf, dest: Path, strip: int = 0, member: str = ""):
    for info in zf.infolist():
        if info.is_dir():
            continue
        name = info.filename
        if member and os.path.basename(name) != member:
            continue
        parts = _member_parts(name, strip)
        if not parts:
            continue
        target = _checked_target(dest, parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        mode = info.external_attr >> 16
        if stat.S_ISLNK(mode):
            linkname = zf.read(info).decode("utf-8", "replace")
            link_parts = _member_parts(linkname, strip)
            if not link_parts:
                link_parts = (os.path.basename(linkname),)
            link_target = _checked_target(dest, link_parts)
            try:
                target.unlink()
            except OSError:
                pass
            try:
                os.symlink(os.path.relpath(link_target, target.parent),
                           target)
            except OSError:
                if link_target.is_file():
                    shutil.copy2(link_target, target)
                else:
                    raise
            continue
        with zf.open(info) as src, open(target, "wb") as out:
            shutil.copyfileobj(src, out)


def _staged_verify(staging: Path):
    """(ok, detail) for a freshly extracted staging dir BEFORE it replaces
    the live install. Catches truncated/missing binaries."""
    exe = ".exe" if os.name == "nt" else ""
    srv = staging / f"llama-server{exe}"
    if not srv.is_file():
        return False, "llama-server missing from staged install"
    try:
        p = subprocess.run([str(srv), "--version"], capture_output=True,
                           text=True, timeout=120)
    except (OSError, subprocess.SubprocessError) as e:
        return False, f"llama-server verify failed: {e}"
    if p.returncode != 0:
        return False, (((p.stdout or "") + (p.stderr or ""))[-500:] or
                       f"llama-server --version exited {p.returncode}")
    rpc = staging / f"ggml-rpc-server{exe}"
    if rpc.is_file():
        try:
            subprocess.run([str(rpc), "--help"], capture_output=True,
                           text=True, timeout=60)
        except (OSError, subprocess.SubprocessError) as e:
            return False, f"ggml-rpc-server verify failed: {e}"
    return True, "ok"


def artifact(config: AgentConfig, urls, dest, archive="tar.gz",
             strip_components=0, member="", mode=None, replace=False) -> dict:
    """Download + extract typed release assets.

    replace=True is a CLEAN install: extract into a sibling staging dir,
    verify llama-server/ggml-rpc-server, then swap it in as `dest`,
    keeping the previous dir as `dest.bak-<ts>`. This is what prevents
    stale backend plugins (e.g. an old libggml-vulkan.so) from surviving
    an upgrade and crashing the new core.
    """
    if isinstance(urls, str):
        urls = [urls]
    if not isinstance(urls, list) or not urls:
        raise AgentInstallError("urls must be a non-empty list")
    urls = [_url(u) for u in urls]
    root = files._path(config, dest, write=True)
    if archive not in ("tar.gz", "zip", "none"):
        raise AgentInstallError("unsupported archive type")
    if member:
        # never allow traversal via a caller-supplied filename
        if (member in (".", "..") or "/" in member or "\\" in member
                or Path(member).name != member):
            raise AgentInstallError("member must be a plain filename")
    if replace:
        root.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=f".{root.name}.staging-",
                                        dir=str(root.parent)))
        target = staging
    else:
        root.mkdir(parents=True, exist_ok=True)
        staging = None
        target = root
    written = []
    try:
        for url in urls:
            fd, tmp = tempfile.mkstemp(prefix="agent-install-")
            os.close(fd)
            try:
                with urllib.request.urlopen(url, timeout=120) as resp, \
                        open(tmp, "wb") as out:
                    shutil.copyfileobj(resp, out)
                if archive == "tar.gz":
                    with tarfile.open(tmp, "r:gz") as tar:
                        _safe_extract_tar(tar, target, int(strip_components),
                                          member)
                elif archive == "zip":
                    with zipfile.ZipFile(tmp) as zf:
                        _safe_extract_zip(zf, target, int(strip_components),
                                          member)
                else:
                    tgt = target / (member or os.path.basename(url))
                    shutil.copy2(tmp, tgt)
                    if mode is not None:
                        os.chmod(tgt, int(mode))
                    written.append(str(tgt))
                if archive != "none":
                    written.append(str(target))
            finally:
                try:
                    os.unlink(tmp)
                except OSError:
                    pass
        if mode is not None:
            for path in target.rglob("*"):
                if path.is_file():
                    try:
                        os.chmod(path, int(mode))
                    except OSError:
                        pass
        if not replace:
            return {"dest": str(root), "urls": urls, "written": written}
        ok, detail = _staged_verify(staging)
        if not ok:
            raise AgentInstallError(f"staged install failed verification: {detail}")
        backup = None
        if root.exists():
            backup = root.parent / (f"{root.name}.bak-"
                                    + __import__("time").strftime("%Y%m%d-%H%M%S"))
            os.replace(root, backup)
        try:
            os.replace(staging, root)
        except OSError as e:
            if backup is not None and backup.exists() and not root.exists():
                try:
                    os.replace(backup, root)
                except OSError:
                    pass
            raise AgentInstallError(f"install swap failed: {e}") from e
        return {"dest": str(root), "urls": urls, "written": written,
                "replaced": True,
                "backup": str(backup) if backup else None,
                "verify": detail}
    finally:
        if staging is not None and staging.exists():
            shutil.rmtree(staging, ignore_errors=True)


def verify(config: AgentConfig, path, args=None) -> dict:
    p = files._path(config, path)
    if not p.is_file():
        raise AgentInstallError("binary not found")
    if p.name not in _ALLOWED_BINARIES:
        raise AgentInstallError("binary is not allowed")
    args = args or ["--version"]
    if not isinstance(args, list) or not all(
            isinstance(a, str) and a and "\x00" not in a for a in args):
        raise AgentInstallError("invalid args")
    try:
        proc = subprocess.run([str(p), *args], capture_output=True,
                              text=True, timeout=120)
    except (OSError, subprocess.SubprocessError) as e:
        raise AgentInstallError(f"verify failed: {e}") from e
    return {"returncode": proc.returncode,
            "output": ((proc.stdout or "") + (proc.stderr or ""))[-1000:]}


