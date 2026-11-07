"""Typed prebuilt artifact install + verification."""
import os
import shutil
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
                     "llama-server.exe", "llama-swap.exe"}


class AgentInstallError(RuntimeError):
    pass


def _url(raw) -> str:
    raw = str(raw or "").strip()
    parsed = urllib.parse.urlparse(raw)
    if parsed.scheme != "https" or parsed.hostname not in _ALLOWED_HOSTS:
        raise AgentInstallError("only GitHub HTTPS release URLs are allowed")
    return raw


def _safe_extract_tar(tar, dest: Path, strip: int = 0, member: str = ""):
    for item in tar.getmembers():
        if not item.isfile():
            continue
        name = item.name
        if member and os.path.basename(name) != member:
            continue
        parts = Path(name).parts
        if strip:
            parts = parts[strip:]
            if not parts:
                continue
        target = dest.joinpath(*parts)
        try:
            target.resolve(strict=False).relative_to(dest.resolve())
        except ValueError as e:
            raise AgentInstallError("archive path escapes destination") from e
        target.parent.mkdir(parents=True, exist_ok=True)
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
        parts = Path(name).parts
        if strip:
            parts = parts[strip:]
            if not parts:
                continue
        target = dest.joinpath(*parts)
        try:
            target.resolve(strict=False).relative_to(dest.resolve())
        except ValueError as e:
            raise AgentInstallError("archive path escapes destination") from e
        target.parent.mkdir(parents=True, exist_ok=True)
        with zf.open(info) as src, open(target, "wb") as out:
            shutil.copyfileobj(src, out)


def artifact(config: AgentConfig, urls, dest, archive="tar.gz",
             strip_components=0, member="", mode=None) -> dict:
    if isinstance(urls, str):
        urls = [urls]
    if not isinstance(urls, list) or not urls:
        raise AgentInstallError("urls must be a non-empty list")
    urls = [_url(u) for u in urls]
    dest = files._path(config, dest, write=True)
    dest.mkdir(parents=True, exist_ok=True)
    if archive not in ("tar.gz", "zip", "none"):
        raise AgentInstallError("unsupported archive type")
    written = []
    for url in urls:
        fd, tmp = tempfile.mkstemp(prefix="agent-install-")
        os.close(fd)
        try:
            with urllib.request.urlopen(url, timeout=120) as resp, \
                    open(tmp, "wb") as out:
                shutil.copyfileobj(resp, out)
            if archive == "tar.gz":
                with tarfile.open(tmp, "r:gz") as tar:
                    _safe_extract_tar(tar, dest, int(strip_components),
                                      member)
            elif archive == "zip":
                with zipfile.ZipFile(tmp) as zf:
                    _safe_extract_zip(zf, dest, int(strip_components),
                                      member)
            else:
                target = dest / (member or os.path.basename(url))
                shutil.copy2(tmp, target)
                if mode is not None:
                    os.chmod(target, int(mode))
                written.append(str(target))
            if archive != "none":
                written.append(str(dest))
        finally:
            try:
                os.unlink(tmp)
            except OSError:
                pass
    if mode is not None:
        for path in dest.rglob("*"):
            if path.is_file():
                try:
                    os.chmod(path, int(mode))
                except OSError:
                    pass
    return {"dest": str(dest), "urls": urls, "written": written}


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


def vllm(config: AgentConfig, python="python3", venv_root="~/vllm-env",
         version="0.30.0", rocm=False) -> dict:
    python = str(python or "python3").strip()
    if os.path.sep in python:
        p = files._path(config, python)
        if not p.is_file():
            raise AgentInstallError("python not found")
        python = str(p)
    elif not shutil.which(python):
        raise AgentInstallError("python not found on PATH")
    if rocm and python == "python3" and shutil.which("python3.12"):
        python = "python3.12"
    root = files._path(config, venv_root, write=True)
    root.mkdir(parents=True, exist_ok=True)
    bindir = root / ("Scripts" if os.name == "nt" else "bin")
    py = bindir / ("python.exe" if os.name == "nt" else "python")
    pip = bindir / ("pip.exe" if os.name == "nt" else "pip")
    if not py.is_file():
        try:
            subprocess.run([python, "-m", "venv", str(root)],
                           capture_output=True, text=True, timeout=300,
                           check=True)
        except (OSError, subprocess.SubprocessError) as e:
            raise AgentInstallError(f"venv creation failed: {e}") from e
    args = [str(pip), "install", f"vllm=={version}"]
    if rocm:
        args += ["--extra-index-url", "https://wheels.vllm.ai/rocm/"]
    try:
        proc = subprocess.run(args, capture_output=True, text=True,
                              timeout=1800)
    except (OSError, subprocess.SubprocessError) as e:
        raise AgentInstallError(f"pip install failed: {e}") from e
    output = ((proc.stdout or "") + (proc.stderr or ""))[-2000:]
    if proc.returncode != 0:
        return {"ok": False, "venv_bin": str(bindir / "vllm"),
                "output": output, "error": "pip install failed"}
    probe = ("import vllm, torch; print('vllm', vllm.__version__, "
             "'backend', 'HIP' if getattr(torch.version, 'hip', None) "
             "else 'CUDA')")
    try:
        v = subprocess.run([str(py), "-c", probe], capture_output=True,
                           text=True, timeout=300)
    except (OSError, subprocess.SubprocessError) as e:
        return {"ok": False, "venv_bin": str(bindir / "vllm"),
                "output": str(e), "error": "import probe failed"}
    text = ((v.stdout or "") + (v.stderr or "")).strip()
    return {"ok": v.returncode == 0, "venv_bin": str(bindir / "vllm"),
            "output": (output + "\n" + text)[-2000:],
            "backend": text.splitlines()[-1] if text else ""}
