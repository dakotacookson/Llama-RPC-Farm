"""Windows release extraction is Python-native and verified.

Regression: the unified installer downloaded `.zip` assets to a `.pkg`
temp name and used PowerShell `Expand-Archive`, trusting its exit code.
On Windows PowerShell 5.1 a failed extract could still exit 0, so the
installer reported success while `llama-server.exe` / `llama-swap.exe`
were missing (Windows release E2E: "llama.cpp install incomplete").
"""

import os
import shutil
import sys
import zipfile

import pytest

SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from installer.env import RunResult                  # noqa: E402
from installer.probe import InstallTool               # noqa: E402
from installer import tools as tools_mod              # noqa: E402


class Run:
    def __init__(self):
        self.cmds = []

    def __call__(self, cmd, timeout=30.0):
        return self.run(cmd, timeout)

    def run(self, cmd, timeout=30.0):
        self.cmds.append(str(cmd))
        return RunResult(str(cmd), 0, "", "")


class ZipDownloader:
    """Copies a prebuilt zip into the installer's temp destination."""

    def __init__(self, archive):
        self.archive = archive
        self.urls = []

    def __call__(self, url, dest):
        self.urls.append(url)
        shutil.copyfile(self.archive, dest)


def _zip(path, members):
    with zipfile.ZipFile(path, "w") as zf:
        for name, data in members.items():
            zf.writestr(name, data)
    return path


def _tempdir(tmp_path, monkeypatch):
    tmp = tmp_path / "tmp"
    tmp.mkdir()
    monkeypatch.setattr(tools_mod.tempfile, "gettempdir",
                        lambda: str(tmp))
    return tmp


def _llcpp_step(dest, name="llama-b20000-bin-win-cpu-x64.zip"):
    return InstallTool(
        "llama.cpp", "prebuilt", "b20000", backend="cpu", dest=str(dest),
        extra={"assets": [name],
               "base": "https://github.com/ggml-org/llama.cpp/releases/"
                       "download/b20000"})


def _swap_step(dest):
    return InstallTool(
        "llama-swap", "prebuilt", "273", dest=str(dest),
        extra={"assets": ["llama-swap_273_windows_amd64.zip"],
               "base": "https://github.com/mostlygeek/llama-swap/releases/"
                       "download/v273"})


def test_windows_llcpp_uses_python_zip_and_lands_binaries(
        tmp_path, monkeypatch):
    tmp = _tempdir(tmp_path, monkeypatch)
    archive = _zip(tmp / "payload.zip", {
        "llama-server.exe": b"server",
        "ggml-rpc-server.exe": b"rpc",
        "ggml.dll": b"dll",
    })
    dest = tmp_path / "llama.cpp" / "bin"
    run = Run()

    entry = tools_mod.install_llcpp(
        _llcpp_step(dest), ZipDownloader(archive), run, lambda m: None,
        "windows", "x86_64")

    assert (dest / "llama-server.exe").read_bytes() == b"server"
    assert (dest / "ggml-rpc-server.exe").read_bytes() == b"rpc"
    assert entry["dest"] == str(dest)
    assert not any("Expand-Archive" in c for c in run.cmds)
    assert not any(c.endswith(".pkg") or ".pkg'" in c for c in run.cmds)


def test_windows_swap_uses_python_zip_and_lands_exe(tmp_path, monkeypatch):
    tmp = _tempdir(tmp_path, monkeypatch)
    archive = _zip(tmp / "swap.zip", {"llama-swap.exe": b"swap"})
    dest = tmp_path / "llama-swap"
    run = Run()

    entry = tools_mod.install_swap(
        _swap_step(dest), ZipDownloader(archive), run, lambda m: None,
        "windows", "x86_64")

    assert (dest / "llama-swap.exe").read_bytes() == b"swap"
    assert entry["dest"] == str(dest / "llama-swap.exe")
    assert not any("Expand-Archive" in c for c in run.cmds)


def test_windows_llcpp_corrupt_zip_fails_loudly(tmp_path, monkeypatch):
    tmp = _tempdir(tmp_path, monkeypatch)
    bad = tmp / "bad.zip"
    bad.write_bytes(b"not a zip archive")

    with pytest.raises(RuntimeError, match="not a readable zip archive"):
        tools_mod.install_llcpp(
            _llcpp_step(tmp_path / "bin"), ZipDownloader(bad), Run(),
            lambda m: None, "windows", "x86_64")


def test_windows_swap_zip_without_exe_fails(tmp_path, monkeypatch):
    tmp = _tempdir(tmp_path, monkeypatch)
    archive = _zip(tmp / "swap.zip", {"README.txt": b"docs"})

    with pytest.raises(RuntimeError, match="no llama-swap.exe"):
        tools_mod.install_swap(
            _swap_step(tmp_path / "bin"), ZipDownloader(archive), Run(),
            lambda m: None, "windows", "x86_64")


def test_windows_zip_rejects_path_traversal(tmp_path, monkeypatch):
    tmp = _tempdir(tmp_path, monkeypatch)
    archive = _zip(tmp / "evil.zip", {
        "llama-server.exe": b"server",
        "ggml-rpc-server.exe": b"rpc",
        "../escaped.txt": b"no",
    })

    with pytest.raises(RuntimeError, match="unsafe archive member"):
        tools_mod.install_llcpp(
            _llcpp_step(tmp_path / "bin"), ZipDownloader(archive), Run(),
            lambda m: None, "windows", "x86_64")


def test_downloader_retries_truncated_body(tmp_path, monkeypatch):
    from installer import execute as execute_mod

    class FakeResponse:
        def __init__(self, body, length):
            self.body = body
            self.headers = {"Content-Length": str(length)}

        def read(self, _n=-1):
            body, self.body = self.body, b""
            return body

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    import urllib.request
    monkeypatch.setattr(
        urllib.request, "urlopen",
        lambda *a, **k: FakeResponse(b"short", 999))
    dest = tmp_path / "asset.zip"

    with pytest.raises(RuntimeError, match="download failed after 2 tries"):
        execute_mod.RealDownloader(retries=2)("https://example.test/a.zip",
                                              str(dest))
    assert not dest.exists()
