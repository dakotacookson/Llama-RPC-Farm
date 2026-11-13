"""The peer agent must do a clean staged install (replace=True): extract
to staging, verify, swap, and keep the old dir as a backup so stale
backend plugins from a previous build cannot survive an upgrade."""
import io
import os
import sys
import tarfile
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "peer_agent"))

from llama_peer_agent.config import AgentConfig  # noqa: E402
from llama_peer_agent import install  # noqa: E402


def _cfg(tmp):
    return AgentConfig(token="t", host="127.0.0.1", port=8198,
                       file_roots=[str(tmp)])


def _tar(path):
    files = [
        ("llama-server", "#!/bin/sh\necho version: test\n", 0o755),
        ("ggml-rpc-server", "#!/bin/sh\nexit 0\n", 0o755),
        ("libggml-cpu.so", "new", 0o644),
    ]
    with tarfile.open(path, "w:gz") as tf:
        for name, body, mode in files:
            data = body.encode()
            info = tarfile.TarInfo(name)
            info.size = len(data)
            info.mode = mode
            tf.addfile(info, io.BytesIO(data))
    return path


def test_clean_replace_removes_stale_files(tmp_path, monkeypatch):
    dest = tmp_path / "bin"
    dest.mkdir()
    (dest / "stale-vulkan.so").write_text("old build")
    (dest / "old-soname.so.0").write_text("old")
    tar = _tar(tmp_path / "asset.tar.gz")
    blob = tar.read_bytes()

    class Resp(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(install, "_url", lambda u: u)
    monkeypatch.setattr(install.urllib.request, "urlopen",
                        lambda url, timeout=120: Resp(blob))
    res = install.artifact(_cfg(tmp_path), ["https://example/a.tar.gz"],
                           str(dest), archive="tar.gz",
                           strip_components=0, mode=0o755, replace=True)
    assert res["replaced"] is True
    assert (dest / "llama-server").is_file()
    assert (dest / "libggml-cpu.so").is_file()
    # stale files are gone from the live dir
    assert not (dest / "stale-vulkan.so").exists()
    assert not (dest / "old-soname.so.0").exists()
    # ...and preserved in the backup
    backup = Path(res["backup"])
    assert backup.is_dir()
    assert (backup / "stale-vulkan.so").read_text() == "old build"


def test_non_replace_keeps_legacy_merge_behavior(tmp_path, monkeypatch):
    dest = tmp_path / "bin"
    dest.mkdir()
    (dest / "keep.txt").write_text("keep")
    tar = _tar(tmp_path / "asset.tar.gz")
    blob = tar.read_bytes()

    class Resp(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(install, "_url", lambda u: u)
    monkeypatch.setattr(install.urllib.request, "urlopen",
                        lambda url, timeout=120: Resp(blob))
    install.artifact(_cfg(tmp_path), ["https://example/a.tar.gz"],
                     str(dest), archive="tar.gz", strip_components=0,
                     mode=0o755, replace=False)
    assert (dest / "keep.txt").exists()
