"""The release app wheel must carry the in-place update payload."""
import os
import subprocess
import sys
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_app_wheel_contains_source_and_ui(tmp_path):
    out = tmp_path / "dist"
    subprocess.run(
        [sys.executable, os.path.join(ROOT, "scripts", "build_app_wheel.py"),
         "--version", "9.9.9", "--out", str(out)],
        check=True, cwd=ROOT, capture_output=True, text=True)
    wheel = out / "llama_rpc_farm-9.9.9-py3-none-any.whl"
    assert wheel.is_file()
    with zipfile.ZipFile(wheel) as zf:
        names = zf.namelist()
        assert len(names) == len(set(names))
        assert "src/app.py" in names
        assert "ui/static/app.js" in names
        assert "peer_agent/pyproject.toml" in names
        assert "llama_rpc_farm-9.9.9.dist-info/RECORD" in names
