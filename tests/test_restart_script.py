"""Restart standardization: dashboard restart uses the same script as the CLI.

POSIX `/api/restart` delegates to `restart.sh --full`, so the dashboard
button and `./restart.sh` perform an identical cluster teardown. Windows
has no bash and keeps its PowerShell fallback.
"""
import os
import subprocess
import sys

import pytest

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import app as app_module  # noqa: E402


@pytest.fixture
def client():
    from fastapi.testclient import TestClient
    return TestClient(app_module.app)


@pytest.mark.skipif(os.name == "nt", reason="POSIX restart path only")
def test_api_restart_delegates_to_full_script(client, monkeypatch):
    calls = {}

    def fake_popen(args, **kwargs):
        calls["args"] = args
        calls["kwargs"] = kwargs
        return object()

    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    r = client.post("/api/restart")
    assert r.status_code == 200
    assert r.json().get("status") == "restarting"

    args = calls["args"]
    assert args[0] == "/bin/bash"
    assert args[1].endswith("restart.sh")
    assert args[2] == "--full"
    assert calls["kwargs"].get("start_new_session") is True


def test_restart_scripts_exist():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    assert os.path.exists(os.path.join(root, "restart.sh"))
    # backwards-compatible alias
    assert os.path.exists(os.path.join(root, "restart-llama-swap.sh"))
