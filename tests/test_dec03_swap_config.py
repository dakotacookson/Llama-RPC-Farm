"""DEC-03 (AUDIT-T6 §1.6): API surface for engine-qualified swap-config
files — POST /api/swap/settings persists swap.yaml's `config_path_vllm`
(two-key pattern alongside `config_path`), machine add/update validate the
per-machine `swap_config_path` / `swap_config_path_vllm` overrides on the
save path (bad values never land in machines.yaml), and the settings GET
surfaces both keys even on fresh installs.
"""
import os
import sys

import pytest
import yaml

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import app as app_module  # noqa: E402
import config_loader  # noqa: E402
import swap_manager  # noqa: E402


@pytest.fixture
def swap_settings_file(tmp_path, monkeypatch):
    """Redirect the swap settings file into the pytest tmp dir."""
    path = tmp_path / "swap.yaml"
    monkeypatch.setattr(swap_manager, "SWAP_SETTINGS_FILE", str(path))
    monkeypatch.setattr(swap_manager, "get_host",
                        lambda name=None: {"name": "aibox1", "ip": "1.2.3.4",
                                           "os": "linux"}, raising=False)
    return path


@pytest.fixture
def machines_file(tmp_path, monkeypatch):
    path = tmp_path / "machines.yaml"
    monkeypatch.setattr(config_loader, "MACHINES_FILE", str(path))
    return path


@pytest.fixture
def client():
    from fastapi.testclient import TestClient
    return TestClient(app_module.app)





def test_machine_add_rejects_bad_swap_config_override_type(client):
    """A non-string machine override is rejected on the save path (422:
    the typed model pins these fields to str) — the bad value never lands
    in machines.yaml."""
    r = client.post("/api/machines", json={
        "name": "b1", "ip": "10.0.0.9", "username": "u", "password": "p",
        "os": "linux", "rpc_port": 50052, "gpu": "",
        "swap_config_path_vllm": 5})
    assert r.status_code == 422
    detail = str(r.json()["detail"])
    assert "swap_config_path_vllm" in detail or "str" in detail