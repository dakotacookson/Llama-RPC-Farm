"""Smoke tests: FastAPI application is importable and serves HTTP responses.

Uses Starlette's TestClient (ships with FastAPI) so no live subprocess or
real network socket is needed. The app module is imported from src/ (already
on sys.path via conftest.py).

These tests verify:
- The FastAPI app object is accessible after import.
- GET / returns HTTP 200 with a non-empty body (the dashboard HTML).
- The server would bind to port 8199 (the configured port constant).

All heavy external I/O (paramiko, config files, local_node detection) is
stubbed at the module level before the app is imported.
"""
import os
import sys
import types

import pytest


# ---------------------------------------------------------------------------
# Stub heavy optional dependencies before importing the app.
# conftest already stubs paramiko; we additionally stub config_loader and
# local_node so the import doesn't attempt filesystem/network operations.
# ---------------------------------------------------------------------------

def _make_stub_cfg():
    """Minimal config_loader stand-in that satisfies app.py's import-time
    accesses without touching the filesystem."""
    stub = types.ModuleType("config_loader")

    class _Machines:
        def all(self):
            return []
        def get(self, name):
            return None
        def require(self, name):
            raise ValueError(name)

    class _Cfg:
        machines = _Machines()
        def get_groups(self):
            return []
        def get_stores(self):
            return {}

    stub.cfg = _Cfg()
    stub.Cfg = _Cfg
    return stub


def _make_stub_local_node():
    stub = types.ModuleType("local_node")
    stub.local_machine = {
        "name": "test-host",
        "ip": "127.0.0.1",
        "role": "master",
        "os": "linux",
        "username": "",
        "password": "",
        "rpc_port": 50052,
        "gpu": "",
        "detected": {"gpu_names": [], "vram_total_gb": 0, "ram_total_gb": 0},
    }
    stub.is_local = lambda m: m.get("name") == "test-host"
    return stub


@pytest.fixture(scope="module", autouse=True)
def _patch_app_deps():
    """Install stubs for config_loader, local_node, rpc_manager, and
    swap_manager before app.py is imported so the module-level startup
    code doesn't touch the network or filesystem."""
    for mod_name, factory in [
        ("config_loader", _make_stub_cfg),
        ("local_node", _make_stub_local_node),
    ]:
        if mod_name not in sys.modules:
            sys.modules[mod_name] = factory()

    # Stub rpc_manager minimally so route registrations don't fail.
    if "rpc_manager" not in sys.modules:
        rm = types.ModuleType("rpc_manager")
        rm.cfg = sys.modules["config_loader"].cfg
        sys.modules["rpc_manager"] = rm

    # Stub swap_manager.
    if "swap_manager" not in sys.modules:
        sys.modules["swap_manager"] = types.ModuleType("swap_manager")

    yield


# ---------------------------------------------------------------------------
# Import app lazily (after stubs are in place)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def app_client():
    """Return a Starlette TestClient wrapping the real FastAPI app."""
    try:
        from fastapi.testclient import TestClient
    except ImportError:
        pytest.skip("fastapi[testclient] not installed")

    import app as app_module  # src/app.py
    client = TestClient(app_module.app, raise_server_exceptions=False)
    return client


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_app_object_exists():
    """The FastAPI application object is accessible after import."""
    import app as app_module
    from fastapi import FastAPI
    assert isinstance(app_module.app, FastAPI), (
        "app.py must expose a FastAPI instance named 'app'"
    )


def test_app_configured_port_is_8199():
    """The server is configured to listen on port 8199 as documented."""
    import app as app_module
    # Accept the port encoded as a module-level constant or as a uvicorn
    # run() call argument.  We look for 8199 in the module source.
    import inspect
    source = inspect.getsource(app_module)
    assert "8199" in source, (
        "Port 8199 not found in app.py source — verify the configured port."
    )


def test_dashboard_returns_200(app_client):
    """GET / must return HTTP 200 with a non-empty body."""
    response = app_client.get("/")
    assert response.status_code == 200, (
        f"Expected 200 from GET /, got {response.status_code}"
    )
    assert len(response.content) > 0, "Dashboard response body must not be empty"


def test_dashboard_body_contains_html(app_client):
    """The dashboard response must contain HTML markup (it serves the
    single-page UI, not a plain-text or JSON response)."""
    response = app_client.get("/")
    if response.status_code != 200:
        pytest.skip("Server did not return 200; covered by test_dashboard_returns_200")
    content_type = response.headers.get("content-type", "")
    body = response.text
    assert "text/html" in content_type or "<html" in body.lower() or "<!doctype" in body.lower(), (
        f"Expected HTML response from GET /; content-type={content_type!r}"
    )
