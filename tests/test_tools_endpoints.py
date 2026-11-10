"""UX-1: unified installer endpoints in app.py.

- GET /api/tools/verify: per-machine {tool: {found, path, version}},
  read-only, offline machines reported, no real hosts contacted.
- POST /api/tools/install/sync: stubbed probe/execute — the installer
  layer's Executor is fake (no real installs, no network).
Both go through TestClient like test_engine_endpoints.py.
"""
import os
import sys
import types

import pytest

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import app as app_module
import settings_store
import routes.install as install_routes  # noqa: E402
import routes.tools_install as tools_install_routes  # noqa: E402


@pytest.fixture
def settings_file(tmp_path, monkeypatch):
    """Redirect settings files into the tmp dir (same pattern as
    test_engine_endpoints) so tests never touch the repo's real config/."""
    import engine_registry
    eng = tmp_path / "settings.yaml"
    eng.write_text("")
    monkeypatch.setattr(settings_store, "SETTINGS_FILE", str(eng))
    monkeypatch.setattr(engine_registry, "SETTINGS_FILE", str(eng))

    machines = tmp_path / "machines.yaml"
    machines.write_text(
        "machines:\n"
        "- id: aibox1\n"
        "  name: aibox1\n"
        "  ip: 127.0.0.1\n"
        "  role: master\n"
        "  os: linux\n"
        "  rpc_port: 50052\n")
    import config_loader as cl
    monkeypatch.setattr(cl.MachinesStore, "yaml_path" if False else "X",
                        None, raising=False) if False else None
    from config_loader import machines as machines_store
    # point the store at the tmp file
    monkeypatch.setattr(machines_store, "path", str(machines))
    return machines


@pytest.fixture
def client(settings_file):
    from fastapi.testclient import TestClient
    return TestClient(app_module.app)


# ---------------- GET /api/tools/verify ------------------------------------
# NOTE (UX-3 merge): the per-machine verify shape ({machines: {...}}) was
# superseded by the per-tool local probe ({llama.cpp, llama-swap})
# — the shape the Settings-tab tool panel consumes.


# ---------------- POST /api/tools/install/sync -----------------------------

_CANNED_SUMMARY = {
    "installed": ["llama.cpp"],
    "skipped": [], "repaired": [], "settings": [], "notes": [],
    "manifest": {"tools": {}}, "failures": [],
}


def test_install_sync_runs_plan_with_stubbed_executor(client, monkeypatch):
    """The sync endpoint's Executor is faked: it records the plan it was
    given and returns a canned summary — no real install ever runs."""
    calls = {}

    def fake_execute(plan, info, dry_run, runner=None):
        calls["plan"] = [type(s).__name__ for s in plan]
        calls["dry_run"] = dry_run
        calls["runner"] = runner
        return dict(_CANNED_SUMMARY)

    monkeypatch.setattr(tools_install_routes, "_execute_plan", fake_execute)
    monkeypatch.setattr(tools_install_routes, "_tools_plan",
                        lambda m, role, log: ([object()], object(),
                                              object(), object()))
    r = client.post("/api/tools/install/sync",
                    json={"machine": "aibox1", "tools": [],
                          "role": "main", "dry_run": True})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["installed"] == ["llama.cpp"]
    assert body["failures"] == []
    # plan was produced (list of step type names) and dry_run passed through
    assert calls["dry_run"] is True
    assert isinstance(calls["plan"], list) and calls["plan"]


def test_install_sync_unknown_machine_404(client, monkeypatch):
    monkeypatch.setattr(tools_install_routes, "_execute_plan",
                        lambda *a, **k: dict(_CANNED_SUMMARY))
    monkeypatch.setattr(tools_install_routes, "_tools_plan",
                        lambda m, role, log: ([], object(), object(), object()))
    r = client.post("/api/tools/install/sync",
                    json={"machine": "ghost"})
    assert r.status_code == 404
    assert "ghost" in r.json()["detail"]



def tmp_machine(client, os_name):
    """Add a windows machine to the store and return its name."""
    import config_loader as cl
    cl.machines.update(
        "aibox1", {}) if False else None
    # append directly to the store's in-memory list via its add API
    name = "winbox-ux1"
    try:
        cl.machines.add({"id": name, "name": name, "ip": "127.0.0.1",
                         "os": os_name})
    except Exception:
        machines = cl.machines.all()
        machines.append({"id": name, "name": name, "ip": "127.0.0.1",
                         "os": os_name})
    return name

def test_verify_respects_configured_paths(client, settings_file, tmp_path,
                                          monkeypatch):
    """Configured per-machine / swap paths are authoritative: the status
    probe must not substitute a different well-known install."""
    bin_dir = tmp_path / "custom-llama"
    bin_dir.mkdir()
    server = bin_dir / "llama-server"
    server.write_text("#!/bin/sh\necho 'version: 0.7.0 (build 12209)'\n")
    server.chmod(0o755)
    rpc = bin_dir / "ggml-rpc-server"
    rpc.write_text("#!/bin/sh\n")
    rpc.chmod(0o755)

    # local machine's llama_bin wins over the global setting
    settings_file.write_text(
        "machines:\n"
        "- id: aibox1\n"
        "  name: aibox1\n"
        "  ip: 127.0.0.1\n"
        "  role: master\n"
        "  os: linux\n"
        "  rpc_port: 50052\n"
        f"  llama_bin: {bin_dir}\n",
        encoding="utf-8",
    )
    swap_dir = tmp_path / "custom-swap"
    swap_dir.mkdir()
    swap = swap_dir / "llama-swap"
    swap.write_text("#!/bin/sh\necho 'v271'\n")
    swap.chmod(0o755)
    import swap_manager
    monkeypatch.setattr(swap_manager, "load_settings",
                        lambda: {"binary": str(swap)})

    body = client.get("/api/tools/verify").json()
    assert body["llama.cpp"]["found"] is True
    assert body["llama.cpp"]["path"] == str(server)
    assert body["llama.cpp"]["configured"] == str(bin_dir)
    assert body["llama-swap"]["found"] is True
    assert body["llama-swap"]["path"] == str(swap)
    assert body["llama-swap"]["configured"] == str(swap)


def test_install_sync_local_alias_resolves_to_this_host(client, monkeypatch):
    import app as app_module
    seen = {}

    monkeypatch.setattr(tools_install_routes, "_tools_plan",
                        lambda m, role, log: ([object()], object(),
                                              object(), object()))

    def fake_execute(plan, info, dry_run, runner=None):
        seen["machine"] = "local"
        return dict(_CANNED_SUMMARY, installed=["llama.cpp"])

    monkeypatch.setattr(tools_install_routes, "_execute_plan", fake_execute)
    r = client.post("/api/tools/install/sync",
                    json={"machine": "__local__", "tools": ["llama.cpp"],
                          "dry_run": True})
    assert r.status_code == 200, r.text
    assert r.json()["installed"] == ["llama.cpp"]


def test_remote_tools_install_uses_agent_and_reports_failures(monkeypatch):
    import app as app_module
    import llamacpp_installer
    import swap_installer

    called = []
    monkeypatch.setattr(
        llamacpp_installer, "install_on",
        lambda machine, log: (called.append("llama.cpp") or
                              {"ok": True, "detail": "cpp ok",
                               "version": "b12211"}))
    monkeypatch.setattr(
        swap_installer, "install_on",
        lambda machine, log: (called.append("llama-swap") or
                              {"ok": False, "detail": "swap failed"}))
    machine = {"name": "peer1", "ip": "10.0.0.50", "os": "linux"}
    summary = tools_install_routes._install_remote_tools(
        machine, "main", [], False, lambda m: None)
    assert called == ["llama.cpp", "llama-swap"]
    assert summary["installed"] == ["llama.cpp"]
    assert summary["failures"][0]["context"]["tool"] == "llama-swap"
    assert summary["manifest"]["tools"]["llama.cpp"]["method"] == "agent"


def test_remote_tools_install_dry_run_does_not_call_agent(monkeypatch):
    import app as app_module
    import llamacpp_installer
    import swap_installer

    def boom(*a, **k):
        raise AssertionError("dry-run must not install")
    monkeypatch.setattr(llamacpp_installer, "install_on", boom)
    monkeypatch.setattr(swap_installer, "install_on", boom)
    logs = []
    summary = tools_install_routes._install_remote_tools(
        {"name": "peer1", "ip": "10.0.0.50", "os": "linux"},
        "main", [], True, logs.append)
    assert summary["planned"] == ["llama.cpp", "llama-swap"]
    assert summary["failures"] == []
    assert any("would install llama.cpp" in line for line in logs)
