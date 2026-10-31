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

import app as app_module  # noqa: E402


@pytest.fixture
def settings_file(tmp_path, monkeypatch):
    """Redirect settings files into the tmp dir (same pattern as
    test_engine_endpoints) so tests never touch the repo's real config/."""
    import engine_registry
    eng = tmp_path / "settings.yaml"
    eng.write_text("")
    monkeypatch.setattr(app_module, "SETTINGS_FILE", str(eng))
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
# superseded by the per-tool local probe ({llama.cpp, llama-swap, vllm})
# — the shape the Settings-tab tool panel consumes. Covered in
# test_engine_endpoints.py (shape + settings-record tests).


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

    def fake_execute(plan, info, dry_run):
        calls["plan"] = [type(s).__name__ for s in plan]
        calls["dry_run"] = dry_run
        return dict(_CANNED_SUMMARY)

    monkeypatch.setattr(app_module, "_execute_plan", fake_execute)
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
    monkeypatch.setattr(app_module, "_execute_plan",
                        lambda *a, **k: dict(_CANNED_SUMMARY))
    r = client.post("/api/tools/install/sync",
                    json={"machine": "ghost"})
    assert r.status_code == 404
    assert "ghost" in r.json()["detail"]


def test_install_sync_windows_vllm_refused(client, monkeypatch, settings_file):
    """A windows machine's plan carries the vLLM refusal (WSL guidance)
    as a PlanNote — the endpoint surfaces it verbatim in `notes`."""
    from installer.probe import PlanNote

    def fake_execute(plan, info, dry_run):
        notes = [(s.kind, s.tool, s.message) for s in plan
                 if isinstance(s, PlanNote)]
        return dict(_CANNED_SUMMARY, notes=notes)

    monkeypatch.setattr(app_module, "_execute_plan", fake_execute)
    win = tmp_machine(client, os_name="windows")
    r = client.post("/api/tools/install/sync",
                    json={"machine": win, "dry_run": True})
    body = r.json()
    refused = [n for n in body["notes"] if n[0] == "refused" and n[1] == "vllm"]
    assert refused, body["notes"]
    assert "WSL" in refused[0][2]


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