"""Unit tests for the farmctl CLI (src/farmctl.py).

farmctl is API-only: every test substitutes Client.request so no app or
network is touched. These pin the wire contracts (paths + payloads), the
redaction behavior of snapshots, and the human/JSON output shapes.
"""
import json
import os

import pytest

import farmctl


def _fake(request_map, calls=None):
    """Build a Client.request stand-in keyed by (method, path)."""
    calls = calls if calls is not None else []

    def request(self, method, path, body=None, timeout=None):
        calls.append({"method": method, "path": path, "body": body,
                      "timeout": timeout})
        key = (method, path)
        value = request_map.get(key, request_map.get(path))
        if isinstance(value, Exception):
            raise value
        if callable(value):
            return value()
        if value is None:
            raise AssertionError(f"unexpected request: {method} {path}")
        return value

    return request, calls


MACHINES = [
    {"name": "m1", "role": "master", "os": "linux"},
    {"name": "m2", "role": "peer", "os": "linux"},
]
STATS = {
    "data": [
        {"name": "m1", "status": "online", "rpc_running": True,
         "cpu_pct": 12.5, "cores": 8, "ram_used_gb": 4, "ram_total_gb": 32,
         "gpus": [{"index": 0, "name": "NVIDIA GeForce RTX 3080",
                   "util": 50, "mem_used_gb": 8, "mem_total_gb": 12,
                   "temp": 60}]},
        {"name": "m2", "status": "offline", "rpc_running": False,
         "cpu_pct": 0, "cores": 4, "ram_used_gb": 1, "ram_total_gb": 16,
         "gpus": []},
    ],
    "totals": {"vram": {"used": 8, "total": 12, "pct": 66.7},
               "ram": {"used": 5, "total": 48, "pct": 10.4},
               "cpu": {"pct": 8.3, "busy": 0.7, "total": 12}},
}
SWAP_UP = {"running": True, "host": "m1", "port": 8080}
LOADED = {"loaded": ["model-a"], "running": True}


def test_status_renders_cluster(monkeypatch, capsys):
    fake, calls = _fake({
        "/api/machines": MACHINES,
        "/api/stats": STATS,
        "/api/swap/status": SWAP_UP,
        "/api/swap/loaded": LOADED,
    })
    monkeypatch.setattr(farmctl.Client, "request", fake)
    assert farmctl.main(["status"]) == 0
    out = capsys.readouterr().out
    assert "llama-swap: UP" in out
    assert "m1" in out and "m2" in out
    assert "Totals:" in out
    assert "model-a" in out
    assert any(c["path"] == "/api/stats" for c in calls)


def test_status_json_and_flags_after_subcommand(monkeypatch, capsys):
    fake, _ = _fake({
        "/api/machines": MACHINES,
        "/api/stats": STATS,
        "/api/swap/status": SWAP_UP,
        "/api/swap/loaded": LOADED,
    })
    monkeypatch.setattr(farmctl.Client, "request", fake)
    assert farmctl.main(["status", "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["swap"]["running"] is True
    assert data["machines"][0]["name"] == "m1"
    assert data["stats"]["totals"]["vram"]["total"] == 12


def test_doctor_ok_and_issues(monkeypatch, capsys):
    ready = {"ready_all": True, "machines": {
        "m1": {"machine": "m1", "ready": True, "local": True,
               "is_swap_host": True,
               "agent": {"ok": True, "version": "2.0.3"},
               "llama_cpp": {"installed": True, "version": "0.9.0",
                             "path": "/bin/llama-server"},
               "llama_rpc": {"installed": True, "path": "/bin/rpc"},
               "llama_swap": {"installed": True, "path": "/bin/swap",
                              "required": True}},
    }}
    fake, _ = _fake({
        "/api/readiness": ready,
        "/api/swap/status": SWAP_UP,
        "/api/machines": MACHINES,
    })
    monkeypatch.setattr(farmctl.Client, "request", fake)
    assert farmctl.main(["doctor"]) == 0
    out = capsys.readouterr().out
    assert "RESULT: OK" in out
    assert "llama.cpp" in out and "rpc" in out

    bad = {"ready_all": False, "machines": {
        "m2": {"machine": "m2", "ready": False,
               "agent": {"ok": False, "error": "auth failed"},
               "llama_cpp": {"installed": False, "error": "missing"},
               "llama_rpc": {"installed": False},
               "llama_swap": {"installed": False, "required": False}}}}
    fake, _ = _fake({
        "/api/readiness": bad,
        "/api/swap/status": {"running": False, "error": "down"},
    })
    monkeypatch.setattr(farmctl.Client, "request", fake)
    assert farmctl.main(["doctor"]) == 1
    out = capsys.readouterr().out
    assert "FAIL" in out
    assert "auth failed" in out
    assert "down" in out


def test_snapshot_redacts_and_writes(monkeypatch, tmp_path):
    mach = [{"name": "m1", "password": "hunter2",
             "_agent_token": "secret-token", "agent_token_set": True}]
    responses = {
        "/api/machines": mach,
        "/api/models?full=1": [{"model_id": "m", "flags": {}}],
        "/api/rpc-map": {"m": {}},
        "/api/swap/settings": {"binary": "llama-swap"},
        "/api/swap/status": SWAP_UP,
        "/api/swap/loaded": LOADED,
        "/api/stats": STATS,
        "/api/settings": {"engine": "llama_cpp", "hf_token_set": True},
    }
    fake, calls = _fake(responses)
    monkeypatch.setattr(farmctl.Client, "request", fake)
    out_file = tmp_path / "snap.json"
    assert farmctl.main(["snapshot", "--fast", "--out", str(out_file)]) == 0
    raw = out_file.read_text()
    assert "hunter2" not in raw
    assert "secret-token" not in raw
    assert (out_file.stat().st_mode & 0o777) == 0o600
    data = json.loads(raw)
    assert data["farmctl_snapshot"] == 1
    assert data["data"]["machines"][0]["agent_token_set"] is True
    # --fast skips readiness
    assert not any(c["path"] == "/api/readiness" for c in calls)


def test_snapshot_stdout_and_section_errors(monkeypatch, capsys):
    responses = {
        "/api/machines": MACHINES,
        "/api/models?full=1": farmctl.FarmError("boom"),
        "/api/rpc-map": {},
        "/api/swap/settings": {},
        "/api/swap/status": SWAP_UP,
        "/api/swap/loaded": LOADED,
        "/api/stats": STATS,
        "/api/settings": {},
    }
    fake, _ = _fake(responses)
    monkeypatch.setattr(farmctl.Client, "request", fake)
    assert farmctl.main(["snapshot", "--fast", "--out", "-"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert "machines" in data["data"]
    assert "models" in data["errors"]


def test_start_daemon_prints_report(monkeypatch, capsys):
    report = {"host": "m1", "port": 8080, "port_rolled": False,
              "failed": False,
              "steps": [{"step": "generate"}, {"step": "swap:start"}],
              "warnings": ["m2: no models"]}
    fake, calls = _fake({"/api/swap/launch": report})
    monkeypatch.setattr(farmctl.Client, "request", fake)
    assert farmctl.main(["start"]) == 0
    out = capsys.readouterr().out
    assert "start: OK" in out
    assert "swap:start" in out
    assert "m2: no models" in out
    assert calls[0]["body"] == {}


def test_start_failure_uses_step_report(monkeypatch, capsys):
    report = {"host": "m1", "failed": True,
              "steps": [{"step": "swap:start", "error": "unit failed"}]}

    def request(self, method, path, body=None, timeout=None):
        # NOTE: the request-body parameter shadows a local named `body`;
        # use `report` so the exception carries the step report.
        raise farmctl.FarmHTTPError(500, report, "http://x/api/swap/launch")

    monkeypatch.setattr(farmctl.Client, "request", request)
    assert farmctl.main(["start"]) == 1
    out = capsys.readouterr().out
    assert "FAIL  swap:start: unit failed" in out
    assert "start: FAILED" in out


def test_launch_requires_model():
    # A bare `launch` must not imply "load every model" — start the daemon
    # with `start`, load one model with `launch --model`.
    with pytest.raises(SystemExit) as exc:
        farmctl.main(["launch"])
    assert exc.value.code == 2


def test_launch_model_uses_swap_load(monkeypatch, capsys):
    fake, calls = _fake({
        "/api/swap/load/model--v1": {"ok": True, "model": "model--v1",
                                     "elapsed_s": 12.3, "preview": "hi"},
    })
    monkeypatch.setattr(farmctl.Client, "request", fake)
    assert farmctl.main(["launch", "--model", "model--v1"]) == 0
    assert calls[0]["path"] == "/api/swap/load/model--v1"
    assert "loaded model--v1" in capsys.readouterr().out


def test_unreachable_app_is_actionable(monkeypatch, capsys):
    def request(self, method, path, body=None, timeout=None):
        raise farmctl.FarmError("cannot reach http://127.0.0.1:8199 (refused)")

    monkeypatch.setattr(farmctl.Client, "request", request)
    assert farmctl.main(["status"]) == 1
    err = capsys.readouterr().err
    assert "cannot reach" in err


def test_stop_prints_daemon_and_machines(monkeypatch, capsys):
    result = {"host": "m1", "stopped": True, "detail": "",
              "machines": [{"name": "m1", "killed": True},
                           {"name": "m2", "killed": False}]}
    fake, calls = _fake({"/api/swap/stop": result})
    monkeypatch.setattr(farmctl.Client, "request", fake)
    assert farmctl.main(["stop"]) == 0
    out = capsys.readouterr().out
    assert "stopped" in out
    assert "m1: stopped" in out
    assert "m2: left running" in out
    assert calls[0]["path"] == "/api/swap/stop"


def test_stop_failure_exit(monkeypatch, capsys):
    fake, _ = _fake({"/api/swap/stop": {"host": "m1", "stopped": False}})
    monkeypatch.setattr(farmctl.Client, "request", fake)
    assert farmctl.main(["stop"]) == 1
    assert "NOT stopped" in capsys.readouterr().out


def test_unload_ok_and_failure(monkeypatch, capsys):
    fake, calls = _fake({"/api/swap/evict/model--v1": {"ok": True}})
    monkeypatch.setattr(farmctl.Client, "request", fake)
    assert farmctl.main(["unload", "model--v1"]) == 0
    assert calls[0]["path"] == "/api/swap/evict/model--v1"
    assert "unloaded model--v1" in capsys.readouterr().out

    fake, _ = _fake({"/api/swap/evict/model": {"ok": False,
                                               "error": "not loaded"}})
    monkeypatch.setattr(farmctl.Client, "request", fake)
    assert farmctl.main(["unload", "model"]) == 1
    assert "not loaded" in capsys.readouterr().out


def test_models_lists_variants_and_json(monkeypatch, capsys):
    models = [{
        "model_id": "base", "servers": ["m1", "m2"],
        "tensor_split_mode": "fit",
        "flags": {"ctx-size": 32768},
        "variants": [{"label": "small", "servers": ["m1"],
                      "flags": {"ctx-size": 8192}}],
    }]
    fake, _ = _fake({"/api/models": models})
    monkeypatch.setattr(farmctl.Client, "request", fake)
    assert farmctl.main(["models"]) == 0
    out = capsys.readouterr().out
    assert "MODEL" in out and "SERVERS" in out
    assert "base" in out
    assert "base--small" in out
    assert "8192" in out

    assert farmctl.main(["models", "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data[0]["model_id"] == "base"
