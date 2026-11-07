"""Regression tests for PR #6 review fixes."""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

import rpc_manager  # noqa: E402
import steps  # noqa: E402
import swap_manager  # noqa: E402


def test_remote_local_single_metal_uses_auto(monkeypatch):
    captured = {}

    def fake_run_steps(report, plan, catch=None):
        for item in plan:
            if item[0].startswith("serve:"):
                captured["model"] = item[2]
        return report

    monkeypatch.setattr(steps, "run_steps", fake_run_steps)
    model = {"model_id": "m", "base_path": "~/x.gguf",
             "servers": ["mac"], "tensor_split_mode": "fit",
             "tensor_split": ""}
    master = {"name": "mac", "ip": "10.0.0.9", "os": "macos",
              "gpu": "Metal"}
    rpc_manager._launch_model_remote_local(model, master, [], 8080)
    assert captured["model"]["tensor_split_mode"] == "auto"
    assert captured["model"]["tensor_split"] == ""


def test_launch_validates_before_stopping(monkeypatch):
    gen = {
        "host": "aibox1", "host_os": "linux", "abs_config": "/tmp/s.yaml",
        "yaml": "models: {}\n", "models": {"m": {}}, "scripts": {},
        "_scripts": {}, "script_mids": [], "entries": [],
        "swap_dir": "/tmp", "scripts_dir": "/tmp/swap", "skipped": [],
        "warnings": [],
    }
    monkeypatch.setattr(swap_manager, "launch_plan",
                        lambda *a, **k: gen)
    monkeypatch.setattr(swap_manager, "load_settings",
                        lambda: {"binary": "llama-swap", "listen_port": 8080,
                                 "config_path": "/tmp/s.yaml",
                                 "host_name": "aibox1"})
    monkeypatch.setattr(swap_manager, "save_settings",
                        lambda *a, **k: None)
    monkeypatch.setattr(swap_manager.cfg.machines, "require",
                        lambda n: {"name": n, "os": "linux",
                                   "ip": "127.0.0.1"})
    monkeypatch.setattr(swap_manager.local_node, "is_local", lambda h: True)
    monkeypatch.setattr(swap_manager.cfg, "list_models", lambda: [])
    stopped = []
    monkeypatch.setattr(swap_manager, "stop",
                        lambda: stopped.append(True) or {"stopped": True})
    monkeypatch.setattr(swap_manager, "_run", lambda *a, **k: (0, "", ""))
    monkeypatch.setattr(swap_manager, "_write_host", lambda *a, **k: None)
    monkeypatch.setattr(swap_manager, "_ensure_host_dir",
                        lambda *a, **k: None)

    def fake_run_steps(report, plan, catch=None):
        report["step_names"] = [s[0] for s in plan]
        report["failed"] = True
        return report

    monkeypatch.setattr(swap_manager, "run_steps", fake_run_steps)
    report = swap_manager.launch()
    names = report["step_names"]
    assert names.index("validate") < names.index("swap:stop")
    assert stopped == []
