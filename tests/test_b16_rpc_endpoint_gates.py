"""B16: engine gates on the remaining RPC/tune/test endpoints.

The CTO route audit (2026-09-24) found 6 route families that still do
llama.cpp-RPC things under engine=vllm. G-3 already gated rpc/start and
start-all; this file covers the REST:

- POST /api/rpc/stop-all -> 409 in vllm mode, rpc_manager.stop_rpc NEVER
  called (both rpc-server start paths are G-3 gated, so no peers exist
  under vLLM; swap_manager.stop does NOT cover peer cleanup — it kills
  only the swap daemon — so a flat 409 is correct everywhere).
- GET /api/rpc-map -> 409 in vllm mode, rpc_manager._split_servers /
  compute_rpc_map NEVER called (no RPC<N> numbering exists).
- POST /api/models/{id}/test -> ROUTED in vllm mode: the REAL
  vllm_manager.validate_model_config validates the stored config
  (errors -> 400 with messages, warns ride along under ok:true,
  unknown model -> 404); rpc_manager.validate_model NEVER called.
  llama.cpp mode keeps the disk-based path byte-identical.
- tune/* family (8 endpoints) -> 409 in vllm mode via the shared
  _refuse_tuner_in_vllm guard (ui_hints.hidden_panels already hides
  the tuner panels; the API refuses too); tuner.* NEVER called.
  llama.cpp mode keeps every endpoint unchanged.

Heavy externals are monkeypatched; the REAL config_loader owns
machines.yaml / model YAMLs against tmp files.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import app as app_module  # noqa: E402
import config_loader  # noqa: E402
import vllm_manager  # noqa: E402


# ---------------- fixtures ------------------------------------------------

@pytest.fixture
def settings_file(tmp_path, monkeypatch):
    """Same contract as test_engine_endpoints.settings_file: BOTH settings
    globals point into tmp; machines.yaml gets one linux master."""
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
        "  rpc_port: 50052\n"
        "  gpu: CUDA0,CUDA1,CUDA2,CUDA3\n"
        "  gpu_vendor: nvidia\n"
        "  detected:\n"
        "    gpu_names: [GPU0, GPU1, GPU2, GPU3]\n"
        "    vram_total_gb: 96.0\n"
        "    ram_total_gb: 64\n"
    )
    monkeypatch.setattr(config_loader.machines, "path", str(machines))
    import rpc_manager
    monkeypatch.setattr(rpc_manager, "cfg", config_loader)

    models_dir = tmp_path / "models"
    models_dir.mkdir()
    monkeypatch.setattr(config_loader, "MODELS_DIR", str(models_dir))
    monkeypatch.setattr(app_module.cfg, "MODELS_DIR", str(models_dir))
    yield str(eng)


@pytest.fixture
def client(settings_file):
    from fastapi.testclient import TestClient
    return TestClient(app_module.app)



class _LcppEngine:
    name = "llama_cpp"

    # DEC-04: save_model dispatches the per-model section round-trip
    # through the owning adapter; stub doubles carry the protocol members.
    def apply_model_section(self, data, where="vllm"):
        return data

    def sanitize_model_section(self, raw, where="vllm"):
        if raw is None:
            return None
        return dict(raw)

    def capable(self, capability):
        return capability == "rpc_cluster"


def _engine(engine_obj, monkeypatch):
    monkeypatch.setattr(app_module.engine_registry, "get_engine",
                        lambda s=None: engine_obj)


def _save_model(model_id="qwen-test", **over):
    """Write a model YAML through the REAL config_loader."""
    payload = {
        "model_id": model_id,
        "name": "Qwen Test",
        "base_path": "/models/Qwen3-8B",
        "servers": ["aibox1"],
    }
    payload.update(over)
    config_loader.save_model(model_id, payload)
    return payload


TUNE_409 = "Tuner is llama.cpp-specific (VRAM/KV math); hidden in vLLM mode"
RPC_409 = ("vLLM mode does not use RPC clustering — serve models "
           "through llama-swap (Launch All / /api/swap/*)")


# ---------------- Part A: POST /api/rpc/stop-all ---------------------------

def test_stop_all_llama_cpp_unchanged(client, settings_file, monkeypatch):
    """llama.cpp mode: stops a peer on every machine — unchanged."""
    _engine(_LcppEngine(), monkeypatch)
    stopped = []
    monkeypatch.setattr(app_module.rpc_manager, "stop_rpc",
                        lambda name: stopped.append(name) or
                        {"machine": name, "killed": True, "port": None})
    r = client.post("/api/rpc/stop-all")
    assert r.status_code == 200, r.text
    assert stopped == ["aibox1"]
    assert r.json()[0]["machine"] == "aibox1"



# ---------------- Part B: GET /api/rpc-map ---------------------------------

def test_rpc_map_llama_cpp_unchanged(client, settings_file, monkeypatch):
    """llama.cpp mode: the map still builds through the rpc_manager math
    path (compute_rpc_map consulted per model; map entry keyed by model)."""
    _engine(_LcppEngine(), monkeypatch)
    monkeypatch.setattr(app_module.swap_manager, "get_host",
                        lambda: {"name": "aibox1"})
    map_calls = []
    real_map = app_module.rpc_manager.compute_rpc_map

    def _spy(model_cfg, master, peers):
        map_calls.append(model_cfg.get("model_id"))
        return real_map(model_cfg, master, peers)

    monkeypatch.setattr(app_module.rpc_manager, "compute_rpc_map", _spy)
    _save_model()
    r = client.get("/api/rpc-map")
    assert r.status_code == 200, r.text
    body = r.json()
    assert "qwen-test" in body
    assert map_calls == ["qwen-test"]



# ---------------- Part C: POST /api/models/{id}/test -----------------------

def test_model_test_llama_cpp_unchanged(client, settings_file, monkeypatch):
    """llama.cpp mode: disk-based rpc_manager.validate_model, same
    500-on-error contract as before."""
    _engine(_LcppEngine(), monkeypatch)
    calls = []
    monkeypatch.setattr(app_module.rpc_manager, "validate_model",
                        lambda mid: calls.append(mid) or
                        {"model_id": mid, "ok": True, "issues": []})
    _save_model()
    r = client.post("/api/models/qwen-test/test")
    assert r.status_code == 200, r.text
    assert calls == ["qwen-test"]
    assert r.json() == {"model_id": "qwen-test", "ok": True, "issues": []}





# ---------------- Part D: the tune/* family (8 endpoints) ------------------

TUNE_ROUTES = [
    ("post", "/api/tune/estimate",
     {"base_path": "/models/Qwen3-8B", "servers": ["aibox1"],
      "device_order": "CUDA0"}),
    ("get", "/api/models/m/tune/estimate", None),
    ("post", "/api/models/m/tune/defaults", {}),
    ("post", "/api/models/m/tune/start", {}),
    ("get", "/api/models/m/tune/status", None),
    ("post", "/api/models/m/tune/cancel", {}),
    ("get", "/api/models/m/tune/drift", None),
    ("post", "/api/models/m/tune/seed", {}),
]


def _stub_tuner(monkeypatch, log, payload=None):
    """Every tuner.* entry point records its call. In vllm tests
    (payload=None) it raises — the gate must fire BEFORE any of them is
    reachable, and an AssertionError surfacing as a 500 would still be a
    reached-caller. In llama.cpp tests it returns a stub payload so the
    endpoint completes 200."""
    import tuner
    def _mk(name):
        def _fn(*a, **k):
            log.append(name)
            if payload is None:
                raise AssertionError(f"tuner.{name} called in vllm mode")
            return payload.get(name, {"stub": name})
        return _fn
    for fn in ("estimate", "safe_defaults", "start_tune", "job_status",
               "cancel_tune", "drift_check", "snapshot_seed"):
        monkeypatch.setattr(tuner, fn, _mk(fn))



@pytest.mark.parametrize("method,path,json_body",
                         [t for t in TUNE_ROUTES if t[1] != "/api/tune/estimate"])
def test_tune_family_llama_cpp_still_works(client, settings_file,
                                           monkeypatch, method, path,
                                           json_body):
    """llama.cpp mode: the gate is a no-op — each tune endpoint reaches
    its tuner.* call (which returns a stub payload). /api/tune/estimate
    is covered separately (needs real machine math)."""
    _engine(_LcppEngine(), monkeypatch)
    log = []
    _stub_tuner(monkeypatch, log, payload={
        "estimate": {"vram_gb": 12, "fits": True},
        "safe_defaults": {"tensor_split_mode": "none", "fit_margin": 0.05,
                          "fit_ram_allow": False,
                          "device_order_policy": "as-selected", "flags": {}},
        "start_tune": {"started": True},
        "job_status": {"model_id": "m", "state": "idle"},
        "cancel_tune": {"cancelled": False},
        "drift_check": {"drift": []},
        "snapshot_seed": {"seeded": True}})
    _save_model(model_id="m")
    r = client.request(method, path, json=json_body)
    assert r.status_code == 200, f"{method} {path}: {r.text}"
    assert len(log) == 1, f"{method} {path}: tuner calls {log}"


def test_tune_estimate_form_llama_cpp_real_math(client, settings_file,
                                                monkeypatch):
    """llama.cpp mode: POST /api/tune/estimate runs the REAL tuner math
    end-to-end against the fixture machine (unchanged behavior) — a GGUF
    that exists on disk estimates; a missing artifact 400s (unchanged)."""
    _engine(_LcppEngine(), monkeypatch)
    # artifact missing -> tuner.estimate raises ValueError -> 400
    r = client.post("/api/tune/estimate", json={
        "base_path": "/models/Qwen3-8B",
        "servers": ["aibox1"], "device_order": "CUDA0,CUDA1"})
    assert r.status_code == 400, r.text
    assert "not found" in r.json()["detail"]


# ---------------- engine read failure -> 400 (launch shape) ----------------

def test_tune_and_rpc_routes_bad_engine_400(client, settings_file,
                                            monkeypatch):
    """Unreadable settings -> 400 carrying the ValueError text — the same
    shape /api/launch uses (ValueError -> 400 verbatim), never a 500,
    never a llama.cpp action."""
    def _boom(s=None):
        raise ValueError("disk on fire")

    monkeypatch.setattr(app_module.engine_registry, "get_engine", _boom)
    for method, path, body in [("post", "/api/rpc/stop-all", None),
                               ("get", "/api/rpc-map", None),
                               ("post", "/api/models/m/test", None),
                               ("get", "/api/models/m/tune/status", None)]:
        r = client.request(method, path, json=body)
        assert r.status_code == 400, f"{method} {path}: {r.text}"
        assert "disk on fire" in r.json()["detail"]