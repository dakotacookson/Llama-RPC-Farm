"""B5a2: engine-aware status / models-save / launch endpoints in app.py.

Covers ARD §4 (endpoint changes) for the three B5a2 surfaces:
- GET /api/machines/{name}/status: llama.cpp mode returns the exact
  machine_status shape (byte-identical, no extra keys); vllm mode adds
  engine + vllm {running, detail} from the adapter probe on the
  llama-swap listen port.
- POST /api/models: vllm-mode adapter validation gates the save —
  errors -> 400 (no file written), warnings ride along in the response;
  the per-model vllm: section round-trips through save/load; llama.cpp
  mode keeps the byte-identical save path (no disk-based validation).
- POST /api/launch: vllm mode -> 409 with explanation (RPC-cluster
  launch is llama.cpp-only; vLLM routes through llama-swap).

Heavy externals (SSH probing, real HTTP /health) are monkeypatched; the
REAL config_loader round-trips the model YAML against tmp files.
"""
import os
import sys
import types

import pytest

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import app as app_module  # noqa: E402


# ---------------- fixtures ------------------------------------------------

@pytest.fixture
def settings_file(tmp_path, monkeypatch):
    """Redirect BOTH settings files into the pytest tmp dir so tests
    never read or write the repo's real config/. app_module.SETTINGS_FILE
    is what POST /api/settings writes; engine_registry.SETTINGS_FILE is
    what get_engine() reads — one tmp path for both keeps them coherent."""
    import engine_registry
    eng = tmp_path / "settings.yaml"
    eng.write_text("")  # empty -> default engine (llama_cpp)
    # BOTH module globals MUST point here: a test that POSTs /api/settings
    # writes through app_module.SETTINGS_FILE — if that still points at
    # the repo file, the suite corrupts the user's real settings.yaml.
    monkeypatch.setattr(app_module, "SETTINGS_FILE", str(eng))
    monkeypatch.setattr(engine_registry, "SETTINGS_FILE", str(eng))
    import access
    monkeypatch.setattr(access, "_settings_path", lambda: str(eng))

    swap = tmp_path / "swap.yaml"
    swap.write_text("listen_port: 8123\n")
    import swap_manager
    monkeypatch.setattr(swap_manager, "SWAP_SETTINGS_FILE", str(swap))

    machines = tmp_path / "machines.yaml"
    machines.write_text(
        "machines:\n"
        "- id: aibox1\n"
        "  name: aibox1\n"
        "  ip: 127.0.0.1\n"
        "  role: master\n"
        "  os: linux\n"
        "  rpc_port: 50052\n"
        "  gpu: CUDA0,CUDA1\n"
        "  gpu_vendor: nvidia\n"
        "  detected:\n"
        "    gpu_names: [GPU0, GPU1]\n"
        "    vram_total_gb: 24.0\n"
        "    ram_total_gb: 64\n"
        "  expose: {mode: both, gpus: []}\n"
    )
    import config_loader
    # _YamlCollection reads self.path at call time -> point the machines
    # collection at the tmp file (no module-global filename attr exists)
    monkeypatch.setattr(config_loader.machines, "path", str(machines))
    import rpc_manager
    monkeypatch.setattr(rpc_manager, "cfg", config_loader)

    # models dir -> tmp (save_model/get_model resolve through it)
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    monkeypatch.setattr(config_loader, "MODELS_DIR", str(models_dir))
    monkeypatch.setattr(app_module.cfg, "MODELS_DIR", str(models_dir))

    # groups: app's save_model checks cfg.groups.get — empty map
    monkeypatch.setattr(config_loader, "groups",
                        types.SimpleNamespace(all=lambda: [],
                                              get=lambda n: None))
    yield str(eng)


@pytest.fixture
def client(settings_file):
    from fastapi.testclient import TestClient
    return TestClient(app_module.app)


def _set_engine(client, engine, tmp_file):
    """Persist the engine via the REAL POST /api/settings (B5a1)."""
    with open(tmp_file, "w", encoding="utf-8") as f:
        f.write(f"engine: {engine}\n")
    # force a fresh read (get_engine() with no args loads the file)
    import engine_registry


# ---------------- GET /api/machines/{name}/status ---------------------------

def test_machine_status_llama_cpp_unchanged(client, settings_file, monkeypatch):
    """llama.cpp mode: the endpoint returns rpc_manager.machine_status
    output EXACTLY as before — no engine key, no probe call."""
    probe_calls = []

    class _Engine:
        # DEC-04: save_model dispatches section round-trip through the
        # adapter; test doubles carry the protocol members (no-op for
        # llama_cpp doubles, pass-through matching VllmAdapter shapes).
        def apply_model_section(self, data, where="vllm"):
            return data

        def sanitize_model_section(self, raw, where="vllm"):
            if raw is None:
                return None
            return dict(raw)

        name = "llama_cpp"

        def capable(self, capability):
            return capability == "rpc_cluster"

        def probe(self, *a, **k):
            probe_calls.append(a)
            raise AssertionError("llama_cpp mode must not probe /health")

    monkeypatch.setattr(app_module.engine_registry, "get_engine",
                        lambda s=None: _Engine())

    def fake_status(name):
        return {"name": name, "ip": "127.0.0.1", "reachable": True,
                "rpc_running": True, "gpus": [], "is_local": True,
                "is_master": True}

    monkeypatch.setattr(app_module.rpc_manager, "machine_status", fake_status)
    r = client.get("/api/machines/aibox1/status")
    assert r.status_code == 200
    body = r.json()
    assert body == {"name": "aibox1", "ip": "127.0.0.1", "reachable": True,
                    "rpc_running": True, "gpus": [], "is_local": True,
                    "is_master": True}
    assert "engine" not in body and "vllm" not in body
    assert probe_calls == []




def test_machine_status_unknown_machine_404(client, settings_file, monkeypatch):
    class _Engine:
        # DEC-04: save_model dispatches section round-trip through the
        # adapter; test doubles carry the protocol members (no-op for
        # llama_cpp doubles, pass-through matching VllmAdapter shapes).
        def apply_model_section(self, data, where="vllm"):
            return data

        def sanitize_model_section(self, raw, where="vllm"):
            if raw is None:
                return None
            return dict(raw)

        name = "llama_cpp"

        def capable(self, capability):
            return capability == "rpc_cluster"

    monkeypatch.setattr(app_module.engine_registry, "get_engine",
                        lambda s=None: _Engine())

    def boom(name):
        raise ValueError(f"machine '{name}' not found")

    monkeypatch.setattr(app_module.rpc_manager, "machine_status", boom)
    r = client.get("/api/machines/nope/status")
    assert r.status_code == 404


# ---------------- POST /api/models ------------------------------------------

def _model_payload(**over):
    base = {
        "model_id": "Qwen3-8B-vllm",
        "name": "Qwen3 8B (vLLM)",
        "base_path": "/models/Qwen3-8B",
        "servers": ["aibox1"],
        "device_order": "aibox1-CUDA0",
        "flags": {"ctx-size": 32768},
        "vllm": {"max_model_len": 32768, "max_num_seqs": 256},
    }
    base.update(over)
    return base






def test_save_model_llama_cpp_path_unchanged(client, settings_file,
                                              monkeypatch):
    """llama.cpp mode: NO adapter validation runs on save (its checks are
    disk-based, POST /api/models/{id}/test owns them) — a payload with
    zero servers saves fine, byte-identical to pre-B5a2 behavior. The
    vllm: key is absent from both request and response."""
    class _Engine:
        # DEC-04: save_model dispatches section round-trip through the
        # adapter; test doubles carry the protocol members (no-op for
        # llama_cpp doubles, pass-through matching VllmAdapter shapes).
        def apply_model_section(self, data, where="vllm"):
            return data

        def sanitize_model_section(self, raw, where="vllm"):
            if raw is None:
                return None
            return dict(raw)

        name = "llama_cpp"

        def capable(self, capability):
            return capability == "rpc_cluster"

        def validate_model_config(self, cfg):
            raise AssertionError(
                "llama_cpp save path must not call validate_model_config")

    monkeypatch.setattr(app_module.engine_registry, "get_engine",
                        lambda s=None: _Engine())
    monkeypatch.setattr(app_module.swap_manager, "status",
                        lambda: {"running": False})
    r = client.post("/api/models", json=_model_payload(vllm=None, servers=[]))
    assert r.status_code == 200, r.text
    body = r.json()
    assert "_validation_warnings" in body
    assert body.get("vllm") is None



# ---------------- DEC-09: save-path translation retired -----------------
# B11c's auto-translation (engine_config_translate.to_vllm / to_llamacpp
# on POST /api/models) is deleted per the user directive: zero
# cross-engine warning groups — separation, not better warnings. Each
# engine saves its own vocabulary verbatim. Replaced below by tests that
# pin the new contract: no translation, no _translation_warnings key.




# ---------------- POST /api/launch ------------------------------------------


def test_launch_llama_cpp_mode_still_works(client, settings_file, monkeypatch):
    """llama.cpp mode: the 409 gate is inert — launch still reaches
    rpc_manager.launch_model with the same arguments."""
    calls = {}

    def fake_launch(model_id, servers, port, variant=None, master_name=None,
                    mode="rpc"):
        calls["args"] = (model_id, servers, port, variant, master_name,
                         mode)
        return {"ok": True, "failed": False}

    monkeypatch.setattr(app_module.rpc_manager, "launch_model", fake_launch)
    r = client.post("/api/launch",
                    json={"model_id": "Qwen3-8B", "servers": ["aibox1"],
                          "port": 8080})
    assert r.status_code == 200
    # unconfigured id: auto falls back to the legacy rpc default
    assert calls["args"] == ("Qwen3-8B", ["aibox1"], 8080, None, None, "rpc")



# ---------------- settings round-trip keeps the engine switch working -------


# ---------------- GET /api/machines (B8/B14: vllm capability payload) --------

def _add_windows_machine(settings_file, monkeypatch):
    """Extend the fixture machines.yaml with a windows peer (winbox)."""
    import config_loader
    machines = config_loader.machines.path
    with open(machines, "a", encoding="utf-8") as f:
        f.write(
            "- id: winbox\n"
            "  name: winbox\n"
            "  ip: 10.0.0.124\n"
            "  role: peer\n"
            "  os: windows\n"
            "  rpc_port: 50052\n"
            "  gpu: CUDA0\n"
            "  gpu_vendor: nvidia\n"
        )
    return machines


def _add_machine(machines_file, **kw):
    """Append one machine entry to the fixture machines.yaml; unknown
    fields default like the add-machine wizard does (os linux, vendor
    nvidia). Returns the file path."""
    fields = {"role": "peer", "os": "linux", "rpc_port": 50052, **kw}
    lines = [f"- id: {fields['name']}", f"  name: {fields['name']}",
             f"  ip: {fields.get('ip', '10.0.0.199')}"]
    for k, v in fields.items():
        if k in ("name", "ip"):
            continue
        if isinstance(v, str):
            lines.append(f"  {k}: {v}")
        else:
            lines.append(f"  {k}: {v}")
    with open(machines_file, "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return machines_file






# ---------------- B14: full hardware capability payload (audit §3b) ----------









_CANNED_REPORT = {
    "machine": "aibox1",
    "delivery": "venv",
    "target": "~/vllm-env/bin/vllm",
    "steps": [
        {"step": "gate:os", "detail": "os linux: supported"},
        {"step": "verify:existing", "detail": "~/vllm-env/bin/vllm not "
                                             "usable yet (no output)"},
        {"step": "gate:gpu", "detail": "nvidia driver ok: 550.54.15, RTX 3090"},
        {"step": "venv:create", "detail": "venv ready: ~/vllm-env via uv"},
        {"step": "venv:install", "detail": "installed vllm==0.30.0 into "
                                           "~/vllm-env"},
        {"step": "verify:version", "detail": "vLLM 0.30.0"},
        {"step": "settings:update", "detail": "settings vllm.venv_bin -> "
                                              "~/vllm-env/bin/vllm"},
    ],
    "failed": False,
    "ok": True,
    "detail": "install complete",
}

_WINDOWS_REFUSAL_REPORT = {
    "machine": "winbox",
    "delivery": "venv",
    "steps": [{"step": "gate:os",
               "failed": True,
               "detail": "vLLM cannot run natively on Windows — install "
                         "WSL2, add the WSL instance to the farm as a "
                         "linux machine, and install vLLM there"}],
    "failed": True,
    "ok": False,
    "detail": "vLLM cannot run natively on Windows — install WSL2, add "
              "the WSL instance to the farm as a linux machine, and "
              "install vLLM there",
}







_CANNED_UNINSTALL_REPORT = {
    "machine": "aibox1",
    "delivery": "venv",
    "target": "~/vllm-env/bin/vllm",
    "steps": [
        {"step": "guard:path", "detail": "venv root ~/vllm-env is "
                                         "farm-managed"},
        {"step": "venv:remove", "detail": "venv ~/vllm-env removed"},
        {"step": "settings:clear", "detail": "settings vllm.venv_bin "
                                             "cleared (was "
                                             "~/vllm-env/bin/vllm)"},
    ],
    "failed": False,
    "ok": True,
    "detail": "vLLM uninstall complete",
}









def test_tools_verify_shape_and_not_found(client, settings_file, monkeypatch):
    """GET /api/tools/verify: read-only status for the three tools with
    the §8 contract shape {found, path, version, expected, configured}.
    Empty tmp settings -> nothing configured, nothing found (tmp home has
    no tool binaries — the RealFS probe runs against the test HOME)."""
    import os
    # RealFS expanduser resolves through os.path.expanduser; pin HOME so
    # the well-known locations (~/.local/bin/...) land in the tmp dir.
    # (monkeypatch fixture: HOME is restored after the test — an unmanaged
    # MonkeyPatch here leaked into later tests and broke installer-probe.)
    monkeypatch.setenv("HOME", os.path.dirname(settings_file))
    r = client.get("/api/tools/verify")
    assert r.status_code == 200, r.text
    body = r.json()
    assert set(body) == {"llama.cpp", "llama-swap", "vllm", "sshpass"}
    for tool, entry in body.items():
        assert set(entry) == {"found", "path", "version", "expected",
                              "configured"}, (tool, entry)
    # tool probes are pinned to the empty tmp HOME; sshpass comes from PATH
    # (optional system binary) so only the three engines are asserted absent.
    for tool in ("llama.cpp", "llama-swap", "vllm"):
        assert body[tool]["found"] is False, (tool, body[tool])
    assert body["sshpass"]["expected"] == "optional (SSH transport)"
    assert body["llama.cpp"]["configured"] is None
    assert body["vllm"]["expected"] == "0.30.0"
    assert body["llama-swap"]["expected"] == "255+"
    assert "b10000" in body["llama.cpp"]["expected"]


def test_tools_verify_settings_record_counts(client, settings_file,
                                               monkeypatch):
    """A configured path that EXISTS counts as found even when the
    well-known locations miss — the settings record (§6 prefill) is
    probed as fallback and its --version becomes the version."""
    # write a fake vllm binary + point settings at it
    import os, stat, yaml as _yaml
    # pin HOME like the shape test so well-known locations miss and the
    # settings record is the only thing the probe can find (fixture-
    # scoped: restored after the test, no leakage into installer-probe)
    monkeypatch.setenv("HOME", os.path.dirname(settings_file))
    bindir = os.path.join(os.path.dirname(settings_file), "bin")
    os.makedirs(bindir, exist_ok=True)
    fake = os.path.join(bindir, "vllm")
    with open(fake, "w") as f:
        f.write("#!/bin/sh\necho 'vLLM 0.30.0'\n")
    os.chmod(fake, 0o755)
    with open(settings_file, "w") as f:
        _yaml.safe_dump({"vllm": {"venv_bin": fake}}, f)
    r = client.get("/api/tools/verify")
    assert r.status_code == 200
    v = r.json()["vllm"]
    assert v["found"] is True, v
    assert v["path"] == fake
    assert v["version"] == "0.30.0"
    assert v["configured"] == fake
