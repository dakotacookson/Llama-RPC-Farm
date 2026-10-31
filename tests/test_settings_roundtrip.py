"""B5a1: settings engine+vllm round-trip through /api/settings.

Covers ARD §2 / §4: GET returns the engine schema (defaults injected,
ui_hints attached), POST persists engine + vllm block and returns the
normalized dict, and validation failures surface as 400 — never silent
fallbacks. Uses the REAL config_loader (inject_engine_defaults /
validate_engine) against a temp settings file; the repo's own
config/settings.yaml is never read or written.
"""
import os
import sys
import yaml

import pytest

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import app as app_module  # noqa: E402


@pytest.fixture
def settings_file(tmp_path):
    """Redirect the app's SETTINGS_FILE into the pytest tmp dir."""
    path = tmp_path / "settings.yaml"
    app_module.SETTINGS_FILE = str(path)
    yield path
    app_module.SETTINGS_FILE = os.path.join(
        app_module.BASE_DIR, "config", "settings.yaml")


@pytest.fixture
def client():
    from fastapi.testclient import TestClient
    return TestClient(app_module.app)


@pytest.fixture(autouse=True)
def vault_env(tmp_path, monkeypatch):
    """Isolate the secrets vault: ENV_PATH -> tmp, secrets stripped from
    os.environ. POST /api/settings writes tokens through secrets_vault;
    without this fixture the write lands in the repo's REAL .env and
    re-encrypts (destroys) the user's machine passwords + HF token.
    secrets_vault is only ever accessed as a module (lazy `import
    secrets_vault` inside app.py), so patching ENV_PATH here covers every
    call site in the request handlers."""
    import secrets_vault
    monkeypatch.setattr(secrets_vault, "ENV_PATH", str(tmp_path / "vault.env"))
    for key in list(os.environ):
        if key.startswith(("FARM_", "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN",
                           "HF_API_TOKEN", "MACHINE_")):
            monkeypatch.delenv(key, raising=False)


def _read(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _write(path, data):
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)


# ---------------- GET ----------------

def test_get_injects_engine_defaults(settings_file, client):
    """A settings file with no engine keys gets the default engine +
    ui_manifest (DEC-02: the engine-owned field inventory the frontend
    renders from) + the deprecated ui_hints projection (DEC-01: the
    vllm: block is engine-qualified — a llama_cpp settings file carries
    NO vllm vocabulary at all; spec §1.4/§3)."""
    _write(settings_file, {"model_dir": "store://llm-models",
                           "llama_bin": "~/llama.cpp/build/bin"})
    r = client.get("/api/settings")
    assert r.status_code == 200
    body = r.json()
    assert body["engine"] == "llama_cpp"
    # vllm: block is engine-qualified: absent in llama_cpp mode
    assert "vllm" not in body
    assert body["model_dir"] == "store://llm-models"
    assert body["hf_token_set"] is False
    man = body["ui_manifest"]
    assert man["labels"]["server"] == "llama-server"
    assert man["hidden_panels"] == []
    assert man["launch"]["mode"] == "rpc_cluster"
    assert man["downloads"]["formats"] == ["gguf"]
    # deprecated shim projects the manifest
    hints = body["ui_hints"]
    assert hints["labels"]["server"] == "llama-server"
    assert hints["hidden_panels"] == []


def test_get_picks_active_engine_and_its_manifest(settings_file, client):
    """Stored engine=vllm -> vLLM manifest (labels + hidden llama.cpp-only
    panels) with the deprecated ui_hints view riding along."""
    _write(settings_file, {"engine": "vllm"})
    body = client.get("/api/settings").json()
    assert body["engine"] == "vllm"
    man = body["ui_manifest"]
    assert man["labels"]["server"] == "vLLM server"
    assert "rpc_peers" in man["hidden_panels"]
    assert man["launch"]["mode"] == "swap_only"
    hints = body["ui_hints"]
    assert hints["labels"]["server"] == "vLLM server"
    assert "rpc_peers" in hints["hidden_panels"]


def test_get_rejects_bad_engine_in_file(settings_file, client):
    """A typo in settings.yaml must NOT fall back silently — 400 with the
    valid values in the detail (ARD §2)."""
    _write(settings_file, {"engine": "villm"})
    r = client.get("/api/settings")
    assert r.status_code == 400
    assert "villm" in r.json()["detail"]
    assert "llama_cpp, vllm" in r.json()["detail"]


def test_get_rejects_unknown_vllm_key(settings_file, client):
    """DEC-01: the vllm: block is validated by its owning adapter — only
    when vllm is the ACTIVE engine. In llama_cpp mode foreign vocabulary
    on disk is dead-but-tolerated (spec §3): GET serves it verbatim, the
    next active-engine save scrubs it (never a validation error)."""
    _write(settings_file, {"engine": "vllm", "vllm": {"nope": 1}})
    assert client.get("/api/settings").status_code == 400
    # the same key under llama_cpp does NOT fail the read
    _write(settings_file, {"engine": "llama_cpp", "vllm": {"nope": 1}})
    body = client.get("/api/settings").json()
    assert body["vllm"] == {"nope": 1}  # dead-but-tolerated, passed through


# ---------------- POST ----------------

def test_post_engine_persists(settings_file, client):
    _write(settings_file, {"model_dir": "store://llm-models"})
    r = client.post("/api/settings", json={"engine": "vllm"})
    assert r.status_code == 200
    body = r.json()
    assert body["engine"] == "vllm"
    # vllm block fully defaulted in both the response and the file
    assert body["vllm"]["delivery"] == "venv"
    assert body["vllm"]["default_gpu_memory_utilization"] == 0.9
    on_disk = _read(settings_file)
    assert on_disk["engine"] == "vllm"
    assert on_disk["vllm"]["delivery"] == "venv"


def test_post_vllm_block_round_trip(settings_file, client):
    _write(settings_file, {"engine": "vllm"})
    r = client.post("/api/settings", json={
        "vllm": {"delivery": "docker", "docker_image": "vllm/vllm-openai:v0"}})
    assert r.status_code == 200
    body = r.json()
    # partial update merged over stored block; missing keys defaulted
    assert body["vllm"]["delivery"] == "docker"
    assert body["vllm"]["docker_image"] == "vllm/vllm-openai:v0"
    assert body["vllm"]["venv_bin"] == "~/vllm-env/bin/vllm"
    assert body["vllm"]["default_gpu_memory_utilization"] == 0.9
    # docker_image survives untouched on disk
    assert _read(settings_file)["vllm"]["docker_image"] == "vllm/vllm-openai:v0"


def test_post_bad_engine_400_file_untouched(settings_file, client):
    _write(settings_file, {"model_dir": "keep"})
    r = client.post("/api/settings", json={"engine": "llamacpp"})
    assert r.status_code == 400
    assert "llama_cpp, vllm" in r.json()["detail"]
    assert _read(settings_file) == {"model_dir": "keep"}


def test_post_unknown_vllm_key_400(settings_file, client):
    _write(settings_file, {"engine": "vllm"})  # block validated by its owner
    r = client.post("/api/settings", json={"vllm": {"bogus": True}})
    assert r.status_code == 400
    assert "bogus" in r.json()["detail"]


def test_post_bad_delivery_400(settings_file, client):
    _write(settings_file, {"engine": "vllm"})  # block validated by its owner
    r = client.post("/api/settings", json={"vllm": {"delivery": "pip"}})
    assert r.status_code == 400
    assert "pip" in r.json()["detail"]


def test_post_bad_gpu_memory_utilization_400(settings_file, client):
    _write(settings_file, {"engine": "vllm"})  # block validated by its owner
    for bad in (0, 1.5, "0.9", True):
        r = client.post("/api/settings",
                        json={"vllm": {"default_gpu_memory_utilization": bad}})
        assert r.status_code == 400, bad


def test_post_vllm_partial_merge_keeps_stored_keys(settings_file, client):
    """A second POST changing only delivery must not drop stored keys."""
    _write(settings_file, {"engine": "vllm", "vllm": {"venv_bin": "/custom/bin/vllm"}})
    r = client.post("/api/settings", json={"vllm": {"delivery": "docker"}})
    assert r.status_code == 200
    assert r.json()["vllm"]["venv_bin"] == "/custom/bin/vllm"


def test_post_engine_and_vllm_together(settings_file, client):
    r = client.post("/api/settings", json={
        "engine": "vllm", "vllm": {"delivery": "docker"}})
    assert r.status_code == 200
    body = r.json()
    assert body["engine"] == "vllm"
    assert body["vllm"]["delivery"] == "docker"


def test_post_omitted_engine_leaves_stored(settings_file, client):
    _write(settings_file, {"engine": "vllm"})
    r = client.post("/api/settings", json={"theme_family": "steel"})
    assert r.status_code == 200
    body = r.json()
    assert body["engine"] == "vllm"  # untouched
    assert body["theme_family"] == "steel"
    assert _read(settings_file)["engine"] == "vllm"


def test_round_trip_get_matches_post(settings_file, client):
    """POST then GET — the normalized on-disk state is what GET serves."""
    client.post("/api/settings", json={
        "engine": "vllm", "vllm": {"default_gpu_memory_utilization": 0.8}})
    got = client.get("/api/settings").json()
    assert got["engine"] == "vllm"
    assert got["vllm"]["default_gpu_memory_utilization"] == 0.8


# ---------------- backward compat (other fields) ----------------

def test_post_other_fields_still_work(settings_file, client):
    """Merged semantics (vault is the source of truth for tokens): a POST
    with hf_token stores it Fernet-encrypted in the vault .env, never in
    settings.yaml; a legacy plaintext token sitting in YAML is removed on
    the next POST; clear_hf_token empties the vault entry."""
    import secrets_vault
    _write(settings_file, {"hf_token": "old-token"})  # legacy YAML residue
    r = client.post("/api/settings", json={"hf_token": "new-token"})
    assert r.status_code == 200
    assert r.json()["hf_token_set"] is True
    assert "hf_token" not in r.json()  # never echoed
    on_disk = _read(settings_file)
    assert "hf_token" not in on_disk  # vault-only: YAML never persists it
    # the vault .env holds the Fernet token, decryptable, never plaintext
    with open(secrets_vault.ENV_PATH, "r", encoding="utf-8") as f:
        env_text = f.read()
    assert "HF_TOKEN_ENC=" in env_text
    assert "new-token" not in env_text
    assert secrets_vault.get_hf_token() == "new-token"
    # clearing removes the vault entry entirely
    r = client.post("/api/settings", json={"clear_hf_token": True})
    assert r.status_code == 200
    assert r.json()["hf_token_set"] is False
    assert "HF_TOKEN_ENC" not in open(secrets_vault.ENV_PATH).read()
    assert "hf_token" not in _read(settings_file)


def test_get_hf_token_set_reflects_vault(settings_file, client):
    """GET's hf_token_set mirrors the vault state (empty vault -> False,
    populated vault -> True). With no vault entry the GET falls back to a
    legacy YAML token for compat (covered by the POST test's cleanup)."""
    import secrets_vault
    _write(settings_file, {"model_dir": "store://llm-models"})
    body = client.get("/api/settings").json()
    assert body["hf_token_set"] is False  # empty isolated vault
    secrets_vault.set_hf_token("vault-token")
    body = client.get("/api/settings").json()
    assert body["hf_token_set"] is True