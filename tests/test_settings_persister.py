"""B9c: settings.yaml round-trip persister (theme switch preserves keys + comments).

Bug (observed live 2026-09-24): switching themes in the UI POSTs to
/api/settings, whose writer did `yaml.safe_dump(current)` — losing every
comment and commented-out block in config/settings.yaml (the B1 engine/vllm
docs block was wiped on the first theme change). The persister now round-trips
through ruamel.yaml (comment-preserving), and unknown top-level keys survive
verbatim. The repo's own config/settings.yaml is never read or written here.
"""
import os
import sys

import pytest
import yaml

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


MARKER_COMMENT = (
    "# Engine selection (ARD: feature/vllm-support).\n"
    "#   llama_cpp — default. RPC clusters of llama-server across machines.\n"
    "# A missing key means llama_cpp; a typo is a STARTUP ERROR listing the\n"
    "# valid values (never a silent fallback).\n"
)

SAMPLE_SETTINGS = (
    "model_dir: store://llm-models\n"
    "llama_bin: ~/llama.cpp/build/bin\n"
    "theme_family: arctic\n"
    "theme_mode: light\n"
    "\n"
    + MARKER_COMMENT +
    "#engine: llama_cpp\n"
    "\n"
    "legacy_unknown_key: keep-me\n"
    "nested_unknown:\n"
    "  a: 1\n"
    "  b: [x, y]\n"
)


def _write_raw(path, text):
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def _read_raw(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


# ---------------- comment + unknown-key preservation ----------------

def test_theme_post_preserves_comments(settings_file, client):
    """A theme switch (POST with only model_dir/llama_bin/theme_*) must
    keep every comment in settings.yaml, including the commented-out
    engine/vllm doc block — the exact live bug from 2026-09-24."""
    _write_raw(settings_file, SAMPLE_SETTINGS)
    r = client.post("/api/settings", json={
        "theme_family": "nord", "theme_mode": "dark"})
    assert r.status_code == 200
    text = _read_raw(settings_file)
    # the marker comment block survived
    assert MARKER_COMMENT.strip().splitlines()[0] in text
    assert "STARTUP ERROR" in text
    # the commented-out (documentation) keys survived
    assert "#engine: llama_cpp" in text
    # the changed key actually changed
    assert "theme_family: nord" in text
    assert "theme_family: arctic" not in text


def test_theme_post_preserves_unknown_keys(settings_file, client):
    """Unknown/future top-level keys (flat and nested) survive a theme POST
    verbatim — never a wholesale rewrite from the request dict."""
    _write_raw(settings_file, SAMPLE_SETTINGS)
    r = client.post("/api/settings", json={
        "theme_family": "nord", "theme_mode": "dark"})
    assert r.status_code == 200
    on_disk = yaml.safe_load(_read_raw(settings_file))
    assert on_disk["legacy_unknown_key"] == "keep-me"
    assert on_disk["nested_unknown"] == {"a": 1, "b": ["x", "y"]}
    # and the schema keys are still normalized into the file
    assert on_disk["engine"] == "llama_cpp"
    # DEC-01: no vllm vocabulary in llama_cpp mode (engine-qualified)
    assert "vllm" not in on_disk


def test_persister_helper_round_trips_comments(tmp_path):
    """Direct unit test of the round-trip helper: load with comments, mutate
    one key, write — comments and key order are byte-stable."""
    p = tmp_path / "settings.yaml"
    _write_raw(p, SAMPLE_SETTINGS)
    data = app_module._load_settings_yaml_preserving(str(p))
    data["theme_mode"] = "dark"
    app_module._dump_settings_yaml_preserving(data, str(p))
    text = _read_raw(p)
    assert "STARTUP ERROR" in text
    assert "#engine: llama_cpp" in text
    assert "theme_mode: dark" in text
    assert "legacy_unknown_key: keep-me" in text
    # key order preserved (ruamel round-trip property)
    assert text.index("model_dir:") < text.index("llama_bin:") \
        < text.index("theme_family:")
    # idempotent: another round trip changes nothing
    data2 = app_module._load_settings_yaml_preserving(str(p))
    app_module._dump_settings_yaml_preserving(data2, str(p))
    assert _read_raw(p) == text


def test_persister_helper_creates_file_when_missing(tmp_path):
    """First save with no settings file: writes a valid mapping."""
    p = tmp_path / "settings.yaml"
    app_module._dump_settings_yaml_preserving({"engine": "vllm"}, str(p))
    assert yaml.safe_load(_read_raw(p))["engine"] == "vllm"


def test_persister_helper_falls_back_without_ruamel(tmp_path, monkeypatch):
    """If ruamel.yaml is missing (bare install), the persister degrades to
    plain safe_dump — keys still survive, comments are accepted as lost."""
    ruamel = pytest.importorskip("ruamel.yaml", minversion=None)  # noqa: F841
    import builtins
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name.startswith("ruamel"):
            raise ImportError("no ruamel here")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    p = tmp_path / "settings.yaml"
    _write_raw(p, SAMPLE_SETTINGS)
    data = app_module._load_settings_yaml_preserving(str(p))
    data["theme_family"] = "steel"
    app_module._dump_settings_yaml_preserving(data, str(p))
    on_disk = yaml.safe_load(_read_raw(p))
    assert on_disk["theme_family"] == "steel"
    assert on_disk["legacy_unknown_key"] == "keep-me"


def test_ruamel_is_a_declared_requirement():
    """requirements.txt must declare ruamel.yaml — the comment-preserving
    persister is not optional on a real install."""
    req = os.path.join(app_module.BASE_DIR, "requirements.txt")
    with open(req, "r", encoding="utf-8") as f:
        text = f.read()
    assert "ruamel.yaml" in text


# ---------------- engine/vllm POSTs keep the fix ----------------

def test_engine_post_preserves_comments_and_unknown_keys(settings_file, client):
    """The comment-preserving round trip applies to every POST, not just
    theme switches."""
    _write_raw(settings_file, SAMPLE_SETTINGS)
    r = client.post("/api/settings", json={"engine": "vllm"})
    assert r.status_code == 200
    text = _read_raw(settings_file)
    assert "STARTUP ERROR" in text
    assert "legacy_unknown_key: keep-me" in text
    on_disk = yaml.safe_load(text)
    assert on_disk["engine"] == "vllm"
    assert on_disk["vllm"]["delivery"] == "venv"


def test_legacy_yaml_token_still_removed_on_post(settings_file, client):
    """The vault-only rule from B9b still holds with the new persister."""
    _write_raw(settings_file, SAMPLE_SETTINGS + 'hf_token: "old-token"\n')
    r = client.post("/api/settings", json={"theme_family": "steel"})
    assert r.status_code == 200
    on_disk = yaml.safe_load(_read_raw(settings_file))
    assert "hf_token" not in on_disk
    assert on_disk["legacy_unknown_key"] == "keep-me"