"""First-run setup: a fresh install has no web login and no access key.

The UI must offer "create admin" (not an unusable sign-in), and the
bootstrap endpoint must create the first admin login + key. These tests
isolate the vault and settings files, so they behave identically on any
host.
"""
import os
import sys

import pytest
import yaml

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import app as app_module          # noqa: E402
import access                     # noqa: E402
import engine_registry            # noqa: E402
import secrets_vault              # noqa: E402


@pytest.fixture
def fresh_install(tmp_path, monkeypatch):
    settings = tmp_path / "settings.yaml"
    settings.write_text("engine: llama_cpp\n", encoding="utf-8")
    monkeypatch.setattr(app_module, "SETTINGS_FILE", str(settings))
    monkeypatch.setattr(engine_registry, "SETTINGS_FILE", str(settings))
    monkeypatch.setenv("FARM_SETTINGS_FILE", str(settings))
    monkeypatch.setattr(access, "_settings_path", lambda: str(settings))

    vault = tmp_path / "vault.env"
    monkeypatch.setattr(secrets_vault, "ENV_PATH", str(vault))
    # Generate a fresh vault key/salt in the tmp vault, never touch the
    # developer's live .env.
    monkeypatch.delenv("FARM_SECRET_KEY", raising=False)
    monkeypatch.delenv("FARM_SALT", raising=False)
    for key in list(os.environ):
        if key.startswith(("LOGIN_", "WEB_SESSION_")):
            monkeypatch.delenv(key, raising=False)
    return settings, vault


@pytest.fixture
def client(fresh_install):
    from fastapi.testclient import TestClient
    return TestClient(app_module.app)


def test_fresh_install_reports_needs_setup(client):
    body = client.get("/api/auth/me").json()
    assert body["has_logins"] is False
    assert body["has_keys"] is False
    assert body["needs_setup"] is True


def test_existing_key_disables_setup_prompt(fresh_install):
    settings, _ = fresh_install
    settings.write_text(
        "engine: llama_cpp\n"
        "access_keys:\n"
        "  sk-existing:\n"
        "    name: operator\n"
        "    models: []\n"
        "    admin: true\n",
        encoding="utf-8",
    )
    from fastapi.testclient import TestClient
    body = TestClient(app_module.app).get("/api/auth/me").json()
    assert body["has_keys"] is True
    assert body["needs_setup"] is False


def test_first_bootstrap_creates_admin_login_and_key(client, fresh_install):
    settings, _ = fresh_install
    r = client.post("/api/auth/bootstrap", json={
        "username": "admin", "password": "long-enough-pass", "key_name": "admin",
    })
    assert r.status_code == 200, r.text

    me = client.get("/api/auth/me").json()
    assert me["has_logins"] is True
    assert me["needs_setup"] is False
    assert me["user"] == "admin"
    assert me["logged_in"] is True

    data = yaml.safe_load(settings.read_text(encoding="utf-8")) or {}
    keys = data.get("access_keys") or {}
    assert any(v.get("name") == "admin" and v.get("admin") is True
               for v in keys.values())


def test_bootstrap_rejects_short_password(client):
    r = client.post("/api/auth/bootstrap", json={
        "username": "admin", "password": "short",
    })
    assert r.status_code == 400
    assert client.get("/api/auth/me").json()["has_logins"] is False
