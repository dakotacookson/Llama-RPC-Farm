"""Machine secret hygiene: plaintext secrets must never reach machines.yaml,
the in-memory machine store, or the /api/machines response."""
import pytest
import yaml

import config_loader
import routes.machines as machines_routes


@pytest.fixture(autouse=True)
def _hermetic_vault(monkeypatch):
    """Vault side effects must not touch the developer's real .env."""
    import secrets_vault
    monkeypatch.setattr(secrets_vault, "get_machine_password", lambda m: "")
    monkeypatch.setattr(secrets_vault, "set_machine_password",
                        lambda mid, pw: None)
    monkeypatch.setattr(secrets_vault, "machine_agent_token_stored",
                        lambda m: False)
    monkeypatch.setattr(secrets_vault, "set_machine_agent_token",
                        lambda mid, tok: None)


def _write_store(tmp_path, monkeypatch, records):
    store = tmp_path / "machines.yaml"
    store.write_text(yaml.safe_dump({"machines": records}), encoding="utf-8")
    monkeypatch.setattr(config_loader.machines, "path", str(store))
    return store


def _read_store(store):
    return yaml.safe_load(store.read_text(encoding="utf-8"))["machines"]


def test_agent_token_never_reaches_machine_yaml(monkeypatch):
    out = config_loader._strip_machine_secret({
        "name": "m1", "id": "m1", "_agent_token": "super-secret",
        "password": "pw",
    })
    assert "_agent_token" not in out
    assert out["password"] == ""


def test_reads_never_include_password_or_token(tmp_path, monkeypatch):
    """The store read path must not resolve the vault password into records."""
    _write_store(tmp_path, monkeypatch, [
        {"name": "m1", "id": "m1", "_agent_token": "legacy", "password": "pw"},
    ])
    rows = config_loader.machines.all()
    assert "password" not in rows[0]
    assert "_agent_token" not in rows[0]
    single = config_loader.machines.get("m1")
    assert "password" not in single and "_agent_token" not in single


def test_update_drops_legacy_and_incoming_agent_tokens(tmp_path, monkeypatch):
    store = _write_store(tmp_path, monkeypatch, [
        {"name": "m1", "id": "m1", "_agent_token": "legacy", "password": "",
         "password_set": False, "agent_token_set": True},
    ])
    config_loader.machines.update("m1", {
        "ip": "10.0.0.9", "_agent_token": "new-secret"})
    saved = _read_store(store)[0]
    assert "_agent_token" not in saved
    assert "password_set" not in saved
    assert "agent_token_set" not in saved
    assert saved["ip"] == "10.0.0.9"


def test_migration_vaults_then_strips_legacy_token(tmp_path, monkeypatch):
    import secrets_vault
    store = _write_store(tmp_path, monkeypatch, [
        {"name": "m1", "id": "m1", "_agent_token": "legacy", "password": ""},
    ])
    vault = {}
    monkeypatch.setattr(secrets_vault, "machine_agent_token_stored",
                        lambda m: (m.get("id") or m.get("name")) in vault)
    monkeypatch.setattr(
        secrets_vault, "set_machine_agent_token",
        lambda mid, tok: vault.__setitem__(mid, tok))

    config_loader.migrate_machine_secrets()

    assert vault == {"m1": "legacy"}
    assert "_agent_token" not in _read_store(store)[0]
    # Idempotent: a second pass is a no-op.
    config_loader.migrate_machine_secrets()
    assert vault == {"m1": "legacy"}


def test_migration_does_not_mistake_farm_token_for_stored_token(
        tmp_path, monkeypatch):
    """The farm-token fallback must not make migration drop a per-machine
    legacy token (the agent may be paired with the machine-specific value)."""
    import secrets_vault
    store = _write_store(tmp_path, monkeypatch, [
        {"name": "m1", "id": "m1", "_agent_token": "machine-specific"},
    ])
    stored = {}
    # Simulate the farm token being resolvable for this machine.
    monkeypatch.setattr(secrets_vault, "get_machine_agent_token",
                        lambda m: "farm-token")
    monkeypatch.setattr(secrets_vault, "machine_agent_token_stored",
                        lambda m: (m.get("id") or m.get("name")) in stored)
    monkeypatch.setattr(
        secrets_vault, "set_machine_agent_token",
        lambda mid, tok: stored.__setitem__(mid, tok))

    config_loader.migrate_machine_secrets()

    assert stored == {"m1": "machine-specific"}
    assert "_agent_token" not in _read_store(store)[0]


def test_migration_vaults_legacy_plaintext_password(tmp_path, monkeypatch):
    import secrets_vault
    store = _write_store(tmp_path, monkeypatch, [
        {"name": "m1", "id": "m1", "password": "plaintext"},
    ])
    writes = []
    monkeypatch.setattr(secrets_vault, "set_machine_password",
                        lambda mid, pw: writes.append((mid, pw)))

    config_loader.migrate_machine_secrets()

    assert writes == [("m1", "plaintext")]
    assert _read_store(store)[0]["password"] == ""


def test_save_does_not_reencrypt_unchanged_password(monkeypatch):
    import secrets_vault
    writes = []
    monkeypatch.setattr(secrets_vault, "get_machine_password",
                        lambda m: "same-secret")
    monkeypatch.setattr(secrets_vault, "set_machine_password",
                        lambda mid, pw: writes.append((mid, pw)))

    out = config_loader._strip_machine_secret(
        {"name": "m1", "id": "m1", "password": "same-secret"})

    assert writes == []  # already in the vault: no .env rewrite
    assert out["password"] == ""


def test_update_with_new_password_vaults_once(tmp_path, monkeypatch):
    import secrets_vault
    store = _write_store(tmp_path, monkeypatch, [
        {"name": "m1", "id": "m1", "password": "", "username": "u"},
    ])
    writes = []
    monkeypatch.setattr(secrets_vault, "set_machine_password",
                        lambda mid, pw: writes.append((mid, pw)))

    config_loader.machines.update("m1", {"password": "new-secret"})

    assert writes == [("m1", "new-secret")]
    assert _read_store(store)[0]["password"] == ""


class _FakeMachines:
    def __init__(self, items):
        self.items = {m["name"]: m for m in items}

    def all(self):
        return list(self.items.values())

    def get(self, name):
        return self.items.get(name)

    def require(self, name):
        if name not in self.items:
            raise ValueError(f"machine '{name}' not found")
        return self.items[name]

    def update(self, name, updates):
        self.items[name].update(updates)
        return self.items[name]


def test_machines_endpoint_never_echoes_agent_token(monkeypatch):
    import asyncio
    import app as app_module
    import secrets_vault
    # Hermetic: the vault/env on the test machine must not decide the flags.
    monkeypatch.setattr(secrets_vault, "machine_agent_token_set",
                        lambda machine: True)
    monkeypatch.setattr(secrets_vault, "machine_password_set",
                        lambda machine: True)
    monkeypatch.setattr(app_module.cfg, "machines", _FakeMachines([
        {"name": "m1", "_agent_token": "super-secret", "password": "pw"},
    ]))
    rows = asyncio.run(machines_routes.list_machines())
    assert rows and rows[0]["name"] == "m1"
    assert "_agent_token" not in rows[0]
    assert "password" not in rows[0]
    assert rows[0]["agent_token_set"] is True
    assert rows[0]["password_set"] is True
