"""Machine secret hygiene: an in-memory agent token must never reach
machines.yaml or the /api/machines response."""
import config_loader


def test_agent_token_never_reaches_machine_yaml(monkeypatch):
    import secrets_vault
    saved = []
    monkeypatch.setattr(secrets_vault, "set_machine_password",
                        lambda mid, pw: saved.append((mid, pw)))
    out = config_loader._strip_machine_secret({
        "name": "m1", "id": "m1", "_agent_token": "super-secret",
        "password": "pw",
    })
    assert "_agent_token" not in out
    assert out["password"] == ""
    assert saved == [("m1", "pw")]


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
    monkeypatch.setattr(app_module.cfg, "machines", _FakeMachines([
        {"name": "m1", "_agent_token": "super-secret", "password": "pw",
         "agent_token_set": True},
    ]))
    rows = asyncio.run(app_module.list_machines())
    assert rows and rows[0]["name"] == "m1"
    assert "_agent_token" not in rows[0]
    assert "password" not in rows[0]
    assert rows[0]["agent_token_set"] is True
