"""Renaming a machine must cascade into model/group/layout references."""
import asyncio
import json

import yaml

import config_loader


def _model_doc(old):
    return {
        "model_id": old,              # must never be treated as a machine ref
        "name": old,
        "servers": [old, "aibox1"],
        "device_order": f"{old}-CUDA0,{old}-RAM,aibox1-CUDA0",
        "expose": {old: {"mode": "ram", "gpus": []}},
        "variants": [{
            "label": "v",
            "servers": [old],
            "device_order": f"{old}-CUDA0",
            "expose": {old: {"mode": "vram", "gpus": []}},
        }],
    }


def test_rename_cascades_models_groups_layout(tmp_path, monkeypatch):
    models = tmp_path / "models"
    models.mkdir()
    (models / f"{'oldbox'}.yaml").write_text(
        yaml.safe_dump(_model_doc("oldbox")), encoding="utf-8")
    monkeypatch.setattr(config_loader, "MODELS_DIR", str(models))

    groups_path = tmp_path / "groups.yaml"
    groups_path.write_text(yaml.safe_dump({"groups": [{
        "name": "cluster", "machines": ["oldbox", "aibox1"]}]}),
        encoding="utf-8")
    monkeypatch.setattr(config_loader.groups, "path", str(groups_path))

    layout_path = tmp_path / "flow_layout.json"
    layout_path.write_text(json.dumps({
        "machine:oldbox": {"x": 1, "y": 2},
        "model:oldbox": {"x": 3, "y": 4},
    }), encoding="utf-8")
    monkeypatch.setattr(config_loader, "FLOW_LAYOUT_FILE", str(layout_path))

    result = config_loader.rename_machine_references("oldbox", "newbox")

    assert result["models"] == ["oldbox"]
    doc = yaml.safe_load((models / "oldbox.yaml").read_text())
    assert doc["model_id"] == "oldbox"      # id/display untouched
    assert doc["name"] == "oldbox"
    assert doc["servers"] == ["newbox", "aibox1"]
    assert doc["device_order"] == "newbox-CUDA0,newbox-RAM,aibox1-CUDA0"
    assert "newbox" in doc["expose"] and "oldbox" not in doc["expose"]
    assert doc["variants"][0]["servers"] == ["newbox"]
    assert doc["variants"][0]["device_order"] == "newbox-CUDA0"
    assert "newbox" in doc["variants"][0]["expose"]

    groups = yaml.safe_load(groups_path.read_text())["groups"]
    assert groups[0]["machines"] == ["newbox", "aibox1"]
    layout = json.loads(layout_path.read_text())
    assert "machine:newbox" in layout and "machine:oldbox" not in layout
    assert "model:oldbox" in layout


def test_update_machine_cascades_rename(monkeypatch):
    import routes.machines as machines_routes

    class FakeMachines:
        def __init__(self):
            self.items = {"oldbox": {"name": "oldbox", "id": "oldbox"}}

        def get(self, name):
            return self.items.get(name)

        def update(self, name, updates):
            item = self.items.pop(name)
            item.update(updates)
            self.items[item["name"]] = item
            return item

    fake = FakeMachines()
    monkeypatch.setattr(machines_routes.cfg, "machines", fake)
    monkeypatch.setattr(machines_routes, "_agent_guard", lambda m: None)
    calls = []
    monkeypatch.setattr(machines_routes.cfg, "rename_machine_references",
                        lambda old, new: calls.append((old, new)))
    import secrets_vault
    monkeypatch.setattr(secrets_vault, "machine_password_set", lambda m: False)
    monkeypatch.setattr(secrets_vault, "machine_agent_token_set", lambda m: False)

    m = machines_routes.MachineIn(name="newbox", ip="10.0.0.9")
    saved = asyncio.run(machines_routes.update_machine("oldbox", m))

    assert saved["name"] == "newbox"
    assert calls == [("oldbox", "newbox")]
