"""Architecture contract: peers are reached ONLY through the peer agent.

These assertions are intentionally installation-independent: they never
read config/machines.yaml or config/settings.yaml, so the suite behaves
the same on anybody's farm.
"""
import pathlib

import pytest

import agent_client
import config_loader
import rpc_manager

REPO = pathlib.Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("legacy", [None, "", "auto", "ssh", "agent", "bogus"])
def test_every_transport_resolves_to_agent(legacy):
    machine = {"name": "node", "ip": "10.0.0.1", "transport": legacy}
    assert agent_client.transport(machine) == "agent"
    assert agent_client.should_use_agent(machine) is True


def test_ssh_runtime_entry_point_is_gone():
    with pytest.raises(RuntimeError):
        rpc_manager._ssh_for({"name": "node", "ip": "10.0.0.1"})


def test_no_ssh_client_dependency():
    req = (REPO / "requirements.txt").read_text(encoding="utf-8").lower()
    assert "paramiko" not in req
    assert "sshpass" not in req


def test_vllm_engine_removed():
    assert config_loader.VALID_ENGINES == ("llama_cpp",)
    assert not (REPO / "src" / "vllm_manager.py").exists()
    assert not (REPO / "peer_agent" / "llama_peer_agent" / "ray.py").exists()
