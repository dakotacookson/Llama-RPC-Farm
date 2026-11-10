"""Setup readiness: what each machine still needs to serve models."""
import os
import sys

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import app as app_module          # noqa: E402
import machine_tokens as mt  # noqa: E402
import routes.machines as machines_routes
import routes.readiness as readiness_module  # noqa: E402
import agent_client               # noqa: E402
import swap_installer             # noqa: E402


def _local_engines(*_a, **_k):
    return {"llama_cpp": {"installed": True, "path": "/bin/llama-server",
                          "version": "b12211"},
            "llama_rpc": {"installed": True, "path": "/bin/ggml-rpc-server"}}


def test_readiness_local_swap_host_requires_swap(monkeypatch):
    monkeypatch.setattr(readiness_module.local_node, "is_local", lambda m: True)
    monkeypatch.setattr(readiness_module, "_local_engine_readiness", _local_engines)
    monkeypatch.setattr(swap_installer, "detect_on", lambda m: (False, None))
    r = readiness_module._readiness_for({"name": "node1", "os": "linux"}, "node1")
    assert r["ready"] is False
    assert r["is_swap_host"] is True
    assert r["llama_swap"]["required"] is True
    assert r["llama_swap"]["installed"] is False


def test_readiness_peer_ready_without_swap(monkeypatch):
    monkeypatch.setattr(readiness_module.local_node, "is_local", lambda m: False)
    monkeypatch.setattr(agent_client, "health",
                        lambda m, timeout=4: {"agent_version": "0.3.4",
                                             "protocol_version": 1})
    monkeypatch.setattr(agent_client, "capabilities", lambda m, timeout=8: {
        "llama_available": True,
        "llama_binary": "/home/u/llama.cpp/bin/llama-server",
        "rpc_available": True,
        "rpc_binary": "/home/u/llama.cpp/bin/ggml-rpc-server",
    })
    monkeypatch.setattr(agent_client, "install_verify",
                        lambda m, path, args=None, timeout=35:
                        {"returncode": 0, "output": "version: b12211"})
    monkeypatch.setattr(swap_installer, "detect_on", lambda m: (False, None))
    r = readiness_module._readiness_for({"name": "peer1", "os": "linux"}, "host1")
    assert r["ready"] is True
    assert r["llama_cpp"]["installed"] is True
    assert r["llama_rpc"]["installed"] is True
    assert r["llama_swap"]["required"] is False


def test_readiness_reports_missing_rpc_and_agent(monkeypatch):
    monkeypatch.setattr(readiness_module.local_node, "is_local", lambda m: False)
    monkeypatch.setattr(agent_client, "health",
                        lambda m, timeout=4: {"agent_version": "0.3.4"})
    monkeypatch.setattr(agent_client, "capabilities", lambda m, timeout=8: {
        "llama_available": True,
        "llama_binary": "/home/u/llama.cpp/bin/llama-server",
        "rpc_available": False,
        "rpc_binary": None,
    })
    monkeypatch.setattr(agent_client, "install_verify",
                        lambda m, path, args=None, timeout=35:
                        {"returncode": 0, "output": "version: b12211"})
    monkeypatch.setattr(swap_installer, "detect_on", lambda m: (False, None))
    r = readiness_module._readiness_for({"name": "peer1", "os": "linux"}, "host1")
    assert r["ready"] is False
    assert r["llama_cpp"]["installed"] is True
    assert r["llama_rpc"]["installed"] is False


def test_agent_token_falls_back_to_farm_vault(monkeypatch):
    import secrets_vault
    monkeypatch.setattr(secrets_vault, "get_machine_agent_token", lambda m: "")
    monkeypatch.setattr(secrets_vault, "get_farm_agent_token", lambda: "farm-tok")
    monkeypatch.delenv("FARM_AGENT_TOKEN", raising=False)
    monkeypatch.delenv("AGENT_TOKEN", raising=False)
    assert agent_client._token({"name": "fresh"}) == "farm-tok"


def test_ensure_agent_ready_pairs_fresh_agent(monkeypatch):
    import secrets_vault
    import agent_client as ac

    def caps(machine, timeout=5):
        if machine.get("_agent_token") != "farm-tok":
            raise ac.AgentError("agent HTTP 401")
        return {"ok": True, "llama_available": True}

    paired = []
    monkeypatch.setattr(ac, "capabilities", caps)
    monkeypatch.setattr(ac, "pair",
                        lambda machine, token, timeout=8: paired.append(token)
                        or {"ok": True})
    monkeypatch.setattr(secrets_vault, "get_farm_agent_token",
                        lambda: "farm-tok")
    ok, err = mt._ensure_agent_ready(
        {"name": "fresh", "ip": "10.0.0.178"}, "")
    assert err is None and ok["llama_available"] is True
    assert paired == ["farm-tok"]


def test_auto_agent_token_pairs_fresh_agent_with_farm(monkeypatch):
    import secrets_vault
    import agent_client as ac
    state = {"paired": False}

    def caps(machine, timeout=5):
        if machine.get("_agent_token") == "farm-tok" and state["paired"]:
            return {"ok": True}
        raise ac.AgentError("agent HTTP 401")

    def pair(machine, token, timeout=8):
        state["paired"] = True
        return {"ok": True}

    monkeypatch.setattr(ac, "capabilities", caps)
    monkeypatch.setattr(ac, "pair", pair)
    monkeypatch.setattr(secrets_vault, "get_machine_agent_token",
                        lambda m: "")
    monkeypatch.setattr(secrets_vault, "get_farm_agent_token",
                        lambda: "farm-tok")
    tok = mt._auto_agent_token(
        {"id": "fresh", "name": "fresh", "ip": "10.0.0.178"})
    assert tok == "farm-tok"
    assert state["paired"] is True


def test_auto_agent_token_keeps_existing_machine_token(monkeypatch):
    import secrets_vault
    import agent_client as ac
    monkeypatch.setattr(ac, "capabilities",
                        lambda m, timeout=5:
                        {"ok": True} if m.get("_agent_token") == "machine-tok"
                        else (_ for _ in ()).throw(ac.AgentError("401")))
    monkeypatch.setattr(ac, "pair",
                        lambda *a, **k: (_ for _ in ()).throw(
                            AssertionError("must not re-pair")))
    monkeypatch.setattr(secrets_vault, "get_machine_agent_token",
                        lambda m: "machine-tok")
    monkeypatch.setattr(secrets_vault, "get_farm_agent_token",
                        lambda: "farm-tok")
    assert mt._auto_agent_token(
        {"id": "m1", "name": "m1", "ip": "10.0.0.178"}) == "machine-tok"


def test_auto_agent_token_fails_when_agent_has_unknown_token(monkeypatch):
    import secrets_vault
    import agent_client as ac
    import pytest
    monkeypatch.setattr(ac, "capabilities",
                        lambda *a, **k: (_ for _ in ()).throw(
                            ac.AgentError("401")))
    monkeypatch.setattr(ac, "pair",
                        lambda *a, **k: (_ for _ in ()).throw(
                            ac.AgentError("already paired")))
    monkeypatch.setattr(secrets_vault, "get_machine_agent_token",
                        lambda m: "")
    monkeypatch.setattr(secrets_vault, "get_farm_agent_token",
                        lambda: "farm-tok")
    with pytest.raises(ValueError, match="could not auto-pair"):
        mt._auto_agent_token(
            {"id": "m1", "name": "m1", "ip": "10.0.0.178"})


def test_delete_machine_clears_vault_agent_token(monkeypatch):
    import asyncio
    import secrets_vault
    calls = []
    monkeypatch.setattr(app_module.cfg.machines, "get",
                        lambda n: {"id": "m1", "name": "node1"})
    monkeypatch.setattr(app_module.cfg.machines, "delete", lambda n: [])
    monkeypatch.setattr(secrets_vault, "set_machine_agent_token",
                        lambda ident, tok: calls.append(("tok", ident, tok)))
    monkeypatch.setattr(secrets_vault, "delete_env_value",
                        lambda key: calls.append(("env", key)))
    monkeypatch.setattr(secrets_vault, "machine_env_key",
                        lambda ident: f"MACHINE_{ident}_PASSWORD_ENC")
    asyncio.run(machines_routes.delete_machine("node1"))
    ids = {c[1] for c in calls if c[0] == "tok"}
    assert {"m1", "node1"} <= ids
    assert all(c[2] == "" for c in calls if c[0] == "tok")
