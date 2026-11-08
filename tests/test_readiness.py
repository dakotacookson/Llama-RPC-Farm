"""Setup readiness: what each machine still needs to serve models."""
import os
import sys

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import app as app_module          # noqa: E402
import agent_client               # noqa: E402
import swap_installer             # noqa: E402


def _local_engines(*_a, **_k):
    return {"llama_cpp": {"installed": True, "path": "/bin/llama-server",
                          "version": "b12211"},
            "llama_rpc": {"installed": True, "path": "/bin/ggml-rpc-server"}}


def test_readiness_local_swap_host_requires_swap(monkeypatch):
    monkeypatch.setattr(app_module.local_node, "is_local", lambda m: True)
    monkeypatch.setattr(app_module, "_local_engine_readiness", _local_engines)
    monkeypatch.setattr(swap_installer, "detect_on", lambda m: (False, None))
    r = app_module._readiness_for({"name": "node1", "os": "linux"}, "node1")
    assert r["ready"] is False
    assert r["is_swap_host"] is True
    assert r["llama_swap"]["required"] is True
    assert r["llama_swap"]["installed"] is False


def test_readiness_peer_ready_without_swap(monkeypatch):
    monkeypatch.setattr(app_module.local_node, "is_local", lambda m: False)
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
    r = app_module._readiness_for({"name": "peer1", "os": "linux"}, "host1")
    assert r["ready"] is True
    assert r["llama_cpp"]["installed"] is True
    assert r["llama_rpc"]["installed"] is True
    assert r["llama_swap"]["required"] is False


def test_readiness_reports_missing_rpc_and_agent(monkeypatch):
    monkeypatch.setattr(app_module.local_node, "is_local", lambda m: False)
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
    r = app_module._readiness_for({"name": "peer1", "os": "linux"}, "host1")
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

    def health(machine, timeout=4, token=None):
        if not (machine.get("_agent_token") or token):
            raise ac.AgentError("agent HTTP 401")
        return {"ok": True, "agent_version": "1.0.0"}

    paired = []
    monkeypatch.setattr(ac, "health", health)
    monkeypatch.setattr(ac, "pair",
                        lambda machine, token, timeout=8: paired.append(token)
                        or {"ok": True})
    monkeypatch.setattr(secrets_vault, "get_farm_agent_token",
                        lambda: "farm-tok")
    ok, err = app_module._ensure_agent_ready(
        {"name": "fresh", "ip": "10.0.0.178"}, "")
    assert err is None and ok["ok"] is True
    assert paired == ["farm-tok"]
