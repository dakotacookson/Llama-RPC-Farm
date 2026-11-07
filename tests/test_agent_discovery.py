"""Agent UDP discovery contract."""
import os
import sys
import threading

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "peer_agent"))

import agent_discovery  # noqa: E402
from llama_peer_agent.config import AgentConfig  # noqa: E402
from llama_peer_agent import discovery as agent_discovery_server  # noqa: E402


def test_discovery_round_trip():
    config = AgentConfig(
        token="t", host="127.0.0.1", port=8198,
        discovery_host="127.0.0.1", discovery_port=0)
    sock = agent_discovery_server.create_socket(config)
    port = sock.getsockname()[1]
    stop = threading.Event()
    thread = threading.Thread(
        target=agent_discovery_server.serve,
        args=(config, sock, stop), daemon=True)
    thread.start()
    try:
        agents = agent_discovery.discover(
            address="127.0.0.1", port=port, timeout=1.0, probes=2)
    finally:
        stop.set()
        sock.close()
        thread.join(timeout=2)
    assert len(agents) == 1
    agent = agents[0]
    assert agent["magic"] == agent_discovery.MAGIC.decode()
    assert agent["agent_port"] == 8198
    assert agent["ip"] == "127.0.0.1"
    assert agent["hostname"]
