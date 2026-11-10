"""Peer connectivity probe used by machine/scanning views."""
import agent_client
import local_node


def _machine_connection_caps(machine):
    """Detect usable transport for one machine: local or peer agent."""
    name = machine.get("name") or machine.get("id") or "?"
    try:
        if local_node.is_local(machine):
            return {"name": name, "mode": "local", "local": True,
                    "agent": False}
    except Exception:
        pass
    try:
        agent_client.health(machine, timeout=3.0)
        return {"name": name, "mode": "agent", "local": False,
                "agent": True}
    except Exception:
        return {"name": name, "mode": "none", "local": False,
                "agent": False}
