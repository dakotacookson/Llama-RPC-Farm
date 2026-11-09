"""Transport seam for peer operations.

Agent-only architecture: main talks to every peer through the peer agent.
Agents are installed from their own platform packages; there is no shell
runtime transport.
"""
import agent_client


class PeerTransport:
    name = "base"

    def status(self, machine):
        raise NotImplementedError

    def stop_all(self, machine):
        raise NotImplementedError

    def logs_tail(self, machine, name, lines=20):
        raise NotImplementedError

    def processes(self, machine):
        raise NotImplementedError


class AgentTransport(PeerTransport):
    name = "agent"

    def status(self, machine):
        return agent_client.status(machine)

    def stop_all(self, machine):
        return agent_client.stop_all(machine)

    def logs_tail(self, machine, name, lines=20):
        return agent_client.logs_tail(machine, name, lines)

    def processes(self, machine):
        return agent_client.processes(machine)


def for_machine(machine) -> PeerTransport:
    return AgentTransport()
