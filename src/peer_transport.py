"""Transport seam for peer operations.

The agent is the preferred transport; SSH remains the compatibility
fallback while endpoint parity is completed. Callers get one object per
machine and do not branch on transport themselves.
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


class SSHTransport(PeerTransport):
    name = "ssh"

    def _rpc(self):
        import rpc_manager
        return rpc_manager

    def status(self, machine):
        return self._rpc().machine_status(machine["name"], use_agent=False)

    def stop_all(self, machine):
        return self._rpc().stop_rpc(machine["name"], None, use_agent=False)

    def logs_tail(self, machine, name, lines=20):
        import local_node
        rm = self._rpc()
        is_win = machine.get("os", "linux") == "windows"
        is_local = local_node.is_local(machine)
        if name.startswith("rpc-server"):
            import re
            m = re.search(r"-(\d+)\.log$", name)
            port = int(m.group(1)) if m else 50052
            text = rm._tail_log(machine, is_win, is_local, port,
                                use_agent=False)
            return {"name": name, "lines": text.splitlines(), "text": text}
        # Generic log names need one bounded tail command. Keep this
        # behind the SSH transport only until a typed agent list endpoint
        # makes the aggregate path unnecessary.
        if is_local:
            from pathlib import Path
            path = Path.home() / name
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                text = ""
        else:
            if is_win:
                cmd = (f'powershell -NoProfile -Command "Get-Content '
                       f'$env:USERPROFILE\\{name} -Tail {int(lines)}"')
            else:
                cmd = f'tail -n {int(lines)} "$HOME/{name}"'
            try:
                _, text, _ = rm._ssh_for(machine).run(cmd, timeout=10)
            except Exception:
                text = ""
        tail = (text or "").splitlines()[-int(lines):]
        return {"name": name, "lines": tail, "text": "\n".join(tail)}

    def processes(self, machine):
        rm = self._rpc()
        return {"rpc": rm.list_rpc_servers(machine["name"], use_agent=False),
                "llama": [], "swap": []}


def for_machine(machine) -> PeerTransport:
    if agent_client.should_use_agent(machine):
        return AgentTransport()
    return SSHTransport()
