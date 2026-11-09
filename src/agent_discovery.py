"""LAN discovery client for the optional peer agent.

Sends a small UDP probe and collects identity-only responses. Discovery
does not authenticate the agent; enrollment still needs the vault token.
"""
import json
import socket
import time

MAGIC = b"LLAMA_FARM_DISCOVER_V1"
DEFAULT_DISCOVERY_PORT = 8199


def _parse(data: bytes, addr) -> dict | None:
    try:
        item = json.loads(data.decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        return None
    if not isinstance(item, dict) or item.get("magic") != MAGIC.decode():
        return None
    item = dict(item)
    item.setdefault("ip", addr[0])
    return item


def discover(address="255.255.255.255", port=DEFAULT_DISCOVERY_PORT,
             timeout=2.0, probes=2) -> list[dict]:
    """Return discovered agents, deduplicated by machine_id/ip:port."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    # Explicit bind keeps the macOS socket usable; bind the wildcard, NOT
    # loopback — a 127.0.0.1 source address makes LAN agents reply to an
    # address this host can't receive (observed: discovery returned 0
    # agents instead of 3).
    sock.bind(("", 0))
    deadline = time.monotonic() + max(0.1, float(timeout))
    found = {}
    try:
        for _ in range(max(1, int(probes))):
            try:
                sock.sendto(MAGIC, (address, int(port)))
            except OSError:
                break
        while time.monotonic() < deadline:
            remaining = deadline - time.monotonic()
            sock.settimeout(max(0.01, min(0.25, remaining)))
            try:
                data, addr = sock.recvfrom(4096)
            except socket.timeout:
                continue
            item = _parse(data, addr)
            if item is None:
                continue
            key = (item.get("machine_id") or "",
                   item.get("ip"), item.get("agent_port"))
            found[key] = item
    finally:
        sock.close()
    return sorted(found.values(), key=lambda a: (
        a.get("hostname") or "", a.get("ip") or ""))
