"""LAN identity helpers + peer-agent discovery (agent-only).

There is no SSH scanning: peers are found through the peer agent's UDP
responder, and MAC/IP healing uses the local ARP cache. Identity is the
MAC, so DHCP re-leases do not break a configured machine.
"""
import ipaddress
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

import agent_discovery


def _normalize_mac(mac):
    """Lowercase, colon-separated MAC (accepts aa-bb-cc and AABBCC forms)."""
    if not mac:
        return ""
    mac = str(mac).strip().lower().replace("-", ":").replace(".", ":")
    if ":" not in mac and len(mac) == 12:  # bare hex form
        mac = ":".join(mac[i:i + 2] for i in range(0, 12, 2))
    parts = [p for p in mac.split(":") if p]
    return ":".join(parts) if len(parts) == 6 else ""


def arp_table():
    """Local ARP cache: {mac: ip} (MAC normalized). Empty when unavailable."""
    entries = {}
    for args in (["ip", " neigh"], ["arp", "-a"]):
        try:
            out = subprocess.run(args, capture_output=True, text=True,
                                 timeout=10).stdout
        except (OSError, subprocess.SubprocessError):
            continue
        for line in out.splitlines():
            ip, mac = "", ""
            toks = line.split()
            if args[0] == "ip":
                if "lladdr" in toks:
                    ip = toks[0]
                    mac = toks[toks.index("lladdr") + 1]
            else:
                if " at " in line and "(" in line:
                    ip = line.split("(")[1].split(")")[0]
                    mac = line.split(" at ")[1].split()[0]
            mac = _normalize_mac(mac)
            if mac and ip and not ip.startswith("127."):
                entries.setdefault(mac, ip)
        if entries:
            break
    return entries


def _ping(ip, timeout_s=1):
    """ICMP ping (exit 0 = alive)."""
    ms = max(1, int(timeout_s)) * 1000
    if sys.platform == "win32":
        args = ["ping", "-n", "1", "-w", str(ms), ip]
    elif sys.platform == "darwin":
        args = ["ping", "-c", "1", "-W", str(ms), ip]
    else:
        args = ["ping", "-c", "1", "-W", str(max(1, int(timeout_s))), ip]
    try:
        return subprocess.run(args, capture_output=True,
                              timeout=timeout_s + 1).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def _warm_arp(subnet):
    try:
        net = ipaddress.ip_network(subnet, strict=False)
    except ValueError:
        return
    hosts = [str(h) for h in net.hosts()]
    with ThreadPoolExecutor(max_workers=128) as pool:
        futs = {pool.submit(_ping, h, 0.5): h for h in hosts}
        for fut in as_completed(futs):
            fut.result()


def resolve_ip_by_mac(mac):
    """Current IP for a MAC from the local ARP cache, or None."""
    mac = _normalize_mac(mac)
    if not mac:
        return None
    table = arp_table()
    if mac not in table:
        sub = local_subnet()
        if not sub:
            return None
        _warm_arp(sub)  # populate the cache so the next read can match
        table = arp_table()
    return table.get(mac)


def local_subnets():
    """All /24 subnets this machine has an interface on (UI dropdown)."""
    import local_node
    return sorted(f"{ip}/24" for ip in local_node.local_ips())


def local_subnet():
    """CIDR of the /24 this machine lives on (first non-loopback IP)."""
    subs = [s for s in local_subnets() if not s.startswith("127.")]
    return subs[0] if subs else None


def discover_agents(port=8199, timeout=2.0, include_known=True):
    """UDP peer-agent discovery. Returns the machine-card result shape."""
    import config_loader as cfg
    known = {m["ip"] for m in cfg.machines.all()}
    try:
        agents = agent_discovery.discover(port=port, timeout=timeout)
    except Exception:
        agents = []
    results = []
    for a in agents:
        ip = a.get("ip")
        if not ip or (not include_known and ip in known):
            continue
        results.append({
            "ip": ip, "port": 0, "ssh_port_open": False,
            "ping_alive": None, "banner": None, "os": a.get("os"),
            "ssh_ok": False, "auth_failed": False,
            "hostname": a.get("hostname"), "gpus": [],
            "gpu_vendor": None, "username": None,
            "mac": a.get("mac", ""), "mac_ip": ip,
            "already_added": ip in known,
            "agent": True, "agent_port": a.get("agent_port"),
            "agent_version": a.get("agent_version"),
            "protocol_version": a.get("protocol_version"),
            "agent_addresses": a.get("addresses") or [],
        })
    results.sort(key=lambda r: (r.get("hostname") or "", r.get("ip") or ""))
    return {"subnet": "peer-agent", "port": 0, "alive": 0,
            "ssh_hosts": 0, "unknown": 0, "agents": len(results),
            "results": results}


def discover(subnet=None, include_known=True, **_kwargs):
    """Compatibility shim: discovery is agent-only."""
    return discover_agents(include_known=include_known)
