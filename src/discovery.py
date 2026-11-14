"""LAN identity helpers + peer-agent discovery (agent-only).

Peers are found through the peer agent's UDP responder; MAC/IP healing
uses the local ARP cache. Identity is the MAC, so DHCP re-leases do not
break a configured machine.
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
        hw = a.get("hardware") or {}
        results.append({
            "ip": ip, "port": 0,
            "ping_alive": None, "banner": None, "os": a.get("os"),
            "hostname": a.get("hostname"),
            "arch": a.get("arch"),
            "board": a.get("board") or hw.get("board") or "",
            "cpu_only": bool(a.get("cpu_only") or hw.get("cpu_only")),
            "gpu_unsupported": (a.get("gpu_unsupported")
                                or hw.get("gpu_unsupported") or ""),
            "gpus": hw.get("gpu_names") or [],
            "gpu_vendor": hw.get("gpu_vendor") or None,
            "ram_total_gb": hw.get("ram_total_gb"),
            "cores": hw.get("cores"),
            "vram_total_gb": hw.get("vram_total_gb"),
            "username": None,
            "mac": a.get("mac", ""), "mac_ip": ip,
            "already_added": ip in known,
            "agent": True, "pairing": bool(a.get("pairing")),
            "agent_port": a.get("agent_port"),
            "agent_version": a.get("agent_version"),
            "protocol_version": a.get("protocol_version"),
            "agent_addresses": a.get("addresses") or [],
        })
    results.sort(key=lambda r: (r.get("hostname") or "", r.get("ip") or ""))
    return {"subnet": "peer-agent", "port": 0, "alive": 0,
            "unknown": 0, "agents": len(results),
            "results": results}


def discover(subnet=None, include_known=True, **_kwargs):
    """Compatibility shim: discovery is agent-only."""
    return discover_agents(include_known=include_known)


def reconcile_machine_ips():
    """Agent UDP discovery -> config IP healing.

    The MAC is the stable identity; the agent's UDP responder reports its
    current addresses. When a configured machine's stored MAC shows up at
    a different IP, update machines.yaml in place and return the changes.
    Only machines with a stored MAC are touched (same safety rule as the
    legacy ARP path); the local master is never rewritten.
    """
    import config_loader as cfg
    import local_node
    try:
        found = discover_agents(include_known=True).get("results") or []
    except Exception:
        return []
    by_mac = {}
    for a in found:
        mac = _normalize_mac(a.get("mac"))
        ip = (a.get("ip") or "").strip()
        if mac and ip:
            by_mac[mac] = ip
    changed = []
    for m in cfg.machines.all():
        try:
            if local_node.is_local(m):
                continue
        except Exception:
            pass
        mac = _normalize_mac(m.get("mac"))
        new_ip = by_mac.get(mac)
        old_ip = (m.get("ip") or "").strip()
        if not new_ip or new_ip == old_ip:
            continue
        try:
            cfg.machines.update(m.get("name") or m.get("id"), {"ip": new_ip})
            changed.append((m.get("name"), old_ip, new_ip))
            print(f"[ip-sync] {m.get('name')}: {old_ip} -> {new_ip} "
                  f"(agent MAC {mac})")
        except Exception:
            pass
    return changed


def pair_agents(include_known=True):
    """Auto-enroll unpaired agents with the farm token (paired once)."""
    import secrets_vault
    import agent_client
    tok = secrets_vault.get_farm_agent_token()
    if not tok:
        return []
    try:
        found = discover_agents(include_known=include_known).get("results") or []
    except Exception:
        return []
    paired = []
    for a in found:
        if not a.get("pairing") or not a.get("ip"):
            continue
        m = {"name": a.get("hostname") or a["ip"], "ip": a["ip"],
             "agent_port": a.get("agent_port") or 8198}
        try:
            agent_client.pair(m, tok)
            paired.append(a["ip"])
            print(f"[pairing] paired {a.get('hostname') or a['ip']} with farm token")
        except Exception as e:
            print(f"[pairing] {a.get('ip')}: failed: {e}")
    return paired
