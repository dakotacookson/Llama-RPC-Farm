"""Detects the machine the app is running on, so it can be seeded as the
default master node and flagged in the UI."""
import socket
import sys

import config_loader as cfg


def local_ips():
    """All IPv4 addresses of this machine (physical, virtual, VPN)."""
    addrs = {local_ip()}
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            addrs.add(info[4][0])
    except socket.gaierror:
        pass
    return {a for a in addrs if not a.startswith("127.")}


def local_ip():
    """Best-guess LAN IP (UDP trick; picks whatever route the OS prefers).
    Callers that care about a specific subnet should use local_ips()."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("10.255.255.255", 1))  # never actually sent
        return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()


def local_hostname():
    return socket.gethostname().split(".")[0].lower()


def local_os():
    if sys.platform == "win32":
        return "windows"
    if sys.platform == "darwin":
        return "macos"
    return "linux"


def primary_mac():
    """MAC of the interface holding our primary LAN IP ("" if unknown).
    The stable identity across DHCP re-leases."""
    import subprocess
    ip = local_ip()
    if ip.startswith("127."):
        return ""
    if sys.platform == "win32":
        # netsh: parse the Physical Address of the iface with our IP
        try:
            out = subprocess.run(["netsh", "interface", "ipv4", "show", "addresses"],
                                  capture_output=True, text=True, timeout=10).stdout
            blocks, cur = [], {}
            for line in out.splitlines():
                s = line.strip()
                if s.startswith("Interface "):
                    if cur.get("ip"):
                        blocks.append(cur)
                    cur = {}
                elif "IP Address:" in s:
                    cur["ip"] = s.split(":", 1)[1].strip()
                elif "Physical Address" in s and "ip" in cur:
                    cur["mac"] = s.split(":", 1)[1].strip()
            for b in blocks:
                if b.get("ip") == ip:
                    return b.get("mac", "").replace("-", ":").lower()
        except (OSError, subprocess.SubprocessError):
            return ""
        return ""
    # POSIX: walk /sys (linux) or ifconfig (macos)
    try:
        out = subprocess.run(
            ["bash", "-c",
             f"for f in /sys/class/net/*/address; do d=$(dirname $f); "
             f"ip addr show $(basename $d) 2>/dev/null | grep -q 'inet {ip}' "
             f"&& cat $f && exit 0; done"],
            capture_output=True, text=True, timeout=10).stdout.strip().lower()
        if out:
            return out
    except (OSError, subprocess.SubprocessError):
        pass
    try:
        out = subprocess.run(
            ["bash", "-c",
             f"for i in $(ifconfig -l 2>/dev/null || ls /sys/class/net); do "
             f"ifconfig $i 2>/dev/null | grep -q 'inet {ip} ' && "
             f"ifconfig $i | awk '/ether/ {{print $2; exit}}' && exit 0; done"],
            capture_output=True, text=True, timeout=10).stdout.strip().lower()
        return out or ""
    except (OSError, subprocess.SubprocessError):
        return ""


def ensure_local_master():
    """Make sure the machine this app runs on exists in the machine list
    with role=master. Returns the local machine entry.

    If no master exists, adds one named after the hostname with the
    detected LAN IP. If a master exists at a different IP (e.g. the user
    moved the app to aibox1), we trust the existing config.
    """
    machines = cfg.machines.all()
    my_ip = local_ip()

    # already registered as master?
    for m in machines:
        if m.get("role") == "master":
            return m

    # no master: seed this machine as the default master
    name = local_hostname()
    if any(m["name"] == name for m in machines):
        return cfg.machines.update(name, {"role": "master", "ip": my_ip, "os": local_os()})
    return cfg.machines.add({
        "name": name,
        "ip": my_ip,
        "username": "",       # local machine: no SSH needed
        "password": "",
        "role": "master",
        "os": local_os(),
    })


def is_local(machine):
    """True if this machine entry refers to the machine running the app.
    Loopback addresses are always local. Otherwise match any of our local
    IPs (physical + VPN + virtual adapters)."""
    if machine.get("ip") in ("127.0.0.1", "localhost", "0.0.0.0"):
        return True
    if machine.get("ip") in local_ips():
        return True
    return (
        machine.get("name") == local_hostname()
        and machine.get("ip") in ("", "127.0.0.1", "localhost")
    )