"""LAN discovery: find live machines, grab their SSH banner pre-auth to
classify the OS (no login required), and — only when credentials are
supplied — SSH in for hostname + GPU details.

No-login OS detection comes from the SSH identification banner every
server sends before authentication, e.g.:
  SSH-2.0-OpenSSH_for_Windows_8.1      -> windows
  SSH-2.0-OpenSSH_9.6p1 Ubuntu-3ubuntu13 -> linux
ICMP ping marks live hosts even when SSH is absent or filtered.
"""
import ipaddress
import subprocess
import socket
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

from ssh_manager import SSHManager, SSHError

IDENTIFY_CMD = (
    'uname -s 2>/dev/null; hostname 2>/dev/null; echo ---; '
    'nvidia-smi --query-gpu=index,name,memory.total '
    '--format=csv,noheader,nounits 2>/dev/null || '
    'rocm-smi --showproductname --showmeminfo vram --json 2>/dev/null || '
    'xpu-smi discovery -j 2>/dev/null || '
    'cmd.exe /c "ver & hostname & (nvidia-smi --query-gpu=index,name,'
    'memory.total --format=csv,noheader,nounits)" 2>/dev/null'
)

# MAC of the interface that answers for the machine's own IP, per OS.
# Used to pin a machine's identity across DHCP re-leases: the MAC is
# stable, the IP is not.
MAC_CMD_LINUX = (
    "for f in /sys/class/net/*/address; do "
    "d=$(dirname $f); br=$(cat $f/brport 2>/dev/null); "
    "[ -n \"$br\" ] && continue; "
    "ip=$(ip -4 addr show $(basename $d) 2>/dev/null | grep -o 'inet [0-9.]*' | awk '{print $2}'); "
    "[ -z \"$ip\" ] && continue; echo \"$ip $(cat $f)\"; done"
)
MAC_CMD_WINDOWS = (
    'powershell -NoProfile -Command "Get-NetIPAddress -AddressFamily IPv4 '
    '| Where-Object {$_.IPAddress -ne \'127.0.0.1\'} '
    '| ForEach-Object { $a = Get-NetAdapter -InterfaceIndex $_.InterfaceIndex '
    '-ErrorAction SilentlyContinue; if ($a -and $a.MacAddress) '
    '{ \"{0} {1}\" -f $_.IPAddress, ($a.MacAddress -replace \'-\', \':\') } }"'
)
MAC_CMD_MACOS = (
    "for i in $(networksetup -listallhardwareports 2>/dev/null "
    "| awk '/Device/ {print $2}'); do "
    "ip=$(ipconfig getifaddr $i 2>/dev/null); "
    "[ -z \"$ip\" ] && continue; "
    "mac=$(ifconfig $i 2>/dev/null | awk '/ether/ {print $2}'); "
    "[ -n \"$mac\" ] && echo \"$ip $mac\"; done"
)


def _normalize_mac(mac):
    """Lowercase, colon-separated MAC (accepts aa-bb-cc and AABBCC forms)."""
    if not mac:
        return ""
    mac = str(mac).strip().lower().replace("-", ":").replace(".", ":")
    if ":" not in mac and len(mac) == 12:  # bare hex form
        mac = ":".join(mac[i:i + 2] for i in range(0, 12, 2))
    parts = [p for p in mac.split(":") if p]
    return ":".join(parts) if len(parts) == 6 else ""


def query_machine_mac(ssh, os_type="linux"):
    """The machine's primary NIC MAC (the interface holding its main IPv4).
    Returns (ip_on_nic, mac) or ("", ""). Best-effort."""
    cmd = {"linux": MAC_CMD_LINUX, "windows": MAC_CMD_WINDOWS,
           "macos": MAC_CMD_MACOS}.get(os_type, MAC_CMD_LINUX)
    try:
        code, out, _ = ssh.run(cmd, timeout=15)
    except (SSHError, OSError):
        return "", ""
    for line in (out or "").splitlines():
        parts = line.split()
        if len(parts) == 2:
            mac = _normalize_mac(parts[1])
            if mac and not mac.startswith(("00:15:5d", "00:1c:14")):
                # skip Hyper-V / VirtualBox virtual NICs: the physical MAC
                # is the stable identity across reboots
                return parts[0], mac
    return "", ""


def arp_table():
    """Local ARP cache: {mac: ip} (MAC normalized). Empty when unavailable."""
    import subprocess
    entries = {}
    for args in (["ip", " neigh"], ["arp", "-a"]):
        try:
            out = subprocess.run(args, capture_output=True, text=True,
                                 timeout=10).stdout
        except (OSError, subprocess.SubprocessError):
            continue
        for line in out.splitlines():
            # "ip neigh": 10.0.0.1 dev ... lladdr aa:bb:cc:dd:ee:ff REACHABLE
            # "arp -a": ? (10.0.0.1) at aa:bb:cc:dd:ee:ff [ether] on eth0
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


def resolve_ip_by_mac(mac):
    """Current IP for a MAC from the local ARP cache, or None.
    Broadcasts a cheap reachability probe first so the cache is warm
    for machines we haven't talked to recently."""
    import socket
    mac = _normalize_mac(mac)
    if not mac:
        return None
    table = arp_table()
    if mac not in table:
        # warm the ARP cache: connect to SSH port of every IP we know about
        # won't help (we don't have an IP), so sweep the /24 quickly
        sub = local_subnet()
        if not sub:
            return None
        net = ipaddress.ip_network(sub, strict=False)
        with ThreadPoolExecutor(max_workers=128) as pool:
            list(pool.map(
                lambda h: _scan_port(h, 22, 0.4),
                [str(h) for h in net.hosts()]))
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


def _ping(ip, timeout_s=1):
    """ICMP ping (exit 0 = alive). No root needed with setuid ping.
    Flags are platform-specific: Windows ping has no -c/-W (uses -n/-w ms);
    macOS/BSD -W is in MILLISECONDS (Linux -W is seconds)."""
    ms = max(1, int(timeout_s)) * 1000
    if sys.platform == "win32":
        args = ["ping", "-n", "1", "-w", str(ms), ip]
    elif sys.platform == "darwin":
        args = ["ping", "-c", "1", "-W", str(ms), ip]
    else:
        args = ["ping", "-c", "1", "-W", str(max(1, int(timeout_s))), ip]
    try:
        return subprocess.run(
            args, capture_output=True, timeout=timeout_s + 1,
        ).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def _grab_banner(ip, port, timeout=2.0):
    """Connect and read the SSH identification string (sent pre-auth).
    Returns banner text or None. This needs no credentials of any kind."""
    try:
        with socket.create_connection((ip, port), timeout=timeout) as s:
            s.settimeout(timeout)
            data = s.recv(96)
            return data.decode("utf-8", errors="replace").strip() or None
    except (OSError, UnicodeDecodeError):
        return None


def _os_from_banner(banner):
    """Classify OS from an SSH banner. Windows OpenSSH always carries
    'for_Windows'; other OpenSSH builds (Linux/BSD/macOS) do not, so
    they land on 'linux' as the pre-auth best guess — post-login
    `uname` (Darwin vs Linux) refines it to 'macos' when creds work."""
    if not banner:
        return None
    b = banner.lower()
    if "windows" in b:
        return "windows"
    if "openssh" in b or "dropbear" in b:
        # dropbear is linux/embedded; plain OpenSSH without 'windows'
        # is overwhelmingly Linux/BSD/macOS
        return "linux"
    return None


def _scan_port(ip, port, timeout):
    """True if a TCP connection to ip:port succeeds."""
    try:
        with socket.create_connection((ip, port), timeout=timeout):
            return True
    except OSError:
        return False


def _ping_hosts(hosts, timeout_s=1, workers=128):
    """Return set of IPs answering ICMP ping."""
    alive = set()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(_ping, h, timeout_s): h for h in hosts}
        for fut in as_completed(futs):
            if fut.result():
                alive.add(futs[fut])
    return alive


def scan_ssh_hosts(subnet, port=22, timeout=0.5, workers=128):
    """Return sorted list of IPs in `subnet` with the SSH port open."""
    net = ipaddress.ip_network(subnet, strict=False)
    hosts = [str(h) for h in net.hosts()]
    open_hosts = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(_scan_port, h, port, timeout): h for h in hosts}
        for fut in as_completed(futs):
            if fut.result():
                open_hosts.append(futs[fut])
    return sorted(open_hosts, key=lambda x: tuple(int(p) for p in x.split(".")))


def _identify(ssh):
    """Run the identify command; parse OS / hostname / GPUs (post-login
    enrichment only — never required for basic discovery)."""
    info = {"os": None, "hostname": None, "gpus": [], "gpu_vendor": None}
    try:
        code, out, err = ssh.run(IDENTIFY_CMD, timeout=20)
    except (SSHError, OSError):
        return info
    text = (out or "") + "\n" + (err or "")
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    head, tail, seen_sep = [], [], False
    for l in lines:
        if l == "---":
            seen_sep = True
            continue
        (head if not seen_sep else tail).append(l)
    for l in head:
        first = l.split()[0:1]
        if first and first[0] == "Darwin":
            info["os"] = "macos"
            break
        if first and first[0] == "Linux":
            info["os"] = "linux"
            break
    blob = "\n".join(lines)
    if info["os"] is None and (
            "Microsoft" in blob or "Windows" in blob
            or "cmd.exe" in blob.lower() or "'uname' is not recognized" in blob):
        info["os"] = "windows"
    # hostname: scan backwards for the first line that isn't the uname
    # answer, a Windows banner/version line, a cmd error, or a GPU row
    for l in reversed(lines):
        low = l.lower()
        if l in ("Linux", "Darwin") or l == "---":
            continue
        if "not recognized" in low or "microsoft" in low or low.startswith("ver"):
            continue
        if l.lstrip().startswith(("{", "[", '"card')):
            continue  # rocm-smi / xpu-smi JSON rows
        parts = [p.strip() for p in l.split(",")]
        if len(parts) == 3 and parts[0].isdigit():
            continue  # nvidia-smi row
        if "nvidia-smi" in low or "cmd.exe" in low:
            continue
        info["hostname"] = l
        break
    for l in lines:
        parts = [p.strip() for p in l.split(",")]
        if len(parts) == 3 and parts[0].isdigit():
            try:
                float(parts[2])
                info["gpus"].append(parts[1])
                info["gpu_vendor"] = "nvidia"
            except ValueError:
                continue
    # AMD path: rocm-smi --json output in the tail
    import gpu_vendor as _gv
    blob = "\n".join(tail)
    if blob.lstrip().startswith(("{", "[")):
        rows = _gv.parse_rocm_json(blob)
        if rows:
            info["gpus"] = [r.split(",")[1] for r in rows]
            info["gpu_vendor"] = "amd"
        else:
            rows = _gv.parse_xpu_smi_json(blob)
            if rows:
                info["gpus"] = [r.split(",")[1] for r in rows]
                info["gpu_vendor"] = "intel"
    elif info["gpus"]:
        info["gpu_vendor"] = "nvidia"
    return info


def _probe_host(ip, port, cred_list, known_ips):
    """Banner-based classification (no login). SSH login only adds
    hostname + GPU preview when credentials work."""
    entry = {
        "ip": ip, "port": port,
        "ssh_port_open": True, "ping_alive": None,
        "banner": None, "os": None,
        "ssh_ok": False, "auth_failed": False,
        "hostname": None, "gpus": [], "gpu_vendor": None, "username": None,
        "mac": "", "mac_ip": "",
        "already_added": ip in known_ips,
    }
    entry["banner"] = _grab_banner(ip, port)
    entry["os"] = _os_from_banner(entry["banner"])
    for user, pwd in cred_list:
        if not user:
            continue
        ssh = SSHManager(ip, user, pwd, port=port, timeout=8)
        try:
            code, out, err = ssh.run("echo ok", timeout=12)
        except SSHError as e:
            msg = str(e)
            if "Permission" in msg or "Authentication" in msg or "auth" in msg.lower():
                entry["auth_failed"] = True
            continue
        if code == 0 and "ok" in out:
            entry["ssh_ok"] = True
            entry["username"] = user
            info = _identify(ssh)
            entry["os"] = info["os"] or entry["os"]  # login result wins
            entry["hostname"] = info["hostname"]
            entry["gpus"] = info["gpus"]
            entry["gpu_vendor"] = info["gpu_vendor"]
            # MAC = stable identity across DHCP re-leases; capture it so
            # later IP drift can be auto-corrected
            mac_ip, mac = query_machine_mac(ssh, entry["os"] or "linux")
            entry["mac"] = mac
            entry["mac_ip"] = mac_ip
            break
    return entry


def discover(subnet=None, port=22, credentials=None, include_known=True,
             scan_timeout=0.5, ping=True):
    """Full scan: ICMP ping sweep + SSH port sweep + banner OS classify
    (+ optional post-login enrichment). Persists nothing."""
    if not subnet:
        subnet = local_subnet()
    if not subnet:
        return {"error": "could not determine local subnet; pass one "
                         "explicitly (e.g. 192.168.1.0/24)"}
    import config_loader as cfg
    known = {m["ip"] for m in cfg.machines.all()}
    if credentials:
        cred_list = list(credentials)
    else:
        seen, cred_list = set(), []
        for m in cfg.machines.all():
            key = (m.get("username"), m.get("password"))
            if key not in seen and m.get("username"):
                seen.add(key)
                cred_list.append(key)
    try:
        net = ipaddress.ip_network(subnet, strict=False)
    except ValueError as e:
        return {"error": f"bad subnet: {e}"}
    hosts = [str(h) for h in net.hosts()]

    alive = _ping_hosts(hosts) if ping else set()
    with ThreadPoolExecutor(max_workers=128) as pool:
        futs = {pool.submit(_scan_port, h, port, scan_timeout): h
                for h in hosts}
        open_hosts = {futs[f] for f in as_completed(futs) if f.result()}

    # report ping-capable hosts even without SSH; SSH hosts always
    targets = (alive | open_hosts) if ping else open_hosts
    if not include_known:
        targets -= known
    targets = sorted(targets, key=lambda x: tuple(int(p) for p in x.split(".")))

    results = []
    with ThreadPoolExecutor(max_workers=min(32, len(targets) or 1)) as pool:
        futs = [pool.submit(_probe_host, h, port, cred_list, known)
                for h in targets]
        for fut in as_completed(futs):
            r = fut.result()
            if ping:
                r["ping_alive"] = r["ip"] in alive
            results.append(r)
    results.sort(key=lambda x: tuple(int(p) for p in x["ip"].split(".")))
    # drop hosts we couldn't fingerprint (no usable banner and no login):
    # they can't be added safely and only clutter the results table
    unknown = [r for r in results if not r["os"]]
    results = [r for r in results if r["os"]]
    return {"subnet": subnet, "port": port, "alive": len(alive),
            "ssh_hosts": len(open_hosts), "unknown": len(unknown),
            "results": results}
