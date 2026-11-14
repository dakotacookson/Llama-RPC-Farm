"""Machine detail used by the dashboard modal."""
import json
import os
import platform
import shutil
import subprocess
import sys
import uuid

from . import processes
from .config import AgentConfig


_BOARD_MODEL_PATHS = ("/proc/device-tree/model",
                      "/sys/firmware/devicetree/base/model")


def _board_model() -> str:
    """SBC/board model from the device tree (Raspberry Pi, etc.).

    `/proc/device-tree/model` is NUL-terminated; desktop/server Linux
    usually has no such file and returns ''."""
    for path in _BOARD_MODEL_PATHS:
        try:
            with open(path, "rb") as f:
                return f.read().replace(b"\x00", b"").decode(
                    "utf-8", "replace").strip()
        except OSError:
            continue
    return ""


def _mac_address() -> str:
    """Stable host MAC for enrollment/detail, matching /info's identity."""
    node = uuid.getnode()
    if not ((node >> 40) & 1):  # multicast bit clear = real hardware MAC
        return ":".join(f"{(node >> i) & 0xff:02x}" for i in range(40, -1, -8))
    # uuid.getnode() can fall back to a random multicast address. On Linux
    # read the first real NIC instead of reporting an empty MAC.
    if sys.platform.startswith("linux"):
        try:
            for name in sorted(os.listdir("/sys/class/net")):
                low = name.lower()
                if low == "lo" or low.startswith(("veth", "br-", "docker", "virbr", "tailscale")):
                    continue
                try:
                    addr = open(f"/sys/class/net/{name}/address",
                                encoding="utf-8").read().strip()
                except OSError:
                    continue
                if addr and addr != "00:00:00:00:00:00":
                    return addr
        except OSError:
            pass
    return ""


def _fmt_bytes(n) -> str:
    try:
        gb = float(n) / 1024 ** 3
    except (TypeError, ValueError):
        return ""
    if gb >= 1024:
        return f"{gb / 1024:.1f} TB"
    if gb >= 10:
        return f"{gb:.0f} GB"
    return f"{gb:.1f} GB"


def _fill_disk_total(free, total, mount):
    """Use the OS disk API when a shell probe omitted the total.

    df/Get-PSDrive parsing differs across agents and virtualized filesystems;
    shutil.disk_usage is the portable fallback and reports bytes.
    """
    free, total = str(free or ""), str(total or "")
    if total:
        return free, total
    try:
        used = shutil.disk_usage(mount or os.path.abspath(os.sep))
    except OSError:
        return free, total
    if not free:
        free = _fmt_bytes(used.free)
    return free, _fmt_bytes(used.total)


def _run(cmd, timeout=30):
    try:
        proc = subprocess.run(["bash", "-c", cmd], capture_output=True,
                              text=True, timeout=timeout)
        return proc.returncode, proc.stdout or "", proc.stderr or ""
    except (OSError, subprocess.SubprocessError) as e:
        return 1, "", str(e)


def _parse_posix(blob):
    out = {}
    for line in (blob or "").splitlines():
        if len(line) > 2 and line[1] == ":":
            out[line[0]] = line[2:].strip()
    return out


def _pcie_tags(raw):
    tags = []
    for line in (raw or "").splitlines():
        cells = [x.strip() for x in line.split(",")]
        if len(cells) == 2 and cells[0].isdigit() and cells[1].isdigit():
            tags.append(f"g{cells[0]}x{cells[1]}")
    return ",".join(tags)


def _rebar_values(raw):
    vals = [x.strip().split()[0] for x in (raw or "").splitlines()
            if x.strip()[:1].isdigit()]
    return ",".join(vals)


def _linux(config: AgentConfig):
    bin_dir = os.path.dirname(processes.llama_binary(config) or
                               shutil.which("llama-server") or "")
    cmd = (
        'K=$(uname -a); '
        "C=$(lscpu | grep 'Model name' | cut -d: -f2 | sed 's/^ *//'); "
        'U=$(uptime -p); '
        "I=$(hostname -I 2>/dev/null | awk '{print $1}'); "
        "S=$(for i in $(ls /sys/class/net | grep -v '^lo'); do "
        'd=$(readlink /sys/class/net/$i/device/driver 2>/dev/null); '
        'case "$i" in veth*|br-*|docker*|virbr*|tailscale*) continue;; esac; '
        's=$(cat /sys/class/net/$i/speed 2>/dev/null); '
        'if [ -n "$s" ] && [ "$s" != "-1" ] && [ "$s" -gt 0 ]; then '
        'if [ "$s" -ge 1000 ]; then echo "$((s/1000)) Gb/s"; '
        'else echo "${s} Mb/s"; fi; break; fi; done); '
        "if [ -z \"$S\" ]; then "
        "IF=$(ip route get 1.1.1.1 2>/dev/null | awk '{for(i=1;i<=NF;i++) if($i==\"dev\"){print $(i+1); exit}}'); "
        "[ -n \"$IF\" ] || IF=$(ls /sys/class/net | grep -v '^lo' | head -1); "
        "S=$(cat /sys/class/net/$IF/speed 2>/dev/null); "
        "if [ -z \"$S\" ] || [ \"$S\" = \"-1\" ]; then "
        "S=$(ethtool \"$IF\" 2>/dev/null | awk -F': ' '/Speed:/{print $2; exit}'); fi; "
        "case \"$S\" in *[Uu]nknown*) S=\"\" ;; esac; "
        "if [ -z \"$S\" ] && command -v iw >/dev/null 2>&1; then "
        "S=$(iw dev \"$IF\" link 2>/dev/null | awk -F': ' '/tx bitrate/{print $2; exit}'); fi; "
        "if [ -n \"$S\" ] && ! echo \"$S\" | grep -q '[A-Za-z]'; then "
        "if [ \"$S\" -ge 1000 ] 2>/dev/null; then S=\"$((S/1000)) Gb/s\"; "
        "else S=\"${S} Mb/s\"; fi; fi; fi; "
        "D=$(df -k / | tail -n1 | awk '{print $4}'); "
        "DT=$(df -k / | tail -n1 | awk '{print $2}'); "
        "V=$(systemd-detect-virt 2>/dev/null || echo bare); "
        "CF=$(lscpu | grep '^Flags:' | cut -d: -f2- | sed 's/^ *//'); "
        # Board model (Raspberry Pi 4/5, other SBCs); empty on most PCs.
        "B=$(tr -d '\\0' < /proc/device-tree/model 2>/dev/null); "
        'G=$(ldd --version 2>&1 | head -1); '
        'DRV=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader '
        '2>/dev/null | head -1); '
        "P=$(nvidia-smi --query-gpu=pcie.link.gen.current,pcie.link.width.current "
        "--format=csv,noheader,nounits 2>/dev/null); "
        "R=$(nvidia-smi --query-gpu=pci.bar1_memory_total "
        "--format=csv,noheader,nounits 2>/dev/null); "
        f'L=$("{bin_dir}/llama-server" --version 2>&1 || '
        f'"{bin_dir}/llama-cli" --version 2>&1); '
        'echo "K:$K"; echo "C:$C"; echo "U:$U"; echo "I:$I"; '
        'echo "S:$S"; echo "D:$D"; echo "DT:$DT"; echo "V:$V"; '
        'echo "CF:$CF"; echo "B:$B"; echo "P:$P"; echo "R:$R"; '
        'echo "L:$L"; echo "DRV:$DRV"'
    )
    code, out, _ = _run(cmd)
    if code != 0:
        return {}
    p = _parse_posix(out)
    disk_free, disk_total = _fill_disk_total(p.get("D", ""), p.get("DT", ""), "/")
    cores = []
    try:
        for line in open("/proc/stat", encoding="utf-8"):
            if line.startswith("cpu") and line.split()[0] != "cpu":
                parts = line.split()
                vals = [int(x) for x in parts[1:]]
                total = sum(vals)
                idle = vals[3] + (vals[4] if len(vals) > 4 else 0)
                cores.append(round((1 - idle / total) * 100, 1) if total else 0.0)
    except (OSError, ValueError):
        pass
    return {
        "kernel": p.get("K", ""), "cpu_model": p.get("C", ""),
        "uptime": p.get("U", ""), "net": {"ip": p.get("I", ""),
                                          "speed": p.get("S", "")},
        "disk": {"free": disk_free, "total": disk_total, "mount": "/"},
        "mac": _mac_address(),
        "llama_version": p.get("L", ""), "cores": os.cpu_count() or 0,
        "cpu_cores": cores, "arch": platform.machine(),
        "board": p.get("B", "") or _board_model(),
        "virt": p.get("V", ""), "cpu_flags": p.get("CF", ""),
        "glibc": p.get("G", ""), "driver_version": p.get("DRV", ""),
        "pcie": _pcie_tags(p.get("P", "")),
        "rebar": _rebar_values(p.get("R", "")),
    }


def _macos(config: AgentConfig):
    bin_dir = os.path.dirname(processes.llama_binary(config) or
                               shutil.which("llama-server") or "")
    cmd = """
echo "K:$(uname -a)"
echo "C:$(sysctl -n machdep.cpu.brand_string)"
echo "U:$(uptime)"
echo "I:$(ipconfig getifaddr en0 2>/dev/null || ipconfig getifaddr en1 2>/dev/null)"
echo "S:$(networksetup -getmedia en0 2>/dev/null | grep -i 'current' | sed 's/.*: //')"
echo "D:$(df -k / | tail -n1 | awk '{print $4}')"
echo "DT:$(df -k / | tail -n1 | awk '{print $2}')"
echo "L:$("BIN/llama-server" --version 2>&1 || "BIN/llama-cli" --version 2>&1)"
""".replace("BIN", bin_dir)
    code, out, _ = _run(cmd)
    if code != 0:
        return {}
    p = _parse_posix(out)
    disk_free, disk_total = _fill_disk_total(p.get("D", ""), p.get("DT", ""), "/")
    return {
        "kernel": p.get("K", ""), "cpu_model": p.get("C", ""),
        "uptime": p.get("U", ""),
        "net": {"ip": p.get("I", ""), "speed": p.get("S", "")},
        "disk": {"free": disk_free, "total": disk_total, "mount": "/"},
        "mac": _mac_address(),
        "llama_version": p.get("L", ""), "driver_version": "",
        "virt": "bare", "cpu_flags": "", "pcie": "", "rebar": "",
        "arch": platform.machine(), "board": "",
        "cores": int(subprocess.run(
            ["sysctl", "-n", "hw.ncpu"], capture_output=True, text=True,
            timeout=10).stdout.strip() or 0),
        "cpu_cores": [],
    }


def _windows(config: AgentConfig):
    bin_path = processes.llama_binary(config)
    ps = (
        "$o=@{};"
        "$o.kernel=(ver);"
        "$o.cpu=(Get-CimInstance Win32_Processor).Name;"
        "$o.uptime=((Get-Date)-(Get-CimInstance Win32_OperatingSystem).LastBootUpTime).ToString();"
        "$o.ip=((Get-NetIPAddress -AddressFamily IPv4 | Where-Object "
        "{$_.IPAddress -notlike '127.*' -and $_.IPAddress -notlike '169.254.*'} "
        "| Select-Object -First 1).IPAddress);"
        "$o.speed=((Get-NetAdapter | Where-Object Status -eq 'Up' | "
        "Select-Object -First 1).LinkSpeed);"
        "$o.disk=(Get-PSDrive -PSProvider FileSystem | Where-Object "
        "{$_.Free -gt 0} | Select-Object Name,Free,Used | ConvertTo-Json -Compress);"
        "$o.cores=(Get-CimInstance Win32_Processor | Measure-Object -Sum "
        "NumberOfLogicalProcessors).Sum;"
        "$o.coreload=(Get-CimInstance Win32_PerfFormattedData_PerfOS_Processor "
        "| Where-Object {$_.Name -ne '_Total'} | "
        "Select-Object -ExpandProperty PercentProcessorTime | "
        "ForEach-Object {[math]::Round($_,1)} | ConvertTo-Json -Compress);"
        f"$o.llama=((& \"{bin_path}\" --version 2>&1 | "
        "Select-String 'version:' | Select-Object -First 1).ToString());"
        "$o | ConvertTo-Json -Compress")
    try:
        proc = subprocess.run(
            ["powershell", "-NoProfile", "-Command", ps],
            capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return {}
    if proc.returncode != 0 or not proc.stdout.strip():
        return {}
    try:
        o = json.loads(proc.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        return {}
    disk = o.get("disk")
    if isinstance(disk, str):
        try:
            disk = json.loads(disk)
        except ValueError:
            disk = None
    free = used = total = ""
    mount = ""
    if isinstance(disk, list) and disk:
        item = disk[0] or {}
        free = str(item.get("Free") or "")
        used = str(item.get("Used") or "")
        mount = str(item.get("Name") or "")
    elif isinstance(disk, dict):
        free = str(disk.get("Free") or "")
        used = str(disk.get("Used") or "")
        mount = str(disk.get("Name") or "")
    try:
        if free and used:
            total = str(int(float(free)) + int(float(used)))
    except (TypeError, ValueError):
        total = ""
    target = f"{mount}:\\" if len(mount) == 1 else os.path.abspath(os.sep)
    free, total = _fill_disk_total(free, total, target)
    cl = o.get("coreload")
    if isinstance(cl, str):
        try:
            cl = json.loads(cl)
        except ValueError:
            cl = []
    return {
        "kernel": o.get("kernel", ""), "cpu_model": o.get("cpu", ""),
        "uptime": o.get("uptime", ""),
        "net": {"ip": o.get("ip", ""), "speed": o.get("speed", "")},
        "disk": {"free": free, "total": total, "mount": mount},
        "mac": _mac_address(),
        "llama_version": o.get("llama", ""), "cores": o.get("cores", 0),
        "cpu_cores": cl or [], "virt": "windows", "cpu_flags": "",
        "arch": platform.machine(), "board": "",
        "driver_version": str(o.get("driver") or ""),
        "pcie": "", "rebar": "",
    }


def detail(config: AgentConfig) -> dict:
    if os.name == "nt":
        return _windows(config)
    if sys.platform == "darwin":
        return _macos(config)
    return _linux(config)


_RDMA_SCRIPT = (
    'for dev in $(ibv_devices 2>/dev/null | awk "NR>2{print \\$1}"); do '
    'for port in 1 2; do '
    'st=$(cat /sys/class/infiniband/$dev/ports/$port/state 2>/dev/null); '
    '[[ "$st" != *ACTIVE* ]] && continue; '
    'gids=/sys/class/infiniband/$dev/ports/$port/gids; '
    'for gf in $(ls $gids 2>/dev/null); do '
    'gid=$(cat $gids/$gf 2>/dev/null); '
    'gt=$(cat /sys/class/infiniband/$dev/ports/$port/gid_attrs/types/$gf 2>/dev/null); '
    'nd=$(cat /sys/class/infiniband/$dev/ports/$port/gid_attrs/ndevs/$gf 2>/dev/null); '
    'norm=$(echo "$gid" | tr -d :); '
    '[[ ${#norm} -ne 32 ]] && continue; [[ "$gt" != *v2* ]] && continue; '
    '[[ "${norm:20:4}" != ffff ]] && continue; '
    'ip=$((16#${norm:24:2})).$((16#${norm:26:2})).$((16#${norm:28:2})).$((16#${norm:30:2})); '
    'echo "$dev|$port|$gf|$ip|$nd"; '
    'done; done; done'
)


def rdma():
    if os.name == "nt" or sys.platform == "darwin":
        # _RDMA_SCRIPT is Linux/netlink-specific; skip it on Windows and
        # macOS instead of burning a 20s timeout and reporting nothing.
        return {}
    code, out, _ = _run(_RDMA_SCRIPT, timeout=20)
    if code != 0:
        return {}
    for line in (out or "").splitlines():
        parts = line.strip().split("|")
        if len(parts) < 4:
            continue
        dev, port, gid, ip = parts[0], parts[1], parts[2], parts[3]
        if ip.startswith("10.0.0.") or (len(parts) > 4 and parts[4]):
            return {"dev": dev, "port": port, "gid": gid, "ip": ip}
    return {}
