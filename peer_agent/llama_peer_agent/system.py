"""Machine detail used by the dashboard modal."""
import json
import os
import shutil
import subprocess
import sys

from . import processes
from .config import AgentConfig


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
        "D=$(df -k / | tail -n1 | awk '{print $4}'); "
        "DT=$(df -k / | tail -n1 | awk '{print $2}'); "
        "V=$(systemd-detect-virt 2>/dev/null || echo bare); "
        "CF=$(lscpu | grep '^Flags:' | cut -d: -f2- | sed 's/^ *//'); "
        'G=$(ldd --version 2>&1 | head -1); '
        "P=$(nvidia-smi --query-gpu=pcie.link.gen.current,pcie.link.width.current "
        "--format=csv,noheader,nounits 2>/dev/null); "
        "R=$(nvidia-smi --query-gpu=pci.bar1_memory_total "
        "--format=csv,noheader,nounits 2>/dev/null); "
        f'L=$("{bin_dir}/llama-server" --version 2>&1 || '
        f'"{bin_dir}/llama-cli" --version 2>&1); '
        'echo "K:$K"; echo "C:$C"; echo "U:$U"; echo "I:$I"; '
        'echo "S:$S"; echo "D:$D"; echo "DT:$DT"; echo "V:$V"; '
        'echo "CF:$CF"; echo "P:$P"; echo "R:$R"; echo "L:$L"'
    )
    code, out, _ = _run(cmd)
    if code != 0:
        return {}
    p = _parse_posix(out)
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
        "disk": {"free": p.get("D", ""), "total": p.get("DT", "")},
        "llama_version": p.get("L", ""), "cores": os.cpu_count() or 0,
        "cpu_cores": cores,
        "virt": p.get("V", ""), "cpu_flags": p.get("CF", ""),
        "glibc": p.get("G", ""),
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
    return {
        "kernel": p.get("K", ""), "cpu_model": p.get("C", ""),
        "uptime": p.get("U", ""),
        "net": {"ip": p.get("I", ""), "speed": p.get("S", "")},
        "disk": {"free": p.get("D", ""), "total": p.get("DT", "")},
        "llama_version": p.get("L", ""),
        "virt": "bare", "cpu_flags": "", "pcie": "", "rebar": "",
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
    free = total = ""
    if isinstance(disk, list) and disk:
        free = str((disk[0] or {}).get("Free") or "")
    elif isinstance(disk, dict):
        free = str(disk.get("Free") or "")
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
        "disk": {"free": free, "total": total},
        "llama_version": o.get("llama", ""), "cores": o.get("cores", 0),
        "cpu_cores": cl or [], "virt": "windows", "cpu_flags": "",
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
    if os.name == "nt":
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
