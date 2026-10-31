"""Per-machine stats collection: GPU util/VRAM, CPU, RAM — mimics
ssh-watch's cluster overview, reusing our SSHManager.

One SSH round-trip per machine using a combined query string (linux)
or a single PowerShell call (windows), instead of ssh-watch's
separate query per metric. GPU queries are vendor-aware:
nvidia-smi (NVIDIA) or rocm-smi JSON (AMD), normalized to the same
csv rows either way.
"""
import subprocess

from ssh_manager import SSHManager
import local_node
import gpu_vendor

GPU_QUERY = gpu_vendor.nvidia_query(
    "index,name,utilization.gpu,memory.used,memory.total,temperature.gpu,"
    "compute_cap,uuid")

AMD_GPU_QUERY = gpu_vendor.rocm_query_json()
INTEL_GPU_QUERY = gpu_vendor.intel_query_json()


def _gpu_section(machine):
    """GPU command for a machine's shell one-liner (linux)."""
    v = gpu_vendor.normalize_vendor(machine.get("gpu_vendor"))
    if v == "amd":
        return AMD_GPU_QUERY
    if v == "intel":
        return INTEL_GPU_QUERY
    return GPU_QUERY

# linux: gpus + cpu% + cores + loadavg + ram in ONE shell invocation
def _linux_all(machine):
    return (
    _gpu_section(machine) + "; echo ---; "
    "read -r _ u1 n1 s1 i1 io1 irq1 si1 st1 _ < /proc/stat; "
    "sleep 0.3; "
    "read -r _ u2 n2 s2 i2 io2 irq2 si2 st2 _ < /proc/stat; "
    "t1=$((u1+n1+s1+i1+io1+irq1+si1+st1)); "
    "t2=$((u2+n2+s2+i2+io2+irq2+si2+st2)); "
    "idle1=$((i1+io1)); idle2=$((i2+io2)); "
    "dt=$((t2-t1)); di=$((idle2-idle1)); "
    "if [ \"$dt\" -gt 0 ]; then "
    "awk -v dt=\"$dt\" -v di=\"$di\" 'BEGIN{printf \"%.1f\", (1-(di/dt))*100}'; "
    "else echo 0.0; fi; "
    "echo ---; nproc; "
    "echo ---; cut -d' ' -f1-3 /proc/loadavg; "
    "echo ---; awk '/^MemTotal:/{print $2}' /proc/meminfo; "
    "awk '/^MemAvailable:/{print $2}' /proc/meminfo; "
    "echo ---; "
    "grep '^cpu' /proc/stat | awk '{u=$2+$3+$4; i=$5+$6; t=u+i; "
    "if (t>0) printf \"%s %.1f\\n\", $1, (u/t)*100}'"
)


# windows: same fields via one PowerShell invocation
def _win_ps(machine):
    _v = gpu_vendor.normalize_vendor(machine.get("gpu_vendor"))
    amd = _v == "amd"
    intel = _v == "intel"
    gpu_cmd = (f"$j = & xpu-smi stats -d -1 -j 2>$null | ConvertFrom-Json; "
               "$g = $j | ForEach-Object { "
               "\"$($_.device_id),Intel GPU $($_.device_id),"
               "$($_.'GPU utilization (%)'),$($_.'GPU Memory Used (MiB)'),0,"
               "$($_.'GPU Core Temperature (C)')\" } "
               if intel else f"$j = & rocm-smi {AMD_GPU_QUERY.split('rocm-smi ')[1]} "
               "2>$null | ConvertFrom-Json; "
               "$g = $j.PSObject.Properties | Where-Object {$_.Name -like 'card*'} | "
               "ForEach-Object { "
               "$i = $_.Name -replace '\\D',''; "
               "$tot = $_.Value.'VRAM TOTAL'; $used = $_.Value.'VRAM TOTAL USED'; "
               "\"$i,AMD GPU $i,$([math]::Round($tot/1MB)),$([math]::Round($used/1MB)),0\" "
               "}" if amd else
                "$g = & nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,"
                "memory.total,temperature.gpu,compute_cap,uuid --format=csv,noheader,nounits 2>$null;")
    return (
    "$o=@{};"
    + gpu_cmd +
    "$o.gpus = ($g -join [Environment]::NewLine);"
    # CPU% like Task Manager: Get-Counter samples ALL logical processors
    # and matches its number, but it costs ~3.5s per poll (perf-counter
    # module autoload + two samples with a 500ms sleep) — the whole
    # reason the dashboard was slow to refresh. WMI's
    # Win32_PerfFormattedData_PerfOS_Processor (locale-independent,
    # ~0.3s) tracks Task Manager closely and needs no sample pair; it's
    # the same source the old non-English fallback used. Dropped
    # Get-Counter entirely after live A/B on winbox: 4.9s → ~1.0s blob,
    # CPU values within a few points of Task Manager.
    "$cpu = Get-CimInstance -ClassName Win32_PerfFormattedData_PerfOS_Processor "
    "| Where-Object {$_.Name -eq '_Total'};"
    "$o.cpu = if ($cpu) { $cpu.PercentProcessorTime } else { 0 };"
    # logical processors, NOT physical cores — Windows scheduling (and
    # Task Manager's "logical processors" count) only cares about these
    "$o.cores = (Get-CimInstance -ClassName Win32_Processor -Property "
    "NumberOfLogicalProcessors | Measure-Object -Sum "
    "NumberOfLogicalProcessors).Sum;"
    "$os = Get-CimInstance -ClassName Win32_OperatingSystem "
    "-Property TotalVisibleMemorySize,FreePhysicalMemory;"
    "$o.ram_total = $os.TotalVisibleMemorySize; $o.ram_free = $os.FreePhysicalMemory;"
    "$o | ConvertTo-Json -Compress"
)


def _parse_gpus(output):
    """Parse GPU rows. Accepts nvidia-smi CSV rows AND rocm-smi JSON
    (normalized to csv rows first) so both vendors share one parser."""
    stripped = (output or "").strip()
    if stripped.startswith("{") or stripped.startswith("["):
        output = "\n".join(gpu_vendor.parse_rocm_json(stripped))
    gpus = []
    for line in (output or "").splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 6:
            continue
        try:
            total = float(parts[4])
            used = float(parts[3])
            # compute_cap is the 7th field on new queries; older cached
            # rows (and AMD/Intel rows) have 6 — absent stays None.
            # uuid is 8th (nvidia only); absent stays None.
            cap = parts[6].strip() if len(parts) > 6 else ""
            uid = parts[7].strip() if len(parts) > 7 else ""
            gpus.append({
                "index": parts[0],
                "name": parts[1],
                "compute_cap": cap if gpu_vendor.parse_compute_cap(cap)
                else None,
                "uuid": uid if uid.startswith("GPU-") else None,
                "util": float(parts[2]),
                "mem_used_gb": round(used / 1024, 1),
                "mem_total_gb": round(total / 1024, 1),
                "mem_pct": round(used / total * 100) if total else 0,
                "temp": int(float(parts[5])),
            })
        except (ValueError, TypeError, ZeroDivisionError):
            continue
    return gpus


def _empty(machine_os):
    return {
        "status": "offline", "os_type": machine_os, "gpus": [],
        "cpu_pct": 0.0, "cores": 0, "load": "", "ram_used_gb": 0,
        "ram_total_gb": 0,
    }


def collect_linux(ssh, machine=None):
    try:
        code, out, _ = ssh.run(_linux_all(machine or {}), timeout=25)
    except Exception:
        return _empty("linux")
    if code != 0:
        return _empty("linux")
    sections = out.split("---")
    stats = {
        "status": "online", "os_type": "linux",
        "gpus": _parse_gpus(sections[0]),
        "cpu_pct": float(sections[1].strip() or 0),
        "cores": int(sections[2].strip() or 0),
        "load": sections[3].strip(),
        "ram_used_gb": 0, "ram_total_gb": 0,
    }
    try:
        total_kb = int(sections[4].split()[0])
        avail_kb = int(sections[4].split()[1])
        stats["ram_total_gb"] = round(total_kb / 1024 / 1024, 1)
        stats["ram_used_gb"] = round((total_kb - avail_kb) / 1024 / 1024, 1)
    except (ValueError, IndexError):
        pass
    # per-core CPU util (section 5): "cpu0 12.3\ncpu1 45.6\n..."
    stats["cpu_cores"] = []
    if len(sections) > 5:
        for line in sections[5].strip().splitlines():
            parts = line.split()
            if len(parts) == 2 and parts[0].startswith("cpu"):
                try:
                    stats["cpu_cores"].append(round(float(parts[1]), 1))
                except ValueError:
                    pass
    return stats


def collect_windows(ssh, machine=None):
    try:
        code, out, _ = ssh.run(
            f'powershell -NoProfile -Command "{_win_ps(machine or {})}"',
            timeout=25)
    except Exception:
        return _empty("windows")
    if code != 0 or not out.strip():
        return _empty("windows")
    import json
    try:
        o = json.loads(out.strip().splitlines()[-1])
    except (ValueError, IndexError):
        return _empty("windows")
    return {
        "status": "online", "os_type": "windows",
        "gpus": _parse_gpus(o.get("gpus", "")),
        "cpu_pct": float(o.get("cpu") or 0),
        "cores": int(o.get("cores") or 0),
        "load": "",
        "ram_total_gb": round(int(o.get("ram_total") or 0) / 1024 / 1024, 1),
        "ram_used_gb": round(
            (int(o.get("ram_total") or 0) - int(o.get("ram_free") or 0))
            / 1024 / 1024, 1),
    }


def collect_local(machine_os, machine=None):
    """Local machine stats without SSH (like ssh-watch's run_local_*)."""
    machine = machine or {}
    if machine_os == "windows":
        try:
            proc = subprocess.run(
                ["powershell", "-NoProfile", "-Command", _win_ps(machine)],
                capture_output=True, text=True, timeout=25,
            )
            if proc.returncode == 0:
                import json
                o = json.loads(proc.stdout.strip().splitlines()[-1])
                return {
                    "status": "online", "os_type": "windows",
                    "gpus": _parse_gpus(o.get("gpus", "")),
                    "cpu_pct": float(o.get("cpu") or 0),
                    "cores": int(o.get("cores") or 0), "load": "",
                    "ram_total_gb": round(int(o.get("ram_total") or 0) / 1024 / 1024, 1),
                    "ram_used_gb": round(
                        (int(o.get("ram_total") or 0) - int(o.get("ram_free") or 0))
                        / 1024 / 1024, 1),
                }
        except (OSError, ValueError, IndexError, subprocess.TimeoutExpired):
            pass
        return _empty("windows")
    # local linux — GPU query per vendor, with auto-fallback when the
    # configured vendor's tool is missing (e.g. mislabeled machine)
    if machine_os == "macos":
        return collect_macos(None, machine)
    vendor = gpu_vendor.normalize_vendor(machine.get("gpu_vendor"))
    if vendor not in ("nvidia", "amd", "intel"):
        vendor = gpu_vendor.detect_vendor_local()
    if vendor == "amd":
        gpu_cmd = (AMD_GPU_QUERY + " || " + GPU_QUERY
                   + " || " + INTEL_GPU_QUERY + " 2>/dev/null")
    elif vendor == "intel":
        gpu_cmd = (INTEL_GPU_QUERY + " || " + GPU_QUERY
                   + " || " + AMD_GPU_QUERY + " 2>/dev/null")
    else:
        gpu_cmd = (GPU_QUERY + " || " + AMD_GPU_QUERY
                   + " || " + INTEL_GPU_QUERY + " 2>/dev/null")
    linux_all = _linux_all({}).replace(_gpu_section({}), gpu_cmd, 1)
    try:
        proc = subprocess.run(
            ["bash", "-c", linux_all], capture_output=True, text=True, timeout=25
        )
        if proc.returncode == 0:
            fake = type("S", (), {"run": lambda self, c, timeout=30: (0, proc.stdout, "")})()
            return collect_linux(fake)
    except (OSError, subprocess.TimeoutExpired):
        pass
    return _empty("linux")


def collect_macos(ssh, machine=None):
    """macOS stats: sysctl/vm_stat/top (no /proc, no nvidia-smi).
    Apple Silicon GPUs report via system_profiler; util/temp are 0
    (powermetrics needs sudo)."""
    MAC_ALL = (
        "sysctl -n hw.ncpu; echo ---; "
        "top -l 1 -n 0 | grep 'CPU usage'; echo ---; "
        "cut -d' ' -f1-3 < <(uptime); echo ---; "
        "sysctl -n hw.memsize; echo ---; "
        "vm_stat | head -6; echo ---; "
        "system_profiler SPDisplaysDataType -json 2>/dev/null | head -c 20000"
    )
    try:
        if ssh:
            code, out, _ = ssh.run(MAC_ALL, timeout=40)
        else:
            proc = subprocess.run(["bash", "-c", MAC_ALL],
                                  capture_output=True, text=True, timeout=40)
            code, out = proc.returncode, proc.stdout
    except Exception:
        return _empty("macos")
    if code != 0 or not out.strip():
        return _empty("macos")
    secs = out.split("---")
    stats = {"status": "online", "os_type": "macos", "gpus": [],
             "cpu_pct": 0.0, "cores": 0, "load": "",
             "ram_used_gb": 0, "ram_total_gb": 0}
    try:
        stats["cores"] = int(secs[0].strip() or 0)
    except (ValueError, IndexError):
        pass
    # "CPU usage: 4.20% user, 6.87% sys, 88.92% idle"
    try:
        import re as _re
        m = _re.search(r"(\d+\.?\d*)%\s*idle", secs[1])
        if m:
            stats["cpu_pct"] = round(100.0 - float(m.group(1)), 1)
    except (ValueError, IndexError):
        pass
    try:
        stats["load"] = secs[2].strip()
    except IndexError:
        pass
    # RAM: total bytes; used = total - (free+speculative pages * 4096)
    try:
        total_b = int(secs[3].strip())
        stats["ram_total_gb"] = round(total_b / 1024 ** 3, 1)
        import re as _re
        free = 0
        for label in ("Pages free", "Pages speculative"):
            m = _re.search(label + r":\s*(\d+)", secs[4])
            if m:
                free += int(m.group(1)) * 4096
        stats["ram_used_gb"] = round(
            max(0, total_b - free) / 1024 ** 3, 1)
    except (ValueError, IndexError):
        pass
    # GPUs from system_profiler JSON (last section)
    try:
        blob = secs[5].strip()
        if blob.startswith("{"):
            import gpu_vendor as _gv
            import json as _json
            data = _json.loads(blob[:blob.rfind("}") + 1])
            rows = []
            idx = 0
            for g in (data.get("SPDisplaysDataType") or []):
                for item in (g.get("spdisplays_ndrvs") or [g]):
                    name = (item.get("_name")
                            or item.get("spdisplays_device_id") or "Apple GPU")
                    name = str(name).replace("Apple ", "").replace(",", "")
                    total = item.get("spdisplays_vram_shared") \
                        or item.get("spdisplays_vram") or ""
                    import re as _re
                    m = _re.search(r"([\d.]+)\s*(GB|MB|TB)", str(total))
                    total_gb = 0.0
                    if m:
                        val, unit = float(m.group(1)), m.group(2).upper()
                        total_gb = round(
                            val * 1024 if unit == "TB"
                            else val if unit == "GB" else val / 1024, 1)
                    rows.append({
                        "index": idx, "name": name, "util": 0.0,
                        "mem_used_gb": 0, "mem_total_gb": total_gb,
                        "mem_pct": 0, "temp": 0,
                    })
                    idx += 1
            stats["gpus"] = rows
    except (ValueError, IndexError):
        pass
    return stats


def machine_stats_light(machine):
    """Fast reachability-only stats for the overview (no SSH collection).
    Returns identity + online/offline + rpc state, with empty gpus/cpu/ram.
    The detail modal calls machine_stats() for the full picture on demand."""
    machine_os = machine.get("os", "linux")
    is_local = local_node.is_local(machine)
    # rpc_running = any ggml-rpc-server process (per-model ports mean the
    # base port alone is no longer a reliable signal)
    import rpc_manager
    rpc = False
    if is_local:
        rpc = rpc_manager._any_rpc_running_local(machine)
    else:
        try:
            ssh = SSHManager(
                machine["ip"], machine["username"], machine["password"],
                port=machine.get("port", 22),
            )
            reachable = ssh.is_reachable()
            rpc = rpc_manager._any_rpc_running_remote(machine, ssh) if reachable else False
        except Exception:
            reachable = False
            rpc = False
    s = _empty(machine_os)
    s["status"] = "online" if (is_local or rpc) else "offline"
    return {**s, "name": machine["name"], "ip": machine["ip"],
            "os": machine_os, "is_local": is_local,
            "is_master": machine.get("role") == "master",
            "rpc_running": rpc,
            "rdma": bool(machine.get("rdma")),
            "rdma_cm": bool(machine.get("rdma_cm"))}


def machine_stats(machine):
    """Collect full stats for one machine config dict, ready for the
    dashboard: health + gpus + cpu/ram + identity (local/master/rpc)."""
    machine_os = machine.get("os", "linux")
    is_local = local_node.is_local(machine)
    try:
        if is_local:
            s = collect_local(machine_os, machine)
        else:
            ssh = SSHManager(
                machine["ip"], machine["username"], machine["password"],
                port=machine.get("port", 22),
            )
            if machine_os == "windows":
                s = collect_windows(ssh, machine)
            elif machine_os == "macos":
                s = collect_macos(ssh, machine)
            else:
                s = collect_linux(ssh, machine)
    except Exception:
        s = _empty(machine_os)
    import rpc_manager
    if is_local:
        rpc = rpc_manager._any_rpc_running_local(machine)
    else:
        try:
            ssh = SSHManager(
                machine["ip"], machine["username"], machine["password"],
                port=machine.get("port", 22),
            )
            rpc = rpc_manager._any_rpc_running_remote(machine, ssh)
        except Exception:
            rpc = False
    return {**s, "name": machine["name"], "ip": machine["ip"],
            "os": machine_os, "is_local": is_local,
            "is_master": machine.get("role") == "master",
            "rpc_running": rpc}


def cluster_totals(stats_list):
    """Aggregate totals across machines, like ssh-watch's compute_totals."""
    vram_used = vram_total = 0
    ram_used = ram_total = 0
    cpu_weighted = 0.0
    cores_total = 0
    for s in stats_list:
        if s["status"] != "online":
            continue
        for g in s["gpus"]:
            vram_used += g["mem_used_gb"]
            vram_total += g["mem_total_gb"]
        ram_used += s["ram_used_gb"]
        ram_total += s["ram_total_gb"]
        cpu_weighted += s["cpu_pct"] * s["cores"]
        cores_total += s["cores"]
    vram_pct = round(vram_used / vram_total * 100, 1) if vram_total else 0
    ram_pct = round(ram_used / ram_total * 100, 1) if ram_total else 0
    cpu_pct = round(cpu_weighted / cores_total, 1) if cores_total else 0
    return {
        "vram": {"used": round(vram_used, 1), "total": round(vram_total, 1),
                 "pct": vram_pct},
        "ram": {"used": round(ram_used, 1), "total": round(ram_total, 1),
                "pct": ram_pct},
        "cpu": {"pct": cpu_pct, "busy": round(cpu_weighted / 100, 1),
                "total": cores_total},
    }


import local_node  # noqa: E402  (needed at bottom: local_node imports config_loader)