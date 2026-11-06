"""Lightweight machine metrics for the agent (stdlib only)."""
import json
import os
import platform
import shutil
import subprocess
import sys
import time

_NVIDIA = ("nvidia-smi", "--query-gpu=index,name,utilization.gpu,"
           "memory.used,memory.total,temperature.gpu,compute_cap,uuid",
           "--format=csv,noheader,nounits")


def _run(args, timeout=20, shell=False):
    try:
        proc = subprocess.run(
            args, capture_output=True, text=True, timeout=timeout,
            shell=shell)
        return proc.returncode, proc.stdout or "", proc.stderr or ""
    except (OSError, subprocess.SubprocessError) as e:
        return 1, "", str(e)


def _nvidia_gpus():
    if not shutil.which(_NVIDIA[0]):
        return []
    code, out, _ = _run(list(_NVIDIA), timeout=15)
    if code != 0:
        return []
    rows = []
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 6:
            continue
        try:
            total = float(parts[4])
            used = float(parts[3])
            rows.append({
                "index": parts[0], "name": parts[1],
                "util": float(parts[2]),
                "mem_used_gb": round(used / 1024, 1),
                "mem_total_gb": round(total / 1024, 1),
                "mem_pct": round(used / total * 100) if total else 0,
                "temp": int(float(parts[5])),
                "compute_cap": parts[6] if len(parts) > 6 else None,
                "uuid": parts[7] if len(parts) > 7 else None,
            })
        except (ValueError, TypeError, ZeroDivisionError):
            continue
    return rows


_gpus = _nvidia_gpus  # backward-compatible name (NVIDIA rows)


def _vendor_from_names(names):
    blob = " ".join(str(n) for n in names if n).lower()
    if any(k in blob for k in ("nvidia", "geforce", "rtx", "tesla", "quadro")):
        return "nvidia"
    if any(k in blob for k in ("amd", "radeon", "instinct", "firepro")):
        return "amd"
    if "apple" in blob:
        return "apple"
    if "intel" in blob or "arc" in blob:
        return "intel"
    return ""


def _amd_gpus():
    if not shutil.which("rocm-smi"):
        return []
    code, out, _ = _run([
        "rocm-smi", "--showproductname", "--showmeminfo", "vram", "--json",
    ], timeout=25)
    if code != 0 or not out.strip():
        return []
    try:
        data = json.loads(out[out.find("{"):out.rfind("}") + 1])
    except (ValueError, TypeError):
        return []
    rows = []
    for i, (_key, val) in enumerate(sorted(data.items())):
        if not isinstance(val, dict):
            continue
        name = next((val[k] for k in (
            "Card series", "Card Series", "Card SKU", "Device Name",
            "GPU Name", "Product Name") if val.get(k)), "AMD GPU")
        total_b = next((val[k] for k in (
            "VRAM Total Memory (B)", "VRAM Total Memory (byte)",
            "VRAM Total Memory (bytes)") if val.get(k)), 0)
        used_b = next((val[k] for k in (
            "VRAM Total Used Memory (B)", "VRAM Total Used Memory (byte)",
            "VRAM Total Used Memory (bytes)") if val.get(k)), 0)
        try:
            total = float(total_b) / 1024 ** 3
        except (TypeError, ValueError):
            total = 0.0
        try:
            used = float(used_b) / 1024 ** 3
        except (TypeError, ValueError):
            used = 0.0
        rows.append({
            "index": str(i), "name": str(name), "util": 0.0,
            "mem_used_gb": round(used, 1),
            "mem_total_gb": round(total, 1),
            "mem_pct": round(used / total * 100) if total else 0,
            "temp": 0,
        })
    return rows


def _intel_gpus():
    if not shutil.which("xpu-smi"):
        return []
    code, out, _ = _run(["xpu-smi", "discovery", "-j"], timeout=25)
    if code != 0 or not out.strip():
        return []
    try:
        data = json.loads(out[out.find("{"):out.rfind("}") + 1])
    except (ValueError, TypeError):
        return []

    found = []

    def walk(obj):
        if isinstance(obj, list):
            for item in obj:
                walk(item)
        elif isinstance(obj, dict):
            name = obj.get("device_name") or obj.get("name")
            mem = (obj.get("memory_physical_size_byte")
                   or obj.get("memory_physical_size")
                   or obj.get("memory_size"))
            if name and mem not in (None, "", 0):
                found.append((str(name), mem))
            for v in obj.values():
                walk(v)

    walk(data)
    rows = []
    seen = set()
    for i, (name, mem) in enumerate(found):
        if name in seen:
            continue
        seen.add(name)
        try:
            total = float(mem) / 1024 ** 3
        except (TypeError, ValueError):
            total = 0.0
        rows.append({
            "index": str(i), "name": name, "util": 0.0,
            "mem_used_gb": 0, "mem_total_gb": round(total, 1),
            "mem_pct": 0, "temp": 0,
        })
    return rows


def _apple_gpus():
    if sys.platform != "darwin":
        return []
    code, out, _ = _run(
        ["system_profiler", "SPDisplaysDataType", "-json"], timeout=35)
    if code != 0 or not out.strip():
        return []
    try:
        data = json.loads(out[out.find("{"):out.rfind("}") + 1])
    except (ValueError, TypeError):
        return []
    rows = []
    idx = 0
    for g in (data.get("SPDisplaysDataType") or []):
        for item in (g.get("spdisplays_ndrvs") or [g]):
            name = item.get("_name") or item.get("sppci_model") or "Apple GPU"
            import re as _re
            total = item.get("spdisplays_vram_shared") \
                or item.get("spdisplays_vram") or ""
            m = _re.search(r"([\d.]+)\s*(GB|MB|TB)", str(total))
            total_gb = 0.0
            if m:
                val, unit = float(m.group(1)), m.group(2).upper()
                total_gb = round(
                    val * 1024 if unit == "TB"
                    else val if unit == "GB" else val / 1024, 1)
            rows.append({
                "index": str(idx), "name": str(name), "util": 0.0,
                "mem_used_gb": 0, "mem_total_gb": total_gb,
                "mem_pct": 0, "temp": 0,
            })
            idx += 1
    return rows


def _win_fallback_gpus():
    ps = ("Get-CimInstance Win32_VideoController | "
          "Select-Object Name,AdapterRAM | ConvertTo-Json -Compress")
    code, out, _ = _run(["powershell", "-NoProfile", "-Command", ps],
                        timeout=25)
    if code != 0 or not out.strip():
        return []
    try:
        data = json.loads(out.strip().splitlines()[-1])
    except (ValueError, IndexError):
        return []
    if isinstance(data, dict):
        data = [data]
    rows = []
    for i, g in enumerate(data or []):
        name = str(g.get("Name") or "GPU")
        try:
            total = float(g.get("AdapterRAM") or 0) / 1024 ** 3
        except (TypeError, ValueError):
            total = 0.0
        rows.append({
            "index": str(i), "name": name, "util": 0.0,
            "mem_used_gb": 0, "mem_total_gb": round(total, 1),
            "mem_pct": 0, "temp": 0,
        })
    return rows


def gpus():
    """(rows, vendor) — NVIDIA first, then ROCm, Intel, Win WMI, Apple."""
    rows = _nvidia_gpus()
    if rows:
        return rows, "nvidia"
    rows = _amd_gpus()
    if rows:
        return rows, "amd"
    rows = _intel_gpus()
    if rows:
        return rows, "intel"
    if os.name == "nt":
        rows = _win_fallback_gpus()
        if rows:
            return rows, _vendor_from_names([r["name"] for r in rows])
    if sys.platform == "darwin":
        rows = _apple_gpus()
        if rows:
            return rows, "apple"
    return [], ""


def _linux():
    def stat():
        try:
            with open("/proc/stat", encoding="utf-8") as f:
                lines = [ln.split() for ln in f if ln.startswith("cpu")]
        except OSError:
            return None
        out = []
        for parts in lines:
            try:
                vals = [int(x) for x in parts[1:]]
            except ValueError:
                continue
            idle = vals[3] + (vals[4] if len(vals) > 4 else 0)
            total = sum(vals)
            out.append((parts[0], total, idle))
        return out

    first = stat()
    time.sleep(0.3)
    second = stat()
    cpu_pct = 0.0
    cores = []
    if first and second and len(first) == len(second):
        for (n1, t1, i1), (n2, t2, i2) in zip(first, second):
            dt, di = t2 - t1, i2 - i1
            pct = round((1 - di / dt) * 100, 1) if dt > 0 else 0.0
            if n1 == "cpu":
                cpu_pct = pct
            else:
                cores.append(pct)
    try:
        load = open("/proc/loadavg", encoding="utf-8").read().split()[:3]
        load = " ".join(load)
    except OSError:
        load = ""
    total_kb = avail_kb = 0
    try:
        with open("/proc/meminfo", encoding="utf-8") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    total_kb = int(line.split()[1])
                elif line.startswith("MemAvailable:"):
                    avail_kb = int(line.split()[1])
    except (OSError, ValueError):
        pass
    rows, vendor = gpus()
    return {
        "status": "online", "os_type": "linux",
        "gpus": rows, "gpu_vendor": vendor, "cpu_pct": cpu_pct,
        "cores": os.cpu_count() or 0, "load": load,
        "ram_total_gb": round(total_kb / 1024 / 1024, 1),
        "ram_used_gb": round((total_kb - avail_kb) / 1024 / 1024, 1),
        "cpu_cores": cores,
    }


def _macos():
    code, cores, _ = _run(["sysctl", "-n", "hw.ncpu"], timeout=10)
    code2, top, _ = _run(["bash", "-c", "top -l 1 -n 0 | grep 'CPU usage'"],
                         timeout=20)
    code3, load, _ = _run(["bash", "-c", "uptime"], timeout=10)
    code4, mem, _ = _run(["sysctl", "-n", "hw.memsize"], timeout=10)
    code5, vm, _ = _run(["bash", "-c", "vm_stat | head -6"], timeout=10)
    pct = 0.0
    if code2 == 0:
        import re
        m = re.search(r"(\d+\.?\d*)%\s*idle", top)
        if m:
            pct = round(100.0 - float(m.group(1)), 1)
    total_gb = used_gb = 0.0
    try:
        total_b = int(mem.strip())
        total_gb = round(total_b / 1024 ** 3, 1)
        import re
        free = 0
        for label in ("Pages free", "Pages speculative"):
            m = re.search(label + r":\s*(\d+)", vm)
            if m:
                free += int(m.group(1)) * 4096
        used_gb = round(max(0, total_b - free) / 1024 ** 3, 1)
    except (ValueError, IndexError):
        pass
    rows = _apple_gpus()
    return {
        "status": "online", "os_type": "macos", "gpus": rows,
        "gpu_vendor": "apple" if rows else "",
        "cpu_pct": pct, "cores": int(cores.strip() or 0) if code == 0 else 0,
        "load": load.strip() if code3 == 0 else "",
        "ram_total_gb": total_gb, "ram_used_gb": used_gb,
    }


def _windows():
    ps = (
        "$o=@{}; "
        "$cpu=Get-CimInstance -ClassName Win32_PerfFormattedData_PerfOS_Processor "
        "| Where-Object {$_.Name -eq '_Total'}; "
        "$o.cpu=if($cpu){$cpu.PercentProcessorTime}else{0}; "
        "$o.cores=(Get-CimInstance Win32_Processor | Measure-Object "
        "NumberOfLogicalProcessors -Sum).Sum; "
        "$os=Get-CimInstance Win32_OperatingSystem; "
        "$o.ram_total=$os.TotalVisibleMemorySize; $o.ram_free=$os.FreePhysicalMemory; "
        "$o.gpus=(& nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,"
        "memory.total,temperature.gpu,compute_cap,uuid --format=csv,noheader,nounits "
        "2>$null) -join \"`n\"; $o | ConvertTo-Json -Compress")
    code, out, _ = _run(["powershell", "-NoProfile", "-Command", ps],
                        timeout=25)
    if code != 0 or not out.strip():
        return {"status": "offline", "os_type": "windows", "gpus": [],
                "cpu_pct": 0.0, "cores": 0, "load": "",
                "gpu_vendor": "", "ram_total_gb": 0, "ram_used_gb": 0}
    try:
        o = json.loads(out.strip().splitlines()[-1])
    except (ValueError, IndexError):
        return {"status": "offline", "os_type": "windows", "gpus": [],
                "cpu_pct": 0.0, "cores": 0, "load": "",
                "gpu_vendor": "", "ram_total_gb": 0, "ram_used_gb": 0}
    rows = []
    for line in str(o.get("gpus") or "").splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 6:
            continue
        try:
            total, used = float(parts[4]), float(parts[3])
            rows.append({"index": parts[0], "name": parts[1],
                         "util": float(parts[2]),
                         "mem_used_gb": round(used / 1024, 1),
                         "mem_total_gb": round(total / 1024, 1),
                         "mem_pct": round(used / total * 100) if total else 0,
                         "temp": int(float(parts[5]))})
        except (ValueError, TypeError, ZeroDivisionError):
            continue
    vendor = "nvidia" if rows else ""
    if not rows:
        rows = _win_fallback_gpus()
        vendor = _vendor_from_names([r["name"] for r in rows]) if rows else ""
    total = int(o.get("ram_total") or 0)
    free = int(o.get("ram_free") or 0)
    return {"status": "online", "os_type": "windows", "gpus": rows,
            "gpu_vendor": vendor,
            "cpu_pct": float(o.get("cpu") or 0),
            "cores": int(o.get("cores") or 0), "load": "",
            "ram_total_gb": round(total / 1024 / 1024, 1),
            "ram_used_gb": round((total - free) / 1024 / 1024, 1)}


def collect() -> dict:
    if os.name == "nt":
        return _windows()
    if sys.platform == "darwin":
        return _macos()
    return _linux()
