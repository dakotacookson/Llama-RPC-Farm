"""GPU vendor support: NVIDIA (CUDA) and AMD (ROCm/HIP).

llama.cpp device naming facts (verified):
  - CUDA builds:      devices CUDA0, CUDA1, ...
  - HIP builds (new): devices ROCm0, ROCm1, ...  (renamed from HIP0
    in llama.cpp b9748; HIP0 kept for older builds)
Env:
  - CUDA_VISIBLE_DEVICES  (nvidia)  <->  HIP_VISIBLE_DEVICES / ROCR_VISIBLE_DEVICES (amd)
  - CUDA_DEVICE_ORDER is NVIDIA-only (no-op on ROCm)
  - GGML_CUDA_P2P works on both (same ggml-cuda code path)
Monitoring:
  - nvidia-smi --query-gpu=...  <->  rocm-smi --showuse --showmeminfo vram --json
    (rocm-smi JSON fields: "GPU use (%)", "VRAM Used (B)", "VRAM Total (B)")
"""
import json
import subprocess

GPU_TOKEN_PREFIXES = ("CUDA", "ROCm", "HIP", "Metal", "SYCL")


def normalize_vendor(v):
    """'amd'/'rocm' -> 'amd', 'nvidia'/'cuda' -> 'nvidia',
    'apple'/'metal' -> 'apple', else ''."""
    v = (v or "").strip().lower()
    if v in ("amd", "rocm", "hip"):
        return "amd"
    if v in ("nvidia", "cuda"):
        return "nvidia"
    if v in ("apple", "metal", "mlx"):
        return "apple"
    if v in ("intel", "sycl", "xpu"):
        return "intel"
    if v == "cpu":
        return "cpu"
    return ""


def gpu_prefix(machine=None):
    """llama.cpp device-name prefix for a machine's GPUs.
    Apple Silicon: the Metal backend registers ONE device literally
    named 'Metal' (no index suffix), so the prefix IS the token."""
    v = normalize_vendor((machine or {}).get("gpu_vendor"))
    if v == "amd":
        return "ROCm"
    if v == "apple":
        return "Metal"
    if v == "intel":
        return "SYCL"
    return "CUDA"


def gpu_token(machine, idx):
    """Full device token: 'CUDA0', 'ROCm2', or plain 'Metal'."""
    p = gpu_prefix(machine)
    return p if p == "Metal" else f"{p}{idx}"


def is_gpu_token(tok):
    """True for any accelerator device token (CUDA*/ROCm*/HIP*)."""
    t = (tok or "").strip()
    return t.startswith(GPU_TOKEN_PREFIXES)


def gpu_index(tok):
    """Numeric index of a device token (CUDA2/ROCm1 -> 2/1), else 0."""
    digits = "".join(ch for ch in (tok or "") if ch.isdigit())
    try:
        return int(digits)
    except ValueError:
        return 0


def cuda_visible_devices(machine):
    """CUDA_VISIBLE_DEVICES value pinning this machine's GPUs to their
    nvidia-smi (PCI-bus) order, as GPU UUIDs — or "" when unavailable.

    Identity beats ordering: CUDA numbers GPUs fastest-first by default
    while nvidia-smi (and our detection, labels, --device tokens) numbers
    by PCI bus, so --device CUDA1 can land on the wrong card of a
    heterogeneous peer. UUIDs select exact cards regardless of either
    enumeration, and unlike CUDA_DEVICE_ORDER they work on old runtimes
    (verified: a v7-era ggml build fails CUDA init outright under
    PCI_BUS_ID but enumerates cleanly under a UUID list).
    All-or-nothing: any GPU without a UUID (AMD/Intel/old detect data)
    yields "" so no card is ever hidden by a partial list."""
    det = (machine or {}).get("detected") or {}
    uuids = det.get("gpu_uuids") or []
    names = det.get("gpu_names") or []
    if not uuids or len(uuids) != len(names) or not names:
        return ""
    if any(not (u or "").startswith("GPU-") for u in uuids):
        return ""
    return ",".join(uuids)


# Compute capability (best-effort, live data only).
#
# What the vLLM community actually does for the bf16 problem (upstream
# PRs #1901/#17265, cuda.py `supported_dtypes`): check the DEVICE's
# compute capability — `torch.cuda.get_device_capability()[0] < 8` — or
# equivalently `nvidia-smi --query-gpu=compute_cap`, and pass
# `--dtype half` explicitly. vLLM deliberately does NO silent
# auto-fallback (accuracy changes must stay a user decision), so neither
# do we: capability data only ever produces a *warning*.
#
# No product-name tables here by design: names rot (every generation
# needs an update), while compute_cap is reported by the driver itself
# for 10-series through 50-series and beyond. Unknown/missing data ->
# None (no claim, never an error).
def parse_compute_cap(text):
    """'7.5'/'8.6' -> (7, 5)/(8, 6); anything else -> None."""
    parts = (text or "").strip().split(".")
    if len(parts) != 2:
        return None
    try:
        return (int(parts[0]), int(parts[1]))
    except ValueError:
        return None


def sm_supports_bf16(major, minor=0):
    """Mirrors vLLM cuda.py: bf16 needs sm >= 8.0 (Ampere+)."""
    try:
        return (int(major), int(minor)) >= (8, 0)
    except (ValueError, TypeError):
        return None


def compute_cap_supports_bf16(compute_cap):
    """'7.5' -> False, '8.6' -> True, unknown/None -> None."""
    sm = parse_compute_cap(compute_cap)
    if sm is None:
        return None
    return sm_supports_bf16(*sm)


def query_compute_caps(runner=None):
    """Live compute caps via nvidia-smi (local or SSH `runner`). Returns
    ['7.5', '8.6', ...] in GPU-index order, [] when unavailable
    (non-NVIDIA host, no driver, old nvidia-smi without the field)."""
    import subprocess as _sp
    cmd = ("nvidia-smi --query-gpu=compute_cap "
           "--format=csv,noheader,nounits")
    try:
        if runner is None:
            out = _sp.run(cmd.split(), capture_output=True, text=True,
                          timeout=15).stdout
        else:
            _code, out, _err = runner(cmd, timeout=15)
    except Exception:
        return []
    caps = []
    for line in (out or "").splitlines():
        line = line.strip()
        if parse_compute_cap(line) is not None:
            caps.append(line)
    return caps


def detect_vendor_local():
    """Best-effort vendor detection on THIS machine: nvidia-smi first,
    then rocm-smi (Linux) / amd-smi; Apple Silicon detected by platform.
    Returns 'nvidia'|'amd'|'apple'|'cpu'."""
    import platform as _pl
    if _pl.system() == "Darwin":
        return "apple"
    for cmd in (["nvidia-smi"], ["rocm-smi"], ["amd-smi"],
                ["xpu-smi", "discovery"]):
        try:
            if subprocess.run(cmd, capture_output=True, timeout=10).returncode == 0:
                if cmd[0] == "nvidia-smi":
                    return "nvidia"
                if cmd[0] in ("rocm-smi", "amd-smi"):
                    return "amd"
                return "intel"
        except (OSError, subprocess.TimeoutExpired):
            continue
    return "cpu"


# ---------------- remote/local query builders ----------------

def nvidia_query(fields):
    """CSV row query for nvidia-smi (fields = comma-joined query-gpu list)."""
    return (f"nvidia-smi --query-gpu={fields} "
            f"--format=csv,noheader,nounits")


def rocm_query_json(show_temp=True):
    """rocm-smi JSON query covering util + VRAM (+temp). Field names in
    the JSON vary by ROCm version; the parser handles known variants."""
    flags = "--showuse --showmeminfo vram --json"
    if show_temp:
        flags += " --showtemp"
    return f"rocm-smi {flags}"


def parse_rocm_json(text):
    """Parse rocm-smi --json into stats-query CSV rows
    ('index,name,util,used_mib,total_mib,temp' — same field order as
    the nvidia-smi stats query) so downstream parsers are
    vendor-agnostic. Returns [] when nothing usable is found."""
    try:
        data = json.loads((text or "").strip())
    except (ValueError, TypeError):
        return []
    rows = []
    for key in sorted(k for k in data if str(k).lower().startswith("card")):
        d = data[key] or {}
        idx = "".join(ch for ch in key if ch.isdigit()) or "0"
        name = (d.get("Card series") or d.get("Device Name")
                or d.get("Product Name") or f"AMD GPU {idx}")
        total = used = temp = util = None
        for k, v in d.items():
            lk = k.lower()
            try:
                if total is None and "vram total" in lk and "used" not in lk:
                    total = float(str(v).replace(",", ""))
                elif used is None and "vram total used" in lk:
                    used = float(str(v).replace(",", ""))
                elif temp is None and ("temperature" in lk or lk == "temp") \
                        and "memory" not in lk:
                    temp = float(str(v).replace(",", "").split()[0])
                elif used is None and "vram used" in lk and "total" not in lk:
                    used = float(str(v).replace(",", ""))
                elif util is None and (
                        "gpu use" in lk or "gpu utilization" in lk
                        or "gpu_busy" in lk):
                    util = float(str(v).replace("%", "").replace(",", "").split()[0])
            except (ValueError, IndexError):
                continue
        if total is None:
            continue
        # rocm-smi meminfo reports BYTES; some versions report MiB.
        # Heuristic: values under ~2^24 are already MiB.
        div = 1 if total < (1 << 24) else 1024 * 1024
        total_mib = total / div if div > 1 else total
        used_mib = (used / div if div > 1 else used) if used is not None else 0
        rows.append(f"{idx},{name},"
                    f"{util if util is not None else 0},"
                    f"{used_mib:.0f},{total_mib:.0f},"
                    f"{temp if temp is not None else 0}")
    return rows


def macos_gpu_rows():
    """GPU info on macOS via system_profiler, normalized to the shared
    csv row shape (index,name,total_mib,used_mib,util,temp). Apple
    Silicon reports unified memory as 'spdisplays_vram_shared'; Intel
    Macs report per-card VRAM. Utilization/temp are unavailable without
    sudo powermetrics, so they report 0."""
    import re as _re
    try:
        proc = subprocess.run(
            ["system_profiler", "SPDisplaysDataType", "-json"],
            capture_output=True, text=True, timeout=20)
        if proc.returncode != 0:
            return []
        data = json.loads(proc.stdout or "{}")
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return []
    rows = []
    idx = 0
    for g in (data.get("SPDisplaysDataType") or []):
        for item in (g.get("spdisplays_ndrvs") or [g]):
            name = (item.get("_name")
                    or item.get("spdisplays_device_id") or "Apple GPU")
            name = str(name).replace("Apple ", "").replace(",", "")
            mem = item.get("spdisplays_vram_shared") or item.get("spdisplays_vram")
            total_mib = 0
            if mem:
                m = _re.search(r"([\d.]+)\s*(GB|MB|TB)", str(mem))
                if m:
                    val, unit = float(m.group(1)), m.group(2).upper()
                    total_mib = int(val * 1024 * 1024 if unit == "TB"
                                    else val * 1024 if unit == "GB" else val)
            rows.append(f"{idx},{name},{total_mib},0,0,0")
            idx += 1
    return rows


def intel_query_json():
    """xpu-smi stats for all devices as JSON (fields: 'GPU utilization
    (%)' etc. per xpumanager docs; parse defensively per version)."""
    return "xpu-smi stats -d -1 -j"


def parse_xpu_smi_json(text):
    """Parse xpu-smi stats JSON into stats-query csv rows
    ('index,name,util,used_mib,total_mib,temp'). xpu-smi reports used
    MiB per device but no total — total falls back to 0 (dashboard
    shows used only until a better source is found)."""
    try:
        data = json.loads((text or "").strip())
    except (ValueError, TypeError):
        return []
    rows = []
    devices = data if isinstance(data, list) else         data.get("device_list") or data.get("stats") or []
    if isinstance(devices, dict):
        devices = list(devices.values())
    for d in devices:
        if not isinstance(d, dict):
            continue
        idx = d.get("device_id", d.get("deviceId", len(rows)))
        name = (d.get("device_name") or d.get("deviceName")
                or f"Intel GPU {idx}")
        name = str(name).replace("Intel(R) ", "").replace("(R)", "") \
            .replace("(TM)", "").replace("(tm)", "")
        util = used = temp = 0.0
        for k, v in d.items():
            lk = k.lower()
            try:
                if "utilization" in lk and "memory" not in lk:
                    util = float(str(v).replace("%", "").split()[0])
                elif "memory used" in lk or lk == "mem_used":
                    used = float(str(v).split()[0])
                elif "temperature" in lk:
                    temp = float(str(v).split()[0])
            except (ValueError, IndexError):
                continue
        rows.append(f"{idx},{name},{util},{used:.0f},0,{temp:.0f}")
    return rows
