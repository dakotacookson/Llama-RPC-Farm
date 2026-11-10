"""Model source scanning + machine inventory views."""
import json
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor

import pydantic
from fastapi import APIRouter, HTTPException
from fastapi.concurrency import run_in_threadpool

import access
import agent_client
import config_loader as cfg
import gguf_inspect
import local_node
import path_resolver
import stats_collector
import stats_history
from runtime_paths import data_dir
from machine_conn import _machine_connection_caps

BASE_DIR = data_dir()

router = APIRouter()


# ---------------- model scanning ----------------

# short-lived stats cache so the 1s overview poll stays snappy (full agent
# collection per machine is slow, esp. Windows). The detail modal forces a
# fresh collection on demand.
_stats_cache = {}  # key -> {"ts": ..., "data": ...}
_STATS_TTL = 10.0  # seconds


@router.get("/api/stats")
async def get_stats(light: bool = False):
    """Get cluster-wide machine stats (GPU, CPU, RAM). Cached briefly so the
    auto-refresh poll doesn't re-query every second. light=true skips the slow
    full agent collection and returns reachability + identity instantly (the
    overview uses this; the detail modal does a full collection on demand).
    Cache is keyed by light so the two never overwrite each other."""
    key = "light" if light else "full"
    now = time.time()
    ent = _stats_cache.get(key)
    if ent is not None and now - ent["ts"] < _STATS_TTL:
        return ent["data"]

    def collect_all():
        machines = cfg.machines.all()
        fn = stats_collector.machine_stats_light if light else stats_collector.machine_stats
        with ThreadPoolExecutor(max_workers=len(machines) or 1) as pool:
            data = list(pool.map(fn, machines))
        return {"data": data, "totals": stats_collector.cluster_totals(data)}
    result = await run_in_threadpool(collect_all)
    _stats_cache[key] = {"ts": time.time(), "data": result}
    return result


@router.get("/api/stats/history")
async def get_stats_history(minutes: int = 15, points: int = 400):
    """Rolling per-machine hardware samples (GPU util/VRAM/temp, CPU, RAM)
    kept in memory by stats_history for the Performance tab, cut to the
    last `minutes` and thinned to at most `points` per machine."""
    minutes = max(1, min(int(minutes or 15), 60))
    points = max(10, min(int(points or 180), 1000))
    hist = stats_history.history(minutes, points)
    machines = []
    for m in cfg.machines.all():
        name = m.get("name")
        machines.append({"name": name, "role": m.get("role"),
                         "samples": hist.get(name, [])})
    return {"enabled": stats_history.INTERVAL > 0,
            "interval": stats_history.INTERVAL, "minutes": minutes,
            "min_interval": stats_history.MIN_INTERVAL,
            "max_interval": stats_history.MAX_INTERVAL,
            "can_set_interval": bool(access.is_admin(None)),
            "machines": machines}


@router.post("/api/stats/history/interval")
async def set_stats_history_interval(body: dict):
    """Change how often the Performance sampler measures every machine
    (admin only; clamped to the sampler's MIN..MAX)."""
    if not access.is_admin(None):
        raise HTTPException(status_code=403, detail="admin only")
    try:
        sec = int(body.get("seconds"))
    except (TypeError, ValueError):
        raise HTTPException(status_code=400, detail="seconds must be a number")
    return {"interval": stats_history.set_interval(sec)}


@router.get("/api/stats/stream")
async def get_stats_stream():
    """Per-machine stats, streamed as NDJSON: one JSON line per machine
    AS ITS collection completes, then a final {totals: ...} line.

    The dashboard consumes this so one slow/offline machine can't drag
    the others: each card paints the moment its own data lands instead
    of waiting for the slowest host (the old /api/stats blocked until
    every agent collection finished — 5s+ on Windows blobs, 6s+ on offline
    boxes). Same short cache as /api/stats per machine."""
    from fastapi.responses import StreamingResponse

    machines = cfg.machines.all()
    per_machine_cache = _stats_cache.setdefault("per-machine", {})
    now = time.time()

    def gen():
        data = {}
        fresh = {}
        with ThreadPoolExecutor(max_workers=len(machines) or 1) as pool:
            futures = {pool.submit(stats_collector.machine_stats, m): m
                       for m in machines}
            for fut in futures:
                m = futures[fut]
                try:
                    s = fut.result()
                except Exception:
                    s = stats_collector._empty(m.get("os", "linux"))
                s["name"] = m["name"]
                s["ip"] = m["ip"]
                data[m["name"]] = s
                fresh[m["name"]] = time.time()
                yield json.dumps(s) + "\n"
        # merge into the per-machine cache, preserving newer entries
        for name, ts in fresh.items():
            old = per_machine_cache.get(name)
            if not old or now - old["ts"] > _STATS_TTL:
                per_machine_cache[name] = {"ts": ts, "data": data[name]}
        totals = stats_collector.cluster_totals(list(data.values()))
        yield json.dumps({"totals": totals, "ts": time.time()}) + "\n"

    return StreamingResponse(gen(), media_type="application/x-ndjson")


def _parse_win_disk(raw):
    """Parse PowerShell Get-PSDrive JSON into (free, total) human strings
    (GB or TB depending on size). Picks the drive with the most free space."""
    if not raw:
        return "", ""
    try:
        data = json.loads(raw)
    except Exception:
        return "", ""
    items = data if isinstance(data, list) else [data]
    if not items:
        return "", ""
    best = max(items, key=lambda d: d.get("Free") or 0)
    free = best.get("Free") or 0
    used = best.get("Used") or 0
    return _fmt_free(free), _fmt_free(free + used)


def _fmt_free(bytes_free):
    gb = bytes_free / 1024 ** 3
    if gb >= 1024:
        return f"{gb / 1024:.1f} TB"
    return f"{gb:.0f} GB"


def _fmt_free_kb(kb_str):
    """df -k free space (KB) -> human GB/TB string."""
    try:
        return _fmt_free(int(kb_str) * 1024)
    except (ValueError, TypeError):
        return ""


def _parse_posix_blob(detail, blob):
    """Parse the combined 'K:...\nC:...\n...' blob from a posix detail query."""
    if not blob:
        return
    for line in blob.splitlines():
        if ":" not in line:
            continue
        key, _, val = line.partition(":")
        val = val.strip()
        if key == "K":
            detail["kernel"] = val
        elif key == "C":
            detail["cpu_model"] = val
        elif key == "U":
            detail["uptime"] = val
        elif key == "I":
            detail["net"]["ip"] = val
        elif key == "S":
            detail["net"]["speed"] = val
        elif key == "D":
            detail["disk"]["free"] = _fmt_free_kb(val)
            detail["disk"]["mount"] = "/"
        elif key == "DT":
            detail["disk"]["total"] = _fmt_free_kb(val)
        elif key == "L":
            detail["llama_version"] = val


_SPEED_CACHE_FILE = os.path.join(BASE_DIR, "config", "net_speed_cache.json")
_SPEED_CACHE_TTL = 3600  # 1 hour


def _load_speed_cache():
    if not os.path.exists(_SPEED_CACHE_FILE):
        return {}
    try:
        with open(_SPEED_CACHE_FILE, "r", encoding="utf-8") as f:
            return json.load(f) or {}
    except Exception:
        return {}


def _save_speed_cache(cache):
    try:
        tmp = _SPEED_CACHE_FILE + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(cache, f, indent=2)
        os.replace(tmp, _SPEED_CACHE_FILE)
    except Exception:
        pass




def _measure_agent_speed(machine, mb=50):
    try:
        return agent_client.speedtest(machine, mb=mb)
    except Exception:
        return None


@router.get("/api/machines/{name}/detail")
async def machine_detail(name: str):
    """Deep system info for one machine: OS/kernel, CPU model, disk, network,
    uptime, llama.cpp version, and the live stats. Used by the dashboard card
    detail modal."""
    try:
        machine = cfg.machines.require(name)
    except ValueError:
        raise HTTPException(status_code=404, detail=f"machine '{name}' not found")
    is_local = local_node.is_local(machine)
    machine_os = machine.get("os", "linux")

    def _run(cmd, timeout=20):
        try:
            p = subprocess.run(cmd, shell=True, capture_output=True,
                               text=True, timeout=timeout)
            return p.stdout.strip()
        except Exception:
            return ""

    detail = {"name": name, "is_local": is_local, "os": machine_os,
              "ip": machine.get("ip", ""), "role": machine.get("role", "peer"),
              "gpu_vendor": machine.get("gpu_vendor", "nvidia"),
              "rpc_port": machine.get("rpc_port", 50052),
              "llama_bin": machine.get("llama_bin", ""),
              "mac": (machine.get("mac") or ""),
              "transport": machine.get("transport") or "auto",
              "agent_port": machine.get("agent_port") or 8198,
              "connection": "local" if is_local else "agent"}

    # ---- combine all detail queries into ONE round-trip per OS ----
    detail["net"] = {"ip": "", "speed": ""}
    detail["disk"] = {"free": "", "mount": ""}
    # RDMA transport info: how this machine's RPC connection is served
    detail["rdma"] = {
        "enabled": bool(machine.get("rdma")),
        "transport": ("RDMA (rdma_cm, native)" if machine.get("rdma") and machine.get("rdma_cm")
                      else "RDMA (auto-negotiate)" if machine.get("rdma")
                      else "TCP"),
        "ip": machine.get("rdma_ip", ""),
        "dev": machine.get("rdma_dev", ""),
        "gid": machine.get("rdma_gid", ""),
    }
    detail["disk"] = {"free": "", "mount": ""}
    try:
        detail.update(_machine_connection_caps(machine))
    except Exception:
        pass

    # Remote machines: the agent's /system/detail is the single source of
    # truth (same fields the local command ladder produces).
    if not is_local:
        try:
            sd = agent_client.system_detail(machine) or {}
        except Exception as e:
            raise HTTPException(status_code=502,
                                detail=f"agent detail failed: {e}")
        detail.update(sd)
        detail.setdefault("net", {"ip": machine.get("ip", ""), "speed": ""})
        detail.setdefault("disk", {"free": "", "mount": ""})
        try:
            detail["stats"] = stats_collector.machine_stats(machine)
        except Exception:
            detail["stats"] = None
        return detail
    agent_detail = None
    if not is_local and agent_client.should_use_agent(machine):
        try:
            agent_detail = agent_client.system_detail(machine)
        except Exception:
            agent_detail = {}
    if agent_detail is not None:
        detail.update(agent_detail)
    else:
        if machine_os == "windows":
            blob = _run(
                "powershell -NoProfile -Command \""
                "$o=@{};"
                "$o.kernel=(ver);"
                "$o.cpu=(Get-CimInstance Win32_Processor).Name;"
                "$o.uptime=((Get-Date)-(Get-CimInstance Win32_OperatingSystem).LastBootUpTime).ToString();"
                "$o.ip=((Get-NetIPAddress -AddressFamily IPv4 | Where-Object {$_.IPAddress -notlike '127.*' -and $_.IPAddress -notlike '169.254.*'} | Select-Object -First 1).IPAddress);"
                "$o.speed=((Get-NetAdapter | Where-Object Status -eq 'Up' | Select-Object -First 1).LinkSpeed);"
                "$o.disk=(Get-PSDrive -PSProvider FileSystem | Where-Object {$_.Free -gt 0} | Select-Object Name,Free,Used | ConvertTo-Json -Compress);"
                "$o.cores=(Get-CimInstance Win32_Processor | Measure-Object -Sum NumberOfLogicalProcessors).Sum;"
                "$o.coreload=(Get-CimInstance Win32_PerfFormattedData_PerfOS_Processor | Where-Object {$_.Name -ne '_Total'} | Select-Object -ExpandProperty PercentProcessorTime | ForEach-Object {[math]::Round($_,1)} | ConvertTo-Json -Compress);"
                "$o.llama=((& (Join-Path $env:USERPROFILE 'llama.cpp\\bin\\llama-server.exe') --version 2>&1 | Select-String 'version:' | Select-Object -First 1).ToString());"
                "if (-not $o.llama) { $o.llama=((& (Join-Path $env:USERPROFILE 'llama.cpp\\llama-server.exe') --version 2>&1 | Select-String 'version:' | Select-Object -First 1).ToString()); }"
                "$o | ConvertTo-Json -Compress\"", timeout=30)
            try:
                o = json.loads(blob)
                detail["kernel"] = o.get("kernel", "")
                detail["cpu_model"] = o.get("cpu", "")
                detail["uptime"] = o.get("uptime", "")
                detail["net"]["ip"] = o.get("ip", "")
                detail["net"]["speed"] = o.get("speed", "")
                detail["disk"]["free"], detail["disk"]["total"] = _parse_win_disk(o.get("disk", ""))
                detail["llama_version"] = o.get("llama", "")
                # coreload comes back as a JSON string; parse it into a list
                cl = o.get("coreload")
                if isinstance(cl, str):
                    try:
                        detail["cpu_cores"] = json.loads(cl)
                    except Exception:
                        detail["cpu_cores"] = []
                else:
                    detail["cpu_cores"] = cl or []
                detail["cores"] = o.get("cores", "")
            except Exception:
                pass
        elif machine_os == "macos":
            blob = _run(
                "echo \"K:$(uname -a)\";"
                "echo \"C:$(sysctl -n machdep.cpu.brand_string)\";"
                "echo \"U:$(uptime)\";"
                "echo \"I:$(ipconfig getifaddr en0 2>/dev/null || ipconfig getifaddr en1 2>/dev/null)\";"
                "echo \"S:$(networksetup -getmedia en0 2>/dev/null | grep -i 'current' | sed 's/.*: //')\";"
                "echo \"D:$(df -k / | awk 'NR==2 {print $4}')\";"
                "echo \"DT:$(df -k / | awk 'NR==2 {print $2}')\";"
                "echo \"L:$(\\\"{bin}/llama-server\\\" --version 2>&1 || "
                "\\\"{bin}/llama-cli\\\" --version 2>&1)\"".format(
                    bin=machine.get('llama_bin', '') or '.'), timeout=30)
            _parse_posix_blob(detail, blob)
        else:
            _bin = machine.get('llama_bin', '')
            blob = _run(
                f"K=$(uname -a); "
                f"C=$(lscpu | grep 'Model name' | cut -d: -f2 | sed 's/^ *//'); "
                f"U=$(uptime -p); "
                f"I=$(hostname -I 2>/dev/null | grep -oF '{machine['ip']}' | head -n1); "
                f"S=$(for i in $(ls /sys/class/net | grep -v '^lo'); do "
                f"d=$(readlink /sys/class/net/$i/device/driver 2>/dev/null); "
                f"case \"$i\" in veth*|br-*|docker*|virbr*|tailscale*) continue;; esac; "
                f"case \"$d\" in *bridge*|*veth|*tun|*tap) continue;; esac; "
                f"s=$(cat /sys/class/net/$i/speed 2>/dev/null); "
                f"if [ -n \"$s\" ] && [ \"$s\" != \"-1\" ] && [ \"$s\" -gt 0 ]; then "
                f"if [ \"$s\" -ge 1000 ]; then echo \"$((s/1000)) Gb/s\"; "
                f"else echo \"${{s}} Mb/s\"; fi; break; fi; "
                f"e=$(ethtool $i 2>/dev/null | grep -i speed | sed 's/.*: *//;s/ *$//'); "
                f"if [ -n \"$e\" ] && [ \"$e\" != \"Unknown!\" ]; then echo \"$e\"; break; fi; "
                f"case \"$d\" in *virtio*) echo \"virtio (speed n/a)\"; break;; esac; done); "
                f"D=$(df -k / | tail -n1 | awk '{{print $4}}'); "
                f"DT=$(df -k / | tail -n1 | awk '{{print $2}}'); "
                f"L=$(\"{_bin}/llama-server\" --version 2>&1 || \"{_bin}/llama-cli\" --version 2>&1); "
                f"echo \"K:$K\"; echo \"C:$C\"; echo \"U:$U\"; echo \"I:$I\"; "
                f"echo \"S:$S\"; echo \"D:$D\"; echo \"DT:$DT\"; echo \"L:$L\"", timeout=30)
            _parse_posix_blob(detail, blob)

    # ---- real throughput over the peer agent (cached 1h) ----
    speed_agent = None
    for peer in cfg.machines.all():
        if peer.get("name") == name or local_node.is_local(peer):
            continue
        speed_agent = peer
        break
    reported = str(detail["net"].get("speed") or "").strip()
    need_measure = (not reported or "n/a" in reported.lower()
                    or "unknown" in reported.lower())
    if need_measure and speed_agent is not None:
        cache = _load_speed_cache()
        ent = cache.get(name)
        now = time.time()
        if ent and now - ent.get("ts", 0) < _SPEED_CACHE_TTL:
            detail["net"]["speed"] = ent.get("mbps")
        else:
            mbps = _measure_agent_speed(speed_agent)
            if mbps:
                detail["net"]["speed"] = mbps
                cache[name] = {"mbps": mbps, "ts": now}
                _save_speed_cache(cache)
        if detail["net"]["speed"]:
            detail["net"]["speed"] = f"{detail['net']['speed']} Mb/s (agent)"

    # live stats ride along so the modal shows current numbers
    try:
        detail["stats"] = stats_collector.machine_stats(machine)
    except Exception:
        detail["stats"] = None
    return detail


class SafeDefaultsIn(pydantic.BaseModel):
    path: str = ""
    group: str = ""
    servers: list[str] = []


@router.get("/api/safe-defaults")
async def safe_defaults(path: str = "", group: str = "", servers: str = ""):
    """Safe defaults for a new model — hardware-aware. Uses live free VRAM/RAM
    (total−used) and GGUF header (is_gemma, has_mtp, layers) to pick ctx, batch, cache, etc.
    Only called from Scan → Configure (not Scan)."""
    import os as _os
    # resolve hardware set
    target_machines = []
    if group:
        g = cfg.groups.get(group)
        if g:
            for nm in g.get("machines") or []:
                try:
                    target_machines.append(cfg.machines.require(nm))
                except Exception:
                    pass
    elif servers:
        for nm in [s.strip() for s in servers.split(",") if s.strip()]:
            try:
                target_machines.append(cfg.machines.require(nm))
            except Exception:
                pass
    if not target_machines:
        target_machines = cfg.machines.all()
    # live stats for those machines
    def get_stats():
        with ThreadPoolExecutor(max_workers=len(target_machines) or 1) as pool:
            data = list(pool.map(stats_collector.machine_stats, target_machines))
        return data, stats_collector.cluster_totals(data)
    try:
        per_machine, totals = await run_in_threadpool(get_stats)
    except Exception:
        totals = {"vram": {"total": 0, "used": 0}, "ram": {"total": 0, "used": 0}}
        per_machine = []
    free_vram = max(0, totals.get("vram", {}).get("total", 0) - totals.get("vram", {}).get("used", 0))
    free_ram = max(0, totals.get("ram", {}).get("total", 0) - totals.get("ram", {}).get("used", 0))
    # heuristic: master cores for threads
    try:
        master = next((m for m in target_machines if m.get("role")=="master"), target_machines[0])
        threads = int((master.get("detected") or {}).get("cores") or master.get("cores") or 8)
    except Exception:
        threads = 8
    threads = max(4, min(16, threads))
    # batch heuristic: more VRAM → larger batch
    batch = 1024
    ubatch = 1024
    # ctx heuristic: keep KV under ~30% of free (VRAM+0.5*RAM) minus model size
    # need model header?
    ctx_size = 16384
    cache_k = "q8_0"
    cache_v = "q8_0"
    is_gemma = False
    is_embed = False
    is_decision = False
    has_mtp = False
    has_mmproj = False
    if path:
        # embed detection: filename contains embed/nomic/bge/gte/minilm
        # or model type is embedding (heuristic)
        pl = path.lower()
        if any(t in pl for t in ("embed", "bge", "gte", "minilm")):
            is_embed = True
        try:
            real = path_resolver.resolve(path)
            real_exp = _os.path.expanduser(real)
            # TypeSafe decision models (laya/clef/...): detected from the
            # GGUF's <arch>.decision.type — the same gate llama.cpp uses.
            # These serve /v1/systemone only (no generation), so chat-side
            # heuristics (KV estimator, gemma cache) must not run for them.
            if gguf_inspect.get_decision_type(real_exp):
                is_decision = True
            # find companion mmproj in same dir
            d = _os.path.dirname(real_exp)
            if _os.path.isdir(d):
                for fn in _os.listdir(d):
                    if fn.startswith("mmproj") and fn.endswith(".gguf"):
                        has_mmproj = True
                        break
            # read header for gemma/mtp
            if _os.path.exists(real_exp):
                try:
                    meta = gguf_inspect.metadata(real_exp)
                    # is_gemma: check arch
                    for k in meta.keys():
                        if "gemma" in str(k).lower() or "gemma" in str(meta.get(k,"")).lower():
                            is_gemma = True
                    # also check model name
                    if "gemma" in path.lower():
                        is_gemma = True
                    has_mtp = gguf_inspect.has_mtp_layers_anywhere(real_exp)
                except Exception:
                    pass
        except Exception:
            pass
    if is_gemma:
        cache_k = "f16"
        cache_v = "f16"
    # ctx: if we can estimate KV bytes, pick safe (chat models only —
    # embed uses fixed small-window values below, never the estimator)
    if path and not is_embed and free_vram + free_ram > 0:
        try:
            real = path_resolver.resolve(path)
            real_exp = _os.path.expanduser(real)
            if _os.path.exists(real_exp):
                meta = gguf_inspect.metadata(real_exp)
                # estimate KV bytes per token: 2*layers*kv_heads*head_dim*bytes
                import struct as _struct
                n_layer = 0
                kv_heads = 0
                head_dim = 128
                for k,v in meta.items():
                    kl = str(k).lower()
                    if kl.endswith("block_count") and isinstance(v,int):
                        n_layer = v
                    if kl.endswith("head_count_kv") and isinstance(v,int):
                        kv_heads = v
                    if kl.endswith("key_length") and isinstance(v,int):
                        head_dim = v
                if n_layer and kv_heads:
                    # bytes for q8_0 vs f16
                    bpe = 1 if cache_k=="q8_0" else 2
                    kv_per_token = 2 * n_layer * kv_heads * head_dim * bpe
                    # model size
                    try:
                        w = sum(gguf_inspect.weight_bytes(p) for p in gguf_inspect.shard_paths(real_exp)) / 1e9
                    except Exception:
                        w = 8
                    free_for_kv = (free_vram + free_ram*0.3) - w - 3  # 3GB OS overhead
                    if free_for_kv > 2:
                        # choose ctx that fits 30% of free_for_kv
                        max_ctx = int((free_for_kv * 1e9 * 0.3) / max(kv_per_token,1))
                        # clamp safe 8192..32768
                        ctx_size = max(8192, min(32768, max_ctx // 1024 * 1024))
                    # HBM-rich hosts can go larger, but safe stays 16k
        except Exception:
            pass
    flags = {
        "ctx-size": ctx_size,
        "threads": threads,
        "flash-attn": "on",
        "cache-type-k": cache_k,
        "cache-type-v": cache_v,
        "batch-size": batch,
        "ubatch-size": ubatch,
        "parallel": 1,
        "jinja": True,
        "split-mode": "layer",
        "n-gpu-layers": 99,
        # verbose logging NO LONGER forced: -v floods thousands of
        # "node #N" sched lines per ubatch (severe slowdown at 27B scale).
        # Add -v/--log-timestamps/--log-prefix per-model only when testing.
    }
    if is_embed:
        # Embedding safe defaults (distinct from chat):
        # - whole prompt must fit in ONE ubatch (llama-server clamps
        #   n_batch down to n_ubatch otherwise), so keep the pair equal
        #   and sized to the window (upstream example serves with -ub 8192)
        # - no jinja/draft/reasoning: embedding-only endpoint, no sampling
        # - flash-attn left at default (auto), not forced on
        # - KV stays q8_0 (tiny for embed), full offload, single slot
        flags["embeddings"] = True
        flags.pop("jinja", None)
        flags.pop("flash-attn", None)
        ctx_size = 8192
        flags["ctx-size"] = 8192
        flags["batch-size"] = 8192
        flags["ubatch-size"] = 8192
        flags["parallel"] = 1
        # nomic v1.5 long-context recipe (2048 native -> 8192 via yarn)
        if "nomic" in path.lower():
            flags["rope-scaling"] = "yarn"
            flags["rope-freq-scale"] = 0.75
    if has_mmproj:
        # leave mmproj to user to pick — don't auto-fill, just hint via flags? keep unset
        pass
    if is_decision:
        # TypeSafe decision-model safe defaults (distinct from chat AND
        # embed). llama.cpp server docs (tools/server/README.md, System
        # One section): these models answer via /v1/systemone only — no
        # text generation, no sampling, no chat template.
        # - for laya/clef the WHOLE prompt is evaluated in one batch, so
        #   --ubatch-size must cover state+questions (>= 2048 headroom;
        #   long prompts are truncated to the model's max_head_tokens)
        # - ctx 8192 = laya's modern-bert.context_length
        # - no jinja/flash-attn/KV-quant/MTP: classifier head, not a
        #   generator; the decision temperatures are baked into the GGUF
        #   (<arch>.decision.temperature.*), sampling flags are no-ops
        # - full offload: 421M (laya) fits any GPU trivially
        for k in ("jinja", "flash-attn", "cache-type-k", "cache-type-v",
                  "batch-size", "ubatch-size", "ctx-size", "threads",
                  "split-mode"):
            flags.pop(k, None)
        flags.update({
            "ctx-size": 8192,
            "ubatch-size": 2048,
            "parallel": 1,
            "n-gpu-layers": 99,
        })
    # never auto-add draft-mtp unless has_mtp
    return {
        "path": path,
        "flags": flags,
        "hardware": {"free_vram": round(free_vram,1), "free_ram": round(free_ram,1), "threads": threads, "is_gemma": is_gemma, "is_embed": is_embed, "is_decision": is_decision, "has_mtp": has_mtp},
        "note": (f"TypeSafe decision model — serves /v1/systemone only (no chat generation)"
                 if is_decision else
                 f"safe for {free_vram:.1f} GB free VRAM + {free_ram:.1f} GB free RAM — tweak ctx/batch/cache based on live free")
    }
