"""Model auto-tune: math-first estimate, live-verify after.

Flow: download -> configure (hardware selected in form is REQUIRED:
servers + device_order chips) -> safe defaults -> Tune -> save winner.

Phase 1 (0 loads, pure math): weights + mmproj + KV(ctx) + SSM state +
MTP draft overhead vs usable VRAM -> max ctx + integer split weights +
safe ubatch. Mirrors vLLM --max-model-len auto / Ollama fit-vram logic.

Phase 2 (staged live search, early exits): baseline at locked ctx,
then MTP variants, KV ladder (current/q8/f16/q4), forced splits,
ubatch/batch/margin neighbors, flash/SWA fallbacks.
Winner saved. ctx-size is NEVER changed: it is the user's.
"""
import os
import time
import threading
import urllib.request
import json as _json

import config_loader as cfg
import gguf_inspect
import path_resolver
import stats_collector

# ctx ladder we walk DOWN from the math estimate until tok/s >= target
CTX_LADDER = [262144, 200000, 160000, 128000, 100000,
              65536, 49152, 32768, 16384, 8192, 4096]
UBATCH_CANDIDATES = [256, 320, 376, 448, 512]
TARGET_TPS_RPC = 20.0    # multi-box over ~1GbE: the physics band
TARGET_TPS_LOCAL = 25.0  # single box, no network hops: aim higher
TARGET_TPS_DEFAULT = TARGET_TPS_RPC
HEADROOM_FRac = 0.90  # keep 10% VRAM free (fragmentation + server overhead)
SLOW_GPU_DERATE = {  # memory-bandwidth / arch derate for split weighting
    "GTX 1080": 0.6, "GTX 10": 0.6, "Pascal": 0.6,
    "RTX 2060": 0.8, "RTX 3050": 0.8, "Turing": 0.85,
}

# advertised memory bandwidth (GB/s) by GPU family — decode is
# bandwidth-bound, so this (not just VRAM size) predicts per-hop speed
# and identifies the pipeline straggler. Curie for live clocks would be
# nvidia-smi clocks.current.sm; throttling shows up as util high +
# clocks low, which stats_collector already surfaces via util/temp.
BANDWIDTH_GBS = [
    ("RTX 4080", 717), ("RTX 3080 TI", 912), ("RTX 3080", 760),
    ("RTX 3070 TI", 608), ("RTX 3070", 448), ("RTX 3060 TI", 448),
    ("RTX 3060", 360), ("RTX 2080 SUPER", 496), ("RTX 2080", 448),
    ("RTX 2070", 448), ("RTX 2060 SUPER", 448), ("RTX 2060", 336),
    ("RTX 3050", 224), ("GTX 1080 TI", 484), ("GTX 1080", 320),
    ("GTX 1070", 256), ("GTX 1660", 288),
]


def _bandwidth_gbs(gpu_name):
    name = (gpu_name or "").upper().replace("NVIDIA GEFORCE ", "")
    for key, bw in BANDWIDTH_GBS:
        if key in name:
            return bw
    return 0  # unknown: excluded from bandwidth math, VRAM weight only


def _derate(gpu_name):
    for key, factor in SLOW_GPU_DERATE.items():
        if key.lower() in (gpu_name or "").lower():
            return factor
    return 1.0


def _model_bytes(resolved_path):
    """weights + mmproj companion bytes (0 when missing)."""
    try:
        total = 0
        for p in gguf_inspect.shard_paths(resolved_path):
            if os.path.exists(p):
                total += gguf_inspect.weight_bytes(p)
        return total
    except Exception:
        try:
            return os.path.getsize(resolved_path)
        except OSError:
            return 0


def _kv_per_token(resolved_path, meta, cache_type="q8_0"):
    """Bytes of KV per context token for full-attention blocks."""
    bpt = {"q8_0": 1.0625, "q4_0": 0.5625, "f16": 2.0,
           "bf16": 2.0, "f32": 4.0}.get(str(cache_type), 1.0625)
    try:
        n_full, _, _ = gguf_inspect.block_attention_kinds(resolved_path)
    except Exception:
        n_full = int(meta.get("qwen35.block_count")
                     or meta.get("llama.block_count") or 32)
    kv_heads = 0
    klen = vlen = 0
    for k, v in meta.items():
        kl = str(k).lower()
        if kl.endswith("attention.head_count_kv") and isinstance(v, int):
            kv_heads = v
        if kl.endswith("attention.key_length") and isinstance(v, int):
            klen = v
        if kl.endswith("attention.value_length") and isinstance(v, int):
            vlen = v
    if not (kv_heads and klen and vlen):
        # fallback: ~128 dim per head, GQA 8
        kv_heads, klen, vlen = 8, 128, 128
    return n_full * (kv_heads * klen + kv_heads * vlen) * bpt


def _hardware_vram(servers, device_order=""):
    """Usable VRAM map for selected chips: [(label, free_gb, total_gb)]."""
    machines = {m["name"]: m for m in cfg.machines.all()}
    selected = [s for s in (servers or []) if s in machines]
    if not selected:
        raise ValueError("no servers selected — pick hardware in the form first")
    # chips restrict to chosen GPUs; empty device_order = all GPUs of servers
    chips = [c.strip() for c in (device_order or "").split(",") if c.strip()]
    out = []
    for name in selected:
        m = machines[name]
        det = m.get("detected") or {}
        names = det.get("gpu_names") or []
        total = float(det.get("vram_total_gb") or 0)
        # live free when reachable, else assume 90% free
        try:
            live = stats_collector.machine_stats(m)
            live_gpus = live.get("gpus") or []
        except Exception:
            live, live_gpus = None, []
        gpu_ids = [d.strip() for d in (m.get("gpu") or "").split(",") if d.strip()]
        if chips:
            wanted = set()
            for tok in chips:
                if "-" in tok:
                    mn, res = tok.rsplit("-", 1)
                    if mn.strip() == name and res.strip().startswith(("CUDA", "ROC", "GPU")):
                        wanted.add(res.strip())
            if wanted:
                gpu_ids = sorted(wanted)
        n = len(gpu_ids) or 1
        per_total = total / n if total else 0
        for i, gid in enumerate(gpu_ids):
            gname = names[i] if i < len(names) else gid
            # Estimate against TOTAL vrAM minus a small system reserve, NOT
            # live free: live free is polluted when a model is already
            # loaded (we'd conclude nothing fits). Live free is reported
            # for display only.
            free = max(0.0, per_total - 0.5)
            live_free = None
            for g in live_gpus:
                if str(g.get("index")) == "".join(c for c in gid if c.isdigit()):
                    live_free = round(max(0.0, g.get("mem_total_gb", 0) - g.get("mem_used_gb", 0)), 1)
            out.append({"label": f"{name}-{gid}", "name": gname,
                        "free_gb": round(free, 1), "total_gb": round(per_total, 1),
                        "live_free_gb": live_free,
                        "bw_gbs": _bandwidth_gbs(gname),
                        "pcie": _pcie_info(m),
                        "weight": round(max(0.5, per_total * _derate(gname)), 2)})
    if not out:
        raise ValueError("no GPUs selected — pick hardware in the form first")
    return out


def safe_defaults(base_path, servers, device_order="", target_os="linux",
                 existing_flags=None):
    """Safe starting flags for a new/existing model (no loads).

    ctx-size is the USER's (locked, non-negotiable); cache-type is only
    a starting point (stage 2b measures q8/f16/q4). Tune optimizes
    placement/spec/batches around them, never sacrifices ctx."""
    existing_flags = existing_flags or {}
    resolved = path_resolver.resolve_for(base_path, target_os)
    try:
        meta = gguf_inspect.metadata(resolved)
    except Exception:
        meta = {}
    try:
        has_mtp = gguf_inspect.has_mtp_layers_anywhere(resolved)
    except Exception:
        has_mtp = False
    model_max = 262144
    for k, v in meta.items():
        if str(k).lower().endswith(("context_length", "n_ctx_train")) \
                and isinstance(v, int) and v > 0:
            model_max = min(model_max, v)
    flags = {
        "flash-attn": "on", "jinja": True,
        "parallel": 1, "split-mode": "layer",
        # verbose logging = ground truth for the Flow Visualizer
        # (upstream token-stream fallback on single-GPU models)
        "-v": True, "log-timestamps": True, "log-prefix": True,
    }
    # batch/ubatch: fill ONLY when missing. A tuned value must survive
    # safe-defaults (overwriting it un-tested the winner and evicted it
    # from the search space entirely). Safe pair: 1024/1024 (uniform,
    # no VRAM heuristic — keeps first-load predictable).
    if not existing_flags.get("batch-size"):
        flags["batch-size"] = 1024
    if not existing_flags.get("ubatch-size"):
        flags["ubatch-size"] = 1024
    if not (existing_flags.get("ctx-size") or existing_flags.get("c")):
        flags["ctx-size"] = min(32768, model_max)
    for ck in ("cache-type-k", "cache-type-v"):
        if not existing_flags.get(ck):
            flags[ck] = "q8_0"
    if has_mtp and not existing_flags.get("spec-type"):
        flags.update({"spec-type": "draft-mtp", "spec-draft-n-max": 2,
                      "spec-draft-ngl": 99})
    return {"tensor_split_mode": "fit", "fit_margin": 1024,
            "fit_ram_allow": 1024, "device_order_policy": "vram-first",
            "flags": flags, "has_mtp": has_mtp, "model_max_ctx": model_max}


def estimate(base_path, servers, device_order="", target_os="linux",
             cache_type="q8_0", mmproj_raw=""):
    """Math-only max-ctx estimate (0 loads). Returns breakdown + split."""
    if not servers:
        raise ValueError("no servers selected — pick hardware in the form first")
    resolved = path_resolver.resolve_for(base_path, target_os)
    if not os.path.exists(resolved):
        raise ValueError(f"model file not found: {resolved}")
    meta = gguf_inspect.metadata(resolved)
    model_max = 262144
    for k, v in meta.items():
        if str(k).lower().endswith(("context_length", "n_ctx_train")) \
                and isinstance(v, int) and v > 0:
            model_max = min(model_max, v)
    weights = _model_bytes(resolved)
    mmproj_b = 0
    if mmproj_raw:
        try:
            mmproj_b = _model_bytes(path_resolver.resolve_for(mmproj_raw, target_os))
        except Exception:
            mmproj_b = 0
    kv_tok = _kv_per_token(resolved, meta, cache_type)
    try:
        ssm_b = gguf_inspect.ssm_state_bytes(resolved, meta)
    except Exception:
        ssm_b = 0
    draft_b = int(weights * 0.06)  # MTP draft heads + buffers, ~6% heuristic
    gpus = _hardware_vram(servers, device_order)
    usable = sum(g["free_gb"] for g in gpus) * 1024**3 * HEADROOM_FRac
    fixed = weights + mmproj_b + ssm_b + draft_b
    # integer split weights proportional to derated VRAM (cf. #18273 4,3,3,4,4)
    w = [g["weight"] for g in gpus]
    scale = 18 / sum(w) if sum(w) else 1
    split = [max(1, round(x * scale)) for x in w]
    # pipeline straggler: slowest known-bandwidth hop gates every token
    known = [g for g in gpus if g.get("bw_gbs")]
    straggler = min(known, key=lambda g: g["bw_gbs"])["label"] if known else ""
    best = 4096
    for ctx in CTX_LADDER:
        if ctx > model_max:
            continue
        need = fixed + kv_tok * ctx
        if need <= usable:
            best = ctx
            break
    # virtualization truth per box (Proxmox/KVM guests lie about PCIe
    # and hide cpufreq): measured Mbps is truth, vPCIe is not. Also flags
    # AVX-less guests (default kvm64 CPU type cripples llama.cpp CPU
    # kernels — Proxmox CPU type should be 'host').
    _by = {m["name"]: m for m in cfg.machines.all()}
    virt = {}
    for _nm in servers:
        _mm = _by.get(_nm)
        if _mm:
            virt[_nm] = virt_info(_mm)
    return {
        "model_max_ctx": model_max,
        "swa": has_swa(meta),
        "virt": virt,
        "recommended_ctx": best,
        "weights_gb": round(weights / 1024**3, 2),
        "mmproj_gb": round(mmproj_b / 1024**3, 2),
        "ssm_gb": round(ssm_b / 1024**3, 2),
        "kv_per_token_kb": round(kv_tok / 1024, 1),
        "kv_at_ctx_gb": round(kv_tok * best / 1024**3, 2),
        "usable_vram_gb": round(usable / 1024**3, 1),
        "gpus": gpus,
        "straggler": straggler,
        "suggested_split": split,
        "suggested_ubatch": 376,
        "ctx_ladder": [c for c in CTX_LADDER if c <= model_max],
    }


# ---------------- live verify (background job, few loads) ----------------
_JOBS = {}
_JOB_LOCK = threading.Lock()


def _job(model_id):
    with _JOB_LOCK:
        return _JOBS.get(model_id)


def job_status(model_id):
    j = _job(model_id)
    if not j:
        return {"model_id": model_id, "state": "idle"}
    return {k: v for k, v in j.items() if k != "thread"}


def _wait_healthy(port, timeout_s=900, interval_s=3):
    """Port-open is NOT enough: llama-server binds the port while still
    loading weights (health -> 503 'Loading model'). Poll /health until
    status ok so the bench never runs against a loading server."""
    import urllib.error
    t0 = time.time()
    last = "no response yet"
    while time.time() - t0 < timeout_s:
        try:
            with urllib.request.urlopen(
                    f"http://127.0.0.1:{port}/health", timeout=10) as r:
                body = r.read().decode("utf-8", "replace")
                if '"status"' in body and '"ok"' in body:
                    return True, body[:120]
                last = body[:120]
        except urllib.error.HTTPError as e:
            try:
                last = f"HTTP {e.code}: {e.read().decode()[:120]}"
            except Exception:
                last = f"HTTP {e.code}"
        except Exception as e:
            last = str(e)[:120]
        time.sleep(interval_s)
    return False, last


def _cache_dir():
    """OS-aware local SSD cache dir (Windows has no /tmp)."""
    import tempfile
    base = tempfile.gettempdir()
    # minimal sanity: must be an absolute path we can create in
    if not os.path.isabs(base):
        base = os.path.expanduser("~")
    return os.path.join(base, "tune-cache")


def _local_file_cache(model_id, *raw_paths, target_os="linux",
                      require_local_master=True):
    """Copy model artifacts to local SSD once per tune so every rung
    loads from disk instead of CIFS. Returns {resolved_orig: local_path}.

    Storage safety (never break the user's disk):
    - OS-aware temp dir (Windows TEMP, not /tmp).
    - pre-flight: TOTAL bytes vs free with 1.2x factor; per-file recheck
      (another process may have filled the disk between).
    - remote-master guard: local copies are pointless when llama-server
      runs on another box (it can't read our disk) — skipped with a log.
    - Skips copies that don't fit — falls back to orig paths."""
    import shutil
    if require_local_master:
        try:
            master = None
            for m in cfg.machines.all():
                if m.get("role") == "master":
                    master = m
                    break
            if master is not None:
                import local_node as _ln
                if not _ln.is_local(master):
                    _log(model_id, "local cache: master is remote — "
                                   "local copies wouldn't be visible to it, "
                                   "loading over network")
                    return {}
        except Exception:
            pass
    cache_dir = _cache_dir()
    mapping = {}
    try:
        os.makedirs(cache_dir, exist_ok=True)
    except OSError as e:
        _log(model_id, f"local cache: cannot create {cache_dir}: {e}")
        return mapping
    # pre-flight total (skip stat failures open: per-file check remains)
    try:
        _sizes = {}
        for raw in raw_paths:
            if not raw:
                continue
            try:
                _o = path_resolver.resolve_for(raw, target_os)
            except Exception:
                continue
            if os.path.exists(_o) and not _o.startswith(cache_dir):
                try:
                    _sizes[_o] = os.path.getsize(_o)
                except OSError:
                    pass
        if _sizes:
            try:
                _free = shutil.disk_usage(cache_dir).free
            except OSError:
                _free = -1
            _need = sum(_sizes.values()) * 1.2
            if 0 <= _free < _need:
                _log(model_id, f"local cache: need {_need/1024**3:.1f}GB, "
                               f"only {_free/1024**3:.1f}GB free on "
                               f"{cache_dir} — loading over network")
                return mapping
    except Exception:
        pass
    for raw in raw_paths:
        if not raw:
            continue
        try:
            orig = path_resolver.resolve_for(raw, target_os)
        except Exception:
            continue
        if not os.path.exists(orig) or orig.startswith(cache_dir):
            continue
        try:
            size = os.path.getsize(orig)
        except OSError:
            continue
        dest = os.path.join(cache_dir, os.path.basename(orig))
        try:
            if (os.path.exists(dest)
                    and os.path.getsize(dest) == size):
                mapping[orig] = dest
                continue
            free = shutil.disk_usage(cache_dir).free
            if free < size * 1.2:
                _log(model_id, f"local cache: only {free/1024**3:.0f}GB free, "
                               f"skipping copy of {os.path.basename(orig)} "
                               f"({size/1024**3:.1f}GB) — loading over network")
                continue
            _log(model_id, f"local cache: copying {os.path.basename(orig)} "
                           f"({size/1024**3:.1f}GB) to SSD...")
            shutil.copyfile(orig, dest)
            mapping[orig] = dest
            _log(model_id, "local cache: copy done")
        except Exception as e:
            _log(model_id, f"local cache copy failed: {e} — using network path")
    return mapping


def _bench_tps(port, timeout=120):
    """True decode speed. Uses server-reported predicted_per_second when
    available (immune to prefill/request overhead); wall-clock fallback
    otherwise. Prompt forces a long numeric run so we never bench a
    3-token 'Paris.' completion (that measured overhead, not tok/s)."""
    body = _json.dumps({
        "prompt": "Write the numbers from 1 to 200 in order, one per line:",
        "n_predict": 64, "temperature": 0,
        "ignore_eos": True,
    }).encode()
    t0 = time.time()
    try:
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/v1/completions", data=body,
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = _json.loads(r.read().decode("utf-8", "replace"))
    except Exception as e:
        return {"ok": False, "error": str(e)[:200]}
    dt = max(0.01, time.time() - t0)
    timings = data.get("timings") or {}
    pred = timings.get("predicted_n", 0) or 0
    text = data.get("content", "") or ""
    if not pred:
        # rough fallback: ~4 chars/token
        pred = max(1, len(text) // 4)
    reported = timings.get("predicted_per_second", 0) or 0
    tps = round(reported, 1) if reported else round(pred / dt, 1)
    # prefill side (turn-time objective: decode-only ranking picks wrong
    # winners for agent/chat workloads — record prompt speed too)
    pp_ms = timings.get("prompt_ms", 0) or 0
    pp_n = timings.get("prompt_n", 0) or 0
    pp_tps = round(pp_n / (pp_ms / 1000), 1) if pp_ms > 0 and pp_n > 0 else 0
    # representative prefill probe: 512-token prompt, 1 gen token.
    # effective throughput for a (P+G) turn: (P+G)/(P/pp + G/tg) is the
    # ranking metric (best TOTAL speed), decode alone is not.
    eff, eff_p, eff_g = tps, pp_n, pred
    try:
        _probe_body = _json.dumps({
            "prompt": ("Explain ambition in one long paragraph. " * 64),
            "n_predict": 1, "temperature": 0}).encode()
        _t0 = time.time()
        _req = urllib.request.Request(
            f"http://127.0.0.1:{port}/v1/completions", data=_probe_body,
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(_req, timeout=timeout) as _r:
            _data = _json.loads(_r.read().decode("utf-8", "replace"))
        _pt = (_data.get("timings") or {})
        _ppn, _ppm = _pt.get("prompt_n", 0) or 0, _pt.get("prompt_ms", 0) or 0
        if _ppn > 100 and _ppm > 0:
            eff_p = _ppn
            pp_tps = round(_ppn / (_ppm / 1000), 1)
            eff_g = pred or 64
            if pp_tps > 0 and tps > 0:
                eff = round((eff_p + eff_g)
                            / (eff_p / pp_tps + eff_g / tps), 1)
    except Exception:
        pass
    if pred < 16:
        # validity guard (DoE practice: reject physically meaningless rows
        # instead of averaging them in) — a <16-token completion is
        # overhead-dominated, not a decode measurement (cf. the 3-token bug)
        return {"ok": False, "error": f"too few tokens ({pred})",
                "tokens": pred}
    accept = None
    for k in ("draft_acceptance", "acceptance_rate"):
        if k in timings:
            accept = timings[k]
    return {"ok": True, "tps": tps, "seconds": round(dt, 1),
            "tokens": pred, "acceptance": accept,
            "pp_tps": pp_tps, "eff": eff, "eff_p": eff_p, "eff_g": eff_g,
            "measured": "server" if reported else "wall"}


def cancel_tune(model_id):
    """Flag a running tune to stop after the current rung (no save) and
    fail-fast the current bench server so it doesn't sit through a load."""
    import subprocess
    with _JOB_LOCK:
        j = _JOBS.get(model_id)
        if not j or j.get("state") != "running":
            return {"model_id": model_id, "state": (j or {}).get("state", "idle"),
                    "cancelled": False}
        j["cancel"] = True
        port = j.get("port", 18399)
    try:
        if os.name == "nt":
            # no pkill on Windows: taskkill by port-matching window title is
            # unreliable; the bench server dies with the trial timeout anyway.
            pass
        else:
            subprocess.run(["pkill", "-f", f"llama-server.*--port {port}"],
                           timeout=10)
    except Exception:
        pass
    return {"model_id": model_id, "state": "cancelling", "cancelled": True}


def _cancelled(model_id):
    with _JOB_LOCK:
        return bool((_JOBS.get(model_id) or {}).get("cancel"))


def _pick_port(wanted):
    """Free test port: skip ports held by other running tunes or anything
    already listening (two concurrent tunes must never share a port —
    benches would measure the wrong server)."""
    import socket as _so
    used = {j.get("port") for j in _JOBS.values()
            if j.get("state") == "running" and j.get("port")}
    for _ in range(50):
        if wanted in used:
            wanted += 1
            continue
        try:
            with _so.create_connection(("127.0.0.1", wanted), timeout=1):
                wanted += 1  # something listening: skip
                continue
        except OSError:
            return wanted
    return wanted


def start_tune(model_id, target_tps=None, port=18399,
               try_mtp_off=True):
    """Background tune: math estimate -> staged live search
    (ctx, then MTP, then ubatch, then fallbacks). Saves winner.
    target None = topology default (25 local single-box, 20 over RPC)."""
    import rpc_manager  # deferred: imports config_loader too
    model = cfg.get_model(model_id)
    if model is None:
        raise ValueError(f"model '{model_id}' not found")
    if not model.get("servers"):
        raise ValueError("no servers selected — pick hardware in the form first")
    if target_tps is None:
        try:
            _master, _peers = rpc_manager._split_servers(model.get("servers"))
            target_tps = (TARGET_TPS_LOCAL if not _peers
                          else TARGET_TPS_RPC)
        except ValueError:
            target_tps = TARGET_TPS_DEFAULT
    with _JOB_LOCK:
        if _JOBS.get(model_id, {}).get("state") == "running":
            return job_status(model_id)
        port = _pick_port(port)
        _JOBS[model_id] = {"model_id": model_id, "state": "running",
                           "log": [], "target_tps": target_tps, "port": port}
    _log(model_id, f"tune port {port}")
    t = threading.Thread(target=_tune_worker,
                         args=(model_id, target_tps, port, try_mtp_off),
                         daemon=True)
    with _JOB_LOCK:
        _JOBS[model_id]["thread"] = t
    t.start()
    return job_status(model_id)


def _code_rev():
    """Source mtime of this file: logged at job start so a result can
    always be traced to the exact code that produced it (edits need an
    app restart — deferred import still caches the module)."""
    try:
        return time.strftime("%H:%M:%S", time.localtime(os.path.getmtime(
            os.path.abspath(__file__))))
    except OSError:
        return "?"


def _log(model_id, msg):
    with _JOB_LOCK:
        j = _JOBS.get(model_id)
        if j is not None:
            j.setdefault("log", []).append(
                f"{time.strftime('%H:%M:%S')} {msg}")
            j["log"] = j["log"][-200:]


def _parse_placement(bench_log):
    """Actual layer placement from llama-server output — the ground truth
    our math only estimates. Presence of 'to CPU' here is the definitive
    RAM-spill answer (better than VRAM snapshots between rungs)."""
    import re as _re
    hits = []
    try:
        with open(bench_log, encoding="utf-8", errors="replace") as f:
            for line in f:
                ll = line.lower()
                if ("offloading" in ll or "buffer size" in ll
                        or "to cpu" in ll or "fit_params" in ll
                        or "failed to fit" in ll or "ggml_gallocr" in ll):
                    hits.append(line.strip().split(" I ", 1)[-1][:160])
    except OSError:
        pass
    return hits[-8:]


def _temp_note():
    """Master GPU temp + VRAM in one line (methodology: record peak
    accelerator state per rung; temp also flags thermal confounding)."""
    try:
        import subprocess
        p = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=temperature.gpu,memory.used",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10)
        temps, mems = [], []
        for line in p.stdout.splitlines():
            parts = [x.strip() for x in line.split(",")]
            if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
                temps.append(int(parts[0]))
                mems.append(int(parts[1]))
        if not temps:
            return ""
        mx = max(temps)
        flag = " THERMAL?" if mx > 83 else ""
        return (f" [gpu °C max={mx}{flag} | "
                f"vram={max(mems)/1024:.1f}GB]")
    except Exception:
        return ""


def _note_temp_c(note):
    """Extract max temp int from a _temp_note string (for history CSV)."""
    import re as _re
    m = _re.search(r"max=(\d+)", note or "")
    return m.group(1) if m else ""


def _history_log(model_id, label, bench, max_temp="", trial=None):
    """Append-only rung history (survives app restarts,unlike the job log).
    results.csv-style: one row per rung, measured numbers only. Carries
    the hardware fingerprint so reuse never crosses topologies."""
    try:
        import csv
        p = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "generated", "tune-history.csv")
        new = not os.path.exists(p)
        with open(p, "a", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["time", "model_id", "label", "tps", "tps2",
                            "tokens", "measured", "max_temp_c", "ok", "hwfp",
                            "pp_tps"])
            w.writerow([time.strftime("%Y-%m-%dT%H:%M:%S"), model_id, label,
                        bench.get("tps", ""), bench.get("tps2", ""),
                        bench.get("tokens", ""), bench.get("measured", ""),
                        max_temp, bool(bench.get("ok")),
                        _hw_fingerprint(trial) if trial is not None else "",
                        bench.get("pp_tps", "")])
    except Exception:
        pass


def _fit_print(model_id, base_path, extra_args=None, target_os="linux"):
    """Exact per-device placement from llama.cpp itself (llama-fit-params
    --fit-print), replacing guesswork for the fit verdict. Returns the raw
    output tail (or '' when the binary lacks the tool). Best-effort."""
    import subprocess
    try:
        resolved = path_resolver.resolve_for(base_path, target_os)
    except Exception:
        return ""
    for cand in (os.path.expanduser("~/llama.cpp/build/bin/llama-fit-params"),):
        if os.path.exists(cand):
            tool = cand
            break
    else:
        return ""
    try:
        p = subprocess.run(
            [tool, "-m", resolved, "--fit-print", "on"] + (extra_args or []),
            capture_output=True, text=True, timeout=120)
        out = (p.stdout or "") + (p.stderr or "")
        lines = [l.strip() for l in out.splitlines()
                 if l.strip().startswith(("llama_params_fit", "llama_fit",
                                          "Printing", "-c ", "-ngl", "-ts "))]
        return "\n".join(lines[-14:])
    except Exception as e:
        return f"fit-print failed: {e}"


def _seed_path(model_id):
    safe = "".join(c if (c.isalnum() or c in "-_.") else "_" for c in model_id)
    return os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "config", "models", f"{safe}.seed")


def snapshot_seed(model_id):
    """Freeze the CURRENT saved config as the known-good reference.
    Tune measures the seed first every run: the live config must BEAT it
    to replace it — a drifted baseline can never again cement itself."""
    import shutil
    model = cfg.get_model(model_id)
    if model is None:
        raise ValueError(f"model '{model_id}' not found")
    src = os.path.join(cfg.MODELS_DIR, f"{model_id}.yaml")
    dst = _seed_path(model_id)
    shutil.copyfile(src, dst)
    return {"model_id": model_id, "seed": dst}


def _load_seed(model_id):
    import yaml as _yaml
    p = _seed_path(model_id)
    if not os.path.exists(p):
        return None
    try:
        with open(p, encoding="utf-8") as f:
            return _yaml.safe_load(f) or None
    except Exception:
        return None


def _priors_path():
    return os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "generated", "tune-priors.json")


def load_priors():
    """Win/run counts per (arch, stage, option): what worked before on
    this hardware family. Orders future stages best-first so a cancelled
    or target-early-exited tune still banked the likely winners."""
    import json as _js
    try:
        p = _priors_path()
        if os.path.exists(p):
            return _js.load(open(p))
    except Exception:
        pass
    return {}


def save_prior(arch, stage, option, won):
    try:
        import json as _js
        priors = load_priors()
        a = priors.setdefault(arch or "unknown", {})
        s = a.setdefault(stage, {})
        o = s.setdefault(option, {"runs": 0, "wins": 0})
        o["runs"] += 1
        if won:
            o["wins"] += 1
        _js.dump(priors, open(_priors_path(), "w"))
    except Exception:
        pass


def prior_rate(priors, arch, stage, option):
    o = ((priors.get(arch) or {}).get(stage) or {}).get(option)
    if not o or not o.get("runs"):
        return None
    return o["wins"] / o["runs"]


def order_by_priors(priors, arch, stage, options, key=None):
    """Best-first by prior win rate; unknown options keep relative order
    after known ones (exploration preserved, exploitation first)."""
    key = key or (lambda o: str(o))
    scored = []
    for i, o in enumerate(options):
        r = prior_rate(priors, arch, stage, key(o))
        scored.append((0 if r is None else 1, -(r or 0), i, o))
    scored.sort()
    return [o for _, _, _, o in scored]
def _hw_fingerprint(trial):
    """Hardware+topology fingerprint for a trial: servers, device order,
    placement mode/weights. History-reuse must match this — the same
    label on different hardware is a DIFFERENT measurement (this exact
    mix-up once condemned an RPC config with local-only numbers)."""
    import hashlib as _hl
    parts = [
        ",".join(trial.get("servers") or []),
        trial.get("device_order") or "",
        trial.get("device_order_policy") or "",
        trial.get("tensor_split_mode") or "",
        trial.get("tensor_split") or "",
    ]
    return _hl.md5("|".join(parts).encode()).hexdigest()[:10]


# Dimensions that caused the 5→24 tok/s incident when they drifted.
# Compared seed-vs-live at every Tune start; placement-affecting drift
# is HIGH severity (it silently re-numbers RPC devices and relocates
# the output head + draft layers).
DRIFT_KEYS_FLAGS = ("ctx-size", "batch-size", "ubatch-size", "spec-type",
                    "spec-draft-n-max", "spec-draft-ngl",
                    "spec-draft-device", "cache-type-k", "cache-type-v",
                    "flash-attn", "split-mode", "n-gpu-layers", "ngl")
DRIFT_KEYS_TOP = ("tensor_split_mode", "tensor_split", "device_order",
                  "device_order_policy", "servers", "fit_margin",
                  "fit_ram_allow")


def drift_check(model_id):
    """Diff live config against its frozen seed. Returns
    {"seed": bool, "drifts": [{"key", "seed", "live", "severity"}]}.
    HIGH = placement/topology (order, numbering, mode, ngl, spec);
    MEDIUM = perf flags (batch, threads); LOW = the rest."""
    seed = _load_seed(model_id)
    if not seed:
        return {"seed": False, "drifts": []}
    live = cfg.get_model(model_id) or {}
    drifts = []

    def _norm(v):
        if isinstance(v, list):
            return ",".join(str(x) for x in v)
        return str(v if v is not None else "")

    for k in DRIFT_KEYS_TOP:
        s, lv = _norm(seed.get(k)), _norm(live.get(k))
        if s != lv:
            sev = ("HIGH" if k in ("tensor_split_mode", "tensor_split",
                                   "device_order", "device_order_policy",
                                   "servers") else "MEDIUM")
            drifts.append({"key": k, "seed": s, "live": lv,
                           "severity": sev})
    sf, lf = seed.get("flags", {}) or {}, live.get("flags", {}) or {}
    for k in DRIFT_KEYS_FLAGS:
        s, lv = _norm(sf.get(k)), _norm(lf.get(k))
        if s != lv:
            sev = ("HIGH" if k in ("spec-type", "spec-draft-n-max",
                                   "spec-draft-ngl", "spec-draft-device",
                                   "split-mode", "n-gpu-layers", "ngl")
                   else "MEDIUM")
            drifts.append({"key": f"flags.{k}", "seed": s, "live": lv,
                           "severity": sev})
    return {"seed": True, "drifts": drifts}


def _history_lookup(model_id, label, trial=None, max_age_days=7):
    """Reuse a recent verified rung instead of reloading it (resume logic:
    identical model+label+HARDWARE measured within N days returns tps)."""
    import csv
    try:
        p = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "generated", "tune-history.csv")
        if not os.path.exists(p):
            return None
        fp = _hw_fingerprint(trial) if trial is not None else None
        cutoff = time.time() - max_age_days * 86400
        best = None
        with open(p, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row.get("model_id") != model_id or row.get("label") != label:
                    continue
                if row.get("ok") != "True":
                    continue
                if fp and row.get("hwfp") and row.get("hwfp") != fp:
                    continue  # same label, different hardware: measure fresh
                try:
                    ts = time.mktime(time.strptime(
                        row.get("time", ""), "%Y-%m-%dT%H:%M:%S"))
                except (ValueError, TypeError):
                    continue
                if ts < cutoff:
                    continue
                try:
                    tps = float(row.get("tps") or 0)
                except (ValueError, TypeError):
                    continue
                if best is None or ts > best[0]:
                    best = (ts, tps)
        return best[1] if best else None
    except Exception:
        return None


def _probe_cache():
    import json as _js
    p = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "generated", "net-probe.json")
    try:
        return (_js.load(open(p)) if os.path.exists(p) else {}), p
    except Exception:
        return {}, p


def _pcie_info(machine):
    """PCIe link ('g4x16') per machine, file-cached 7d, best-effort.
    VMs often report nothing (host-controlled) — empty string then."""
    import subprocess
    import json as _js
    cache, cache_p = _probe_cache()
    key = f"pcie-{machine.get('name', '?')}"
    hit = cache.get(key)
    if hit and time.time() - hit.get("ts", 0) < 7 * 86400:
        return hit.get("val", "")
    val = ""
    q = ("nvidia-smi --query-gpu=pcie.link.gen.current,"
         "pcie.link.width.current --format=csv,noheader,nounits")
    try:
        from ssh_manager import SSHManager
        import local_node as _ln
        if _ln.is_local(machine):
            p = subprocess.run(q.split(), capture_output=True, text=True,
                               timeout=10)
            out = p.stdout
        else:
            ssh = SSHManager(machine["ip"], machine["username"],
                             machine["password"],
                             port=machine.get("port", 22))
            code, out, _ = ssh.run(q, timeout=15)
            if code != 0:
                out = ""
        tags = []
        for line in (out or "").splitlines():
            cells = [x.strip() for x in line.split(",")]
            if len(cells) == 2 and cells[0].isdigit() and cells[1].isdigit():
                tags.append(f"g{cells[0]}x{cells[1]}")
        val = ",".join(tags)
    except Exception:
        val = ""
    cache[key] = {"val": val, "ts": time.time()}
    try:
        _js.dump(cache, open(cache_p, "w"))
    except Exception:
        pass
    return val


def probe_link_mbps(machine, size_mb=32):
    """Effective master->machine throughput (Mbps), file-cached 7d.
    Small dd-over-SSH transfer: measures the real path RPC tensors take,
    not the NIC label (virtio reports speed -1)."""
    import subprocess
    import json as _js
    cache, cache_p = _probe_cache()
    name = machine.get("name", "?")
    hit = cache.get(name)
    if hit and time.time() - hit.get("ts", 0) < 7 * 86400:
        return hit["mbps"]
    mbps = 0
    try:
        from ssh_manager import SSHManager
        import local_node as _ln
        if _ln.is_local(machine):
            mbps = 0  # local: not a network hop
        else:
            t0 = time.time()
            # pump zeros over the same SSH path RPC tensors take.
            # sshpass -e reads the password from $SSHPASS (environment),
            # never from the command line — -p '...' would expose it in
            # `ps aux` to every user on the box. Password itself comes
            # from the in-memory vault-resolved machine dict.
            import os as _os
            cmd = (f"dd if=/dev/zero bs=1M count={size_mb} 2>/dev/null | "
                   f"sshpass -e ssh "
                   f"-o StrictHostKeyChecking=no "
                   f"-o ConnectTimeout=8 -c aes128-gcm@openssh.com "
                   f"{machine['username']}@{machine['ip']} "
                   f"-p {machine.get('port', 22)} 'cat > /dev/null'")
            env = dict(_os.environ, SSHPASS=machine['password'] or "")
            r = subprocess.run(["bash", "-c", cmd], timeout=120, env=env)
            dt = max(0.5, time.time() - t0)
            if r.returncode == 0:
                mbps = round(size_mb * 8 / dt)
    except Exception:
        mbps = 0
    cache[name] = {"mbps": mbps, "ts": time.time()}
    try:
        _js.dump(cache, open(cache_p, "w"))
    except Exception:
        pass
    return mbps


def virt_info(machine=None):
    """Virtualization + CPU truth for a box (Proxmox/KVM guests lie about
    PCIe link state and hide cpufreq — measured Mbps is truth, vPCIe is
    not). Returns dict; best-effort, never raises."""
    import subprocess
    info = {"virt": "", "cpu": "", "rebar": ""}
    try:
        from ssh_manager import SSHManager
        import local_node as _ln
        if machine is None or _ln.is_local(machine):
            def run(cmd):
                p = subprocess.run(cmd.split(), capture_output=True,
                                   text=True, timeout=10)
                return p.stdout.strip()
        else:
            ssh = SSHManager(machine["ip"], machine["username"],
                             machine["password"],
                             port=machine.get("port", 22))

            def run(cmd):
                code, out, _ = ssh.run(cmd, timeout=15)
                return out.strip() if code == 0 else ""
        info["virt"] = run("systemd-detect-virt") or "bare?"
        cpu = run("lscpu -J")
        if cpu:
            import json as _js
            try:
                fields = {e.get("field", "").rstrip(":"): e.get("data", "")
                          for e in _js.loads(cpu).get("lscpu", [])}
                info["cpu"] = (fields.get("Model name", "") + " " +
                               fields.get("Flags", ""))[:160]
            except Exception:
                pass
        # Resizable BAR: large VRAM windows need it for full-speed access
        try:
            def _rebar():
                import subprocess as _sp
                q = ("nvidia-smi --query-gpu=pci.bar1_memory_total "
                     "--format=csv,noheader,nounits")
                if machine is None or _ln.is_local(machine):
                    p = _sp.run(q.split(), capture_output=True, text=True,
                                timeout=10)
                    return p.stdout.strip().splitlines()
                code, out, _ = ssh.run(q, timeout=15)
                return out.strip().splitlines() if code == 0 else []
            bars = _rebar()
            if bars:
                nums = [b.strip().split()[0] for b in bars
                        if b.strip()[:1].isdigit()]
                if nums:
                    info["rebar"] = ",".join(nums)
        except Exception:
            pass
    except Exception:
        pass
    return info


def has_moe_tensors(first_path):
    """True when the GGUF actually carries MoE expert tensors (dense
    FFN files like our Qwen27B/gemma4 return False despite MoE-ish
    names — checked, not assumed). Gates expert-placement variants."""
    try:
        import re as _re
        pat = _re.compile(r"(exps|expert|moe)", _re.IGNORECASE)
        for p in gguf_inspect.shard_paths(first_path):
            for n in gguf_inspect.tensor_names(p):
                if pat.search(n):
                    return True
        return False
    except Exception:
        return False


_FLAGS_CACHE = {}


def supports_flag(flag):
    """Does this build's llama-server accept --flag? (version guard:
    exotic flags must skip, not burn a load, on builds that lack them.)"""
    import subprocess
    if flag in _FLAGS_CACHE:
        return _FLAGS_CACHE[flag]
    try:
        # no hardcoded build dirs: expanduser finds the local checkout on
        # any machine (Windows resolves to %USERPROFILE%\llama.cpp).
        cand = os.path.expanduser("~/llama.cpp/build/bin/llama-server")
        if os.name == "nt":
            cand += ".exe"
        p = subprocess.run(
            [cand, "--help"],
            capture_output=True, text=True, timeout=30)
        ok = flag in (p.stdout or "")
    except Exception:
        ok = True  # fail open: a rung failure is still logged, not fatal
    _FLAGS_CACHE[flag] = ok
    return ok


def has_swa(meta):
    """True when the GGUF declares sliding-window attention (any arch):
    gemma3/4 hybrid stacks carry <arch>.attention.sliding_window[_pattern].
    Name-matching the arch would miss the next SWA family; read the spec."""
    try:
        return any("sliding_window" in str(k).lower()
                   for k in (meta or {}).keys())
    except Exception:
        return False


# Minimum free VRAM per GPU for a config to count as SAFE (optimize for
# BOTH capacity and latency: a rung 2% faster with 200MB headroom is one
# background process away from OOM — reported, never saved while a safe
# alternative exists).
SAFE_MIN_FREE_GB = 0.75


def measure_headroom(servers):
    """Min free VRAM (GB) across the selected servers' GPUs, right now.
    Lightweight nvidia-smi poll (local + one SSH each), not full stats."""
    import subprocess
    q = "nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits"
    frees = []
    for name in servers or []:
        try:
            m = cfg.machines.require(name)
        except ValueError:
            continue
        try:
            from ssh_manager import SSHManager
            import local_node as _ln
            if _ln.is_local(m):
                p = subprocess.run(q.split(), capture_output=True, text=True,
                                   timeout=10)
                out = p.stdout
            else:
                ssh = SSHManager(m["ip"], m["username"], m["password"],
                                 port=m.get("port", 22))
                code, out, _ = ssh.run(q, timeout=15)
                if code != 0:
                    continue
            for line in (out or "").splitlines():
                line = line.strip()
                if line.replace(".", "", 1).isdigit():
                    frees.append(float(line) / 1024)
        except Exception:
            continue
    return round(min(frees), 2) if frees else -1.0


def _kv_pair_per_token(resolved_path, meta, k_type, v_type):
    """KV bytes/token for a (possibly mixed) K/V precision pair."""
    _BPT = {"q8_0": 1.0625, "q4_0": 0.5625, "f16": 2.0,
            "bf16": 2.0, "f32": 4.0, "q5_0": 0.6875, "q5_1": 0.6875}
    try:
        n_full, _, _ = gguf_inspect.block_attention_kinds(resolved_path)
    except Exception:
        n_full = 32
    kv_heads = klen = vlen = 0
    for k, v in (meta or {}).items():
        kl = str(k).lower()
        if kl.endswith("attention.head_count_kv") and isinstance(v, int):
            kv_heads = v
        if kl.endswith("attention.key_length") and isinstance(v, int):
            klen = v
        if kl.endswith("attention.value_length") and isinstance(v, int):
            vlen = v
    if not (kv_heads and klen and vlen):
        kv_heads, klen, vlen = 8, 128, 128
    return (n_full * kv_heads
            * (klen * _BPT.get(str(k_type), 1.0625)
               + vlen * _BPT.get(str(v_type), 1.0625)))


def _parse_acceptance(bench_log):
    """Draft acceptance from the bench server's slot timing lines
    (bench API timings omit it). Returns (rate, accepted, generated) or
    None. None + MTP active = suspicious (drafts never verifying)."""
    import re as _re
    last = None
    try:
        with open(bench_log, encoding="utf-8", errors="replace") as f:
            for line in f:
                m = _re.search(
                    r"draft acceptance\s*=\s*([\d.]+)\s*\(\s*(\d+)\s*accepted\s*/\s*(\d+)\s*generated",
                    line)
                if m:
                    last = (float(m.group(1)), int(m.group(2)),
                            int(m.group(3)))
    except OSError:
        pass
    return last


def _trial_os(trial):
    """Master OS for a trial (for path resolution)."""
    try:
        import rpc_manager as _rm
        return _rm._split_servers(trial.get("servers") or [])[0].get("os", "linux")
    except Exception:
        return "linux"


def _run_trial(model_id, trial, port, label, path_map=None):
    """One load+bench cycle, SINGLE launch (tune owns its trials end to end).
    Returns (load_ok, bench_dict)."""
    import rpc_manager
    import subprocess
    import copy
    import platform as _platform
    # Trials execute generated POSIX shell scripts (bash, sshpass, /dev/tcp
    # probes) — Windows can't run them locally. macOS works (bash exists).
    if _platform.system() == "Windows":
        return False, {"ok": False, "error": "tuning runs locally only on Linux/macOS (POSIX shell required)"}
    tmp_id = f"{model_id}__tune_tmp"
    if _cancelled(model_id):
        return False, {"ok": False, "error": "cancelled"}
    # NOTE: no history-reuse. Reused numbers once crowned a 130 tok/s
    # local-only rung as the reference inside an RPC tune (same labels,
    # different hardware) and every downstream stage inherited the
    # phantom. History CSV stays as audit-only; each rung loads fresh.
    # bench server output goes here (not DEVNULL) so dead-rung
    # failures (e.g. fa=off refusing to bind) stay diagnosable
    bench_log = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "generated", f"bench-{tmp_id}.log")
    trial = copy.deepcopy(trial)
    if path_map:
        # point trials at local SSD copies (mapped by resolved orig path)
        for key in ("base_path", "path"):
            if trial.get(key):
                try:
                    r = path_resolver.resolve_for(trial[key], _trial_os(trial))
                    if r in path_map:
                        trial[key] = path_map[r]
                except Exception:
                    pass
        fl = trial.get("flags", {}) or {}
        if fl.get("mmproj"):
            try:
                r = path_resolver.resolve_for(fl["mmproj"], _trial_os(trial))
                if r in path_map:
                    fl["mmproj"] = path_map[r]
            except Exception:
                pass
        trial["flags"] = fl
    cfg.save_model(tmp_id, {**trial, "model_id": tmp_id})
    blog = open(bench_log, "w", encoding="utf-8")
    try:
        _log(model_id, f"load {label} ...")
        servers = trial.get("servers") or []
        m, peers = rpc_manager._split_servers(servers)
        if peers:
            script = rpc_manager.generate_script(trial, servers)
            # generated scripts are vault-only (no baked password): supply
            # $SSH_PASS from the in-memory vault-resolved master password.
            # Env-only — never on the command line (ps-visible) or on disk.
            import os as _os2
            trial_env = dict(_os2.environ,
                             SSH_PASS=m.get("password") or "")
            proc = subprocess.Popen(
                ["bash", script, str(port)],
                stdout=blog, stderr=subprocess.STDOUT,
                start_new_session=True, env=trial_env)
        else:
            cmd = rpc_manager.build_llama_command(
                trial, servers).replace('"$PORT"', str(port))
            proc = subprocess.Popen(
                ["bash", "-c", cmd],
                stdout=blog, stderr=subprocess.STDOUT,
                start_new_session=True)
        try:
            up, dead_tail = False, ""
            for _ in range(150):
                if rpc_manager._port_open(m, port,
                                          __import__("local_node").is_local(m)):
                    up = True
                    break
                if proc.poll() is not None:
                    break  # exited prematurely: fail fast, keep the log
                time.sleep(2)
            if not up:
                try:
                    blog.flush()
                    with open(bench_log, encoding="utf-8",
                              errors="replace") as f:
                        dead_tail = f.read()[-400:]
                except OSError:
                    pass
                _log(model_id, f"  load FAILED: {dead_tail[-300:]}")
                return False, {"ok": False, "error": "load failed"}
            healthy, hinfo = _wait_healthy(port, timeout_s=900)
            if not healthy:
                tail = ""
                try:
                    blog.flush()
                    with open(bench_log, encoding="utf-8",
                              errors="replace") as f:
                        tail = f.read()[-600:]
                except OSError:
                    pass
                _log(model_id, f"  server never healthy: {hinfo}\n"
                               f"  bench log tail: {tail[-400:]}")
                return True, {"ok": False, "error": "never healthy"}
            b = _bench_tps(port)
            # second consecutive run: nearly free, exposes run-to-run
            # variance (single-rep gen numbers have fooled people before)
            b2 = _bench_tps(port)
            if b.get("ok") and b2.get("ok"):
                b["tps2"] = b2["tps"]
                b["tps"] = round((b["tps"] + b2["tps"]) / 2, 1)
            # capacity half of the score: headroom right now, same GPUs
            b["min_free_gb"] = measure_headroom(trial.get("servers") or [])
            _temps = _temp_note()
            _acc = _parse_acceptance(bench_log)
            if _acc is not None:
                b["acceptance"] = round(_acc[0], 3)
                b["acc_n"] = f"{_acc[1]}/{_acc[2]}"
            _log(model_id, f"  -> {b}{_temps}")
            _history_log(model_id, label, b, _note_temp_c(_temps), trial)
            for pl in _parse_placement(bench_log):
                _log(model_id, f"  placement: {pl}")
            return True, b
        finally:
            try:
                if proc.poll() is None:
                    proc.terminate()
                    proc.wait(timeout=15)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
            try:
                blog.close()
            except Exception:
                pass
    finally:
        try:
            cfg.delete_model(tmp_id)
        except Exception:
            pass


UB_LADDER = [128, 256, 376, 512, 768, 1024]
B_LADDER = [512, 1024, 2048, 4096, 8192]
MARGIN_LADDER = [256, 512, 1024, 2048]
KV_CANDIDATES = ["q8_0", "f16", "q4_0"]


def _ladder_neighbors(current, ladder):
    """Dynamic offsets around ANY live value: nearest rung below, the
    value itself, nearest rung above. Whatever the baseline number is,
    both directions get probed once — no fixed grids anywhere."""
    try:
        cur = int(current)
    except (TypeError, ValueError):
        cur = ladder[len(ladder) // 2]
    if cur in ladder:
        i = ladder.index(cur)
        lo = ladder[i - 1] if i > 0 else cur
        hi = ladder[i + 1] if i < len(ladder) - 1 else cur
    else:
        below = [x for x in ladder if x < cur]
        above = [x for x in ladder if x > cur]
        lo = below[-1] if below else cur
        hi = above[0] if above else cur
    return list(dict.fromkeys([lo, cur, hi]))  # dedupe, keep order


def _ub_neighbors(current):
    return _ladder_neighbors(current, UB_LADDER)


def _same_mtp(a_flags, b_flags):
    """Same speculation setup? (spec-type, n-max, draft-ngl)."""
    keys = ("spec-type", "spec-draft-n-max", "spec-draft-ngl")
    norm = lambda f, k: str(f.get(k, "") or "").lower()
    return all(norm(a_flags, k) == norm(b_flags, k) for k in keys)


def _same_split(a_trial, b_trial):
    """Same placement? (mode, weights, order, policy)."""
    keys = ("tensor_split_mode", "tensor_split", "device_order",
            "device_order_policy")
    return all(str(a_trial.get(k, "") or "") == str(b_trial.get(k, "") or "")
               for k in keys)


def _apply_variant(trial, variant):
    """Mutate a trial's flags per variant dict. Returns label string."""
    import copy
    t = copy.deepcopy(trial)
    f = t.get("flags", {}) or {}
    bits = []
    if "ctx" in variant:
        f["ctx-size"] = variant["ctx"]
        bits.append(f"ctx={variant['ctx']}")
    if "kv" in variant:
        _kv = variant["kv"]
        if isinstance(_kv, (tuple, list)):
            f["cache-type-k"], f["cache-type-v"] = _kv[0], _kv[1]
            bits.append(f"kv={_kv[0]}/{_kv[1]}")
        else:
            f["cache-type-k"] = _kv
            f["cache-type-v"] = _kv
            bits.append(f"kv={_kv}")
    if "mtp" in variant:
        if variant["mtp"] == "off":
            f.pop("spec-type", None)
            f.pop("spec-draft-n-max", None)
            f.pop("spec-draft-ngl", None)
            f.pop("spec-draft-device", None)
            bits.append("mtp=off")
        elif variant["mtp"] == "combo":
            # champion combo elsewhere: MTP + ngram-map-k4v, defaults
            f["spec-type"] = "draft-mtp,ngram-map-k4v"
            f["spec-draft-n-max"] = 2
            f.setdefault("spec-draft-ngl", 99)
            bits.append("mtp=combo")
        elif isinstance(variant["mtp"], str) and variant["mtp"].startswith("nogl-"):
            # MTP on but WITHOUT --spec-draft-ngl: draft placement falls
            # back to llama.cpp default (partial/CPU) instead of forcing
            # the whole draft stack onto GPU (suspected RAM-spill trigger)
            n = int(variant["mtp"].split("-", 1)[1] or 2)
            f["spec-type"] = "draft-mtp"
            f["spec-draft-n-max"] = n
            f.pop("spec-draft-ngl", None)
            bits.append(f"mtp=n{n}-nongl")
        else:
            f["spec-type"] = "draft-mtp"
            f["spec-draft-n-max"] = variant["mtp"]
            f.setdefault("spec-draft-ngl", 99)
            bits.append(f"mtp=n{variant['mtp']}")
    if "ubatch" in variant:
        f["ubatch-size"] = variant["ubatch"]
        bits.append(f"ub={variant['ubatch']}")
        # constraint: batch must cover ubatch (llama.cpp requires b>=ub)
        try:
            if int(f.get("batch-size", 2048)) < int(variant["ubatch"]):
                f["batch-size"] = int(variant["ubatch"])
                bits.append(f"(b bumped to {variant['ubatch']})")
        except (TypeError, ValueError):
            pass
    if "batch" in variant:
        f["batch-size"] = variant["batch"]
        bits.append(f"b={variant['batch']}")
    if "margin" in variant:
        t["fit_margin"] = variant["margin"]
        bits.append(f"margin={variant['margin']}")
    if "flash" in variant:
        f["flash-attn"] = variant["flash"]
        bits.append(f"fa={variant['flash']}")
    if "draft_device" in variant:
        # pin the MTP catch-up decode to one fast device (+8% prefill
        # measured elsewhere; gen effect unknown — measured here)
        f["spec-draft-device"] = variant["draft_device"]
        bits.append(f"dd={variant['draft_device']}")
    if variant.get("swafull"):
        # sliding-window archs (gemma3/4): full-size SWA cache. Past the
        # window a shared-prefix workload otherwise loses prompt-cache
        # reuse entirely (measured 0% vs 90% elsewhere).
        f["swa-full"] = True
        bits.append("swa=on")
    if "threads" in variant:
        f["threads"] = variant["threads"]
        bits.append(f"t={variant['threads']}")
    if "moe_cpu" in variant:
        # MoE expert-placement (layer-split alternative): keep expert
        # weights on CPU, dense/attention on GPU. For MoE-over-RPC this
        # usually beats layer-split — the token stream only crosses the
        # wire for the CPU expert hop, and VRAM stays fully busy.
        mc = variant["moe_cpu"]
        if mc == "off":
            f.pop("cpu-moe", None)
            f.pop("n-cpu-moe", None)
            bits.append("moecpu=off")
        elif mc == "all":
            f["cpu-moe"] = True
            f.pop("n-cpu-moe", None)
            bits.append("moecpu=all")
        else:
            f["n-cpu-moe"] = int(mc)
            f.pop("cpu-moe", None)
            bits.append(f"moecpu={mc}")
    if "split" in variant:
        # Full placement spec verbatim: {mode, device_order, policy,
        # weights} (+ optional servers reorder for peer-swap).
        sp = variant["split"]
        t["tensor_split_mode"] = sp["mode"]
        t["tensor_split"] = sp.get("weights", "")
        if sp.get("device_order"):
            t["device_order"] = sp["device_order"]
        if sp.get("policy"):
            t["device_order_policy"] = sp["policy"]
        if sp.get("servers"):
            t["servers"] = list(sp["servers"])
            bits.append("peerswap")
        if sp["mode"] == "auto":
            # ngl-order candidates reproduce the last-good shape
            # (explicit -ngl, explicit order, no fit): ensure ngl present
            f.setdefault("n-gpu-layers", 99)
        bits.append(f"split={sp.get('weights') or sp['mode']}")
    if "split2" in variant:
        # Device-SET override (stage 3c): full placement spec that also
        # REPLACES the servers list (not just reorders) — solo GPU,
        # master-only, or master+peer-subset trials. Kept separate from
        # "split" so the baseline set stays the dedupe floor.
        sp = variant["split2"]
        t["tensor_split_mode"] = sp["mode"]
        t["tensor_split"] = sp.get("weights", "")
        t["device_order"] = sp.get("device_order", "")
        t["device_order_policy"] = sp.get("policy", "vram-first")
        if sp.get("servers"):
            t["servers"] = list(sp["servers"])
        bits.append(f"devset={sp.get('device_order') or sp['mode']}")
    t["flags"] = f
    return t, " ".join(bits)


def _tune_worker(model_id, target_tps, port, try_mtp_off):
    import rpc_manager
    import copy

    def finish(state, **kw):
        with _JOB_LOCK:
            _JOBS[model_id].update({"state": state, **kw})

    try:
        # ONE-button flow: safe defaults first (locked ctx kept,
        # MTP iff present, vram-first fit), so Tune works straight
        # from fresh form inputs.
        _log(model_id, f"tuner rev {_code_rev()} | target {target_tps} tok/s")
        try:
            import subprocess as _sp
            _br = _sp.run(["git", "-C", os.path.expanduser("~/llama.cpp"),
                           "log", "-1", "--format=%h %ad", "--date=short"],
                          capture_output=True, text=True, timeout=15)
            _log(model_id, f"llama.cpp build: "
                           f"{_br.stdout.strip() or 'unknown'}")
        except Exception:
            _log(model_id, "llama.cpp build: unknown")
        model = cfg.get_model(model_id)
        master0 = rpc_manager._split_servers(model.get("servers"))[0]
        d = safe_defaults(model.get("base_path") or model.get("path", ""),
                          model.get("servers"), model.get("device_order") or "",
                          master0.get("os", "linux"),
                          model.get("flags", {}) or {})
        model["tensor_split_mode"] = d["tensor_split_mode"]
        model["fit_margin"] = d["fit_margin"]
        model["fit_ram_allow"] = d["fit_ram_allow"]
        model["device_order_policy"] = d["device_order_policy"]
        flags0 = model.get("flags", {}) or {}
        flags0.update(d["flags"])
        model["flags"] = flags0
        cfg.save_model(model_id, model)
        _log(model_id, f"safe defaults applied (MTP "
                       f"{'on' if d['has_mtp'] else 'off'}, model cap {d['model_max_ctx']})")
        # drift test: diff live config vs frozen seed BEFORE measuring.
        # HIGH-severity drift (order, numbering, mode) caused the 5→24
        # incident — surface it here, not after an hour of rungs.
        try:
            _dc = drift_check(model_id)
            if not _dc["seed"]:
                _log(model_id, "drift: no seed frozen — consider Seed button "
                               "once this config proves itself")
            for _dr in _dc["drifts"]:
                _log(model_id, f"drift [{_dr['severity']}] {_dr['key']}: "
                               f"seed={_dr['seed'][:60]} live={_dr['live'][:60]}")
            if not _dc["drifts"] and _dc["seed"]:
                _log(model_id, "drift: live matches seed — clean")
        except Exception as e:
            _log(model_id, f"drift check failed: {e}")
        model = cfg.get_model(model_id)
        flags = model.get("flags", {}) or {}
        master = rpc_manager._split_servers(model.get("servers"))[0]
        est = estimate(model.get("base_path") or model.get("path", ""),
                       model.get("servers"), model.get("device_order") or "",
                       master.get("os", "linux"),
                       (flags.get("cache-type-k") or "q8_0"),
                       flags.get("mmproj", ""))
        _log(model_id, f"math: weights {est['weights_gb']}GB + "
                       f"KV {est['kv_per_token_kb']}KB/tok, usable "
                       f"{est['usable_vram_gb']}GB; ctx LOCKED "
                       f"({flags.get('ctx-size')}), "
                       f"KV pairs tested (uniform q8/f16/q4 + k4-f16 mixes)")
        for _vn, _vi in (est.get("virt") or {}).items():
            _fl = _vi.get("cpu", "")
            import re as _re2
            _avx = bool(_re2.search(r"(^|\s)avx(\s|$)", _fl))
            _log(model_id, f"virt {_vn}: {_vi.get('virt') or '?'} "
                           f"cpu={(_fl.split('  ')[0] if _fl else '?')[:40]} "
                           f"avx={'yes' if _avx else 'NO (set Proxmox CPU type to host!)'} "
                               f"rebar={_vi.get('rebar') or '?'}")
        has_mtp = bool(d.get("has_mtp"))  # file carries MTP tensors
        # exact fit verdict from llama.cpp itself (llama-fit-params):
        # per-device plan + the CLI it would emit. Ground truth next to
        # our estimate.
        _fp = _fit_print(
            model_id, model.get("base_path") or model.get("path", ""),
            ["-c", str(flags.get("ctx-size") or 32768),
             "-ctk", str(flags.get("cache-type-k") or "q8_0"),
             "-ctv", str(flags.get("cache-type-v") or "q8_0")],
            master.get("os", "linux"))
        if _fp:
            for _line in _fp.splitlines():
                _log(model_id, f"fit-print: {_line}")
        # static validation first (instant, catches bad flags/devices
        # before burning a single load)
        val = rpc_manager.validate_model(model_id)
        for issue in val.get("issues", []):
            if issue.get("level") == "error":
                finish("done", ok=False,
                       detail=f"config invalid: {issue.get('msg')}")
                _log(model_id, f"config ERROR: {issue.get('msg')}")
                return
        # SSD cache: copy once, every rung loads local (not CIFS)
        pmap = _local_file_cache(
            model_id, model.get("base_path") or model.get("path", ""),
            flags.get("mmproj", ""), target_os=master.get("os", "linux"))
        base = copy.deepcopy(model)
        bf = base.get("flags", {}) or {}
        # ctx is the user's (locked): Tune optimizes everything else
        # around it, never sacrifices it. KV starts locked, stage 2b
        # measures the ladder.
        # ubatch keeps the hardware-aware safe default (512 on capable
        # cards); only fill when missing.
        bf.setdefault("ubatch-size", 256)
        bf.setdefault("batch-size", 2048)
        base["flags"] = bf
        locked_ctx = bf.get("ctx-size")
        locked_kv = (bf.get("cache-type-k"), bf.get("cache-type-v"))

        best = None  # (tps, variant_label, trial_dict, min_free_gb)
        best_safe = None  # fastest rung with min_free >= SAFE_MIN_FREE_GB

        def _ctx_of(trial):
            try:
                return int((trial.get("flags", {}) or {}).get("ctx-size") or 0)
            except (TypeError, ValueError):
                return 0

        def consider(b, label, trial, trusted=False):
            # ranked on EFFECTIVE throughput (turn time for a representative
            # P+G request), not decode alone — best total speed at this ctx.
            # Quality tie-break: a rung with MORE context within 5% of the
            # best speed wins — "the best context/speed/KV balance" means a
            # 4% slower config holding 2x the ctx is the better model
            # (decode-only ranking picked fast-but-tiny winners before).
            nonlocal best, best_safe
            tps = (b or {}).get("eff", (b or {}).get("tps"))
            if tps is None:
                return
            free = (b or {}).get("min_free_gb", -1.0)
            if best is not None and tps <= best[0] and tps > best[0] * 0.95 \
                    and _ctx_of(trial) > _ctx_of(best[2]):
                tps = best[0] + 0.01  # nudge past best: same speed band, more ctx wins
            if best is None or tps > best[0]:
                best = (tps, label, copy.deepcopy(trial), free)
            if (free >= SAFE_MIN_FREE_GB or trusted) and (
                    best_safe is None or tps > best_safe[0]):
                best_safe = (tps, label, copy.deepcopy(trial), free)

        def _pick():
            """Winner: fastest SAFE rung; fastest overall only when
            nothing safe exists (flagged unsafe in the report)."""
            if best_safe is not None:
                return best_safe, True
            return best, False

        # ---- save best ----
        def _save_winner():
            (tps, label, win_trial, free), safe = _pick()
            win_flags = win_trial.get("flags", {}) or {}
            model = cfg.get_model(model_id)
            model["flags"] = win_flags
            model["tensor_split_mode"] = win_trial.get("tensor_split_mode", "fit")
            model["tensor_split"] = win_trial.get("tensor_split", "")
            model["device_order"] = win_trial.get("device_order", "")
            model["device_order_policy"] = win_trial.get(
                "device_order_policy", "vram-first")
            if "fit_margin" in win_trial:
                model["fit_margin"] = win_trial["fit_margin"]
            model["expose"] = win_trial.get("expose", model.get("expose", {}))
            cfg.save_model(model_id, model)
            ok = tps >= target_tps
            winner = {"ctx-size": win_flags.get("ctx-size"),
                      "kv": win_flags.get("cache-type-k"),
                      "mtp": win_flags.get("spec-type", "off"),
                      "mtp_n": win_flags.get("spec-draft-n-max"),
                      "split": win_trial.get("tensor_split") or win_trial.get("tensor_split_mode", "fit"),
                      "ubatch": win_flags.get("ubatch-size"),
                      "threads": win_flags.get("threads"),
                      "flash": win_flags.get("flash-attn"),
                      "min_free_gb": free,
                      "safe": safe,
                      "tps": tps}
            detail = ""
            if not safe:
                detail = (f"no rung kept >= {SAFE_MIN_FREE_GB}GB headroom; "
                          f"saved fastest-unsafe {tps} tok/s —verify headroom!")
            elif not ok:
                detail = f"best {tps} tok/s under target {target_tps}"
            finish("done", ok=(ok and safe), winner=winner, detail=detail)
            _log(model_id, f"WINNER {label} = {tps} tok/s "
                           f"(headroom {free}GB, "
                           f"{'SAFE' if safe else 'UNSAFE'}) — saved")
            # push winner into the swap scripts so the NEXT load serves it
            # (already-loaded instances keep serving until evicted).
            try:
                import swap_manager as _sm
                _pushed = _sm.push_scripts()
                _log(model_id, f"swap scripts pushed "
                               f"({len(_pushed.get('written', []))} files)")
            except Exception as e:
                _log(model_id, f"swap push failed: {e} — relaunch swap "
                               f"from the UI to pick up the winner")

        def _best_flags():
            return (best[2].get("flags", {}) or {}) if best else {}

        def _best_mtp():
            f = _best_flags()
            st = str(f.get("spec-type", "")).lower()
            if "draft-mtp" in st:
                if "ngram" in st:
                    return "combo"
                try:
                    n = int(f.get("spec-draft-n-max", 1))
                except (TypeError, ValueError):
                    n = 1
                # no draft-ngl = default placement variant
                if "spec-draft-ngl" not in f:
                    return f"nogl-{n}"
                return n
            return "off"

        def _best_split():
            """Full placement spec of the current best trial (reproducible)."""
            if not best:
                return {"mode": "fit", "device_order": "",
                        "policy": "vram-first", "weights": ""}
            t = best[2]
            return {"mode": t.get("tensor_split_mode", "fit"),
                    "device_order": t.get("device_order", ""),
                    "policy": t.get("device_order_policy", "vram-first"),
                    "weights": t.get("tensor_split", "") or ""}

        # ---- dedupe runner: an identical config never burns a load twice.
        # Signature covers every tuned dimension; stages share it, so a
        # THOROUGH candidate equal to an already-measured LIGHT rung (or
        # to the model's own current config) skips with a log line.
        class _ExitWorker(Exception):
            pass

        _measured = []

        def _sig(trial):
            f = trial.get("flags", {}) or {}
            return (str(f.get("spec-type", "")).lower(),
                    str(f.get("spec-draft-n-max", "")),
                    "ngl" if "spec-draft-ngl" in f else "-",
                    str(f.get("spec-draft-device", "")),
                    str(f.get("n-gpu-layers", "")),
                    ",".join(trial.get("servers") or []),
                    trial.get("tensor_split_mode", ""),
                    trial.get("tensor_split", "") or "",
                    trial.get("device_order", "") or "",
                    trial.get("device_order_policy", ""),
                    str(f.get("ubatch-size", "")),
                    str(f.get("batch-size", "")),
                    str(trial.get("fit_margin", "")),
                    str(f.get("cache-type-k", "")) + "/"
                    + str(f.get("cache-type-v", "")),
                    str(f.get("flash-attn", "")),
                    "swa" if f.get("swa-full") else "-")

        def _probe(stage, variant, key=None):
            if _stop():
                raise _ExitWorker()
            trial, label = _apply_variant(base, {"ctx": ctx_star, **variant})
            sig = _sig(trial)
            if sig in _measured:
                _log(model_id, f"{stage} {label} ... already measured — skipped")
                return
            _measured.append(sig)
            load_ok, b = _run_trial(
                model_id, trial, port, f"{stage} {label}", path_map=pmap)
            if load_ok and b.get("ok"):
                consider(b, f"{stage} {label}", trial)
                if key is not None:
                    _note_result(stage, key, b["tps"])

        # ---- stage 0: seed reference (known-good snapshot, if any) ----
        # Measured FIRST so best starts at proven speed, not at whatever
        # the live config drifted to. A drifted baseline can never again
        # cement itself: the winner must beat the seed.
        _seed = _load_seed(model_id)
        if _seed and _seed.get("servers"):
            _log(model_id, "seed reference found — measuring it first")
            _strial = copy.deepcopy(_seed)
            _strial["model_id"] = model_id
            _slabel = "seed reference"
            _ssig = _sig(_strial)
            _measured.append(_ssig)
            load_ok, b = _run_trial(model_id, _strial, port,
                                    f"stage0 {_slabel}", path_map=pmap)
            if load_ok and b.get("ok"):
                # seed is user-blessed production truth: exempt from the
                # headroom gate (it runs at 40MB free and holds 24 tok/s —
                # a heuristic must not overrule measured production).
                consider(b, f"stage0 {_slabel}", _strial, trusted=True)
                _log(model_id, f"stage0: seed {b['tps']} tok/s — "
                               f"the number to beat")
            else:
                _log(model_id, "stage0: seed failed to load "
                               "(hardware changed?) — continuing")

        # ---- priors: what won before on this arch family ----
        # orders every stage best-first (cancel-safe: partial runs bank
        # the likely winners) and records every rung's max for next time.
        try:
            _arch = str((gguf_inspect.metadata(
                path_resolver.resolve_for(
                    model.get("base_path") or model.get("path", ""),
                    master.get("os", "linux"))
            ) or {}).get("general.architecture", "") or "unknown")
        except Exception:
            _arch = "unknown"
        _priors = load_priors()
        _arch_priors = _priors.get(_arch, {})

        def _note_result(stage, key, tps):
            try:
                a = _priors.setdefault(_arch, {})
                s = a.setdefault(stage, {})
                if tps > float(s.get(key, 0) or 0):
                    s[key] = round(float(tps), 1)
                import json as _js
                _js.dump(_priors, open(_priors_path(), "w"))
            except Exception:
                pass

        def _order(stage, opts, keyfn):
            scored = []
            for i, o in enumerate(opts):
                v = _arch_priors.get(stage, {}).get(keyfn(o))
                scored.append((0 if v is None else 1, -(v or 0), i, o))
            scored.sort()
            return [o for _, _, _, o in scored]

        _log(model_id, f"priors[{_arch}]: "
                       f"{ {s: len(v) for s, v in _arch_priors.items()} } "
                       f"stages known")

        # ---- stage 1: baseline = CURRENT config verbatim ----
        # methodology: the model as-saved (safe-defaults only fill gaps)
        # is the number to beat AND the fallback that can never be lost:
        # best starts here, so the winner can only improve on it.
        ctx_star = locked_ctx
        trial, label = _apply_variant(base, {})
        load_ok, b = _run_trial(model_id, trial, port, f"stage1 baseline {label}", path_map=pmap)
        if not load_ok or not b.get("ok"):
            finish("done", ok=False,
                   detail="baseline failed to load/bench; check log (likely OOM/split)")
            return
        consider(b, f"stage1 baseline {label}", trial)
        _log(model_id, f"stage1: baseline {b['tps']} tok/s @ "
                       f"ctx={locked_ctx} kv={locked_kv[0]}/{locked_kv[1]}")

        # ---- hardware-aware time prediction ----
        # probe peer links (cached 7d): per-load minutes ≈ weights crossing
        # the wire / slowest link. LIGHT ≈ 4 loads, THOROUGH ≈ +10.
        _by_name = {m["name"]: m for m in cfg.machines.all()}
        _links = {}
        for _nm in (model.get("servers") or []):
            _mm = _by_name.get(_nm)
            if _mm and not __import__("local_node").is_local(_mm):
                _links[_nm] = probe_link_mbps(_mm)
        _wires = [v for v in _links.values() if v]
        _min_per_load = 2.0
        if _wires:
            _slow = min(_wires)
            # ~half the weights cross the wire on a fresh load
            _min_per_load = max(
                2.0, est["weights_gb"] * 1024 * 8 * 0.5 / max(1, _slow) / 60)
        _log(model_id, f"links: {_links or 'local-only'} pcie: "
                       f"{[(g['label'], g.get('pcie') or '?') for g in est['gpus']]}")
        _log(model_id, f"est ~{_min_per_load:.0f} min/load → "
                       f"LIGHT ~{4*_min_per_load:.0f} min, "
                       f"THOROUGH ~+{10*_min_per_load:.0f} min")

        # ---- split-spec builder (shared by LIGHT plain + THOROUGH) ----
        import rpc_manager as _rm2
        _master_name = _rm2._split_servers(base.get("servers") or [])[0]["name"]
        _gpus = est["gpus"]
        _mfirst = [g for g in _gpus if g["label"].startswith(_master_name + "-")]
        _rest = [g for g in _gpus if g not in _mfirst]
        _ordered = _mfirst + _rest

        def _split_weights(mult_master=1.0, min_peer=1):
            # capacity-first (VRAM must physically hold the layers);
            # heavy/hostfirst variants additionally bias by bandwidth share
            # so fast hops hold more layers and the straggler holds fewer
            bws = [g.get("bw_gbs") or 0 for g in _ordered]
            use_bw = mult_master != 1.0 and all(bws)
            if use_bw:
                tot = sum(bws)
                w = [(b / tot) * len(_ordered) * 3 for b in bws]
                w = [x * (mult_master if g in _mfirst else 1.0)
                     for x, g in zip(w, _ordered)]
            else:
                w = [g["weight"] * (mult_master if g in _mfirst else 1.0)
                     for g in _ordered]
            scale = 18 / sum(w) if sum(w) else 1
            out = []
            for x, g in zip(w, _ordered):
                lo = min_peer if g not in _mfirst else 1
                out.append(str(max(lo, round(x * scale))))
            return ",".join(out)

        _labels = ",".join(g["label"] for g in _ordered)
        _by_label = {g["label"]: g for g in _ordered}
        # fastest-last: peers in RPC order, then master GPUs slowest
        # first (fastest silicon last = output head + draft tail).
        _peer_labels = [g["label"] for g in _ordered
                        if not g["label"].startswith(_master_name + "-")]
        _master_by_bw = sorted(
            [g for g in _ordered
             if g["label"].startswith(_master_name + "-")],
            key=lambda g: g.get("bw_gbs") or 0)
        _flast_labels = ",".join(
            _peer_labels + [g["label"] for g in _master_by_bw])
        # peer-swap: reversed peer order renumbers the RPC devices
        # (servers list drives --rpc order, not just chip order).
        _peers = [p["name"] for p in _rm2._split_servers(
            base.get("servers") or [])[1]]
        _swap_servers = ([_master_name]
                         + list(reversed(_peers)))
        _swap_labels = ",".join(
            [g["label"] for g in _master_by_bw or _mfirst]
            + [l for _pn in reversed(_peers) for l in _peer_labels
               if l.startswith(_pn + "-")])
        _split_specs = {
            # current: the model's live placement verbatim — the floor no
            # candidate may fall below (already measured as baseline, so it
            # dedupes free; listed so reports name it explicitly)
            "current": {"mode": base.get("tensor_split_mode", "fit"),
                        "device_order": base.get("device_order", ""),
                        "policy": base.get("device_order_policy",
                                           "as-selected"),
                        "weights": base.get("tensor_split", "") or ""},
            # plain: llama.cpp defaults (ngl, explicit order, no fit) —
            # the shape of the 24 tok/s last-good launch
            "plain": {"mode": "auto",
                      "device_order": base.get("device_order", ""),
                      "policy": base.get("device_order_policy",
                                         "as-selected"),
                      "weights": ""},
            "fit": {"mode": "fit", "device_order": "",
                    "policy": "vram-first", "weights": ""},
            "math": {"mode": "split", "device_order": _labels,
                     "policy": "as-selected",
                     "weights": _split_weights(1.0)},
            "heavy": {"mode": "split", "device_order": _labels,
                      "policy": "as-selected",
                      "weights": _split_weights(2.0)},
            # host-first: master ~4x, peers starved to 1 share each —
            # fastest silicon does ~80% of layers, weak hops barely used
            "hostfirst": {"mode": "split", "device_order": _labels,
                          "policy": "as-selected",
                          "weights": _split_weights(4.0)},
            # ngl-order set (explicit -ngl, explicit order, no fit):
            # master-first, fastest-last (output head on fastest),
            # peer-swapped (renumbered RPC devices).
            "mfirst": {"mode": "auto", "device_order": _labels,
                       "policy": "as-selected", "weights": ""},
            "flast": {"mode": "auto", "device_order": _flast_labels,
                      "policy": "as-selected", "weights": ""},
            "peerswap": {"mode": "auto",
                         "device_order": _swap_labels,
                         "policy": "as-selected", "weights": "",
                         "servers": _swap_servers},
        }
        kv_star = (locked_kv[0], locked_kv[1])  # refreshed after 2b
        _log(model_id, f"split candidates: math={_split_specs['math']['weights']} "
                       f"heavy={_split_specs['heavy']['weights']} "
                       f"hostfirst={_split_specs['hostfirst']['weights']}")

        def _stop():
            if _cancelled(model_id):
                finish("cancelled", ok=False,
                       detail="cancelled by user; nothing saved")
                _log(model_id, "cancelled — nothing saved")
                return True
            return False

        # ---- LIGHT PASS (dynamic: from the model's live config) ----
        # mtp-2 → plain order → ub neighbor. Variants equal to the current
        # config dedupe automatically (already measured as baseline).
        # If LIGHT hits target SAFE: save, done. Else THOROUGH digs deeper.
        _cur_ub = base.get("flags", {}).get("ubatch-size", 256)
        _ubs = _ub_neighbors(_cur_ub)
        _light_ub = _ubs[-1] if len(_ubs) > 1 else _ubs[0]
        if has_mtp:
            _probe("light", {"mtp": 2}, key="mtp=2")
        else:
            _log(model_id, "light: no MTP tensors in file — skipped")
        _probe("light", {"mtp": _best_mtp(), "split": _split_specs["plain"],
                         "kv": kv_star}, key="split=plain")
        _probe("light", {"mtp": _best_mtp(), "split": _best_split(),
                         "kv": kv_star, "ubatch": _light_ub},
               key=f"ub={_light_ub}")
        _log(model_id, f"light pass done: best {best[0]} tok/s "
                       f"({best[1]}) vs target {target_tps}")
        if best_safe is not None and best_safe[0] >= target_tps:
            _log(model_id, "light pass hit target SAFE — saving, done")
            _save_winner()
            return

        _log(model_id, "light fell short — THOROUGH pass")

        # ---- THOROUGH: MTP numerics around the LIVE n-max ----
        # dynamic offsets (not a fixed set): current ±1 clamped to 1..4,
        # plus no-draft-ngl at current n and the MTP+ngram combo.
        # Duplicates of LIGHT/baseline dedupe automatically.
        _cur_n = 2
        try:
            _cur_n = int(base.get("flags", {}).get("spec-draft-n-max", 2))
        except (TypeError, ValueError):
            _cur_n = 2
        _n_opts = []
        if has_mtp:
            for _n in (_cur_n - 1, _cur_n, _cur_n + 1):
                if 1 <= _n <= 4:
                    _n_opts.append(_n)
            _n_opts.append(f"nogl-{min(4, max(1, _cur_n))}")
            _n_opts.append("combo")
        for opt in _order("mtp", _n_opts, keyfn=lambda o: f"mtp={o}"):
            _probe("stage2", {"mtp": opt}, key=f"mtp={opt}")
        mtp_star = _best_mtp()
        _log(model_id, f"stage2: mtp={mtp_star} fastest so far ({best[0]} tok/s)")

        # ---- stage 2c: draft-device pin (only if MTP won) ----
        # --spec-draft-device <fastest master GPU>: keeps the per-ubatch
        # catch-up decode off the slow hops. Carries current best KV.
        _bf = _best_flags()
        kv_star = (_bf.get("cache-type-k", "q8_0"),
                   _bf.get("cache-type-v", "q8_0"))
        if str(_bf.get("spec-type", "")).lower() != "off" \
                and "draft-mtp" in str(_bf.get("spec-type", "")) \
                and supports_flag("--spec-draft-device"):
            import rpc_manager as _rm3
            _m3 = _rm3._split_servers(base.get("servers") or [])[0]
            _mgpus = [d.strip() for d in (_m3.get("gpu", "") or "").split(",")
                      if d.strip()]
            if _mgpus:
                _probe("stage2c", {"mtp": mtp_star, "kv": kv_star,
                                   "draft_device": _mgpus[0]},
                       key=f"draft-device={_mgpus[0]}")
                _log(model_id, f"stage2c: draft-device fastest so far "
                               f"({best[0]} tok/s)")

        # ---- stage 2b: KV-cache pairs (uniform + mixed) ----
        # KV precision moves BOTH memory and speed (and MTP acceptance).
        # Uniform (q8/f16/q4) plus mixed k/v probes: mismatched pairs can
        # hit a kernel-fallback pit (13x prefill collapse seen elsewhere),
        # so a collapse here is a measurement, not a bug. Each candidate
        # math-gated (skip what can't fit); identical-to-current deduped
        # by _probe automatically.
        if True:
            try:
                _tgt = master.get("os", "linux")
                _res = path_resolver.resolve_for(
                    base.get("base_path") or base.get("path", ""), _tgt)
                _meta = gguf_inspect.metadata(_res)
                _fixed = ((est["weights_gb"] + est["mmproj_gb"]
                           + est["ssm_gb"]) * 1024**3
                          + est["weights_gb"] * 1024**3 * 0.06)
                _usable = est["usable_vram_gb"] * 1024**3
                _ck = str((_best_flags().get("cache-type-k")
                           or "q8_0")).lower()
                _cv = str((_best_flags().get("cache-type-v")
                           or "q8_0")).lower()
                _pairs = [(_ck, _cv), ("q8_0", "q8_0"), ("f16", "f16"),
                          ("q4_0", "q4_0"), ("q8_0", "f16"),
                          ("q4_0", "f16")]
                for _kp, _vp in _order(
                        "kv", list(dict.fromkeys(_pairs)),
                        keyfn=lambda p: f"kv={p[0]}/{p[1]}"):
                    _per = _kv_pair_per_token(_res, _meta, _kp, _vp)
                    _need = _fixed + _per * int(ctx_star or 0)
                    if _need > _usable:
                        _log(model_id, f"stage2b: kv={_kp}/{_vp} needs "
                                       f"{_need/1024**3:.1f}GB > usable "
                                       f"{est['usable_vram_gb']}GB — skipped")
                        continue
                    _probe("stage2b", {"mtp": mtp_star,
                                       "kv": (_kp, _vp)},
                           key=f"kv={_kp}/{_vp}")
            except Exception as e:
                _log(model_id, f"stage2b: skipped ({e})")

        # ---- stage 3: forced tensor-split (THOROUGH; plain already ran
        # in LIGHT) ----
        # fit vs math integers vs master-heavy vs host-first.
        kv_star = (_best_flags().get("cache-type-k", "q8_0"),
                   _best_flags().get("cache-type-v", "q8_0"))
        _fit_specs = {k: v for k, v in _split_specs.items() if k != "plain"}
        for _sname, spec in _order(
                "split", list(_fit_specs.items()),
                keyfn=lambda item: f"split={item[0]}"):
            _probe("stage3", {"mtp": mtp_star, "split": spec, "kv": kv_star},
                   key=f"split={_sname}")
        split_star = _best_split()
        _log(model_id, f"stage3: split="
                       f"{split_star['weights'] or split_star['mode']} "
                       f"fastest so far ({best[0]} tok/s)")

        # ---- stage 3b: fit-margin neighbors (fit mode only) ----
        # margin only steers the fitter; under plain/split modes it is
        # inert, so the stage self-skips there. Dynamic around live value.
        if split_star.get("mode") == "fit":
            _cur_m = base.get("fit_margin", 1024)
            try:
                _cur_m = int(_cur_m)
            except (TypeError, ValueError):
                _cur_m = 1024
            for _mg in _order("margin", _ladder_neighbors(_cur_m, MARGIN_LADDER),
                               keyfn=lambda m: f"margin={m}"):
                _probe("stage3b", {"mtp": mtp_star, "split": split_star,
                                   "kv": kv_star, "margin": _mg},
                       key=f"margin={_mg}")
        else:
            _log(model_id, "stage3b: margin N/A outside fit mode — skipped")

        # ---- stage 3c: DEVICE-SET hill climb (with backoff) ----
        # The form's hardware selection is a starting point, not gospel:
        # solo master GPUs, master-only, master+peer subsets, and the full
        # set all get measured. Order (priors + bandwidth):
        #   1. solo rungs per master GPU (which chip alone is fastest?)
        #   2. master-only (all local GPUs, peers idle)
        #   3. every proper subset: master + peers, largest first
        #   4. full set (= the form's selection; dedupes vs baseline)
        # BACKOFF: subsets are tried biggest-first from the full set;
        # a dropping step (full -> full-minus-one) that doesn't beat the
        # incumbent stops the climb (removing a hop can only shed more
        # capacity once the straggler is gone — don't burn loads proving it
        # again per subset).
        _dev_stop = False
        if len(_mfirst) > 1 or _peers:
            # solo master GPUs: is one chip enough? (single-box configs
            # often beat split-over-RPC on speed)
            for g in _order("devset", _mfirst,
                            keyfn=lambda x: f"devset=solo-{x['label']}"):
                if _dev_stop:
                    break
                _tps_before = best[0] if best else 0
                _probe("stage3c", {"mtp": mtp_star, "split": split_star,
                                   "kv": kv_star,
                                   "split2": {"mode": "fit",
                                              "device_order": g["label"],
                                              "policy": "vram-first",
                                              "weights": "",
                                              "servers": [_master_name]}},
                       key=f"devset=solo-{g['label']}")
                if best is not None and best[0] <= _tps_before:
                    _dev_stop = True
                    _log(model_id, "stage3c: solo rung did not improve — "
                                   "backoff, skipping remaining solos")
            if not _dev_stop and len(_mfirst) > 1:
                _tps_before = best[0]
                _mo_labels = ",".join(g["label"] for g in _mfirst)
                _probe("stage3c",
                       {"mtp": mtp_star, "split": split_star, "kv": kv_star,
                        "split2": {"mode": "fit", "device_order": _mo_labels,
                                   "policy": "vram-first", "weights": "",
                                   "servers": [_master_name]}},
                       key="devset=master-only")
                if best[0] <= _tps_before:
                    _dev_stop = True
                    _log(model_id, "stage3c: master-only rung did not "
                                   "improve — backoff, skipping subsets")
            # subset ladder: shrink the full set one peer at a time
            # (slowest removed first). BACKOFF: the first shrink that
            # doesn't beat the incumbent ends the walk — going smaller
            # only sheds more capacity after that.
            if _peers and not _dev_stop:
                for k in range(len(_peers), 0, -1):
                    if k == len(_peers):
                        continue  # full set = baseline, already measured
                    if _dev_stop:
                        break
                    _tps_before = best[0]
                    sub = _peers[:k]
                    sub_labels = ",".join(
                        [l for _pn in sub for l in _peer_labels
                         if l.startswith(_pn + "-")])
                    _probe("stage3c",
                           {"mtp": mtp_star, "split": split_star,
                            "kv": kv_star,
                            "split2": {"mode": split_star.get("mode", "fit"),
                                       "device_order": sub_labels,
                                       "policy": "vram-first",
                                       "weights": "",
                                       "servers": [_master_name] + sub}},
                           key=f"devset=m+{k}peers")
                    if best[0] <= _tps_before:
                        _dev_stop = True
                        _log(model_id, f"stage3c: m+{k}peers did not "
                                       f"improve — backoff, device-set "
                                       f"climb done")
        else:
            _log(model_id, "stage3c: single GPU local — device-set N/A")

        # ---- stage 3d: CONTEXT-LENGTH balance walk (with backoff) ----
        # ctx was LOCKED at baseline (the user's number); once speed is
        # known we test whether MORE ctx is affordable at ~the same speed
        # (quality up) or whether LESS ctx buys a real speed win (only a
        # strict improvement moves down — a faster-but-tinier config is a
        # regression unless it clearly beats the incumbent).
        # BACKOFF: one non-improving step per direction ends that direction.
        if locked_ctx:
            _ctx_up_stop = _ctx_dn_stop = False
            _up = _dn = locked_ctx
            for _i in range(2):  # max 2 walk steps per direction
                if not _ctx_up_stop:
                    _nxt = next((c for c in CTX_LADDER if c > _up
                                 and c <= est["model_max_ctx"]), None)
                    if _nxt is None:
                        _ctx_up_stop = True
                    else:
                        _before = best[0] if best else 0
                        _probe("stage3d", {"ctx": _nxt},
                               key=f"ctx={_nxt}")
                        if best[0] <= _before * 0.97:
                            # bigger ctx cost >3% speed: stop walking up
                            # (monotone-ish: next rung only costs more)
                            _ctx_up_stop = True
                        else:
                            _up = _nxt
                if not _ctx_dn_stop:
                    _nxt = next((c for c in reversed(CTX_LADDER)
                                 if c < _dn), None)
                    if _nxt is None:
                        _ctx_dn_stop = True
                    else:
                        _before = best[0]
                        _probe("stage3d", {"ctx": _nxt}, key=f"ctx={_nxt}")
                        if best[0] <= _before:
                            # down-step didn't STRICTLY improve: quality
                            # regression for nothing — stop and revert
                            _ctx_dn_stop = True
                        else:
                            _dn = _nxt
                if _ctx_up_stop and _ctx_dn_stop:
                    break
            _log(model_id, f"stage3d: ctx walk done "
                           f"(up_stop={_ctx_up_stop} dn_stop={_ctx_dn_stop})")

        # ---- stage 4: ubatch neighbors of the LIVE value ----
        # dynamic, not a fixed grid: current ± nearest ladder rungs.
        for ub in _order("ub", _ub_neighbors(
                base.get("flags", {}).get("ubatch-size", 256)),
                keyfn=lambda u: f"ub={u}"):
            _probe("stage4", {"mtp": mtp_star, "split": split_star,
                              "kv": kv_star, "ubatch": ub},
                   key=f"ub={ub}")
        ub_star = int(_best_flags().get("ubatch-size", 256) or 256)
        _log(model_id, f"stage4: ubatch={ub_star} fastest so far ({best[0]} tok/s)")

        # ---- stage 4b: batch neighbors of the LIVE value ----
        # dynamic offsets like ubatch; constrained to batch >= ub_star
        # (llama.cpp requires b>=ub — smaller candidates drop out here).
        for _b in _order("batch", [b for b in _ladder_neighbors(
                base.get("flags", {}).get("batch-size", 2048), B_LADDER)
                if b >= ub_star], keyfn=lambda b: f"b={b}"):
            _probe("stage4b", {"mtp": mtp_star, "split": split_star,
                               "kv": kv_star, "ubatch": ub_star,
                               "batch": _b},
                   key=f"b={_b}")

        # ---- stage 5: SWA rung for sliding-window archs (any family
        # declaring it in metadata). Flash-attn stays ON and threads stay
        # as configured (experience: both all downside to touch).
        try:
            _moe = has_moe_tensors(path_resolver.resolve_for(
                base.get("base_path") or base.get("path", ""),
                master.get("os", "linux")))
        except Exception:
            _moe = False
        _log(model_id, f"moe expert tensors: {'yes' if _moe else 'no'} "
                       f"{'(expert-placement N/A — dense FFN)' if not _moe else ''}")
        # ---- stage 5b: MoE expert placement (layer-split alternative) ----
        # MoE models with RAM in the device list: keeping expert weights
        # on CPU (--n-cpu-moe / --cpu-moe) usually beats layer-split over
        # RPC — only the expert hop crosses the wire, and VRAM stays
        # fully busy. Ladder: all-on-CPU, then the first-half fractions.
        if _moe and best[0] < target_tps:
            if supports_flag("--n-cpu-moe"):
                for _nc in _order("moecpu", [24, 32, 12],
                                  keyfn=lambda n: f"moecpu={n}"):
                    _probe("stage5b", {"mtp": mtp_star, "split": split_star,
                                       "kv": kv_star, "ubatch": ub_star,
                                       "moe_cpu": _nc},
                           key=f"moecpu={_nc}")
                    if _stop():
                        break
                if best[0] < target_tps and supports_flag("--cpu-moe"):
                    _probe("stage5b", {"mtp": mtp_star, "split": split_star,
                                       "kv": kv_star, "ubatch": ub_star,
                                       "moe_cpu": "all"},
                           key="moecpu=all")
            else:
                _log(model_id, "stage5b: build lacks --n-cpu-moe — skipped")
        if best[0] < target_tps:
            # sliding-window archs (any family declaring it in metadata):
            # full-size SWA cache rung
            try:
                _meta = (gguf_inspect.metadata(
                    path_resolver.resolve_for(
                        base.get("base_path") or base.get("path", ""),
                        master.get("os", "linux"))
                ) or {})
            except Exception:
                _meta = {}
            if has_swa(_meta):
                _probe("stage5", {"mtp": mtp_star, "split": split_star,
                                  "kv": kv_star, "ubatch": ub_star,
                                  "swafull": True},
                       key="swa=on")

        # ---- save best (falls back to the live config: Tune can only
        # improve on what you had, never lose it) ----
        _save_winner()
    except _ExitWorker:
        return
    except Exception as e:
        finish("failed", ok=False, detail=str(e)[:500])
        _log(model_id, f"ERROR {e}")
