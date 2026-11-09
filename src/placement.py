"""Recommend a baseline hardware placement + flags for a model.

Math-only (no loads): reads the GGUF header (real weight bytes + KV per
token) and the farm's last-detected GPU/RAM inventory, then picks the
smallest sensible topology — one box first (no network hops), then a
multi-box RPC spread, host RAM spill last. Returns exactly the shapes the
model editor saves (servers, device_order, tensor_split_mode, flags), so
a single UI click can fill the form.

This is a *baseline*, not a tune: it mirrors tuner.safe_defaults (fit +
vram-first + q8_0 KV + flash-attn + jinja) because those are the
known-good fleet defaults, and Tune still refines batches/KV/split live.

Per-GPU VRAM is derived from each machine's detected totals (sum / GPU
count), the same approximation tuner.estimate uses, because the persisted
`detected` block has no per-card figure. A live Tune is the precise path.
"""
import os

import config_loader as cfg
import gguf_inspect
import gpu_vendor
import path_resolver
import stats_collector
import tuner

BASELINE_CTX = 32768      # same starting ctx as tuner.safe_defaults
GPU_RESERVE_GB = 0.5      # driver/context overhead per card
RAM_USABLE_FRAC = 0.8     # never plan against all system RAM
HEADROOM = tuner.HEADROOM_FRac

# Verbose logging is great for the visualizer but floods thousands of
# sched lines at 27B scale (see safe_defaults history), so a *baseline*
# recommendation ships without it. Tune/visualizer can add it per-model.
_VERBOSE_FLAGS = ("-v", "log-timestamps", "log-prefix")


def _model_max_ctx(meta):
    model_max = 262144
    for k, v in meta.items():
        if str(k).lower().endswith(("context_length", "n_ctx_train")) \
                and isinstance(v, int) and v > 0:
            model_max = min(model_max, v)
    return model_max


def _local_resolved(base_path, target_os):
    """Path the *app* can read: try the consuming OS first (matches what
    the generated command will use), then fall back to this host's
    resolve so a Windows peer target still inspects on a Linux master."""
    resolved = path_resolver.resolve_for(base_path, target_os)
    if not os.path.exists(resolved):
        local = path_resolver.resolve(base_path)
        if os.path.exists(local):
            return local
    return resolved


def _model_needs(base_path, target_os, target_ctx=None,
                 cache_type="q8_0", mmproj_raw=""):
    """Weights + KV at ctx + SSM + draft overhead, in bytes.

    Raises ValueError with a readable message when the GGUF can't be read
    (the caller turns that into a 400).
    """
    if not base_path:
        raise ValueError("no model path — pick a model file first")
    resolved = _local_resolved(base_path, target_os)
    if not os.path.exists(resolved):
        raise ValueError(f"model file not found: {resolved}")
    try:
        meta = gguf_inspect.metadata(resolved)
    except Exception as e:
        raise ValueError(f"could not read GGUF header: {e}")
    model_max = _model_max_ctx(meta)
    ctx = int(target_ctx) if target_ctx else min(BASELINE_CTX, model_max)
    ctx = max(1024, min(ctx, model_max))
    weights = tuner._model_bytes(resolved)
    try:
        kv_tok = tuner._kv_per_token(resolved, meta, cache_type)
    except Exception:
        kv_tok = 0
    try:
        ssm = gguf_inspect.ssm_state_bytes(resolved, meta)
    except Exception:
        ssm = 0
    mmproj = 0
    if mmproj_raw:
        try:
            mmproj = tuner._model_bytes(
                _local_resolved(mmproj_raw, target_os))
        except Exception:
            mmproj = 0
    try:
        has_mtp = gguf_inspect.has_mtp_layers_anywhere(resolved)
    except Exception:
        has_mtp = False
    draft = int(weights * 0.06)  # MTP draft heads + buffers heuristic
    need = weights + draft + ssm + mmproj + int(kv_tok * ctx)
    return {
        "resolved": resolved,
        "meta": meta,
        "arch": str(meta.get("general.architecture") or "unknown"),
        "model_max_ctx": model_max,
        "ctx": ctx,
        "has_mtp": has_mtp,
        "weights_gb": round(weights / 1024**3, 2),
        "kv_gb": round(kv_tok * ctx / 1024**3, 2),
        "ssm_gb": round(ssm / 1024**3, 2),
        "mmproj_gb": round(mmproj / 1024**3, 2),
        "draft_gb": round(draft / 1024**3, 2),
        "need_gb": round(need / 1024**3, 2),
        "need_bytes": need,
    }


def _token_index(tok):
    digits = "".join(ch for ch in (tok or "") if ch.isdigit())
    return int(digits) if digits else 0


def _machine_inventory(machine, live=None):
    """Per-card VRAM/RAM for one machine, with usable capacity.

    Prefers LIVE per-GPU totals (agent/local metrics) because persisted
    `detected.vram_total_gb` is a sum with no per-card split — on a mixed
    box (e.g. 2060S + P100 + 2060) dividing the total evenly picks the
    wrong card. Falls back to the detected total / count when the box is
    offline, mirroring tuner._hardware_vram. The fit budget is
    VRAM*HEADROOM minus a small per-card reserve (total, not free: a
    loaded model must not make us conclude nothing fits).
    """
    det = machine.get("detected") or {}
    names = [str(n) for n in (det.get("gpu_names") or [])]
    tokens = [t.strip() for t in (machine.get("gpu") or "").split(",")
              if t.strip()]
    if not tokens:
        tokens = [gpu_vendor.gpu_token(machine, i) for i in range(len(names))]
    live_gpus = {}
    if isinstance(live, dict):
        for g in (live.get("gpus") or []):
            try:
                live_gpus[int(g.get("index"))] = g
            except (TypeError, ValueError):
                continue
    n = len(tokens)
    total = float(det.get("vram_total_gb") or 0)
    per = total / n if n else 0.0
    gpus = []
    for i, tok in enumerate(tokens):
        idx = gpu_vendor.gpu_index(tok)
        live_g = live_gpus.get(idx)
        if live_g and float(live_g.get("mem_total_gb") or 0) > 0:
            vram = float(live_g["mem_total_gb"])
            gname = str(live_g.get("name") or (names[i] if i < len(names) else tok))
        else:
            vram = per
            gname = names[i] if i < len(names) else tok
        usable = max(0.0, vram * HEADROOM - GPU_RESERVE_GB) if vram else 0.0
        gpus.append({
            "token": tok,
            "name": gname,
            "vram_gb": round(vram, 1),
            "usable_gb": round(usable, 1),
            "bw_gbs": tuner._bandwidth_gbs(gname),
            "weight": round(max(0.5, vram * tuner._derate(gname)), 2),
        })
    ram_total = float((live or {}).get("ram_total_gb") or 0) if isinstance(
        live, dict) else 0
    if ram_total <= 0:
        ram_total = float(det.get("ram_total_gb") or 0)
    return {
        "name": machine.get("name", ""),
        "role": machine.get("role", "peer"),
        "os": machine.get("os", "linux"),
        "gpus": gpus,
        "ram_total_gb": round(ram_total, 1),
        "ram_usable_gb": round(ram_total * RAM_USABLE_FRAC, 1),
    }


def _live_stats(machines):
    """Per-machine live stats, in parallel, best-effort.

    Offline boxes are skipped (fallback inventory) so one dead host never
    blocks the recommendation. Machines with no address are skipped
    outright (unit tests / not-yet-configured entries).
    """
    targets = [m for m in machines if m.get("ip")]
    if not targets:
        return {}
    out = {}
    try:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=min(len(targets), 8)) as pool:
            results = list(pool.map(stats_collector.machine_stats, targets))
        for m, s in zip(targets, results):
            if isinstance(s, dict) and s.get("status") == "online":
                out[m.get("name")] = s
    except Exception:
        pass
    return out


def _inventory(machines, live_by_name=None):
    live_by_name = live_by_name or {}
    return [m for m in (
        _machine_inventory(x, live_by_name.get(x.get("name"))) for x in machines)
        if m["name"]]


def _master_name():
    for m in cfg.machines.all():
        if m.get("role") == "master":
            return m.get("name", "")
    return ""


def _split_weights(gpus):
    """Integer split proportional to derated VRAM (tuner's 18-unit scale
    convention), offered as a reference when the user wants forced split.
    """
    weights = [max(1, int(round(g["weight"]))) for g in gpus]
    total = sum(weights) or 1
    scale = 18 / total
    return [max(1, int(round(w * scale))) for w in weights]


def _best_single(inv, need_gb, master):
    """Fastest single box whose cards alone fit need_gb.

    Returns (score, machine, chosen_gpus, capacity) or None. Cards are
    ordered bandwidth-first so a slow straggler is only added when the
    capacity is actually needed. Ties prefer the master (no proxy hop).
    """
    best = None
    for m in inv:
        if not m["gpus"]:
            continue
        ordered = sorted(
            m["gpus"], key=lambda g: (-(g["bw_gbs"] or 0), -g["usable_gb"]))
        pick, cap = [], 0.0
        for g in ordered:
            pick.append(g)
            cap += g["usable_gb"]
            if cap >= need_gb:
                break
        if cap < need_gb:
            continue
        bw = sum(g["bw_gbs"] or 0 for g in pick) or sum(g["weight"] for g in pick)
        # throughput first; master breaks near-ties; tighter fit after that
        key = (round(bw, 1), 1 if m["name"] == master else 0, -(cap - need_gb))
        if best is None or key > best[0]:
            best = (key, m, pick, cap)
    return best


def _expose_for(used, chosen, ram_machines, host_name):
    expose = {}
    for m in used:
        gpus = [g["token"] for g in chosen.get(m["name"], [])]
        want_ram = m["name"] in ram_machines and m["name"] != host_name
        if gpus and want_ram:
            expose[m["name"]] = {"mode": "both", "gpus": gpus}
        elif gpus:
            expose[m["name"]] = {"mode": "vram", "gpus": gpus}
        elif want_ram:
            expose[m["name"]] = {"mode": "ram", "gpus": []}
    return expose


def recommend(base_path, servers=None, target_ctx=None, allow_ram=True,
              cache_type="q8_0", mmproj_raw=""):
    """Pick a baseline placement for one model.

    servers: optional allow-list of machine names (e.g. a group's members).
             Empty/None = the whole farm.
    """
    machines = cfg.machines.all()
    if servers:
        wanted = {s for s in servers if s}
        filtered = [m for m in machines if m.get("name") in wanted]
        if filtered:
            machines = filtered
    if not machines:
        raise ValueError("no machines configured — add hardware first")

    master = _master_name()
    by_name = {m.get("name"): m for m in machines}
    # Prefer the real master's OS for path resolution; else the first box.
    host_guess = by_name.get(master) or machines[0]
    target_os = host_guess.get("os", "linux")
    needs = _model_needs(base_path, target_os, target_ctx,
                         cache_type, mmproj_raw)
    need_gb = needs["need_gb"]

    inv = _inventory(machines, _live_stats(machines))
    # Candidate GPUs, fastest first (decode is bandwidth-bound). Unknown
    # bandwidth (0) sorts last but stays available as capacity.
    candidates = []
    for m in inv:
        for g in m["gpus"]:
            candidates.append((m, g))
    candidates.sort(key=lambda mg: (
        -(mg[1]["bw_gbs"] or 0),
        -mg[1]["usable_gb"],
        0 if mg[0]["name"] == master else 1,
        _token_index(mg[1]["token"]),
    ))

    reasons = []
    warnings = []
    used, ram_machines = [], set()
    chosen = {}  # machine name -> [gpu] actually selected

    # 1) Prefer ONE box when it fits on its own: no network hops and the
    #    host's RAM is natively addressable. Pick the fastest sufficient
    #    box (a small subset of its cards, not necessarily all of them).
    best_single = _best_single(inv, need_gb, master)
    cap = 0.0
    if best_single is not None:
        _key, m, pick, cap = best_single
        used = [m]
        chosen[m["name"]] = pick

    # 2) Otherwise spread across boxes: add the fastest cards until the
    #    model fits. Cards we don't need are left idle (a slow straggler
    #    would gate every token).
    if not used:
        for m, g in candidates:
            if cap >= need_gb:
                break
            if m not in used:
                used.append(m)
            chosen.setdefault(m["name"], []).append(g)
            cap += g["usable_gb"]

    # 3) RAM spill (last resort): only the host's RAM is implicit to
    #    llama-server; a peer's RAM needs an explicit <peer>-RAM chip
    #    (RPC CPU backend). Count host RAM before peers so we prefer the
    #    local box's memory.
    host = next((m for m in used if m["name"] == master), used[0] if used else None)
    if cap < need_gb and allow_ram and host is not None:
        add = min(host["ram_usable_gb"], need_gb - cap)
        if add > 0:
            cap += add
            reasons.append(
                f"host RAM spill: +{add:.1f} GB from {host['name']} "
                f"(implicit to llama-server)")
        for m in used:
            if cap >= need_gb:
                break
            if m["name"] == host["name"]:
                continue
            add = min(m["ram_usable_gb"], need_gb - cap)
            if add > 0:
                cap += add
                ram_machines.add(m["name"])
                reasons.append(
                    f"peer RAM spill: +{add:.1f} GB from {m['name']} "
                    f"(RPC CPU backend)")

    # 3) Nothing with GPUs at all: degrade to a CPU/RAM box rather than
    #    returning an empty placement.
    cpu_only = not used and bool(inv)
    if cpu_only:
        cpu = max(inv, key=lambda m: m["ram_usable_gb"])
        used = [cpu]
        ram_machines.add(cpu["name"])
        cap = cpu["ram_usable_gb"]
        warnings.append(
            "no usable GPUs detected — recommending a CPU/RAM-only box; "
            "expect slow chat but fine embeddings/ASR")

    used_names = [m["name"] for m in used]
    if master in used_names:  # llama-server (RPC host) on the master
        used_names = [master] + [n for n in used_names if n != master]
    used = [next(m for m in inv if m["name"] == n) for n in used_names]

    # A Windows box can only be an RPC peer (it cannot read store:// paths
    # as a remote-launch host), so when it is the only accelerator source
    # keep/add the master as the llama-server host so the RPC chain works.
    if any(m["os"] == "windows" for m in used):
        master_obj = next((m for m in inv if m["name"] == master), None)
        if master_obj is not None and master_obj not in used:
            used.insert(0, master_obj)
            used_names = [master] + [n for n in used_names if n != master]
        warnings.append(
            "a Windows box is included: it can only run ggml-rpc-server "
            "(RPC peer), not host llama-server over SMB model paths")
    host_name = used_names[0] if used_names else ""

    # Emit chips only for the GPUs we counted; a peer RAM chip is explicit
    # (RPC CPU backend). Host RAM stays implicit, except on a CPU-only
    # recommendation where the RAM chip is what shows the choice in the UI.
    device_chips = []
    for m in used:
        for g in sorted(chosen.get(m["name"], []),
                        key=lambda g: _token_index(g["token"])):
            device_chips.append(f"{m['name']}-{g['token']}")
        if m["name"] in ram_machines and m["name"] != host_name:
            device_chips.append(f"{m['name']}-RAM")
    if cpu_only and host_name:
        device_chips = [f"{host_name}-RAM"]

    fits = cap >= need_gb
    single_box = len(used_names) == 1
    if not fits:
        warnings.append(
            f"model needs ~{need_gb:.1f} GB but the farm can provide "
            f"~{cap:.1f} GB with this placement — it may fail to load")
    elif single_box:
        reasons.append(
            f"single box: {host_name} — {len(chosen.get(host_name, []))} "
            f"GPU(s), ~{cap:.1f} GB usable, no network hops")
    else:
        reasons.append(
            f"multi-box RPC: {', '.join(used_names)} — {len(device_chips)} "
            f"device(s), ~{cap:.1f} GB usable (pooled over the LAN)")

    # 4) Baseline flags: reuse the fleet's known-good safe defaults, then
    #    reconcile ctx / MTP / verbose / offload with what we selected.
    try:
        d = tuner.safe_defaults(
            base_path, used_names,
            ",".join(device_chips), target_os, existing_flags={})
        flags = dict(d.get("flags") or {})
    except Exception:
        flags = {}
    flags["ctx-size"] = needs["ctx"]
    for k in _VERBOSE_FLAGS:
        flags.pop(k, None)
    if needs["has_mtp"] and not flags.get("spec-type"):
        flags.update({"spec-type": "draft-mtp", "spec-draft-n-max": 2,
                      "spec-draft-ngl": 99})
    all_gpus = [g for m in used for g in chosen.get(m["name"], [])]
    if all_gpus:
        flags["n-gpu-layers"] = 99
    else:
        flags["n-gpu-layers"] = 0
        cores = int((by_name.get(host_name, {}) or {}).get("detected", {})
                    .get("cores") or 8)
        flags["threads"] = max(4, min(16, cores))

    suggested_split = _split_weights(all_gpus) if len(all_gpus) > 1 else []

    reasons.append(
        f"needs ~{need_gb:.1f} GB at ctx {needs['ctx']} "
        f"(weights {needs['weights_gb']} + KV {needs['kv_gb']}"
        + (f" + mmproj {needs['mmproj_gb']}" if needs["mmproj_gb"] else "")
        + (f" + SSM {needs['ssm_gb']}" if needs["ssm_gb"] else "")
        + f" + draft {needs['draft_gb']})")
    reasons.append(
        "per-GPU VRAM uses live metrics when a box is reachable, detected "
        "totals otherwise; Tune confirms the exact fit")

    expose = _expose_for(used, chosen, ram_machines, host_name)
    if cpu_only and host_name:
        expose[host_name] = {"mode": "ram", "gpus": []}

    return {
        "ok": True,
        "base_path": base_path,
        "arch": needs["arch"],
        "ctx": needs["ctx"],
        "model_max_ctx": needs["model_max_ctx"],
        "servers": used_names,
        "host": host_name,
        "single_box": single_box,
        "device_order": ",".join(device_chips),
        "device_order_policy": "vram-first",
        "tensor_split_mode": "fit",
        "tensor_split": "",
        "suggested_split": suggested_split,
        "fit_margin": 1024,
        "fit_ram_allow": 1024,
        "flags": flags,
        "expose": expose,
        "needs_gb": need_gb,
        "usable_gb": round(cap, 1),
        "weights_gb": needs["weights_gb"],
        "kv_gb": needs["kv_gb"],
        "fits": fits,
        "has_mtp": needs["has_mtp"],
        "reasons": reasons,
        "warnings": warnings,
    }
