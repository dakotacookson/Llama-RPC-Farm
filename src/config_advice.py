"""Non-blocking configuration advice for models, serve groups and machine groups.

These are warnings, not hard gates: the config still saves. They exist because
the farm is deliberately flexible — single-box, RPC, peer-local, custom and
MLX runtimes plus machine/model grouping — and some combinations are allowed
but will not do what a user expects. Keep messages short and actionable.
"""


def _chip(token):
    """'aibox3-CUDA0' -> ('aibox3', 'CUDA0'); 'aibox1-RAM' -> ('aibox1','RAM')."""
    token = str(token or "").strip()
    if "-" not in token:
        return "", token
    return token.rsplit("-", 1)


def _models_by_id():
    try:
        import config_loader as cfg
        out = {}
        for m in cfg.list_models():
            mid = m.get("model_id")
            if not mid:
                continue
            out[mid] = m
            # serve groups reference deployment labels: base--<label>
            for v in (m.get("variants") or []):
                label = (v or {}).get("label")
                if label:
                    out[f"{mid}--{label}"] = m
        return out
    except Exception:
        return {}


def _machines_by_name():
    try:
        import config_loader as cfg
        return {m.get("name"): m for m in cfg.machines.all()}
    except Exception:
        return {}


def model_warnings(model, machines_by_name=None):
    """Advice for one model config. Returns [str] (warnings only)."""
    model = model or {}
    runtime = str(model.get("runtime") or "llama").lower()
    servers = [str(s).strip() for s in (model.get("servers") or []) if str(s).strip()]
    order = str(model.get("device_order") or "").strip()
    launch = str(model.get("launch_mode") or "auto").lower()
    out = []

    def w(msg):
        out.append(msg)

    if runtime == "mlx":
        if len(servers) > 1:
            w(f"MLX is single-Mac: extra servers ({', '.join(servers[1:])}) "
              f"are ignored.")
        if order:
            w("MLX ignores device_order — it has no --rpc/--device graph.")
        if (model.get("expose") or {}):
            w("MLX ignores expose/GPU sharing — it always runs on the Mac "
              "itself.")
        if launch not in ("", "auto", "local"):
            w(f"MLX always runs locally; launch_mode '{launch}' is ignored.")
        return out

    if runtime == "custom":
        if len(servers) != 1:
            w(f"Custom runtimes need exactly one target box in servers "
              f"(got {len(servers) or 0}). The farm starts/stops one "
              f"external OpenAI server and proxies it.")
        return out

    # llama.cpp deployments
    if not servers:
        w("No servers configured — the model has nowhere to run.")
        return out
    names = set(servers)
    if launch == "local":
        if len(servers) > 1:
            w("launch_mode 'local' ignores every server except the first "
              "(single-box, no RPC). Use 'rpc' or 'auto' to spread it.")
    elif launch == "rpc" and len(servers) == 1:
        w("launch_mode 'rpc' has no peers — it will run as a single-box "
          "model anyway.")
    if order:
        stray = []
        for tok in order.split(","):
            tok = tok.strip()
            if not tok:
                continue
            mname, _res = _chip(tok)
            if mname and mname not in names:
                stray.append(mname)
        if stray:
            w("device_order references machines not in servers: "
              + ", ".join(sorted(set(stray))) + ".")
    # Split/fit/flag/path checks live in rpc_manager.validate_model and are
    # surfaced on save too — don't duplicate them here with different
    # wording (it created noisy double warnings).
    flags = model.get("flags") or {}
    dev_tokens = {str(t).strip().rsplit("-", 1)[-1]
                  for t in order.split(",") if t.strip()}
    mmdev = str(flags.get("mmproj-device") or "").strip()
    if flags.get("mmproj") and mmdev and dev_tokens and mmdev not in dev_tokens:
        w(f"mmproj-device '{mmdev}' is not one of the selected device tokens "
          f"({', '.join(sorted(dev_tokens))}).")
    if launch in ("auto", "local") and machines_by_name:
        win = [s for s in servers
               if str((machines_by_name.get(s) or {}).get("os", "")).lower()
               == "windows"]
        if win and launch == "local":
            w(f"Windows peer(s) {', '.join(win)} cannot run a remote-local "
              f"llama-server reliably (SMB model paths). Use the RPC chain "
              f"or a model path local to that box.")
    return out


def serve_group_warnings(group, models_by_id=None):
    """Advice for a llama-swap serve group."""
    group = group or {}
    models = models_by_id if models_by_id is not None else _models_by_id()
    members = [str(m).strip() for m in (group.get("members") or [])
               if str(m).strip()]
    out = []

    def w(msg):
        out.append(msg)

    if not members:
        w("Serve group has no members.")
        return out
    missing = [m for m in members if models and m not in models]
    if missing:
        w("Unknown model(s) in this group: " + ", ".join(missing) + ".")
    if group.get("exclusive") and group.get("persistent"):
        w("exclusive + persistent conflict: exclusive evicts every other "
          "model, while persistent wants this set always loaded.")
    if group.get("swap") and group.get("persistent"):
        w("swap + persistent conflict: swap makes members take turns, "
          "persistent keeps them all warm.")
    if group.get("persistent") and len(members) > 1:
        w("Persistent groups keep every member loaded at once. If they do "
          "not all fit in VRAM together, loads will fail — verify with Tune "
          "or use a swap group instead.")
    return out


def machine_group_warnings(group, machines_by_name=None):
    """Advice for a machine (deployment affinity) group."""
    group = group or {}
    machines = machines_by_name if machines_by_name is not None \
        else _machines_by_name()
    members = [str(m).strip() for m in (group.get("machines") or [])
               if str(m).strip()]
    out = []

    def w(msg):
        out.append(msg)

    if not members:
        w("Machine group has no machines.")
        return out
    missing = [m for m in members if machines and m not in machines]
    if missing:
        w("Unknown machine(s) in this group: " + ", ".join(missing) + ".")
    dupes = sorted({m for m in members if members.count(m) > 1})
    if dupes:
        w("Duplicate machine(s): " + ", ".join(dupes) + ".")
    return out
