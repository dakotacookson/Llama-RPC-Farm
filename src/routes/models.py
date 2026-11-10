"""Model configs, tuning, placement advice, and model discovery."""
import os
from concurrent.futures import ThreadPoolExecutor

import pydantic
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.concurrency import run_in_threadpool

import access
import config_loader as cfg
import engine_registry
import path_resolver
import rpc_manager
import swap_manager
from hf_icons import _load_model_icons
from swap_apply import apply_swap_config
import settings_store
from settings_store import _dump_settings_yaml_preserving, _load_settings_yaml_preserving, _view_name_for_ip
from web_shared import _access_filter_models

router = APIRouter()


# ---------------- models ----------------
class VariantIn(pydantic.BaseModel):
    label: str
    servers: list[str] | None = None
    group: str | None = None
    device_order: str | None = None
    device_order_policy: str | None = None
    tensor_split_mode: str | None = None
    tensor_split: str | None = None
    fit_margin: int | None = None
    fit_ram_allow: int | None = None
    llama_bin: str | None = None
    expose: dict | None = None
    flags: dict | None = None
    pinned: bool | None = None
    ttl: int | None = None
    launch_mode: str | None = None          # auto | local | rpc
    launch_mode_local_port: int | None = None


class ModelConfigIn(pydantic.BaseModel):
    model_id: str
    original_model_id: str = ""  # set when renaming: old file is removed after save
    name: str = ""
    base_path: str = ""
    tensor_split_mode: str = "auto"   # auto | fit | split (manual is legacy -> fit)
    tensor_split: str = ""            # explicit layer weights for split mode (e.g. 4,4,4,2,2,2); unused by auto/fit
    fit_margin: int = 1024           # MiB reserve per device when mode=fit
    fit_ram_allow: int = 1024      # MiB of each RAM pool fit may use; rest reserved
    llama_bin: str = "~/llama.cpp/build/bin/llama-server"
    servers: list[str] = []
    group: str = ""                  # optional group affinity — pins model to group's machines
    pinned: bool = False             # keep-warm: never auto-unload when pinned
    ttl: int = 0                     # auto-unload after ttl seconds idle (0 = swap default, pinned forces 0)
    device_order: str = ""
    device_order_policy: str = "as-selected"  # as-selected | vram-first
    launch_mode: str = "auto"        # auto | local | rpc (swap deployment)
    launch_mode_local_port: int | None = None
    expose: dict = {}
    flags: dict = {}
    # custom runtime (external server, e.g. winbox .bat passthrough):
    # round-trips verbatim so custom models are editable/deletable saved
    # configs like any other model; validated below on save.
    runtime: str = "llama"
    custom: dict | None = None
    variants: list[VariantIn] = []
    reset: bool = False  # true = stop all RPC/llama processes + daemon before pushing, so the edit applies to a clean cluster


@router.get("/api/models")
async def list_saved_models(full: bool = False):
    # DEC-11 (T4-G6): the Models table lists only models the ACTIVE
    # engine serves — adapter filter (llama.cpp: GGUF), the same
    # downloads.vocabulary gate /api/scan applies to sources.
    # full=1: the ACCESS-TAB grant grid needs the WHOLE catalog (base +
    # every variant) so the operator can grant anything — allowed only
    # for control identities (trusted console / admin key). Scoped
    # viewers get the view-filtered table (?full=1 silently ignored).
    if full and ((access.gate_open() and access.view_allow() is None)
                 or access.is_admin(None)):
        import os as _os
        import yaml as _yaml
        out = []
        for fname in sorted(_os.listdir(cfg.MODELS_DIR)):
            if not fname.endswith(".yaml"):
                continue
            if "__tune_tmp" in fname or "__ab_single" in fname:
                continue
            try:
                with open(_os.path.join(cfg.MODELS_DIR, fname), "r",
                          encoding="utf-8") as f:
                    d = _yaml.safe_load(f) or {}
            except Exception:
                continue
            d["_file"] = fname
            if "variants" not in d or d["variants"] is None:
                d["variants"] = []
            d["architecture"] = cfg._get_model_architecture(d.get("base_path"))
            d["decision_type"] = cfg._get_decision_type(d.get("base_path"))
            mid = (d.get("model_id") or "").lower()
            name = (d.get("name") or "").lower()
            arch = (d.get("architecture") or "").lower()
            flags = d.get("flags") or {}
            tags = []
            if d["decision_type"]:
                tags.append("decision")
            else:
                if ("asr" in mid or "whisper" in mid or "audio" in mid
                        or flags.get("audio")):
                    tags.append("speech")
                if "embed" in mid or "embed" in name or "bert" in arch:
                    tags.append("embedding")
                if ("speech" not in tags
                        and (flags.get("mmproj") or "vision" in mid
                             or ("vl" in arch and arch != "unknown"))):
                    tags.append("vision")
                if not tags or tags == ["vision"]:
                    tags.append("chat")
            d["model_tags"] = tags
            out.append(d)
        return out
    try:
        engine = engine_registry.get_engine()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from None
    models = engine.filter_models(cfg.list_models())

    def _ids(fn):
        try:
            return {m.get("model_id") for m in (fn(models) or [])}
        except Exception:
            return set()

    emb_ids = _ids(engine.filter_embedding_models)
    aud_ids = _ids(engine.filter_audio_models)
    vis_ids = _ids(engine.filter_vision_models)
    dec_ids = _ids(engine.filter_decision_models)
    for m in models:
        mid = m.get("model_id") or ""
        base = mid.split("--")[0]
        arch = (m.get("architecture") or "").lower()
        tags = []
        if m.get("decision_type") or base in dec_ids:
            tags.append("decision")
        else:
            is_speech = base in aud_ids
            if is_speech:
                tags.append("speech")
            # ASR models also carry a projector, but it is audio-side; do
            # not label them vision just because mmproj exists.
            if not is_speech and (base in vis_ids
                                  or "mmproj" in (m.get("flags") or {})):
                tags.append("vision")
            if base in emb_ids or "bert" in arch or "embed" in arch:
                tags.append("embedding")
            if not tags or (tags == ["vision"]):
                tags.append("chat")
        m["model_tags"] = tags
    return _access_filter_models(models)


@router.get("/api/rpc-map")
async def rpc_map_all():
    """RPC<N> -> {machine, resource} for every model, exactly as the
    generated scripts count devices (peer order = --rpc order; per peer:
    selected GPUs by index, then CPU iff RAM selected). The visualizer
    resolves sched-log RPC tokens through this instead of guessing."""
    out = {}
    host = swap_manager.get_host()
    for m in cfg.list_models():
        mid = m.get("model_id")
        servers = m.get("servers") or []
        if not servers or not mid:
            continue
        try:
            master, peers = rpc_manager._split_servers(
                [host["name"]] + [s for s in servers if s != host["name"]],
                host["name"])
        except (ValueError, KeyError):
            continue
        # launch-mode parity with the swap plan (swap_manager
        # _resolve_launch_mode): a local-mode deployment runs llama-server
        # on ONE box with no RPC peers — no RPC<N> tokens exist in its
        # sched logs, so the visualizer must see an empty map, not the
        # peer-chaining map the old code invented.
        def _mode(deploy):
            return swap_manager._resolve_launch_mode(
                deploy, host["name"],
                [s for s in (deploy.get("servers") or servers)
                 if s != host["name"]])
        variants = {"": {} if _mode(m) == "local"
                    else rpc_manager.compute_rpc_map(m, master, peers)}
        for v in (m.get("variants") or []):
            label = (v.get("label") or "").strip()
            if not label:
                continue
            merged = rpc_manager._resolve_variant(m, label)
            try:
                vmaster, vpeers = rpc_manager._split_servers(
                    [host["name"]] + [s for s in (merged.get("servers") or servers)
                                      if s != host["name"]], host["name"])
                variants[label] = {} if _mode(merged) == "local" \
                    else rpc_manager.compute_rpc_map(merged, vmaster, vpeers)
            except (ValueError, KeyError):
                continue
        out[mid] = {"file": (m.get("base_path") or "").split("/").pop(),
                    "maps": variants}
    return out


@router.get("/api/models/icons")
async def model_icons_all():
    """Return all permanently-cached model icons (config/model_icons.json)."""
    return _load_model_icons()


@router.get("/api/models/{model_id}")
async def get_model(model_id: str):
    m = cfg.get_model(model_id)
    if m is None:
        raise HTTPException(status_code=404, detail=f"model '{model_id}' not found")
    return m


@router.post("/api/models")
async def save_model(m: ModelConfigIn, request: Request):
    mid = (m.model_id or "").strip()
    if not mid:
        raise HTTPException(status_code=400, detail="model_id is required")
    # View auto-join: a restricted (view-scoped) user who just downloaded
    # and configured a NEW model keeps access to it — save_model files
    # the config, so this adds the fresh mid to their own view's
    # allowlist immediately (both wizard steps ride ONE request, so the
    # same request's contextvar gate identifies the viewer). Trusted
    # users are never view-scoped; nothing happens for them.
    import access as _access
    if not _access.gate_open() and not cfg.get_model(mid):
        ip = _access.request_ip(request)
        view = _access.view_for_ip(ip)
        if view is not None:
            view_name = _view_name_for_ip(ip)
            if view_name:
                current = _load_settings_yaml_preserving(settings_store.SETTINGS_FILE)
                views = current.get(_access.VIEWS_SETTINGS_KEY) or {}
                entry = views.get(view_name) or {}
                models_list = [str(x) for x in (entry.get("models") or [])]
                if mid not in models_list:
                    models_list.append(mid)
                    entry["models"] = models_list
                    views[view_name] = entry
                    current[_access.VIEWS_SETTINGS_KEY] = views
                    _dump_settings_yaml_preserving(current, settings_store.SETTINGS_FILE)
    # manual tensor-split removed: fit mode supersedes it (llama-server's
    # fitter assigns whole layers by live free memory at load). Legacy
    # manual payloads are coerced so old clients keep working.
    if m.tensor_split_mode == "manual":
        m.tensor_split_mode = "fit"
        m.tensor_split = ""
    # Variant-write protection (strict mode): the editor round-trips the
    # variant list it was SHOWN — and for a scoped key that list is
    # narrowed to their grants (config_loader._filtered_variants). Saving
    # it back verbatim would silently DELETE ungranted variants from the
    # config (observed shape: the zazi key editing dad's model wipes
    # --dad). Rule: a scoped (restricted) editor keeps on-disk variants
    # their token does not cover, posted entries replace granted ones.
    # Gate-open editors (operator console / unrestricted) keep the old
    # replace-everything semantics — a trusted user's empty variant list
    # means "remove all variants".
    disk_variants = []
    if not _access.gate_open() and not (m.original_model_id or "").strip():
        try:
            import yaml as _yaml
            with open(os.path.join(cfg.MODELS_DIR, f"{mid}.yaml"),
                      "r", encoding="utf-8") as _f:
                disk_variants = (_yaml.safe_load(_f) or {}).get(
                    "variants") or []
        except Exception:
            disk_variants = []
        allow = _access.view_allow()
        if allow is not None:
            posted = {((v.label or "")).strip() for v in (m.variants or [])}
            kept = [v for v in disk_variants
                    if ((v.get("label") or "").strip() not in posted
                        and f"{mid}--{(v.get('label') or '').strip()}"
                        not in allow)]
            m.variants = list(m.variants or []) + kept
    # optional group affinity: must exist if set
    if m.group:
        g = cfg.groups.get(m.group)
        if not g:
            raise HTTPException(status_code=400, detail=f"group '{m.group}' not found")
        # if servers explicitly set, they must be subset of group's machines (warn, not block — group may be intent)
        if m.servers:
            allowed = set(g.get("machines") or [])
            unknown = [s for s in m.servers if s not in allowed]
            if unknown:
                raise HTTPException(status_code=400, detail=f"servers {unknown} not in group '{m.group}' (group has {sorted(allowed)})")
    # validate variants: labels unique, non-empty, servers subset of group if group set
    if m.variants:
        seen = set()
        for v in m.variants:
            lab = (v.label or "").strip()
            if not lab:
                raise HTTPException(status_code=400, detail="variant label is required")
            if "/" in lab or "\\" in lab or lab in (".", ".."):
                raise HTTPException(status_code=400, detail=f"variant label '{lab}' contains invalid characters")
            if lab in seen:
                raise HTTPException(status_code=400, detail=f"duplicate variant label '{lab}'")
            seen.add(lab)
            if v.group:
                gv = cfg.groups.get(v.group)
                if not gv:
                    raise HTTPException(status_code=400, detail=f"variant '{lab}': group '{v.group}' not found")
                if v.servers:
                    allowed = set(gv.get("machines") or [])
                    unknown = [s for s in v.servers if s not in allowed]
                    if unknown:
                        raise HTTPException(status_code=400, detail=f"variant '{lab}': servers {unknown} not in group '{v.group}' (group has {sorted(allowed)})")
            # servers must exist (if any). v.servers may be None from
            # clients that skip the hardware step; treat it as empty.
            valid_machines = {mm["name"] for mm in cfg.machines.all()}
            unknown_v = [s for s in (v.servers or []) if s not in valid_machines]
            if unknown_v:
                raise HTTPException(status_code=400, detail=f"variant '{lab}': unknown machine(s): {', '.join(unknown_v)}")
    try:
        data = m.model_dump()
        # runtime: custom (external server) — validate the spec now (400)
        # so a bad custom config can never be filed; the plan-time branch
        # re-validates and skips with a warning as second defense. A
        # custom model needs no GGUF: base_path stays empty by design.
        if (data.get("runtime") or "") == "custom":
            spec = data.get("custom") or {}
            swap_manager.validate_custom_spec(spec, where=f"model '{mid}'")
            if not data.get("custom"):
                data.pop("custom", None)
            # winbox-bat is single-Windows-box by construction — fail fast
            # at save (400) instead of skipping silently at plan time
            if (spec.get("kind") or "") == "winbox-bat":
                try:
                    host_name = swap_manager.load_settings().get("host")
                except Exception:
                    host_name = None
                peers = [s for s in (data.get("servers") or [])
                         if s != host_name]
                if len(peers) != 1:
                    raise ValueError(
                        f"model '{mid}': custom winbox-bat needs exactly one "
                        f"target box in servers (got {peers})")
                try:
                    box = cfg.machines.require(peers[0])
                except ValueError:
                    raise ValueError(
                        f"model '{mid}': unknown machine '{peers[0]}'")
                if (box.get("os", "linux")) != "windows":
                    raise ValueError(
                        f"model '{mid}': custom winbox-bat needs a windows "
                        f"box, '{peers[0]}' is {box.get('os', 'linux')}")
        elif data.get("custom") is None:
            data.pop("custom", None)
        if not data.get("runtime") or data.get("runtime") == "llama":
            data.pop("runtime", None)  # default stays unfiled (legacy files)
        orig = (data.pop("original_model_id", "") or "").strip()
        validation_warnings = []
        saved = cfg.save_model(mid, data)
        # Validation is disk-based, so run it AFTER the save (the config
        # exists now). It never blocks the save: errors and warnings ride
        # back to the UI so a user sees a doomed placement/flag combination
        # at configuration time instead of at first load.
        try:
            import config_advice
            validation_warnings += config_advice.model_warnings(saved)
        except Exception as e:
            validation_warnings.append(f"warn: advice unavailable: {e}")
        try:
            from engine_registry import get_engine
            issues = await run_in_threadpool(
                get_engine().validate_model_config, saved)
            for it in issues or []:
                if isinstance(it, dict):
                    if it.get("level") == "info":
                        continue  # e.g. "custom runtime: checks skipped"
                    validation_warnings.append(
                        f"{'error' if it.get('level') == 'error' else 'warn'}: "
                        f"{it.get('msg') or it}")
                elif str(it).strip():
                    validation_warnings.append(f"warn: {it}")
        except Exception as e:
            validation_warnings.append(f"warn: validation skipped: {e}")
        # de-dupe identical lines (advice and validator overlap)
        validation_warnings = list(dict.fromkeys(validation_warnings))
        if orig and orig != mid:
            # rename: remove the orphaned old file (plus its seed, if any)
            try:
                cfg.delete_model(orig)
                import tuner as _tuner
                import os as _os
                _sp = _tuner._seed_path(orig)
                if _os.path.exists(_sp):
                    _os.remove(_sp)
            except Exception:
                pass
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    # optional hard-reset: user asked that a model edit resets everything.
    # Stop the daemon + every RPC/llama process cluster-wide BEFORE pushing,
    # so the regenerated config/scripts apply to a clean state (no stale
    # peers keeping the old device map alive). Then push + restart daemon.
    if m.reset:
        reset_result = {"stopped": [], "errors": []}
        try:
            def _stop_all(mm_name):
                try:
                    return {"machine": mm_name,
                            "r": rpc_manager.stop_rpc(mm_name)}
                except Exception as e:
                    return {"machine": mm_name, "error": str(e)}
            # daemon first, then all peers in parallel; idempotent + fast
            try:
                reset_result["stopped"].append(swap_manager.stop())
            except Exception as e:
                reset_result["errors"].append(f"swap stop: {e}")
            names = [mm["name"] for mm in cfg.machines.all()]
            with ThreadPoolExecutor(max_workers=len(names) or 1) as pool:
                reset_result["stopped"].extend(pool.map(_stop_all, names))
        except Exception as e:
            reset_result["errors"].append(str(e))
    # Cascade the save into the running deployment: regenerate the swap
    # config/scripts and reload the daemon in the background.
    push_result = await apply_swap_config(f"model save {mid}", reload=True)
    reloading = bool(push_result.get("reloading"))
    reset_result = None
    if isinstance(saved, dict):
        saved = {**saved, "_swap_push": push_result, "_swap_reload": reloading,
                 "_reset": reset_result if m.reset else None,
                 "_validation_warnings": validation_warnings}
    return saved


@router.post("/api/models/cleanup")
async def cleanup_models():
    """Delete any blank/invalid model configs (model_id empty or file is .yaml).
    Returns deleted filenames. Safe to call anytime."""
    import glob as _glob
    deleted = []
    errors = []
    # .yaml is the blank-id file; also catch any file where loaded model_id is empty
    for path in _glob.glob(os.path.join(cfg.MODELS_DIR, "*.yaml")):
        try:
            import yaml as _yaml
            with open(path, "r", encoding="utf-8") as f:
                data = _yaml.safe_load(f) or {}
            mid = (data.get("model_id") or "").strip()
            fname = os.path.basename(path)
            # blank file or mismatched id
            if not mid or fname == ".yaml" or fname == ".yml":
                os.remove(path)
                deleted.append(fname)
        except Exception as e:
            errors.append(f"{os.path.basename(path)}: {e}")
    if deleted:
        await apply_swap_config("model cleanup")
    return {"deleted": deleted, "errors": errors, "remaining": len(cfg.list_models())}


def _file_entry(path):
    exists = os.path.exists(path)
    return {"path": path, "exists": exists,
            "size_gb": round(os.path.getsize(path) / 1e9, 1) if exists else 0}


@router.delete("/api/models/scan-file")
async def delete_scanned_file(delete_file: bool = False, path: str = ""):
    """Delete a scanned artifact file (store:// form) from the store.
    delete_file=False previews (exists/size); the path must resolve
    inside a configured store so arbitrary paths are rejected."""
    if not path or not path.startswith(path_resolver.PREFIX):
        raise HTTPException(status_code=400, detail="store:// path required")
    local = path_resolver.resolve(path)
    store_name = path[len(path_resolver.PREFIX):].split("/")[0]
    root = path_resolver.resolve(f"store://{store_name}")
    norm_local = os.path.normpath(local)
    norm_root = os.path.normpath(root)
    if not norm_local.startswith(norm_root):
        raise HTTPException(status_code=400, detail="path outside store roots")

    files = await run_in_threadpool(
        lambda: {"files": [_file_entry(local)]})
    if not delete_file:
        return files
    target = files["files"][0]
    if not target["exists"]:
        return {"deleted": [], "errors": [f"{local}: not found"]}
    try:
        await run_in_threadpool(os.remove, local)
    except OSError as e:
        return {"deleted": [], "errors": [str(e)]}
    return {"deleted": [local], "errors": []}


@router.delete("/api/models/{model_id}")
async def delete_model(model_id: str):
    out = {"deleted": cfg.delete_model(model_id)}
    # stale seeds must not haunt a future model with the same id
    try:
        import tuner, os
        sp = tuner._seed_path(model_id)
        if os.path.exists(sp):
            os.remove(sp)
            out["seed_removed"] = True
    except Exception:
        pass
    if out["deleted"]:
        await apply_swap_config(f"model delete {model_id}")
    return out


@router.post("/api/models/{model_id}/test")
async def test_model(model_id: str):
    """Instant static config check (no side effects). Live verification
    is covered by Tune, which loads and benches the real artifact."""
    report = await run_in_threadpool(rpc_manager.validate_model, model_id)
    status = 200 if report.get("ok") else 500
    return JSONResponse(report, status_code=status)


# ---------------- auto-tune (math first, live-verify after) ----------------
class TuneEstimateIn(pydantic.BaseModel):
    base_path: str = ""
    servers: list[str] = []
    device_order: str = ""
    cache_type: str = "q8_0"
    mmproj: str = ""


@router.post("/api/tune/estimate")
async def tune_estimate_form(t: TuneEstimateIn):
    """Math-only estimate for the model form (unsaved): hardware MUST be
    selected in the form (servers + device_order chips). 0 loads."""
    import tuner
    try:
        master = None
        if t.servers:
            try:
                master = rpc_manager._split_servers(t.servers)[0]
            except ValueError:
                master = None
        target_os = (master or {}).get("os", "linux") if master else "linux"
        est = await run_in_threadpool(
            tuner.estimate, t.base_path, t.servers, t.device_order,
            target_os, t.cache_type, t.mmproj)
        return est
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/api/models/{model_id}/tune/estimate")
async def tune_estimate_saved(model_id: str):
    import tuner
    m = cfg.get_model(model_id)
    if m is None:
        raise HTTPException(status_code=404, detail=f"model '{model_id}' not found")
    try:
        master = rpc_manager._split_servers(m.get("servers") or [])[0]
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    flags = m.get("flags", {}) or {}
    try:
        return await run_in_threadpool(
            tuner.estimate, m.get("base_path") or m.get("path", ""),
            m.get("servers") or [], m.get("device_order") or "",
            master.get("os", "linux"),
            flags.get("cache-type-k") or "q8_0", flags.get("mmproj", ""))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/api/models/{model_id}/tune/defaults")
async def tune_apply_defaults(model_id: str):
    """Apply safe defaults (q8 KV, MTP on iff file has draft tensors,
    vram-first fit) to a saved model. No loads."""
    import tuner
    m = cfg.get_model(model_id)
    if m is None:
        raise HTTPException(status_code=404, detail=f"model '{model_id}' not found")
    if not m.get("servers"):
        raise HTTPException(status_code=400,
                            detail="no servers selected — pick hardware in the form first")
    try:
        master = rpc_manager._split_servers(m.get("servers"))[0]
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    d = await run_in_threadpool(
        tuner.safe_defaults, m.get("base_path") or m.get("path", ""),
        m.get("servers"), m.get("device_order") or "",
        master.get("os", "linux"), m.get("flags", {}) or {})
    m["tensor_split_mode"] = d["tensor_split_mode"]
    m["fit_margin"] = d["fit_margin"]
    m["fit_ram_allow"] = d["fit_ram_allow"]
    m["device_order_policy"] = d["device_order_policy"]
    flags = m.get("flags", {}) or {}
    flags.update(d["flags"])
    m["flags"] = flags
    saved = cfg.save_model(model_id, m)
    return {"saved": saved["model_id"] if isinstance(saved, dict) else model_id,
            "defaults": d}


class RecommendPlacementIn(pydantic.BaseModel):
    base_path: str = ""
    servers: list[str] | None = None   # restrict to these boxes (group)
    target_ctx: int | None = None
    allow_ram: bool = True
    cache_type: str = "q8_0"
    mmproj: str = ""


@router.post("/api/models/recommend")
async def recommend_placement(r: RecommendPlacementIn):
    """Recommend a baseline hardware placement + flags for an (unsaved)
    model file: reads the GGUF header, picks the best single box or
    multi-box RPC spread from detected hardware, and returns the same
    fields the model editor saves. Math-only — no loads, no side effects.
    """
    import placement
    try:
        return await run_in_threadpool(
            placement.recommend, r.base_path, r.servers, r.target_ctx,
            r.allow_ram, r.cache_type, r.mmproj)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


class TuneStartIn(pydantic.BaseModel):
    target_tps: float | None = None  # None = topology default (25 local, 20 RPC)
    try_mtp_off: bool = True
    port: int = 18399


@router.post("/api/models/{model_id}/tune/start")
async def tune_start(model_id: str, t: TuneStartIn):
    import tuner
    if "__tune_tmp" in model_id or "__ab_single" in model_id:
        raise HTTPException(
            status_code=400,
            detail=f"'{model_id}' is a tuner scratch config — save the form "
                   f"first (it auto-saves under the proper model id) then tune")
    if cfg.get_model(model_id) is None:
        raise HTTPException(status_code=404, detail=f"model '{model_id}' not found (did you delete it? re-save the form first)")
    try:
        return await run_in_threadpool(
            tuner.start_tune, model_id, t.target_tps, t.port, t.try_mtp_off)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/api/models/{model_id}/tune/status")
async def tune_status(model_id: str):
    import tuner
    return tuner.job_status(model_id)


@router.post("/api/models/{model_id}/tune/cancel")
async def tune_cancel(model_id: str):
    import tuner
    return await run_in_threadpool(tuner.cancel_tune, model_id)


@router.get("/api/models/{model_id}/tune/drift")
async def tune_drift(model_id: str):
    """Seed-vs-live config drift (the 5→24 incident class). No loads."""
    import tuner
    return await run_in_threadpool(tuner.drift_check, model_id)


@router.post("/api/models/{model_id}/tune/seed")
async def tune_seed(model_id: str):
    """Freeze the current saved config as the known-good seed.
    Future tunes measure it first and must beat it to replace it."""
    import tuner
    try:
        return await run_in_threadpool(tuner.snapshot_seed, model_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


class DeleteModelIn(pydantic.BaseModel):
    delete_file: bool = False


@router.delete("/api/models/{model_id}/file")
async def delete_model_file(model_id: str, d: DeleteModelIn):
    """Model file deletion: removes the artifact file(s) the model
    config points at (base_path gguf + mmproj/mtp companions) from the
    store, then removes the config itself. With delete_file=False it
    only previews what would be deleted."""
    m = cfg.get_model(model_id)
    if m is None:
        raise HTTPException(status_code=404, detail=f"model '{model_id}' not found")

    def collect():
        seen = set()
        raws = [m.get("base_path")]
        flags = m.get("flags") or {}
        raws += [flags.get(k) for k in ("mmproj", "mtp") if flags.get(k)]
        files = []
        for raw in raws:
            if not raw:
                continue
            path = path_resolver.resolve(raw)
            if path not in seen:
                seen.add(path)
                files.append(_file_entry(path))
        return files

    files = await run_in_threadpool(collect)
    if not d.delete_file:
        return {"files": files}

    def really_delete():
        deleted, errors = [], []
        for f in files:
            if not f["exists"]:
                continue
            try:
                os.remove(f["path"])
                deleted.append(f["path"])
            except OSError as e:
                errors.append(f"{f['path']}: {e}")
        if not errors:
            cfg.delete_model(model_id)
        return {"deleted": deleted, "errors": errors,
                "config_removed": not errors}
    result = await run_in_threadpool(really_delete)
    if result.get("config_removed"):
        await apply_swap_config(f"model file delete {model_id}")
    return result
