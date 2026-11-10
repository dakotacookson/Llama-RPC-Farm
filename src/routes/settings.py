"""Farm settings (comment-preserving) and engine switch orchestration."""
import os
import sys
import tempfile

import pydantic
from fastapi import APIRouter, HTTPException, Request

import access
import config_loader as cfg
import engine_registry
import settings_store
import swap_manager
from settings_store import (_dump_settings_yaml_preserving,
                            _load_settings_yaml_preserving)

router = APIRouter()


# ---------------- app settings ----------------

# ---------------- settings.yaml persister (comment-preserving) --------------
# B9c (DEC-14): the round-trip pair lives in settings_persister.py so
# writers can persist settings.yaml through the SAME comment-preserving
# persister instead of plain yaml.safe_dump.

@router.get("/api/settings")
async def get_settings():
    import yaml
    defaults = {"model_dir": "store://llm-models",
                "llama_bin": "~/llama.cpp/build/bin",
                "theme_family": "nord", "theme_mode": "dark"}
    if os.path.exists(settings_store.SETTINGS_FILE):
        with open(settings_store.SETTINGS_FILE, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        defaults.update(data)
    # engine schema (ARD §2): inject engine defaults so the frontend never
    # None-checks. A typo in settings.yaml (bad engine) must NOT fall back
    # silently — ValueError -> 400 with the valid values in the detail.
    try:
        engine_registry.cfg.inject_engine_defaults(defaults)
    except ValueError as e:
        raise HTTPException(400, str(e)) from None
    # Identity-scoped settings: theme state is PER-KEY (access_keys ui:
    # block); engine/model_dir stay farm-wide and are admin-console
    # material. Scoped (non-admin) callers get their ui overlay + no
    # topology write surface (POST refuses engine from them).
    _key_ui = {}
    try:
        rec = access.current_key_record()
        if rec and isinstance(rec.get("ui"), dict):
            _key_ui = rec["ui"]
    except Exception:
        _key_ui = {}
    if _key_ui:
        defaults.update({k: v for k, v in _key_ui.items()
                         if k in ("theme_family", "theme_mode")})
    # never leak the raw token to the browser — expose only set-state.
    # Source of truth is the encrypted vault (.env); legacy YAML read for compat.
    token = str(defaults.pop("hf_token", "") or "")
    try:
        import secrets_vault as _vault
        defaults["hf_token_set"] = _vault.hf_token_set() or bool(token.strip())
        defaults["farm_agent_token_set"] = _vault.farm_agent_token_set()
    except Exception:
        defaults["hf_token_set"] = bool(token.strip())
        defaults["farm_agent_token_set"] = False
    # UI field manifest (AUDIT-T6 §1.5, DEC-02) — the complete
    # engine-owned inventory the frontend renders from. The legacy
    # ui_hints view rides along DEPRECATED (its last JS consumer is the
    # flag-presets reader; Card 15 retires the key): ui_manifest() is
    # now the contract, app.js reads appSettings.ui_manifest.
    engine_adapter = engine_registry.get_engine(defaults)
    defaults["ui_manifest"] = engine_adapter.ui_manifest()
    # deprecated shim (retires with the ui_hints key in Card 15): derived
    # from the same manifest so the two views cannot drift
    defaults["ui_hints"] = engine_adapter.ui_hints()
    return defaults


class AppSettingsIn(pydantic.BaseModel):
    model_dir: str = "store://llm-models"
    llama_bin: str = "~/llama.cpp/build/bin"
    theme_family: str = "nord"
    theme_mode: str = "dark"
    hf_token: str | None = None  # only a non-empty value updates the stored token
    clear_hf_token: bool = False
    engine: str | None = None  # llama_cpp — None leaves the stored value


async def _restart_everything_for_engine(new_engine: str):
    """Engine changed: restart llama-swap (engine-qualified config), kill all
    model servers, and restart this web app so every reader picks up the new
    engine. Same detached-restart machinery as POST /api/restart."""
    import subprocess as _sp
    import shlex as _shlex
    app_path = os.path.abspath(__file__)
    project_root = os.path.dirname(os.path.dirname(app_path))
    py = sys.executable
    # llama-swap settings: engine-qualified config path per DEC-03
    try:
        s = swap_manager.load_settings()
    except Exception:
        s = {}
    sw_port = s.get("listen_port") or 8080
    raw_cfg = s.get("config_path") or "~/.config/llama-swap/config.yaml"
    raw_cfg = os.path.expanduser(raw_cfg)
    cfg_engine = new_engine
    try:
        eff, _ = swap_manager.engine_swap_config_path(cfg_engine, raw_cfg)
    except Exception:
        eff = raw_cfg
    log = os.path.join(tempfile.gettempdir(), "llama-web.log")
    if os.name == "nt":
        ps = (
            "Start-Sleep 1; "
            "Get-Process llama-swap,llama-server -ErrorAction SilentlyContinue | Stop-Process -Force; "
            "$c = Get-NetTCPConnection -LocalPort 8199 -State Listen -ErrorAction SilentlyContinue; "
            "if ($c) { Stop-Process -Id $c.OwningProcess -Force -ErrorAction SilentlyContinue }; "
            "Start-Sleep 1; "
            f"Start-Process -FilePath '{py}' -ArgumentList '\"{app_path}\"' "
            f"-WindowStyle Hidden -RedirectStandardOutput '{log}' -RedirectStandardError '{log}.err'"
        )
        _sp.Popen(["powershell", "-NoProfile", "-Command", ps],
                  start_new_session=True, close_fds=True)
    else:
        q = _shlex.quote
        # 1) Re-provision the daemon the SAME way the app does it: swap_manager
        #    .launch() regenerates the engine-qualified config, rewrites the
        #    systemd unit's ExecStart, and restarts it — beating the unit's
        #    Restart=always properly instead
        #    of racing it with pkill+setsid (which lost the race: systemd
        #    resurrected the OLD-config daemon within 5s).
        # stop FIRST so the old daemon frees the configured port (the
        # launch path rolls to 8081+ when the port is still held by the
        # dying daemon — persisting the wrong port into swap.yaml)
        try:
            swap_manager.stop()
            import time as _tm
            _tm.sleep(2)
        except Exception as e:
            print(f"[engine-flip] swap stop failed: {e}", file=sys.stderr)
        # restore the configured port if a previous flip rolled it
        try:
            st = swap_manager.load_settings()
            st["listen_port"] = 8080
        except Exception:
            pass
        try:
            swap_manager.launch(None, 8080, None, None)
        except Exception as e:
            print(f"[engine-flip] swap relaunch failed: {e}", file=sys.stderr)
        # 2) Kill the old engine's model servers (they serve stale configs).
        # 3) Restart this web app detached so every reader picks up the new
        #    engine (same machinery as POST /api/restart).
        cmd = (
            "sleep 1; "
            "pkill llama-server 2>/dev/null || true; "
            "(fuser -k 8199/tcp > /dev/null 2>&1 || true); "
            "sleep 1; "
            f"setsid -f {q(py)} {q(app_path)} > {q(log)} 2>&1 "
            f"|| nohup {q(py)} {q(app_path)} > {q(log)} 2>&1 &"
        )
        _sp.Popen(["/bin/bash", "-c", cmd], start_new_session=True, close_fds=True)
    return {
        "status": "restarting",
        "message": f"Engine switched to {new_engine} — restarting llama-swap "
                   f"(config: {eff}) and the web app (~3s at http://127.0.0.1:8199)",
    }


@router.post("/api/settings")
async def save_settings(s: AppSettingsIn, r: Request):
    # Identity split: THEMES are per-key (any identity sets their own
    # ui); ENGINE/model_dir/llama_bin/hf_token are farm-wide state — only
    # trusted console or an ADMIN key may write those. A scoped non-admin
    # POST carrying engine/etc. is refused 404-shaped.
    _control_identity = (access.gate_open() and access.view_allow() is None) \
        or access.is_admin(None)
    _key_name_for_ui = access.key_display_name()
    if not _control_identity and (s.engine is not None
                                  or s.hf_token or s.clear_hf_token):
        raise HTTPException(404, "not found")
    # merge over the CURRENT FILE so unknown/future keys survive the write;
    # loaded via the comment-preserving persister (B9c) — a plain
    # yaml.safe_load here is what stripped every comment on theme switches
    current = _load_settings_yaml_preserving(settings_store.SETTINGS_FILE)
    engine_changed = False
    update = s.model_dump()
    tok = (update.pop("hf_token", None) or "").strip()
    clear_tok = update.pop("clear_hf_token", False)
    engine = update.pop("engine", None)
    # per-key UI state: scoped identities write theme_family/theme_mode
    # into THEIR access_keys ui: block instead of the shared file —
    # farm-wide state untouched.
    if not _control_identity and _key_name_for_ui and \
            (update.get("theme_family") or update.get("theme_mode")):
        _want = {k: update[k] for k in ("theme_family", "theme_mode")
                 if update.get(k)}
        _keys = current.get(access.KEYS_SETTINGS_KEY)
        if not isinstance(_keys, dict):
            _keys = {}
        for _t, _e in _keys.items():
            if isinstance(_e, dict) and _e.get("name") == _key_name_for_ui:
                ui = _e.get("ui") or {}
                ui.update(_want)
                _e["ui"] = ui
                break
        current[access.KEYS_SETTINGS_KEY] = _keys
        update.pop("theme_family", None)
        update.pop("theme_mode", None)
    current.update(update)
    if engine is not None:
        # enum check with the valid values in the message (never a silent
        # fallback — ARD §2); ValueError -> 400
        try:
            cfg.validate_engine(engine, where="POST /api/settings")
        except ValueError as e:
            raise HTTPException(400, str(e)) from None
        engine_changed = engine != (current.get("engine") or cfg.DEFAULT_ENGINE)
        current["engine"] = engine
    # normalize the merged settings (injects defaults for missing keys) so
    # the file on disk always round-trips through the same schema the
    # loader enforces; failure -> 400, file left untouched
    try:
        cfg.inject_engine_defaults(current)
    except ValueError as e:
        raise HTTPException(400, str(e)) from None
    # never persist raw token in YAML — source of truth is the encrypted
    # vault (.env), legacy YAML token ignored/removed
    current.pop("hf_token", None)
    import secrets_vault as _vault
    if clear_tok:
        _vault.set_hf_token("")
    elif tok:
        _vault.set_hf_token(tok)
    # don't echo the raw token back; write via the comment-preserving
    # persister (B9c) so unrelated keys, key order, and all comments
    # survive — never a wholesale rewrite from the request dict
    # DEC-15: the _engine_save_warnings channel is retired with the
    # cross-engine save-path translation (DEC-09) — no warning groups
    # remain. Keep the scrub-pop one release long: a pre-DEC-15 build
    # may have persisted the key into settings.yaml, and the persister
    # rewrites the whole file from this dict, so dropping it here
    # scrubs the stale copy from disk.
    current.pop("_engine_save_warnings", None)
    _dump_settings_yaml_preserving(current, settings_store.SETTINGS_FILE)
    out = dict(current)
    out.pop("hf_token", None)  # response-only: never echoed
    out["hf_token_set"] = _vault.hf_token_set()
    out["farm_agent_token_set"] = _vault.farm_agent_token_set()
    if engine_changed:
        # Engine flip: the running daemons still serve the OLD engine's
        # configs/scripts. Full restart (same as /api/restart) — but the
        # app process restarting itself re-reads settings.yaml on boot,
        # so the new engine takes effect everywhere in one shot.
        resp = await _restart_everything_for_engine(engine)
        # keep the settings echo (engine/hf_token_set) in the
        # same response — the client renders the saved state AND the
        # restart notice; a bare {"status": "restarting"} lost the
        # normalized settings the caller just saved.
        merged = dict(out)
        merged.update(resp)
        return merged
    return out
