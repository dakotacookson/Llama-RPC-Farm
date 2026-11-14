"""First-run setup walkthrough: storage choice + completion state.

The account step itself is the existing `/api/auth/bootstrap` form. This
module drives the next two steps (model storage, then the Machines page)
and records whether the walkthrough has been completed so it only shows
for genuinely fresh installs.
"""
import pydantic
from fastapi import APIRouter, HTTPException

import access
import model_sync
import path_resolver
import settings_store
from settings_store import (_dump_settings_yaml_preserving,
                            _load_settings_yaml_preserving)

router = APIRouter()


def _control():
    return ((access.gate_open() and access.view_allow() is None)
            or access.is_admin(None))


def _require_control():
    if not _control():
        raise HTTPException(404, "not found")


class StorageIn(pydantic.BaseModel):
    mode: str = "shared"          # shared | sync
    model_dir: str = "store://llm-models"
    default_dir: str = ""
    store_name: str = "llm-models"
    store_path: str = ""          # current-platform root for shared mode
    platform: str = ""


@router.get("/api/setup/status")
async def setup_status():
    _require_control()
    current = _load_settings_yaml_preserving(settings_store.SETTINGS_FILE)
    cfg = model_sync.sync_config(current)
    stores = path_resolver._load_stores()
    return {
        "required": current.get("setup_wizard_done") is False,
        "done": current.get("setup_wizard_done") is not False,
        "platform": path_resolver.LOCAL_PLATFORM,
        "stores": stores,
        "store_name": "llm-models",
        "model_dir": current.get("model_dir") or "store://llm-models",
        "storage_mode": cfg["mode"],
        "default_dir": cfg["default_dir"],
    }


@router.post("/api/setup/storage")
async def setup_storage(s: StorageIn):
    _require_control()
    mode = str(s.mode or "").strip().lower()
    if mode not in ("shared", "sync"):
        raise HTTPException(400, "mode must be shared or sync")
    store_name = (s.store_name or "llm-models").strip() or "llm-models"
    platform = (s.platform or path_resolver.LOCAL_PLATFORM).strip().lower()
    if s.store_path.strip():
        try:
            path_resolver.save_store(store_name, platform,
                                     s.store_path.strip())
        except ValueError as e:
            raise HTTPException(400, str(e)) from None
    current = _load_settings_yaml_preserving(settings_store.SETTINGS_FILE)
    current["model_dir"] = (s.model_dir or "store://llm-models").strip()
    current["storage_mode"] = mode
    ms = current.get("model_sync")
    if not isinstance(ms, dict):
        ms = {}
    ms["enabled"] = mode == "sync"
    ms["default_dir"] = str(s.default_dir or "").strip()
    current["model_sync"] = ms
    _dump_settings_yaml_preserving(current, settings_store.SETTINGS_FILE)
    return {"ok": True, "mode": mode, "model_dir": current["model_dir"],
            "default_dir": ms["default_dir"], "store_name": store_name,
            "platform": platform, "store_path": s.store_path.strip()}


@router.post("/api/setup/complete")
async def setup_complete():
    _require_control()
    current = _load_settings_yaml_preserving(settings_store.SETTINGS_FILE)
    current["setup_wizard_done"] = True
    _dump_settings_yaml_preserving(current, settings_store.SETTINGS_FILE)
    return {"ok": True, "done": True}
