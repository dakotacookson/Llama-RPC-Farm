"""Small per-user UI state: custom themes, flow layout, accordion."""
import json
import os
import re

import pydantic
from fastapi import APIRouter, HTTPException
from runtime_paths import data_dir

BASE_DIR = data_dir()

router = APIRouter()


# ---------------- custom themes (user-built palettes) ----------------
# User themes live in config/custom_themes.json: {id: {"name": str, "tokens": {css-var: #hex}}}.
# The frontend applies them as inline custom properties over the base dark
# tokens, so no CSS regeneration is needed and previews apply instantly.

CUSTOM_THEMES_FILE = os.path.join(BASE_DIR, "config", "custom_themes.json")
CUSTOM_THEME_TOKENS = (
    "--bg", "--surface", "--raised", "--hover", "--inset",
    "--border", "--border-soft", "--border-strong",
    "--text", "--text-2", "--muted",
    "--accent", "--ribbon", "--ok", "--warn", "--bad",
)
# numeric (non-color) theme settings: ribbon FX multipliers
CUSTOM_THEME_NUMERIC = ("--ribbon-glow", "--ribbon-sat", "--ribbon-alpha")


def _load_custom_themes():
    if not os.path.exists(CUSTOM_THEMES_FILE):
        return {}
    try:
        with open(CUSTOM_THEMES_FILE, "r", encoding="utf-8") as f:
            data = json.load(f) or {}
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _save_custom_themes(themes):
    tmp = CUSTOM_THEMES_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(themes, f, indent=2)
    os.replace(tmp, CUSTOM_THEMES_FILE)


@router.get("/api/themes/custom")
async def list_custom_themes():
    """All user-built themes (name + token map each)."""
    return {"themes": _load_custom_themes()}


class CustomThemeIn(pydantic.BaseModel):
    id: str = ""      # empty = derive from name
    name: str = "Custom"
    tokens: dict = {}


@router.post("/api/themes/custom")
async def save_custom_theme(t: CustomThemeIn):
    """Create or replace a user theme. Only known tokens + #rrggbb values
    are stored; anything else is rejected."""
    name = (t.name or "").strip()[:40] or "Custom"
    tid = (t.id or "").strip().lower()
    if not tid:
        tid = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")[:32] or "custom"
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,31}", tid):
        raise HTTPException(status_code=400, detail="theme id must match [a-z0-9-]")
    if not isinstance(t.tokens, dict):
        raise HTTPException(status_code=400, detail="tokens must be an object")
    clean = {}
    for k, v in t.tokens.items():
        if k in CUSTOM_THEME_NUMERIC:
            try:
                f = float(v)
            except (TypeError, ValueError):
                raise HTTPException(status_code=400, detail=f"token '{k}' must be a number")
            if not 0 <= f <= 2.5:
                raise HTTPException(status_code=400, detail=f"token '{k}' must be 0-2.5")
            clean[k] = str(f)
            continue
        if k not in CUSTOM_THEME_TOKENS:
            raise HTTPException(status_code=400, detail=f"unknown token '{k}'")
        if not isinstance(v, str) or not re.fullmatch(r"#[0-9a-fA-F]{6}", v):
            raise HTTPException(status_code=400, detail=f"token '{k}' must be #rrggbb")
        clean[k] = v.lower()
    if len(clean) < len(CUSTOM_THEME_TOKENS) + len(CUSTOM_THEME_NUMERIC):
        missing = [k for k in (*CUSTOM_THEME_TOKENS, *CUSTOM_THEME_NUMERIC) if k not in clean]
        raise HTTPException(status_code=400, detail=f"missing tokens: {', '.join(missing)}")
    themes = _load_custom_themes()
    themes[tid] = {"name": name, "tokens": clean}
    try:
        _save_custom_themes(themes)
    except OSError as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {"id": tid, "name": name, "tokens": clean}


@router.delete("/api/themes/custom/{tid}")
async def delete_custom_theme(tid: str):
    themes = _load_custom_themes()
    if tid not in themes:
        raise HTTPException(status_code=404, detail=f"custom theme '{tid}' not found")
    del themes[tid]
    try:
        _save_custom_themes(themes)
    except OSError as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {"deleted": tid, "remaining": list(themes)}


ACCORDION_FILE = os.path.join(BASE_DIR, "config", "accordion_state.json")


def _load_accordion_state():
    if not os.path.exists(ACCORDION_FILE):
        return {}
    try:
        with open(ACCORDION_FILE, "r", encoding="utf-8") as f:
            return json.load(f) or {}
    except Exception:
        return {}


def _save_accordion_state(state):
    tmp = ACCORDION_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)
    os.replace(tmp, ACCORDION_FILE)


FLOW_LAYOUT_FILE = os.path.join(BASE_DIR, "config", "flow_layout.json")


def _load_flow_layout():
    if not os.path.exists(FLOW_LAYOUT_FILE):
        return {}
    try:
        with open(FLOW_LAYOUT_FILE, "r", encoding="utf-8") as f:
            return json.load(f) or {}
    except Exception:
        return {}


def _save_flow_layout(layout):
    try:
        tmp = FLOW_LAYOUT_FILE + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(layout, f, indent=2)
        os.replace(tmp, FLOW_LAYOUT_FILE)
    except Exception:
        pass


@router.get("/api/flow/layout")
async def get_flow_layout():
    return _load_flow_layout()


class FlowLayoutIn(pydantic.BaseModel):
    layout: dict


@router.post("/api/flow/layout")
async def set_flow_layout(inp: FlowLayoutIn):
    _save_flow_layout(inp.layout)
    return {"ok": True}


@router.get("/api/accordion")
async def get_accordion_state():
    return _load_accordion_state()


class AccordionIn(pydantic.BaseModel):
    key: str
    open: bool


@router.post("/api/accordion")
async def set_accordion_state(a: AccordionIn):
    state = _load_accordion_state()
    state[a.key] = a.open
    _save_accordion_state(state)
    return {"ok": True, "key": a.key, "open": a.open}
