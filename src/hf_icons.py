"""Permanent model-icon cache (config/model_icons.json)."""
import json
import os

from runtime_paths import data_dir

BASE_DIR = data_dir()


MODEL_ICONS_FILE = os.path.join(BASE_DIR, "config", "model_icons.json")
_ICON_CACHE = None  # ((mtime_ns, size), dict)


def _sanitize_icon_url(v):
    """Coerce a cached icon value to a clean https URL or ''.

    Old cache entries were polluted by the HF avatar scraper
    (URL + '&quot;, ...' JSON leakage). Strip at the first HTML
    entity/quote/whitespace and reject non-http(s) values (e.g.
    dicts from the former double-wrapped _resolve_icon_url return).
    """
    if not isinstance(v, str):
        return ""
    v = v.replace("&amp;", "&").strip()
    for sep in ("&quot;", '"', "'", " ", "<", "\\n", "\\t"):
        if sep in v:
            v = v.split(sep)[0]
    v = v.strip()
    if not (v.startswith("https://") or v.startswith("http://") or v == ""):
        return ""
    return v


def _load_model_icons():
    global _ICON_CACHE
    try:
        st = os.stat(MODEL_ICONS_FILE)
        key = (st.st_mtime_ns, st.st_size)
    except OSError:
        return {}
    if _ICON_CACHE and _ICON_CACHE[0] == key:
        return _ICON_CACHE[1]
    try:
        with open(MODEL_ICONS_FILE, "r", encoding="utf-8") as f:
            raw = json.load(f) or {}
    except Exception:
        return {}
    # sanitize on load so one polluted entry can't break the UI
    cleaned = {}
    dirty = False
    for k, v in (raw.items() if isinstance(raw, dict) else []):
        sv = _sanitize_icon_url(v)
        cleaned[k] = sv
        if sv != v:
            dirty = True
    if dirty:
        try:
            _save_model_icons(cleaned)
        except Exception:
            pass
    try:
        st = os.stat(MODEL_ICONS_FILE)
        _ICON_CACHE = ((st.st_mtime_ns, st.st_size), cleaned)
    except OSError:
        _ICON_CACHE = None
    return cleaned


def _save_model_icons(icons):
    global _ICON_CACHE
    try:
        tmp = MODEL_ICONS_FILE + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(icons, f, indent=2)
        os.replace(tmp, MODEL_ICONS_FILE)
        st = os.stat(MODEL_ICONS_FILE)
        _ICON_CACHE = ((st.st_mtime_ns, st.st_size), icons)
    except Exception:
        pass
