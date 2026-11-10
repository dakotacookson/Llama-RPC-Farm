"""settings.yaml location + comment-preserving load/save + view lookup."""
import os

import settings_persister
from runtime_paths import data_dir

BASE_DIR = data_dir()
SETTINGS_FILE = os.path.join(BASE_DIR, "config", "settings.yaml")


def _view_name_for_ip(ip: str):
    """Reverse lookup: which configured view (settings.yaml views:) an IP
    maps to, by ACCESS-TABLE name — used by the save-model auto-join (a
    restricted user configuring a new model keeps seeing it)."""
    ip = (ip or "").strip()
    for name, ent in _access_views_table().items():
        if ip in ent["ips"]:
            return name
    return None


def _access_views_table():
    import access as _a
    return _a._views_table()


_SETTINGS_CACHE = {}


def _settings_key(path):
    try:
        st = os.stat(path)
        return (path, st.st_mtime_ns, st.st_size)
    except OSError:
        return None


def _load_settings_yaml_preserving(path):
    """settings.yaml as a mutable mapping, comments attached (mtime-cached)."""
    key = _settings_key(path)
    if key is not None:
        hit = _SETTINGS_CACHE.get(path)
        if hit and hit[0] == key:
            return hit[1]
    data = settings_persister.load_yaml_preserving(path)
    if key is not None:
        _SETTINGS_CACHE[path] = (key, data)
    return data


def _dump_settings_yaml_preserving(data, path):
    """Write settings.yaml back, comments intact. Atomic via tmp+replace."""
    settings_persister.dump_yaml_preserving(data, path)
    key = _settings_key(path)
    if key is not None:
        _SETTINGS_CACHE[path] = (key, data)
