"""Comment-preserving settings.yaml persister (B9c pattern, DEC-14).

settings.yaml is user-documented: rewriting it with plain yaml.safe_dump
wipes every comment and commented-out key. ALL writers go through this
round-trip pair: ruamel.yaml keeps comments/quotes/key order; plain
PyYAML is only a fallback when ruamel is absent (bare install —
comments lost but unknown keys still preserved).

Extracted from app.py (DEC-14) so installers can write settings.yaml
through the SAME persister instead of plain yaml.safe_dump, which
silently destroyed all comments on every install or uninstall run.
"""

import os


def _chmod_private(path):
    """settings.yaml holds plaintext access keys; keep it owner-only."""
    if os.name != "posix":
        return
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass


def ensure_private(path):
    """Tighten permissions on an existing settings.yaml (startup migration)."""
    if os.name != "posix" or not path or not os.path.exists(path):
        return
    try:
        mode = os.stat(path).st_mode & 0o777
        if mode & 0o077:
            os.chmod(path, 0o600)
    except OSError:
        pass


def load_yaml_preserving(path):
    """settings.yaml as a mutable mapping, comments attached."""
    try:
        from ruamel.yaml import YAML
    except ImportError:  # bare install: degrade to plain PyYAML
        import yaml
        if not os.path.exists(path):
            return {}
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    y = YAML()          # rt loader by default: comments + ordering preserved
    y.preserve_quotes = True
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return y.load(f) or {}


def dump_yaml_preserving(data, path):
    """Write settings.yaml back, comments intact. Atomic via tmp+replace."""
    try:
        from ruamel.yaml import YAML
    except ImportError:  # bare install: degrade to plain PyYAML
        import yaml
        tmp = path + ".tmp"
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, default_flow_style=False, sort_keys=False)
        os.replace(tmp, path)
        _chmod_private(path)
        return
    y = YAML()
    y.preserve_quotes = True
    y.default_flow_style = False
    tmp = path + ".tmp"
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        y.dump(data, f)
    os.replace(tmp, path)
    _chmod_private(path)