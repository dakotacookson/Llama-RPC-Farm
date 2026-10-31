"""Comment-preserving settings.yaml persister (B9c pattern, DEC-14).

settings.yaml is user-documented: rewriting it with plain yaml.safe_dump
wipes every comment and commented-out key. ALL writers go through this
round-trip pair: ruamel.yaml keeps comments/quotes/key order; plain
PyYAML is only a fallback when ruamel is absent (bare install —
comments lost but unknown keys still preserved).

Extracted from app.py (DEC-14) so vllm_installer / vllm_uninstaller can
write settings.yaml through the SAME persister instead of plain
yaml.safe_dump, which silently destroyed all comments on every install
or uninstall run.
"""

import os


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
        with open(tmp, "w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, default_flow_style=False, sort_keys=False)
        os.replace(tmp, path)
        return
    y = YAML()
    y.preserve_quotes = True
    y.default_flow_style = False
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        y.dump(data, f)
    os.replace(tmp, path)