"""DEC-14: installer/uninstaller settings writes preserve comments (B9c).

Bug (found in T1): vllm_installer._update_settings and
vllm_uninstaller._clear_settings_key wrote settings.yaml with plain
yaml.safe_dump, bypassing the comment-preserving ruamel persister
app.py uses (B9c) — so every vllm install or uninstall silently wiped
all comments and commented-out keys from config/settings.yaml (the
B1 engine/vllm docs block). Both now route through the SAME persister
(src/settings_persister.py, extracted from app.py in DEC-14).

No real hosts, no real subprocesses; settings files live in tmp_path —
the repo's own config/settings.yaml is never read or written.
"""
import os
import sys

import yaml

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

COMMENTS = """\
# farm settings — comments here are load-bearing docs (B1)
model_dir: store://llm-models  # inline comment survives too

# engine/vllm block (B9c):
engine: llama
vllm:
  delivery: venv
  # venv_bin: commented-out default
"""
COMMENTS_VENV = COMMENTS.replace(
    "  # venv_bin: commented-out default",
    "  venv_bin: ~/vllm-env/bin/vllm")


def write(path, text):
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def read(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


# ---------------- installer: _update_settings ----------------

def test_update_settings_preserves_comments(tmp_path):
    import vllm_installer
    sp = str(tmp_path / "settings.yaml")
    write(sp, COMMENTS)
    res = vllm_installer._update_settings(sp, "~/vllm-env/bin/vllm")
    assert res["detail"].endswith("~/vllm-env/bin/vllm")
    text = read(sp)
    assert "# farm settings — comments here are load-bearing docs (B1)" in text
    assert "# venv_bin: commented-out default" in text
    assert "model_dir: store://llm-models" in text
    data = yaml.safe_load(read(sp))
    assert data["vllm"]["venv_bin"] == "~/vllm-env/bin/vllm"
    assert data["engine"] == "llama"


def test_update_settings_noop_when_already_set(tmp_path):
    import vllm_installer
    sp = str(tmp_path / "settings.yaml")
    write(sp, COMMENTS_VENV)
    before = read(sp)
    res = vllm_installer._update_settings(sp, "~/vllm-env/bin/vllm")
    assert "already" in res["detail"]
    assert read(sp) == before  # untouched on the no-op path


def test_update_settings_missing_file(tmp_path):
    import vllm_installer
    sp = str(tmp_path / "settings.yaml")
    res = vllm_installer._update_settings(sp, "~/vllm-env/bin/vllm")
    data = yaml.safe_load(read(sp))
    assert data["vllm"]["venv_bin"] == "~/vllm-env/bin/vllm"


# ---------------- uninstaller: _clear_settings_key ----------------

def test_clear_settings_key_preserves_comments(tmp_path):
    from vllm_uninstaller import _clear_settings_key
    sp = str(tmp_path / "settings.yaml")
    write(sp, COMMENTS_VENV)
    res = _clear_settings_key(sp, "~/vllm-env")
    assert "cleared" in res["detail"]
    text = read(sp)
    assert "# farm settings — comments here are load-bearing docs (B1)" in text
    # note: the "# venv_bin: ..." comment line was attached to the popped
    # key, so ruamel drops it with the key — that's the expected round-trip
    # behavior; every OTHER comment survives.
    assert "# engine/vllm block (B9c):" in text
    data = yaml.safe_load(read(sp))
    assert "venv_bin" not in data["vllm"]
    assert data["engine"] == "llama"


def test_clear_settings_key_foreign_pointer_untouched(tmp_path):
    from vllm_uninstaller import _clear_settings_key
    sp = str(tmp_path / "settings.yaml")
    write(sp, COMMENTS_VENV)
    res = _clear_settings_key(sp, "~/other-venv")
    assert "elsewhere" in res["detail"]
    data = yaml.safe_load(read(sp))
    assert data["vllm"]["venv_bin"] == "~/vllm-env/bin/vllm"


def test_clear_settings_key_idempotent(tmp_path):
    from vllm_uninstaller import _clear_settings_key
    sp = str(tmp_path / "settings.yaml")
    write(sp, COMMENTS)
    res = _clear_settings_key(sp, "~/vllm-env")
    assert "absent" in res["detail"] or "already clear" in res["detail"]


# ---------------- shared persister module ----------------

def test_persister_roundtrip_preserves_comments(tmp_path):
    import settings_persister
    sp = str(tmp_path / "settings.yaml")
    write(sp, COMMENTS)
    data = settings_persister.load_yaml_preserving(sp)
    data["vllm"]["venv_bin"] = "~/vllm-env/bin/vllm"
    settings_persister.dump_yaml_preserving(data, sp)
    text = read(sp)
    assert "# farm settings — comments here are load-bearing docs (B1)" in text
    assert yaml.safe_load(text)["vllm"]["venv_bin"] == "~/vllm-env/bin/vllm"


def test_persister_missing_file(tmp_path):
    import settings_persister
    assert settings_persister.load_yaml_preserving(
        str(tmp_path / "nope.yaml")) == {}