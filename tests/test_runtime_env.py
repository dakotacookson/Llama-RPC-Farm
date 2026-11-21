"""Packaged and source builds both load config/runtime.env."""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

import runtime_paths


def test_load_runtime_env_source_and_packaged_layout(monkeypatch, tmp_path):
    monkeypatch.setattr(runtime_paths, "data_dir", lambda: str(tmp_path))
    monkeypatch.delenv("FARM_SKIP_RUNTIME_ENV", raising=False)
    monkeypatch.delenv("FARM_REQUIRE_AUTH", raising=False)
    monkeypatch.delenv("FARM_TRUSTED_PROXIES", raising=False)
    cfg = tmp_path / "config"
    cfg.mkdir()
    (cfg / "runtime.env").write_text(
        "# comment\nFARM_REQUIRE_AUTH=1\n"
        "export FARM_TRUSTED_PROXIES='127.0.0.1'\n",
        encoding="utf-8")
    runtime_paths.load_runtime_env()
    assert os.environ["FARM_REQUIRE_AUTH"] == "1"
    assert os.environ["FARM_TRUSTED_PROXIES"] == "127.0.0.1"


def test_real_environment_wins(monkeypatch, tmp_path):
    monkeypatch.setattr(runtime_paths, "data_dir", lambda: str(tmp_path))
    monkeypatch.delenv("FARM_SKIP_RUNTIME_ENV", raising=False)
    monkeypatch.setenv("FARM_REQUIRE_AUTH", "0")
    cfg = tmp_path / "config"
    cfg.mkdir()
    (cfg / "runtime.env").write_text("FARM_REQUIRE_AUTH=1\n",
                                     encoding="utf-8")
    runtime_paths.load_runtime_env()
    assert os.environ["FARM_REQUIRE_AUTH"] == "0"
