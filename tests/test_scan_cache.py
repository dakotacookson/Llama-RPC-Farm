"""The Source scan caches the directory walk; dirty refreshes bypass it."""
import asyncio


class _Engine:
    def ui_manifest(self):
        return {"downloads": {"formats": ["gguf"]}}

    def filter_models(self, models):
        return models


def test_scan_results_are_cached_until_forced(monkeypatch, tmp_path):
    import routes.inference as inf

    inf._SCAN_CACHE.clear()
    calls = {"n": 0}

    def fake_scan(path, recursive=False):
        calls["n"] += 1
        return [], []

    monkeypatch.setattr(inf.model_parser, "scan_directory", fake_scan)
    monkeypatch.setattr(inf.path_resolver, "resolve", lambda p: p)
    monkeypatch.setattr(inf.path_resolver, "to_alias", lambda p: p)
    monkeypatch.setattr(inf.cfg, "MODELS_DIR", str(tmp_path))
    monkeypatch.setattr(inf.cfg, "list_models", lambda: [])
    monkeypatch.setattr(inf.engine_registry, "get_engine", lambda: _Engine())
    monkeypatch.setattr(inf.access, "gate_open", lambda: True)

    asyncio.run(inf.scan_models(str(tmp_path), False))
    asyncio.run(inf.scan_models(str(tmp_path), False))
    assert calls["n"] == 1, "second identical scan should hit the cache"

    asyncio.run(inf.scan_models(str(tmp_path), False, refresh=True))
    assert calls["n"] == 2, "refresh=True must bypass the cache"
