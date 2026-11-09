"""Collection picker hides collections with no GGUF repos."""
import asyncio

import hf_browse


COLLECTIONS = [
    {"slug": "o/gguf-col", "title": "GGUF collection", "items": [
        {"type": "model", "id": "author/gguf-model"}]},
    {"slug": "Qwen/source-col", "title": "Qwen3.8", "items": [
        {"type": "model", "id": "Qwen/Qwen3.8-27B"}]},
    {"slug": "o/empty-col", "title": "Empty", "items": []},
]


def _install(monkeypatch, verdicts):
    class FakeResp:
        status_code = 200

        @staticmethod
        def json():
            return COLLECTIONS

        @staticmethod
        def raise_for_status():
            pass

    class FakeClient:
        def __init__(self, *a, **kw):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def get(self, url, params=None):
            return FakeResp()

    async def fake_count(repo_id):
        return verdicts.get(repo_id)

    monkeypatch.setattr(hf_browse.httpx, "AsyncClient", FakeClient)
    monkeypatch.setattr(hf_browse, "_repo_gguf_count", fake_count)
    hf_browse._cache.clear()


def test_non_gguf_collections_are_hidden(monkeypatch):
    _install(monkeypatch, {"author/gguf-model": 3, "Qwen/Qwen3.8-27B": 0})
    out = asyncio.run(hf_browse.list_collections("Qwen", limit=100))
    assert [c["slug"] for c in out["collections"]] == ["o/gguf-col"]
    assert out["gguf_only"] is True


def test_unknown_repo_fails_open(monkeypatch):
    _install(monkeypatch, {"author/gguf-model": 0, "Qwen/Qwen3.8-27B": None})
    out = asyncio.run(hf_browse.list_collections("Qwen", limit=100))
    assert [c["slug"] for c in out["collections"]] == ["Qwen/source-col"]


def test_gguf_only_false_returns_everything(monkeypatch):
    _install(monkeypatch, {})
    out = asyncio.run(hf_browse.list_collections("Qwen", limit=100,
                                                 gguf_only=False))
    assert len(out["collections"]) == 3
