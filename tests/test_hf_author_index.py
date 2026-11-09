"""Full GGUF publisher index: cursor parsing, merge, and company listing."""
import asyncio

import hf_browse


def test_index_next_cursor_extracts_cursor():
    link = ('<https://huggingface.co/api/models?filter=gguf&limit=1000'
            '&cursor=abc%3D%3D>; rel="next"')
    assert hf_browse._index_next_cursor(link) == "abc=="
    assert hf_browse._index_next_cursor("") is None
    assert hf_browse._index_next_cursor('<x>; rel="prev"') is None


def test_merge_author_batch_accumulates():
    state = {"authors": {}, "models": 0}
    hf_browse._merge_author_batch(state, [
        {"id": "a/one", "author": "a", "downloads": 10, "likes": 1},
        {"id": "a/two", "author": "a", "downloads": 5, "likes": 2},
        {"id": "b/three", "downloads": 100},
        {"id": ""},
    ])
    assert state["models"] == 3
    assert state["authors"]["a"] == {"count": 2, "downloads": 15, "likes": 3}
    assert state["authors"]["b"]["count"] == 1


def test_all_companies_uses_full_index_and_exhaustive_search(monkeypatch):
    snap = {
        "authors": {
            "mradermacher": {"count": 72901, "downloads": 9, "likes": 0},
            "Qwen": {"count": 3, "downloads": 1, "likes": 0},
            "qwen-helper": {"count": 7, "downloads": 2, "likes": 0},
        },
        "models": 72911, "done": True, "running": False,
        "error": None, "updated": None,
    }
    monkeypatch.setattr(hf_browse, "ensure_author_index", lambda: snap)
    hf_browse._cache.clear()

    all_companies = asyncio.run(hf_browse.list_gguf_companies(limit=10))
    assert {c["author"] for c in all_companies["companies"]} == {
        "mradermacher", "Qwen", "qwen-helper"}
    assert all_companies["totalUnique"] == 3
    assert all_companies["complete"] is True

    # Substring search must reach any indexed author, not just the top N.
    hf_browse._cache.clear()
    found = asyncio.run(hf_browse.list_gguf_companies(search="qwen", limit=10))
    assert {c["author"] for c in found["companies"]} == {"Qwen", "qwen-helper"}


def test_partial_index_is_not_cached(monkeypatch):
    snap = {"authors": {"a": {"count": 1, "downloads": 0, "likes": 0}},
            "models": 1, "done": False, "running": True,
            "error": None, "updated": None}
    monkeypatch.setattr(hf_browse, "ensure_author_index", lambda: snap)
    hf_browse._cache.clear()
    first = asyncio.run(hf_browse.list_gguf_companies(limit=10))
    assert first["scanning"] is True and first["complete"] is False
    assert not hf_browse._cache, "partial company list must not be cached"
