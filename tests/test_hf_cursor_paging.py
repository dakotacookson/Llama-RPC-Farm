"""GGUF search deep paging must use HF's cursor chain, never deprecated skip.

HF returns 400 once `skip` gets large ("deprecated ... use the links
response header"), which silently truncated the catalogue. These tests
fake the HF listing so cursor paging is exercised end-to-end.
"""
import asyncio

import hf_browse


def _model(i, name=None):
    return {
        "id": name or f"author/model-{i}",
        "author": "author",
        "downloads": i,
        "likes": 0,
        "pipeline_tag": "text-generation",
        "siblings": [{"rfilename": f"model-{i}-Q4_K_M.gguf"}],
    }


class _Resp:
    def __init__(self, batch, next_cursor):
        self._batch = batch
        self.status_code = 200
        self.headers = {}
        if next_cursor:
            self.headers["link"] = (
                f'<https://huggingface.co/api/models?cursor={next_cursor}>;'
                ' rel="next"')

    def json(self):
        return self._batch

    def raise_for_status(self):
        pass


def _install(monkeypatch, pages):
    """pages: cursor -> (batch, next_cursor). Tracks requested params."""
    seen_params = []

    class FakeClient:
        def __init__(self, *a, **kw):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def get(self, url, params=None):
            params = params or {}
            seen_params.append(dict(params))
            cur = params.get("cursor")
            batch, nxt = pages[cur]
            return _Resp(batch, nxt)

    monkeypatch.setattr(hf_browse.httpx, "AsyncClient", FakeClient)
    hf_browse._cache.clear()
    return seen_params


def test_cursor_chain_pages_without_skip(monkeypatch):
    pages = {
        None: ([_model(i) for i in range(50)], "c1"),
        "c1": ([_model(i) for i in range(50, 100)], None),
    }
    params_seen = _install(monkeypatch, pages)

    first = asyncio.run(hf_browse.search_gguf(limit=50))
    assert len(first["models"]) == 50
    assert first["next_cursor"] == "c1"
    assert first["paged_by"] == "cursor"
    assert first["has_more"] is True

    hf_browse._cache.clear()
    second = asyncio.run(hf_browse.search_gguf(limit=50, cursor="c1"))
    assert len(second["models"]) == 50
    assert second["next_cursor"] is None
    assert second["has_more"] is False

    # The retired `skip` parameter must never be sent again.
    assert all("skip" not in p for p in params_seen)
    assert params_seen[0].get("cursor") is None
    assert params_seen[1].get("cursor") == "c1"


def test_sparse_quant_advances_cursor_without_losing_rows(monkeypatch):
    # First batch has zero Q8_0 matches; the page must advance and refill
    # rather than return empty, and must keep the cursor pointing past the
    # rows it consumed.
    class _M(_Resp):
        pass

    def q8(i):
        m = _model(i)
        m["siblings"] = [{"rfilename": f"model-{i}-Q8_0.gguf"}]
        return m

    pages = {
        None: ([_model(i) for i in range(50)], "c1"),
        "c1": ([q8(50), q8(51)], "c2"),
        "c2": ([q8(52)], None),
    }
    _install(monkeypatch, pages)
    out = asyncio.run(hf_browse.search_gguf(limit=50, quant="Q8_0"))
    assert [m["id"] for m in out["models"]] == [
        "author/model-50", "author/model-51"]
    assert out["next_cursor"] == "c2"
