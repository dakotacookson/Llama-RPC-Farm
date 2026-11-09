"""HF collection browse: anonymous fallback for entity-scoped tokens.

A fine-grained HF token can read its owner's collections but HF returns
403 for another owner's PUBLIC collection. get_collection must retry
anonymously and only surface the token error when the anonymous call
fails too. No network: httpx.AsyncClient is faked.
"""
import asyncio

import pytest

import hf_browse


def _fake_httpx(monkeypatch, auth_status, anon_status, token="hf_test"):
    calls = []

    class FakeResp:
        def __init__(self, status):
            self.status_code = status

        def json(self):
            return {"slug": "Qwen/qwen38", "title": "Qwen3.8",
                    "items": [{"type": "model", "id": "Qwen/model"}]}

        def raise_for_status(self):
            if self.status_code >= 400:
                raise RuntimeError(f"HTTP {self.status_code}")

    class FakeClient:
        def __init__(self, *a, **kw):
            self.headers = kw.get("headers") or {}
            self.timeout = kw.get("timeout")

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def get(self, url, params=None):
            authed = "Authorization" in self.headers
            calls.append(authed)
            return FakeResp(auth_status if authed else anon_status)

    monkeypatch.setattr(hf_browse.httpx, "AsyncClient", FakeClient)
    monkeypatch.setattr(hf_browse, "_token", lambda: token)
    hf_browse._cache.clear()
    return calls


def test_public_collection_falls_back_to_anonymous(monkeypatch):
    calls = _fake_httpx(monkeypatch, auth_status=403, anon_status=200)
    out = asyncio.run(hf_browse.get_collection("Qwen/qwen38", gguf_only=False))
    assert [it["id"] for it in out["items"]] == ["Qwen/model"]
    assert calls == [True, False]  # token first, then anonymous retry


def test_private_collection_keeps_token_error(monkeypatch):
    calls = _fake_httpx(monkeypatch, auth_status=403, anon_status=404)
    with pytest.raises(RuntimeError, match="HTTP 403"):
        asyncio.run(hf_browse.get_collection("me/private", gguf_only=False))
    assert calls == [True, False]


def test_no_token_single_anonymous_call(monkeypatch):
    calls = _fake_httpx(monkeypatch, auth_status=200, anon_status=200,
                        token=None)
    out = asyncio.run(hf_browse.get_collection("Qwen/qwen38", gguf_only=False))
    assert out["items"]
    assert calls == [False]
