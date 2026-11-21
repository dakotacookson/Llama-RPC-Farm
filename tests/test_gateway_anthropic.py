"""Gateway :8090 — Anthropic Messages API support.

`/v1/messages` and `/v1/messages/count_tokens` must cross the same
access-key + model-allowlist gate as OpenAI generate paths, and streamed
Anthropic usage must be recorded with exact token counts.
"""
import asyncio
import json
import os
import sys

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import pytest
from starlette.requests import Request
from starlette.testclient import TestClient

import swap_gateway as gw

SCOPED = "s" * 32
ADMIN = "a" * 32


@pytest.fixture
def key_table(monkeypatch):
    table = {
        SCOPED: {"name": "client", "models": {"m1"}, "admin": False},
        ADMIN: {"name": "admin", "models": set(), "admin": True},
    }
    monkeypatch.setattr(gw, "access_keys", lambda: table)
    return table


class FakeResp:
    def __init__(self, body=b'{"ok":true}', ctype="application/json"):
        self.status_code = 200
        self.headers = {"content-type": ctype}
        self._body = body

    async def aread(self):
        return self._body

    async def aclose(self):
        pass

    async def aiter_bytes(self):
        yield self._body


class FakeClient:
    def __init__(self, resp):
        self.resp = resp
        self.requests = []

    def build_request(self, *a, **kw):
        self.requests.append((a, kw))
        return object()

    async def send(self, *a, **kw):
        return self.resp


def _client_for(monkeypatch, resp):
    fake = FakeClient(resp)
    monkeypatch.setattr(gw, "client", fake)
    monkeypatch.setattr(gw, "request_key_name", lambda r: "")
    return fake


def test_anthropic_paths_are_model_scoped():
    assert gw._is_model_path("/v1/messages")
    assert gw._is_model_path("/v1/messages/count_tokens")
    assert gw._is_generate_path("/v1/messages")
    assert not gw._is_generate_path("/v1/messages/count_tokens")


def test_anthropic_usage_includes_cache_tokens():
    got = gw._anthropic_usage({
        "input_tokens": 4,
        "cache_read_input_tokens": 12,
        "cache_creation_input_tokens": 1,
        "output_tokens": 20,
    })
    assert got == {"prompt_tokens": 17, "completion_tokens": 20}


def test_request_key_accepts_anthropic_x_api_key():
    req = Request({"type": "http", "method": "POST", "path": "/v1/messages",
                   "headers": [(b"x-api-key", b"sk-anth")],
                   "client": ("192.0.2.5", 1234)})
    assert gw.request_key(req) == "sk-anth"


def test_anthropic_error_shape(key_table, monkeypatch):
    _client_for(monkeypatch, FakeResp())
    c = TestClient(gw.app)
    r = c.post("/v1/messages",
               headers={"content-type": "application/json"},
               json={"model": "m1", "messages": []})
    assert r.status_code == 401
    body = r.json()
    assert body["type"] == "error"
    assert body["error"]["type"] == "authentication_error"


def test_anthropic_model_allowlist_is_enforced(key_table, monkeypatch):
    _client_for(monkeypatch, FakeResp())
    c = TestClient(gw.app)
    r = c.post("/v1/messages",
               headers={"content-type": "application/json",
                        "x-api-key": SCOPED},
               json={"model": "not-granted", "messages": []})
    assert r.status_code == 404
    body = r.json()
    assert body["type"] == "error"
    assert body["error"]["type"] == "not_found_error"


def test_anthropic_allowed_model_reaches_upstream(key_table, monkeypatch):
    _client_for(monkeypatch, FakeResp(b'{"type":"message"}'))

    async def _canon(mid):
        return mid
    monkeypatch.setattr(gw, "canonical_model", _canon)

    c = TestClient(gw.app)
    r = c.post("/v1/messages",
               headers={"content-type": "application/json",
                        "x-api-key": SCOPED},
               json={"model": "m1", "messages": []})
    assert r.status_code == 200
    assert r.json() == {"type": "message"}


def test_count_tokens_is_gated_but_not_counted_as_generate(key_table,
                                                           monkeypatch):
    fake = _client_for(monkeypatch, FakeResp(b'{"input_tokens":7}'))

    async def _canon(mid):
        return mid
    monkeypatch.setattr(gw, "canonical_model", _canon)

    c = TestClient(gw.app)
    r = c.post("/v1/messages/count_tokens",
               headers={"content-type": "application/json",
                        "x-api-key": SCOPED},
               json={"model": "m1", "messages": []})
    assert r.status_code == 200
    assert fake.requests


def test_count_tokens_rejects_ungranted_model(key_table, monkeypatch):
    _client_for(monkeypatch, FakeResp())

    async def _canon(mid):
        return mid
    monkeypatch.setattr(gw, "canonical_model", _canon)

    c = TestClient(gw.app)
    r = c.post("/v1/messages/count_tokens",
               headers={"content-type": "application/json",
                        "x-api-key": SCOPED},
               json={"model": "other", "messages": []})
    assert r.status_code == 404


def _run_counting(monkeypatch, chunks):
    import usage_collector
    seen = {}

    def _record(name, path, model, status, dur, tokens, estimated=False):
        seen.update(name=name, path=path, model=model, status=status,
                    dur=dur, tokens=tokens, estimated=estimated)
    monkeypatch.setattr(usage_collector, "record", _record)

    class R:
        async def aiter_bytes(self):
            for c in chunks:
                yield c

    async def _go():
        out = []
        async for c in gw._counting_stream(
                R(), {"name": "client", "path": "v1/messages",
                      "model": "m1", "status": 200, "dur": 0.1}):
            out.append(c)
        return b"".join(out)
    asyncio.run(_go())
    return seen


def test_anthropic_stream_usage_records_exact_tokens(monkeypatch):
    chunks = [
        b'event: message_start\ndata: {"type":"message_start","message":'
        b'{"usage":{"input_tokens":4,"cache_read_input_tokens":12,'
        b'"output_tokens":0}}}\n\n',
        b'event: content_block_delta\ndata: {"type":"content_block_delta",'
        b'"delta":{"type":"text_delta","text":"hi"}}\n\n',
        b'event: message_delta\ndata: {"type":"message_delta",'
        b'"usage":{"output_tokens":20}}\n\n',
    ]
    seen = _run_counting(monkeypatch, chunks)
    assert seen["tokens"] == {"prompt_tokens": 16, "completion_tokens": 20}
    assert seen["estimated"] is False
    assert seen["path"] == "v1/messages"


def test_openai_timings_still_win(monkeypatch):
    chunks = [
        b'data: {"choices":[{"delta":{"content":"hi"}}]}\n\n',
        b'data: {"choices":[{"delta":{}}],"timings":'
        b'{"predicted_n":9,"prompt_n":3,'
        b'"predicted_per_second":50.0}}\n\n',
    ]
    seen = _run_counting(monkeypatch, chunks)
    assert seen["tokens"] == {"prompt_tokens": 3, "completion_tokens": 9}
    assert seen["estimated"] is False


def test_request_key_accepts_bearer_for_anthropic():
    req = Request({"type": "http", "method": "POST", "path": "/v1/messages",
                   "headers": [(b"authorization", b"Bearer sk-anth")],
                   "client": ("192.0.2.5", 1234)})
    assert gw.request_key(req) == "sk-anth"


def test_usage_kind_maps_messages_to_chat():
    import usage_collector
    assert usage_collector.kind_for("v1/messages") == "chat"
    assert usage_collector.kind_for("/v1/messages/count_tokens") == "chat"


class FakeStreamResp:
    def __init__(self, chunks, ctype="text/event-stream"):
        self.status_code = 200
        self.headers = {"content-type": ctype}
        self._chunks = chunks

    async def aclose(self):
        pass

    async def aread(self):
        return b"".join(self._chunks)

    async def aiter_bytes(self):
        for c in self._chunks:
            yield c


ANTHROPIC_STREAM = [
    b'event: message_start\ndata: {"type":"message_start","message":'
    b'{"usage":{"input_tokens":4,"cache_read_input_tokens":12,'
    b'"output_tokens":0}}}\n\n',
    b'event: content_block_delta\ndata: {"type":"content_block_delta",'
    b'"delta":{"type":"text_delta","text":"hi"}}\n\n',
    b'event: message_delta\ndata: {"type":"message_delta",'
    b'"usage":{"output_tokens":20}}\n\n',
    b'event: message_stop\ndata: {"type":"message_stop"}\n\n',
]


def _capture_usage(monkeypatch):
    import usage_collector
    rows = []

    def _record(name, path, model, status, dur, tokens, estimated=False):
        rows.append({"name": name, "path": path, "model": model,
                     "status": status, "dur": dur, "tokens": tokens,
                     "estimated": estimated})
    monkeypatch.setattr(usage_collector, "record", _record)
    return rows


def test_anthropic_stream_through_proxy_bytes_and_usage(key_table,
                                                        monkeypatch):
    """Full proxy path: key gate → upstream SSE → unchanged bytes → one
    exact usage row."""
    _client_for(monkeypatch, FakeStreamResp(ANTHROPIC_STREAM))
    monkeypatch.setattr(gw, "request_key_name", lambda r: "client")
    rows = _capture_usage(monkeypatch)

    async def _canon(mid):
        return mid
    monkeypatch.setattr(gw, "canonical_model", _canon)

    c = TestClient(gw.app)
    with c.stream("POST", "/v1/messages",
                  headers={"content-type": "application/json",
                           "x-api-key": SCOPED,
                           "anthropic-version": "2023-06-01"},
                  json={"model": "m1", "stream": True,
                        "messages": []}) as r:
        assert r.status_code == 200
        body = b"".join(r.iter_bytes())

    assert b"event: message_start" in body
    assert b'"type":"message_stop"' in body
    assert len(rows) == 1
    assert rows[0]["path"] == "v1/messages"
    assert rows[0]["model"] == "m1"
    assert rows[0]["tokens"] == {"prompt_tokens": 16,
                                 "completion_tokens": 20}
    assert rows[0]["estimated"] is False


def test_anthropic_nonstream_proxy_records_input_output_tokens(
        key_table, monkeypatch):
    body = (b'{"type":"message","usage":{"input_tokens":9,'
            b'"output_tokens":21}}')
    _client_for(monkeypatch, FakeResp(body))
    monkeypatch.setattr(gw, "request_key_name", lambda r: "client")
    rows = _capture_usage(monkeypatch)

    async def _canon(mid):
        return mid
    monkeypatch.setattr(gw, "canonical_model", _canon)

    c = TestClient(gw.app)
    r = c.post("/v1/messages",
               headers={"content-type": "application/json",
                        "x-api-key": SCOPED},
               json={"model": "m1", "messages": []})
    assert r.status_code == 200
    assert len(rows) == 1
    assert rows[0]["tokens"] == {"prompt_tokens": 9,
                                 "completion_tokens": 21}


def test_count_tokens_creates_no_usage_row(key_table, monkeypatch):
    _client_for(monkeypatch, FakeResp(b'{"input_tokens":7}'))
    monkeypatch.setattr(gw, "request_key_name", lambda r: "client")
    rows = _capture_usage(monkeypatch)

    async def _canon(mid):
        return mid
    monkeypatch.setattr(gw, "canonical_model", _canon)

    c = TestClient(gw.app)
    r = c.post("/v1/messages/count_tokens",
               headers={"content-type": "application/json",
                        "x-api-key": SCOPED},
               json={"model": "m1", "messages": []})
    assert r.status_code == 200
    assert rows == []


def test_admin_key_can_use_messages(key_table, monkeypatch):
    _client_for(monkeypatch, FakeResp(b'{"type":"message"}'))

    async def _canon(mid):
        return mid
    monkeypatch.setattr(gw, "canonical_model", _canon)

    c = TestClient(gw.app)
    r = c.post("/v1/messages",
               headers={"content-type": "application/json",
                        "authorization": f"Bearer {ADMIN}"},
               json={"model": "anything", "messages": []})
    assert r.status_code == 200


def test_quota_exhaustion_is_429_openai_and_anthropic(key_table, monkeypatch):
    _client_for(monkeypatch, FakeResp())
    import usage_collector
    monkeypatch.setattr(usage_collector, "check_quota",
                        lambda name, limits: {
                            "period": "day", "used": 5, "limit": 5,
                            "reset_in": 30, "reset_at": 9999999999})

    async def _canon(mid):
        return mid
    monkeypatch.setattr(gw, "canonical_model", _canon)
    c = TestClient(gw.app)
    r = c.post("/v1/chat/completions",
               headers={"content-type": "application/json",
                        "authorization": f"Bearer {SCOPED}"},
               json={"model": "m1", "messages": []})
    assert r.status_code == 429
    assert r.headers.get("retry-after") == "30"

    r = c.post("/v1/messages",
               headers={"content-type": "application/json",
                        "x-api-key": SCOPED},
               json={"model": "m1", "messages": []})
    assert r.status_code == 429
    assert r.json()["error"]["type"] == "rate_limit_error"
