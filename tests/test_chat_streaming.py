"""Chat test endpoint: streaming passthrough + long-load keepalives.

The Chat tab's upstream can be silent for minutes (model load / prefill)
before the first token. The route must keep the SSE channel alive and
surface upstream errors as SSE frames instead of dropping the stream.
"""
import asyncio
import json

import pytest

import httpx
import routes.chat as chat_mod
from routes.chat import ChatIn


class FakeStream:
    def __init__(self, lines=(), delay=0.0, fail=None):
        self.lines = list(lines)
        self.delay = delay
        self.fail = fail
        self.status_code = 200

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def aread(self):
        return b""

    async def aiter_lines(self):
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.fail:
            raise self.fail
        for line in self.lines:
            yield line


class FakeClient:
    stream_obj = None

    def __init__(self, *args, **kwargs):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    def stream(self, *args, **kwargs):
        return FakeClient.stream_obj


async def _collect(monkeypatch, stream_obj, keepalive=0.05):
    FakeClient.stream_obj = stream_obj
    monkeypatch.setattr(httpx, "AsyncClient", FakeClient)
    monkeypatch.setattr(chat_mod, "_SSE_KEEPALIVE_SECONDS", keepalive)
    monkeypatch.setattr(chat_mod, "_gateway_url", lambda p: "http://gw")
    monkeypatch.setattr(chat_mod.access, "gate_open", lambda: True)
    monkeypatch.setattr(chat_mod.swap_manager, "status",
                        lambda: {"running": True})
    resp = await chat_mod.chat_complete(
        ChatIn(model="m", messages=[{"role": "user", "content": "hi"}]),
        request=None)
    out = []
    async for chunk in resp.body_iterator:
        out.append(chunk)
    return "".join(out)


def test_streams_sse_and_emits_keepalive_while_upstream_is_quiet(
        monkeypatch):
    stream = FakeStream(
        ['data: {"choices":[{"delta":{"content":"hi"}}]}'],
        delay=0.18)
    text = asyncio.run(_collect(monkeypatch, stream))

    assert text.startswith(": connected\n\n")
    assert ": keepalive\n\n" in text
    assert 'data: {"choices":[{"delta":{"content":"hi"}}]}' in text


def test_passes_reasoning_content_through(monkeypatch):
    stream = FakeStream([
        'data: {"choices":[{"delta":{"reasoning_content":"think"}}]}',
        'data: {"choices":[{"delta":{"content":"answer"}}]}',
    ])
    text = asyncio.run(_collect(monkeypatch, stream))

    assert '"reasoning_content":"think"' in text
    assert '"content":"answer"' in text


def test_upstream_failure_becomes_sse_error_frame(monkeypatch):
    stream = FakeStream(fail=RuntimeError("boom"))
    text = asyncio.run(_collect(monkeypatch, stream))

    assert "data: " in text
    payload = text.split("data: ", 1)[1].strip()
    assert json.loads(payload)["error"] == "swap error: boom"


def test_non_200_upstream_returns_error_frame(monkeypatch):
    stream = FakeStream()
    stream.status_code = 503

    async def _aread():
        return b'{"error":{"message":"still loading"}}'
    stream.aread = _aread
    text = asyncio.run(_collect(monkeypatch, stream))

    payload = text.split("data: ", 1)[1].strip()
    assert json.loads(payload)["error"] == "still loading"
