"""Major config mutations cascade into the swap deployment."""
import asyncio


def test_apply_pushes_and_reloads(monkeypatch):
    import swap_apply
    import swap_manager

    calls = []

    async def fake_threadpool(fn, *args, **kwargs):
        return fn(*args, **kwargs)

    monkeypatch.setattr(swap_apply, "run_in_threadpool", fake_threadpool)
    monkeypatch.setattr(swap_manager, "status", lambda: {"running": True})
    monkeypatch.setattr(swap_manager, "push_scripts",
                        lambda: calls.append("push") or {
                            "models": ["m"], "warnings": ["w"]})
    monkeypatch.setattr(swap_manager, "reload_daemon_async",
                        lambda: calls.append("reload") or {"running": True})

    async def go():
        result = await swap_apply.apply_swap_config("test", reload=True)
        await asyncio.sleep(0)
        return result

    result = asyncio.run(go())
    assert result["pushed"] is True
    assert result["models"] == ["m"] and result["warnings"] == ["w"]
    assert result["reloading"] is True
    assert calls == ["push", "reload"]


def test_apply_noop_when_swap_down(monkeypatch):
    import swap_apply
    import swap_manager

    monkeypatch.setattr(swap_manager, "status", lambda: {"running": False})
    monkeypatch.setattr(swap_manager, "push_scripts",
                        lambda: (_ for _ in ()).throw(
                            AssertionError("must not push")))
    result = asyncio.run(swap_apply.apply_swap_config("test"))
    assert result["pushed"] is False
    assert "reason" in result
