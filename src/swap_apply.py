"""Cascade a config change into the running llama-swap deployment.

Major mutations (machine/group/model changes) call this so the generated
swap config/scripts are refreshed immediately and, when the daemon is
running, it reloads in the background. Best-effort: failures are reported
in the returned dict and never break the caller's write.
"""
import asyncio

from fastapi.concurrency import run_in_threadpool

import swap_manager


async def apply_swap_config(reason="", reload=True):
    result = {"pushed": False, "models": [], "warnings": [],
              "reloading": False}
    try:
        status = await run_in_threadpool(swap_manager.status)
    except Exception as exc:
        print(f"[swap-apply] status failed ({reason}): {exc}")
        return {**result, "error": str(exc)}
    if not status.get("running"):
        return {**result, "reason": "swap not running — Launch All to apply"}
    try:
        report = await run_in_threadpool(swap_manager.push_scripts)
    except Exception as exc:
        print(f"[swap-apply] push failed ({reason}): {exc}")
        return {**result, "error": str(exc)}
    result.update(pushed=True,
                  models=report.get("models", []) or [],
                  warnings=report.get("warnings", []) or [])
    if reload:
        async def _reload():
            try:
                await run_in_threadpool(swap_manager.reload_daemon_async)
            except Exception as exc:
                print(f"[swap-apply] reload failed ({reason}): {exc}")

        try:
            asyncio.get_running_loop().create_task(_reload())
            result["reloading"] = True
        except RuntimeError:
            pass
    return result
