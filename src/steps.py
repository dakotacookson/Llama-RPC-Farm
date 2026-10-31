"""Shared orchestration scaffolding: run named steps, collect a report.

Both the single-model live launch and the llama-swap launch use this,
so step reports evolve in one place instead of drifting apart.
"""

def run_steps(report, steps, catch=(Exception,), stop_on_fail=True):
    """Execute [(name, fn, *args), ...] appending to report["steps"].

    Failures are detected per-step semantics: an "error" key, a truthy
    "failed" key, or a falsy success indicator (killed / running /
    valid) when that key is present. stop_on_fail=True aborts the
    remaining steps on the first failure. Returns the report dict.
    """
    report.setdefault("steps", [])
    report["failed"] = False
    for name, fn, *args in steps:
        try:
            result = fn(*args)
        except catch as e:
            result = {"error": str(e)}
        if not isinstance(result, dict):
            result = {"result": result}
        report["steps"].append({"step": name, **result})
        failed = bool(result.get("error") or result.get("failed"))
        for ok_key in ("killed", "running", "valid"):
            if ok_key in result and not result[ok_key]:
                failed = True
        if failed:
            report["failed"] = True
            if stop_on_fail:
                break
    return report