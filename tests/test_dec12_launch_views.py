"""DEC-12 (closes T4-G8): Launch tab per-engine views.

The Launch tab renders engine-native views driven by the adapter's
launch semantics (ui_manifest.launch.mode, DEC-06) instead of engine-name
gates:

* rpc_cluster  — llama.cpp's own copy (RPC peers prose + per-model grid)
* swap_only    — vLLM's own view (daemon-only prose; no RPC grid)

The frontend has no harness, so these tests pin the greppable contract:
the template carries data-launch-view spans/cards keyed by launch.mode,
applyEngineHints renders them from the manifest's launch.mode, and the
'# warnings' preview header is retired (DEC-08/09 retired save-time
warnings; the preview shows scripts + yaml only).
"""
import os

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _src(rel):
    with open(os.path.join(HERE, rel), encoding="utf-8") as f:
        return f.read()


def test_template_has_launch_view_nodes():
    """Both engines' Launch-tab views exist in the template, keyed by
    launch.mode (rpc_cluster / swap_only) — the same vocabulary the
    adapters' ui_manifest.launch.mode emits."""
    tpl = _src(os.path.join("ui", "templates", "index.html"))
    views = [l for l in tpl.splitlines() if "data-launch-view=" in l]
    assert any('data-launch-view="rpc_cluster"' in l for l in views), views
    assert any('data-launch-view="swap_only"' in l for l in views), views
    # the vLLM view is its own card (engine-native view, not a relabel)
    assert any('id="vllm-launch-view"' in l
               and 'data-launch-view="swap_only"' in l for l in views), views


def test_apply_engine_hints_renders_launch_views_from_manifest_mode():
    """applyEngineHints gates the launch views on ui_manifest.launch.mode
    (the adapter's launch semantics) — data-launch-view, not an
    engine-name comparison."""
    appjs = _src(os.path.join("ui", "static", "app.js"))
    fn = appjs.split("function applyEngineHints", 1)[1]
    assert 'el.dataset.launchView !== launchMode' in fn, \
        "launch views must render from the manifest's launch.mode"
    # launchMode itself comes from the manifest's launch.mode (DEC-02),
    # with the engine-name fallback only for stale caches
    assert "(hints.launch || {}).mode" in fn


def test_vllm_launch_view_is_swap_only_rpc_view_is_llama_cpp():
    """The adapters declare their launch modes: vLLM is swap_only (models
    serve THROUGH llama-swap; direct /api/launch 409s), llama.cpp is
    rpc_cluster (peers + per-model grid). These drive the template's
    data-launch-view keys — one grammar, both sides."""
    import vllm_manager
    from engine_registry import LlamaCppAdapter
    assert vllm_manager.VllmAdapter().ui_manifest()["launch"]["mode"] \
        == "swap_only"
    assert LlamaCppAdapter().ui_manifest()["launch"]["mode"] \
        == "rpc_cluster"


def test_preview_warnings_header_retired():
    """DEC-08/09 retired save-time warnings, so the Launch tab's preview
    no longer prefixes '# warnings' noise — the output is scripts +
    yaml only. The backend plan still carries warnings (host-gate
    errors), surfaced where the operator acts, not in the preview."""
    appjs = _src(os.path.join("ui", "static", "app.js"))
    preview = appjs.split("swap-preview", 1)[1].split("swap-launch", 1)[0]
    assert "# warnings" not in preview
    assert 'r.scripts.join' in preview and "r.yaml" in preview
    # and the retired header is gone from app.js entirely
    assert '"# warnings:' not in appjs