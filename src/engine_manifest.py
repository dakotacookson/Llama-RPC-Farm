"""UI field manifest protocol (AUDIT-T6 §1.5, DEC-02).

The model editor, launch page, and downloads panes render from a
per-adapter field manifest instead of toggling hardcoded DOM nodes.
This module owns the protocol + the ui_hints compatibility shim so
engine_registry.py stays out of the Cards-1/2/6 hotspot (spec CARD 2
NOTE: "put Protocol additions in a new module engine_manifest.py to
avoid the hotspot entirely — recommended").

Design constraints carried from the spec:

* `ui_manifest()` is the COMPLETE field inventory the frontend renders
  from — sections/fields/chips are emitted (or skipped) from the
  manifest, not per-element toggled (T4-G10; Cards 10/11 build on it).
* `ui_hints()` survives as a DEPRECATED shim (spec §2.2 step 3 "keep":
  no consumer is migrated in this wave): the GET /api/settings payload
  and tests still carry the legacy {labels, hidden_panels, hidden_tabs,
  flag_presets} view, derived from the manifest so the two can never
  drift. The key retires in Card 15 (dead-key sweep).
* `ui_manifest()` must be self-contained — importing only typing here,
  so neither engine_registry nor vllm_manager gains an import edge.
"""
from typing import Protocol, runtime_checkable


@runtime_checkable
class EngineUxProtocol(Protocol):
    """UI-vocabulary slice of the adapter protocol (AUDIT-T6 §1.3
    `ui_manifest` + the deprecated `ui_hints` shim). Mixins below provide
    the shared defaults; adapters override `ui_manifest()`."""

    def ui_manifest(self) -> dict:
        """Complete field inventory the model editor + downloads + launch
        pages render from. Structure (§1.5):

            labels:         {server, peers, parallelism, probe}
            hidden_panels:  [str]   llama.cpp-only panels (vllm lists them)
            hidden_tabs:    [str]   engine-exclusive test tabs
            model_editor:   {sections: [{id, flag_keys?, fields?}], ...}
            flag_presets:   {name: {flag, placeholder?, choices?, note?}}
            launch:         {mode, probe_label}
            downloads:      {formats, browse_source}
        """
        ...

    def ui_hints(self) -> dict:
        """DEPRECATED legacy view — kept only because GET /api/settings
        and the test suite still speak it (retires in Card 15). Derived
        from ui_manifest() so the two views cannot drift."""
        ...


class UiHintsMixin:
    """Shared adapter defaults: the manifest's legacy projection + the
    llama.cpp-shaped empty manifest. Engines with their own vocabulary
    override ui_manifest() (VllmAdapter does); UiHintsMixin.ui_hints()
    keeps projecting it."""

    def ui_manifest(self) -> dict:
        """Default (llama.cpp-shaped) manifest: every section the editor
        knows about is listed, nothing hidden, generic flags editing."""
        return {
            "labels": {},
            "hidden_panels": [],
            "hidden_tabs": [],
            "model_editor": {
                "sections": [
                    {"id": "flags"},
                    {"id": "tensor_split"},
                    {"id": "rpc_peers"},
                    {"id": "rpc_launch"},
                    {"id": "tuner"},
                ],
            },
            "flag_presets": {},
            "launch": {"mode": "rpc_cluster", "probe_label": "RPC port"},
            "downloads": {"formats": ["gguf"], "browse_source": "gguf_search"},
        }

    def ui_hints(self) -> dict:
        """Legacy view of the manifest (DEPRECATED shim — Card 15 removes
        the key). Same four keys ui_hints() always returned; the frontend
        falls back to it until every consumer reads ui_manifest."""
        manifest = self.ui_manifest()
        return {
            "labels": manifest.get("labels") or {},
            "hidden_panels": list(manifest.get("hidden_panels") or []),
            "hidden_tabs": list(manifest.get("hidden_tabs") or []),
            "flag_presets": dict(manifest.get("flag_presets") or {}),
        }


def hidden_panels(manifest: dict) -> list:
    """Panels the active engine's manifest hides (llama.cpp-only vocab)."""
    return list((manifest or {}).get("hidden_panels") or [])


def hidden_tabs(manifest: dict) -> list:
    """Test tabs the active engine's manifest hides (engine-exclusive)."""
    return list((manifest or {}).get("hidden_tabs") or [])


def model_editor_sections(manifest: dict) -> list:
    """The model_editor.sections list (may be absent on legacy callers —
    Cards 10/11 render from it; until then the template keeps its
    hardcoded sections and only the gated ones consult the manifest)."""
    editor = (manifest or {}).get("model_editor") or {}
    return list(editor.get("sections") or [])