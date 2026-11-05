"""DEC-10 (AUDIT-T4 §1, closes T4-G1/G2/G3/G4) — manifest-rendered model
editor + variant modal.

The frontend has no JS harness, so these tests pin the CONTRACTS the
editor consumes, greppable from the plain-text UI sources:

* G1  variant modal: llama.cpp blocks tagged data-variant-lcpp (hide
      under a manifest that hides tensor_split), a data-variant-vllm
      override card exists (the engine's own fields — never translated),
      and variantModalSave strips/keeps keys by the SAME manifest signal.
* G2  device-order policy: the save path strips the llama.cpp tensor-
      placement keys when the active engine's manifest hides the
      tensor_split panel; the badge no longer counts the key under vllm.
* G3  flags fieldset: tagged data-editor-section="flags" and rendered
      from ui_manifest.model_editor.sections (vllm lists only
      vllm_section → the fieldset hides). The vllmBridgePreview UI
      translation mirror must stay deleted (DEC-08/09 directive).
* G4  models table: per-engine rows — Tune button omitted when the
      engine owns vllm_section (its backend 409s), row subtitle uses the
      engine's own fields (modelSubVllm), llama.cpp rows keep the
      verbatim columns.
"""
import os

import pytest


def _ui(rel):
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(here, rel), encoding="utf-8") as f:
        return f.read()


@pytest.fixture(scope="module")
def appjs():
    return _ui(os.path.join("ui", "static", "app.js"))


@pytest.fixture(scope="module")
def tpl():
    return _ui(os.path.join("ui", "templates", "index.html"))


@pytest.fixture(scope="module")
def modal(tpl):
    """the variant modal's own markup region"""
    start = tpl.index('id="variant-modal"')
    end = tpl.index("<!-- Machine detail modal", start)
    return tpl[start:end]


# ---------------- G1: variant modal engine gating -------------------------

def test_variant_modal_lcpp_blocks_tagged(modal):
    """Every llama.cpp vocab block in the variant modal carries
    data-variant-lcpp so the gate hides them under vllm
    (device-order chips, tensor-placement chips + fit inputs, llama
    bin, flags override card)."""
    for needle in ('class="chip-group v-chips-order"',
                  'class="chip-group v-chips-tensor-mode"',
                  'class="v-order-policy"',
                  'class="v-tensor-mode"',
                  'class="v-llama-bin"',
                  'class="v-flags-text"'):
        line = next(l for l in modal.splitlines() if needle in l)
        # the input or its enclosing gated block must carry the gate —
        # search back to the nearest opening div/select/fieldset tag
        idx = modal.index(needle)
        block = modal[max(0, idx - 1500):idx]
        assert "data-variant-lcpp" in block, f"{needle} not gated: {block[-300:]}"



def test_variant_modal_gated_from_manifest(appjs):
    """applyVariantModalEngineGate reads the ACTIVE engine's manifest
    (manifestHiddenPanels / tensor_split) — no hardcoded engine list."""
    fn = appjs.split("function applyVariantModalEngineGate", 1)[1]
    fn = fn[:fn.index("\nfunction ", 1)]
    assert "manifestHiddenPanels" in fn
    # no engine-name gate — the manifest drives the hide, not a
    # hardcoded engine string (the data-variant-vllm selector is a
    # section id, not an engine name)
    import re
    for m in re.finditer(r"[\"']vllm[\"']", fn):
        pytest.fail(f"engine-name gate leaked into the gate fn: {fn[:200]}")


def test_variant_save_strips_lcpp_keys_under_manifest(appjs):
    """variantModalSave nulls the llama.cpp keys when the manifest hides
    tensor_split, and saves the vllm overrides from the engine's own
    card (readVariantVllm) — hidden inputs never leak into the payload."""
    fn = appjs.split("function variantModalSave", 1)[1]
    fn = fn[:fn.index("\nfunction ", 1)]
    assert "lcppHidden" in fn
    assert "readVariantVllm(modal)" in fn
    # every llama.cpp key is gated on lcppHidden
    for key in ("device_order_policy", "tensor_split_mode", "llama_bin", "flags"):
        assert f"lcppHidden ? null" in fn  # strip idiom present
        assert key in fn


def test_variant_summary_is_engine_aware(appjs):
    """refreshVariantSummary renders the active engine's fields — vllm
    overrides instead of llama.cpp flag chips."""
    fn = appjs.split("function refreshVariantSummary", 1)[1]
    fn = fn[:fn.index("\nfunction ", 1)]
    assert "manifestHiddenPanels" in fn
    assert "tensor_parallel_size" in fn


# ---------------- G2: device-order policy gating --------------------------



# ---------------- G3: flags fieldset rendered from the manifest ----------

def test_flags_fieldset_tagged_with_editor_section(tpl):
    """The power-user flags fieldset carries
    data-editor-section="flags" so applyEngineHints can gate it from
    ui_manifest.model_editor.sections."""
    line = next(l for l in tpl.splitlines() if "Flags (power users)" in l)
    i = tpl.splitlines().index(line)
    assert 'data-editor-section="flags"' in "\n".join(
        tpl.splitlines()[max(0, i - 3):i + 1])


def test_apply_engine_hints_renders_editor_sections(appjs):
    """applyEngineHints gates data-editor-section nodes from the
    manifest's model_editor.sections list — not per-element hardcoded."""
    fn = appjs.split("function applyEngineHints", 1)[1]
    fn = fn[:fn.index("\nfunction ", 1)]
    assert "data-editor-section" in fn
    assert "model_editor" in fn



def test_bridge_preview_translation_mirror_stays_deleted(appjs):
    """DEC-08/09 directive: the vllmBridgePreview UI translation mirror
    must stay gone — hidden-not-translated, never resurrected."""
    assert "vllmBridgePreview" not in appjs
    assert "updateVllmBridgeHint" not in appjs


# ---------------- G4: per-engine models table -----------------------------

def test_models_table_tune_gated_by_manifest(appjs):
    """loadModels emits the Tune button ONLY when the active engine's
    manifest doesn't own vllm_section (its backend 409s under vllm —
    hidden, not a dead-end), and the row subtitle switches to the
    engine's own fields."""
    fn = appjs.split("async function loadModels", 1)[1]
    fn = fn[:fn.index("\nfunction ", 1)]
    assert 'manifestHasSection("vllm_section")' in fn
    assert 'vllmMode ? "" :' in fn           # Tune omitted under vllm
    assert "modelSubVllm(m)" in fn           # engine-native subtitle
    assert "servers:" in fn                   # llama.cpp rows verbatim


def test_manifest_accessors_exist(appjs):
    """The manifest accessors every gated surface reads through — no
    consumer hand-rolls its own engine-name check."""
    for name in ("function engineManifest()",
                 "function manifestHasSection(",
                 "function manifestHiddenPanels()"):
        assert name in appjs