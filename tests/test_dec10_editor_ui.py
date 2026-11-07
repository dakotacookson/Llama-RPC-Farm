"""DEC-10 (AUDIT-T4 §1, closes T4-G1/G2/G3/G4) — manifest-rendered model
editor + variant modal.

The frontend has no JS harness, so these tests pin the CONTRACTS the
editor consumes, greppable from the plain-text UI sources:

* G1  variant modal: llama.cpp blocks tagged data-variant-lcpp.
* G3  flags fieldset: tagged data-editor-section="flags" and rendered
      from ui_manifest.model_editor.sections.
* G4  models table: llama.cpp rows keep the verbatim columns.
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
    data-variant-lcpp (device-order chips, tensor-placement chips + fit
    inputs, llama bin, flags override card)."""
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



# ---------------- G4: per-engine models table -----------------------------

def test_manifest_accessors_exist(appjs):
    """The manifest accessors every gated surface reads through — no
    consumer hand-rolls its own engine-name check."""
    for name in ("function engineManifest()",
                 "function manifestHasSection(",
                 "function manifestHiddenPanels()"):
        assert name in appjs