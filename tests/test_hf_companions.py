"""Companion selection for HF downloads: one mmproj + matching MTP only.

Regression for unsloth/Qwen3.8-Flash-Next-GGUF, where selecting one quant
pulled every MTP head (6 variants) + both mmproj precisions.
"""
import os
import sys

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import hf_downloader as hd  # noqa: E402

R = "Qwen3.8-Flash-Next"
REPO_FILES = [
    f"BF16/{R}-BF16-00001-of-00008.gguf",
    f"Q8_0/{R}-Q8_0-00001-of-00006.gguf",
    f"UD-IQ1_S/{R}-UD-IQ1_S-00001-of-00003.gguf",
    f"UD-IQ1_S/{R}-UD-IQ1_S-00002-of-00003.gguf",
    f"UD-IQ1_S/{R}-UD-IQ1_S-00003-of-00003.gguf",
    f"UD-Q4_K_XL/{R}-UD-Q4_K_XL-00001-of-00004.gguf",
    f"MTP/mtp-{R}-BF16.gguf",
    f"MTP/mtp-{R}-Q4_K_M.gguf",
    f"MTP/mtp-{R}-Q8_0.gguf",
    f"MTP/mtp-{R}-shared-BF16.gguf",
    f"MTP/mtp-{R}-shared-Q4_K_M.gguf",
    f"MTP/mtp-{R}-shared-Q8_0.gguf",
    "mmproj-BF16.gguf",
    "mmproj-F16.gguf",
    f"mtp-{R}-Q8_0.gguf",
    "config.json", "tokenizer.json", "README.md", "LICENSE",
]


def _sel(quant_dir, quant):
    return [f for f in REPO_FILES
            if f.startswith(quant_dir + "/") and quant in f]


def test_selected_iq1s_skips_all_mtp_and_picks_one_mmproj():
    main = _sel("UD-IQ1_S", "UD-IQ1_S")
    comps, warnings = hd._select_companions(main[0], REPO_FILES, main)
    assert comps == ["mmproj-F16.gguf"]           # F16 preferred
    assert not any("mtp" in c.lower() for c in comps)
    assert any("not for" in w and "MTP" in w for w in warnings)


def test_selected_q8_downloads_one_matching_mtp():
    main = [f for f in REPO_FILES if f.startswith("Q8_0/")]
    comps, _ = hd._select_companions(main[0], REPO_FILES, main)
    mtp = [c for c in comps if "mtp" in os.path.basename(c).lower()]
    assert mtp == [f"mtp-{R}-Q8_0.gguf"]          # root copy, non-shared
    assert "mmproj-F16.gguf" in comps
    assert len(comps) == 2                         # never the 6-variant pile


def test_selected_bf16_picks_non_shared_bf16_mtp():
    main = [f for f in REPO_FILES if f.startswith("BF16/")]
    comps, _ = hd._select_companions(main[0], REPO_FILES, main)
    mtp = [c for c in comps if "mtp" in os.path.basename(c).lower()]
    assert mtp == [f"MTP/mtp-{R}-BF16.gguf"]
    assert "mmproj-F16.gguf" in comps


def test_no_other_full_quants_are_ever_companions():
    main = _sel("UD-IQ1_S", "UD-IQ1_S")
    comps, _ = hd._select_companions(main[0], REPO_FILES, main)
    for c in comps:
        assert c not in REPO_FILES or c.startswith("mmproj") \
            or "mtp" in c.lower()


def test_single_draft_head_is_used_when_unambiguous():
    files = ["m-Q4_K_M.gguf", "mmproj-F16.gguf", "draft-head.gguf"]
    comps, _ = hd._select_companions("m-Q4_K_M.gguf", files, ["m-Q4_K_M.gguf"])
    assert "draft-head.gguf" in comps


def test_no_companions():
    files = ["m-Q4_K_M.gguf", "config.json"]
    comps, warnings = hd._select_companions("m-Q4_K_M.gguf", files,
                                            ["m-Q4_K_M.gguf"])
    assert comps == []
    assert warnings == []


def test_quant_tags():
    assert hd._quant_tag(f"UD-IQ1_S/{R}-UD-IQ1_S-00001-of-00003.gguf") \
        == "IQ1S"
    assert hd._quant_tag(f"{R}-Q4_K_M.gguf") == "Q4KM"
    assert hd._quant_tag("mmproj-F16.gguf") == "F16"
    assert hd._quant_tag("model.gguf") == ""
