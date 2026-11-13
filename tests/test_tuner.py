"""Unit tests for tuner._pick_port: bench ports must skip running tunes
and anything already listening, so two concurrent tunes never share a
port (benches would measure the wrong server).
"""
import socket

import tuner


def test_pick_port_free_returns_wanted():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        free = s.getsockname()[1]
    tuner._JOBS.clear()
    assert tuner._pick_port(free) == free


def test_pick_port_skips_listening():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        busy = s.getsockname()[1]
        s.listen(1)
        tuner._JOBS.clear()
        got = tuner._pick_port(busy)
        assert got != busy


def test_pick_port_skips_running_tune_ports():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        free = s.getsockname()[1]
    tuner._JOBS.clear()
    tuner._JOBS["m1"] = {"state": "running", "port": free}
    try:
        assert tuner._pick_port(free) != free
    finally:
        tuner._JOBS.clear()


def test_pick_port_ignores_idle_tune_ports():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        free = s.getsockname()[1]
    tuner._JOBS.clear()
    tuner._JOBS["m1"] = {"state": "idle", "port": free}
    try:
        assert tuner._pick_port(free) == free
    finally:
        tuner._JOBS.clear()


def test_cancel_tune_unknown_model():
    res = tuner.cancel_tune("no-such-model")
    assert res["cancelled"] is False


def test_moe_cache_helpers_gate_on_cpu_experts():
    """The two cache flags are meaningless for GPU-resident MoE: the
    tuner must only probe them when expert weights are on the CPU."""
    assert tuner.is_moe_expert_cached({"cpu-moe": True})
    assert tuner.is_moe_expert_cached({"n-cpu-moe": 12})
    assert not tuner.is_moe_expert_cached({})
    assert not tuner.is_moe_expert_cached({"n-gpu-layers": 99})
    assert tuner.moe_cache_trials({}) == ([], [])
    assert tuner.moe_cache_trials({"n-gpu-layers": 99}) == ([], [])


def test_moe_cache_trials_returns_ladders_for_cpu_experts():
    mibs, shares = tuner.moe_cache_trials({"n-cpu-moe": 12})
    assert mibs == tuner.MOE_CACHE_MIB_CANDIDATES
    assert shares == tuner.MOE_CACHE_CPU_SHARE_CANDIDATES


def test_moe_cache_ladders_sane():
    assert tuner.MOE_CACHE_MIB_CANDIDATES == sorted(tuner.MOE_CACHE_MIB_CANDIDATES)
    assert all(n > 0 for n in tuner.MOE_CACHE_MIB_CANDIDATES)
    assert all(0.0 <= s <= 1.0 for s in tuner.MOE_CACHE_CPU_SHARE_CANDIDATES)
    assert 0.0 in tuner.MOE_CACHE_CPU_SHARE_CANDIDATES


def test_trial_signature_includes_ctx_and_moe_placement():
    """Regression: ctx-size must be part of the signature or every ctx
    rung dedupes against the baseline and the walk never runs."""
    a = {"flags": {"ctx-size": 4096, "cache-type-k": "q8_0"}, "servers": ["a"]}
    b = {"flags": {"ctx-size": 8192, "cache-type-k": "q8_0"}, "servers": ["a"]}
    assert tuner.trial_signature(a) != tuner.trial_signature(b)
    c = {"flags": {"cpu-moe": True}, "servers": ["a"]}
    d = {"flags": {"n-cpu-moe": 12}, "servers": ["a"]}
    assert tuner.trial_signature(c) != tuner.trial_signature(d)
    e = {"flags": {"moe-cache-mib": 4096}, "servers": ["a"]}
    f = {"flags": {"moe-cache-mib": 8192}, "servers": ["a"]}
    assert tuner.trial_signature(e) != tuner.trial_signature(f)


def test_moe_cpu_ladder_derives_from_layer_count():
    assert tuner._moe_cpu_ladder(0) == [24, 32, 12]  # metadata fallback
    got = tuner._moe_cpu_ladder(40)
    assert got == sorted(got, reverse=True)
    assert all(1 <= n <= 40 for n in got)
    assert max(got) == int(40 * 0.9)


def test_moe_cache_mib_ladder_scales_to_headroom():
    assert tuner.moe_cache_mib_ladder(0) == tuner.MOE_CACHE_MIB_CANDIDATES
    scaled = tuner.moe_cache_mib_ladder(32)
    assert scaled == sorted(scaled)
    assert all(n >= 512 for n in scaled)
    # a 4 GB card must not be offered a >4 GB cache
    assert max(tuner.moe_cache_mib_ladder(4)) <= 4 * 1024


def test_moe_cache_trials_scales_with_free_vram():
    mibs, shares = tuner.moe_cache_trials({"n-cpu-moe": 8}, free_gb=32)
    assert mibs == tuner.moe_cache_mib_ladder(32)
    assert shares == tuner.MOE_CACHE_CPU_SHARE_CANDIDATES


def test_model_modality_detection():
    assert tuner._model_modality({"embeddings": True}, {}) == "embedding"
    assert tuner._model_modality(
        {}, {"general.architecture": "whisper"}) == "audio"
    assert tuner._model_modality(
        {}, {"general.architecture": "qwen3"}) == "text"


def test_kv_rank_orders_precision():
    assert tuner._kv_rank("q4_0", "q4_0") < tuner._kv_rank("q8_0", "q8_0")
    assert tuner._kv_rank("q8_0", "q8_0") < tuner._kv_rank("f16", "f16")
    # min() over the pair: a lossy side makes the pair lossy
    assert tuner._kv_rank("f16", "q4_0") == tuner._kv_rank("q4_0", "q4_0")


def test_resolve_bin_dir_accepts_file_or_dir():
    assert tuner._resolve_bin_dir(
        {"llama_bin": "/x/build/bin/llama-server"}) == "/x/build/bin"
    assert tuner._resolve_bin_dir({"llama_bin": "/x/build/bin"}) == "/x/build/bin"
    assert tuner._resolve_bin_dir({}, {"llama_bin": "/y/bin"}) == "/y/bin"


def test_supports_flag_exact_match_and_fail_closed(tmp_path):
    tuner._FLAGS_CACHE.clear()
    tuner._HELP_CACHE.clear()
    exe = tmp_path / "llama-server"
    exe.write_text("#!/bin/sh\necho '--moe-cache-mib N'\n"
                   "echo '--load-mode MODE'\n")
    exe.chmod(0o755)
    assert tuner.supports_flag("--moe-cache-mib", bin_dir=str(tmp_path))
    # exact token, not a prefix of a longer flag
    assert not tuner.supports_flag("--moe", bin_dir=str(tmp_path))
    # unknown/unreadable build fails closed
    assert not tuner.supports_flag("--moe-cache-mib",
                                   bin_dir=str(tmp_path / "missing"))
