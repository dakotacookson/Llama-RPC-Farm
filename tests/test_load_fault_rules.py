"""Regression: load-fault classifier must not flag models without mmproj/MTP.

The top-right load-failure notices are driven by scanning llama.cpp /
llama-swap startup logs. The rules used to match broad phrases anywhere in
the stream (a bare "MTP context", "vision.*error", "clip.*error"), so
healthy embedding/decision models that have neither an mmproj nor MTP
layers showed bogus "Vision projector (mmproj) failed..." notices.
"""
import os
import sys

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import app as app_module


BENIGN = [
    # startup chatter from models with neither a projector nor MTP layers
    "load_hparams: projector: qwen3a",
    "MTP context: 4 layers enabled",
    "clip_model_load: loaded successfully",
    "vision encoder ready",
    "nomic-embed-text-v1.5.f16.gguf",
    "loading model: Laya-Q8_0.gguf",
]

FAULTS = {
    "failed to load CLIP model from /models/mmproj.gguf": "mmproj",
    "clip_model_load: failed to load model": "mmproj",
    "failed to load mmproj model": "mmproj",
    "error loading projector: shape mismatch": "mmproj",
    "MTP context creation failed": "mtp",
    "doesn't contain MTP layers": "mtp",
    "failed to create MTP context": "mtp",
}


def test_benign_lines_are_not_load_faults():
    for line in BENIGN:
        assert app_module._parse_load_fault(line) is None, line


def test_real_projector_and_mtp_failures_are_classified():
    for line, code in FAULTS.items():
        got = app_module._parse_load_fault(line)
        assert got is not None, line
        assert got[0] == code, (line, got)


def test_streamed_response_chatter_is_ignored():
    # model output is arbitrary text — never treat it as a startup fault
    lines = [
        'srv operator(): http: streamed chunk: data: {"choices":'
        '[{"delta":{"content":"vision error"}}]}',
        '[INFO] Request 127.0.0.1 "POST /v1/chat/completions HTTP/1.1" 200',
        '[INFO] Request 127.0.0.1 "GET /v1/models HTTP/1.1" 200',
    ]
    for line in lines:
        assert app_module._TRAFFIC_RE.search(line), line


def test_model_ready_clears_the_load_window():
    lines = [
        "srv  server_run_i: model loaded",
        "srv  server_on_li: listening on http://0.0.0.0:19001",
        "server is listening on 127.0.0.1:8080",
    ]
    for line in lines:
        assert app_module._MODEL_READY_RE.search(line), line
