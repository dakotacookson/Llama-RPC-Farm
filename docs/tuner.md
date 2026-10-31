# Model auto-tune (src/tuner.py)

## The flow

```
download → configure (hardware REQUIRED: servers + device_order chips)
  → safe defaults → Tune → winner saved into the model config
```

ctx-size is NEVER changed by the tuner — it is the user's decision.

## Phase 1 — math estimate (zero loads)

`estimate()` / `safe_defaults()` compute, purely from GGUF metadata +
hardware facts:

- weight bytes + mmproj + KV cache per token (per quant) + SSM state +
  MTP draft overhead vs usable VRAM (90% headroom kept: fragmentation +
  server overhead)
- max safe ctx, integer split weights per device, safe ubatch
- per-GPU bandwidth table predicts decode tok/s and identifies the
  pipeline straggler (decode is bandwidth-bound)

Mirrors vLLM `--max-model-len` auto / Ollama fit-vram logic. Exposed at
`GET /api/safe-defaults` and the Tune estimate endpoints.

## Phase 2 — staged live search

`start_tune()` runs a background thread (one job per model id):

1. baseline trial at locked ctx (bench server on a dedicated port
   18399, health-checked up to 15 min)
2. MTP (draft) variants
3. KV ladder walk: current → q8 → f16 → q4 (ctx fixed)
4. forced split variants
5. ubatch/batch/margin neighbors
6. flash-attn / SWA fallbacks

Early exits as soon as target tok/s reached (`TARGET_TPS_RPC = 20` over
~1 GbE multi-box; `TARGET_TPS_LOCAL = 25` single box). Every trial's
placement is parsed from the bench log and the winner (highest tok/s
with SAFE headroom) is written back into the model config.

Slow-GPU derates (`SLOW_GPU_DERATE`: GTX 10-series 0.6, RTX 20/30 lower
tiers) weight split distribution by memory bandwidth, not just VRAM.

## Safety properties

- Trials run with `SSH_PASS` in the subprocess env only (never argv/disk).
- `cancel_tune` kills the bench server (Windows: deliberately skipped —
  the bench process is killed by timeout instead; documented in code).
- Trials refuse to run from a Windows master (POSIX scripts required).
- VRAM headroom + measured temps (`_history_log` records max temp) feed
  the SAFE/UNSAFE verdict; unsafe winners are not saved.
- `tune/drift` compares historical best vs current hardware state
  (detects GPU swaps, thermal regressions).

## Bandwidth probe

`probe_link_mbps()` measures SSH throughput between master and each peer
(7-day cached in `config/net_speed_cache.json`). Used by estimates to
model RPC hop costs. On a Windows master it degrades gracefully to 0.

## Where results land

Winning flags (ubatch, batch, cache types, flash, split) are merged into
the model's `flags` in `config/models/<id>.yaml`; the ctx-size the user
set is preserved. `_note_temp_c` / history go to the tune cache dir, not
the model config.