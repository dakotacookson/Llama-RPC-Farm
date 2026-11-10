# Benchmarks & performance reports

There is no single "tokens/sec" number for Llama RPC Farm: it depends on
the model, quant, context, KV cache type, tensor split, GPU mix, and the
network between boxes. This page records **real measured runs** and gives
you a template to add yours.

## What the numbers mean

The farm reports **decode tokens/sec** from llama.cpp's own
`timings.predicted_per_second` (prefill excluded). The auto-tuner records
each trial in `generated/tune-history.csv` on the machine running the app.

Rows below are real auto-tune runs on the maintainer's mixed fleet. They
are **not a controlled benchmark**: only 64 tokens were generated, and the
network path was the existing LAN. Treat them as a sanity band, not a
leaderboard.

## Maintainer's fleet — measured runs

### 4B-class Q4_K over RPC

| Field | Value |
|---|---|
| Model | 4B-class Q4_K (Gemma-class, `Q4_K`) |
| Placement | 3 Linux boxes, 6 GPUs total (host 3 + peers 2 + 1) |
| Tensor split | `8,3,3,2,1,1` |
| Context | 16,384 |
| KV cache | `q8_0` / `q8_0` |
| Batch / ubatch | 2048 / 256 |
| MTP / spec decode | off |
| Measured decode | **~130.7 tok/s** (best trial) |
| Max temp seen | 46 °C |

### Qwen3.8-27B-Uncensored Q8_0 over RPC + MTP

| Field | Value |
|---|---|
| Model | `Qwen3.8-27B-Uncensored-Q8_0` |
| Placement | 3 Linux boxes, 6 GPUs total (host 3 + peers 2 + 1) |
| Device order | host 3 GPUs + 3 RPC GPUs from two peers |
| Context | 180,000 |
| KV cache | `q8_0` / `q8_0` |
| Spec decode | MTP draft, `n_max=2` |
| Measured decode | **~24.6 tok/s** (best trial) |
| Max temp seen | 42 °C |

The same weights on one box will be faster per token. RPC buys reach, not
speed: use it for models that do not fit on a single machine, and keep
models that do fit local.

## Reproduce on your farm

1. Open a model in the UI → **Tune** (or the Tune endpoints in
   [docs/api.md](api.md)).
2. Let the baseline run finish; the tuner writes each trial to
   `generated/tune-history.csv`.
3. Report the **best safe** row, not the fastest unsafe one.

A quick manual check against a loaded model:

```bash
curl -s http://127.0.0.1:8090/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"YOUR-MODEL","messages":[{"role":"user","content":"hi"}],
       "max_tokens":64}' | jq '.timings'
```

`timings.predicted_per_second` is the decode rate; `timings.prompt_n` /
`timings.predicted_n` are the token counts.

## Submit a report

Open an issue or PR with this table filled in:

```markdown
| Field | Value |
|---|---|
| Model / quant | |
| Placement (boxes, GPUs, RAM) | |
| Tensor split / fit mode | |
| Context | |
| KV cache | |
| MTP / spec decode | |
| GPU vendor + driver | |
| Network (1 GbE / 2.5 GbE / 10 GbE / Wi-Fi / Tailscale) | |
| Decode tok/s (server timings) | |
| Prefill tok/s (if reported) | |
| Max temp / power limit | |
| Notes (throttling, P2P, errors) | |
```

Reports for ROCm, Intel SYCL, Apple Metal, CPU-only, and mixed
Windows/Linux fleets are especially useful — see the 🧪 rows in the
[hardware table](../README.md#supported-hardware).
