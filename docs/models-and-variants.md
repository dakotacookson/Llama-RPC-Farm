# Models, flags, device order & variants

## Model config anatomy

One YAML file per model in `config/models/`. Key fields:

- `model_id` — the identity everywhere (llama-swap key, API paths,
  visualizer node ids). `name` is display only.
- `base_path` — `store://llm-models/<file>`; sharded models point at the
  FIRST shard (`-00001-of-00003`); the scanner collapses shards into one
  model (`model_parser.py`).
- `runtime: llama` (GGUF) or `mlx` (Apple Silicon; base_path is an HF
  repo id, served via `python -m mlx_lm server`, no RPC/--device).
- `servers` — machines this model deploys across; first is the master
  for this model (llama-server host), rest are RPC peers. In vllm mode
  exactly ONE machine (multi-machine serving is llama.cpp RPC only;
  multi-GPU is tensor-parallel within the host).

## Model formats (GGUF vs HF/safetensors)

The scanner (`model_parser.py`) recognizes both:

- **GGUF** — a file (or shard set `model-00001-of-00003.gguf` / `part 01
  of 03`) is ONE model entry; llama.cpp serves the file.
- **HF/safetensors** — a directory containing `config.json` plus at
  least one `*.safetensors` is ONE model: `vllm serve <directory>`
  takes the directory itself as base_path. Its
  `-000XX-of-000YY.safetensors` shards collapse into that single entry
  (same shard pattern as GGUF splits); stray safetensors files
  (`adapter_model.safetensors`) are not weight shards and are left out.
  A safetensors file alone is not identifiable — vLLM reads the
  architecture from `config.json`.

Every scanned entry carries `format`: `"gguf"` | `"hf"`, so
engine-aware surfaces (model wizard, vLLM validation) can treat
GGUF-on-vLLM as experimental without re-statting the path.

**GGUF on vLLM is experimental** (vllm-gguf-plugin, opt-in — the
installer never auto-installs it): `vllm serve model.gguf` without a
tokenizer fails at load, so a GGUF base_path requires
`vllm.tokenizer` pointing at the base HF model directory — missing it
is a validation **error**; with it set, validation still warns
"experimental — expect limited support". HF/safetensors models need
none of that and are the native vLLM path.

## Per-model vllm section (engine: vllm)

Each model config may carry a `vllm:` section next to `flags:` — same
vocabulary validated by `apply_vllm_section`
(`max_model_len`, `max_num_seqs`, `tensor_parallel_size`,
`pipeline_parallel_size`,
`kv_cache_dtype` (auto|fp8), `cpu_offload_gb`, `quantization`,
`tokenizer`, `extra_args`); unknown keys are rejected on save. See
configuration.md for the full key table. Device chips behave the same
as in llama.cpp mode — in vllm mode they map to
`CUDA_VISIBLE_DEVICES` index lists (RAM chips have no effect; use
`cpu_offload_gb` instead) and `tensor_parallel_size` is cross-checked
against the number of GPUs the host actually exposes (world size is
`tensor_parallel_size × pipeline_parallel_size`). Ray and other
distributed-executor backends are not schema keys — pass them via
`extra_args` (see configuration.md).

Explicit chip order: `<machine>-<token>` comma-joined, e.g.
`aibox1-CUDA0,aibox1-CUDA1,aibox1-CUDA2,aibox3-CUDA0,aibox3-CUDA1`.

- `as-selected` policy: used verbatim (peer order = --rpc order).
- `vram-first` policy: `_vram_first_key` reorders — all GPUs before any
  CPU; host leads; then machines by total VRAM desc (or RAM desc for
  CPU devices). Stable sort preserves chip order within ties.
- Empty device_order → auto from machine detection (`_auto_device_order`).

The editor's hardware chips write this string; `renderDeviceOrderPreview`
shows the resulting token sequence live.

## flags (three synced views)

The model editor shows flags in three places that must stay in sync live:

1. **Common inputs**: `#flag-ctx-size` (answers to the `--c` alias too),
   `#flag-cache-k` / `#flag-cache-v` dropdowns, flash-attn / jinja chips.
2. **Chip rows**: `#flags-container` key/value rows.
3. **Raw text**: `#flags-text`, one `--flag [value]` per line, `#`
   comments allowed.

Editing any one updates the other two immediately:

- ctx: `setCtxViews` (common ↔ rows ↔ text, `upsertFlagLine` writes the
  text view; `upsertFlagRow` handles duplicate `ctx-size`/`c` keys)
- KV cache: `setKvCacheViews` + generic `upsertFlagTextLine(key, value)`
- Save path: `syncCommonFlagsToRows` folds common inputs into rows so
  `readFlags()` sees everything regardless of active view.

Flag value coercion: `true`/`false` → bool, numeric → number, else
string. Bool `true` renders as a bare flag; flags stored as bool but
requiring a value upstream get a default (only `--cors-origins` today).

## Variants (additional configs)

Each variant is a sub-config with its own `label`, optional
`servers`/`device_order`/`expose`/`flags`/... overrides. Effective
config = base merged with variant (variant value wins; `null` inherits).

Examples from this fleet:

- `qwen3-8-27b-...-128k` (aibox1-only, ctx 128000) with variant
  `rpc-X3` (aibox1 + aibox3, ctx 300000, parallel 3).
- In llama-swap config both appear as `model_id` and
  `model_id--<label>` entries.

**Critical invariant:** variants share the base model's GGUF filename.
Any filename → servers mapping (e.g. `buildFlowModelMaps` in app.js)
must never let a variant's data overwrite the BASE filename keys, or
streaming the plain model lights the variant's machines (this bug
happened — the visualizer showed aibox3 while only aibox1 served).

Variant save diffs against the primary's flags so untouched prefilled
values stay inherited (`variantModalSave`).

## Serve groups (llama-swap)

`config/swap_groups.yaml` groups MODELS (not machines):

- `swap: true` → llama-swap group: loading one member evicts the others
- `exclusive: true` → member loads evict ALL models
- `persistent: true` → llama-swap keeps the group's slot warm

Generation maps these into llama-swap's `groups:` section at push time
(`swap_manager.generate()`); the `_swap_key` namespaces model ids so
co-served models don't collide.

## Model editor wizard (3 steps)

1. **Source** — pick a scanned GGUF (scan root from Settings) or paste a
   path; safe defaults apply on Configure (not Scan).
2. **Hardware** — group affinity (optional) + GPU/RAM chips → servers +
   device_order.
3. **Details** — ids, runtime, advanced panel (tensor placement, order
   policy, flags views, pinned/TTL, variants card).

Advanced badge counts customized advanced fields live.