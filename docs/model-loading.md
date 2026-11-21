# How a model can load

This is the short answer page for **“how do I run this model?”** There are
several supported deployment shapes. Pick by where the model fits and
whether you want one copy or several.

## Quick decision table

| Goal | Use | Key fields |
|---|---|---|
| Model fits on one machine | **Single deployment** | `servers: [machine]`, `device_order: machine-CUDA0,...` |
| Model fits on one remote machine, keep the API on the master | **Remote-local deployment** | `servers: [remote]`, `launch_mode: local` |
| Model does not fit on one machine | **RPC sharding** | `servers: [a,b,c]`, chips from all three, `launch_mode: rpc` |
| Model fits on several machines and you want throughput | **Load-balanced full replicas** | `load_balance: true`, chips from each machine, `lb_keep_warm: true` |
| Several different models on different machines | **Normal model routing** | one model config per model, llama-swap routes by request model |
| Same model with different settings/placements | **Variants** | `variants: [{label, servers, device_order, flags}]` |
| External server you already manage | **Custom runtime** | `runtime: custom`, `custom:` |

## 1. Single deployment

The normal case. The model loads once on one machine.

```yaml
model_id: qwen3-8b
servers: [aibox1]
device_order: aibox1-CUDA0,aibox1-CUDA1
launch_mode: auto
flags:
  ctx-size: 32768
  n-gpu-layers: 99
```

`launch_mode: auto` chooses local execution when the selected chips belong
to one machine. Add `launch_mode: local` to force it.

## 2. Remote-local deployment

The API stays on the master, but llama-server runs on one peer using that
peer’s own GPUs/CPU. No RPC traffic.

```yaml
servers: [aibox3]
device_order: aibox3-CUDA0,aibox3-CUDA1
launch_mode: local
```

The peer agent starts/stops llama-server; the farm proxies requests to it.
The model file must be readable on that machine (`store://` shared storage
or Model sync).

## 3. RPC sharding across machines

Use when the model is too large for any one machine. llama-server runs on
the master and uses RPC backends on the other machines.

```yaml
servers: [aibox1, aibox3, Aibox2]
device_order: aibox1-CUDA0,aibox3-CUDA0,Aibox2-CUDA0
launch_mode: rpc
```

Choose tensor placement in the model editor:

- **auto** — even placement
- **fit** — VRAM-aware fitting
- **split** — explicit layer weights, e.g. `8,10,10`

RPC is slower per token than a single-box load. Use it for models that do not
fit anywhere else.

## 4. Load-balanced full replicas

Use when the model **does** fit on multiple machines and you want more
throughput. Each selected machine loads a complete copy; the gateway balances
requests under one public model name.

```yaml
model_id: qwen3-8b
load_balance: true
lb_strategy: least_inflight   # or round_robin
lb_keep_warm: true
lb_max_inflight: 2
servers: [aibox1, aibox3]
device_order: aibox1-CUDA0,aibox3-CUDA0
```

This is **not RPC sharding**. Each machine has its own full model.

- `lb_keep_warm` keeps TTL at 0 and creates a persistent co-serve group.
- `lb_max_inflight` is the gateway’s per-replica routing cap. The editor
  keeps it equal to llama.cpp `--parallel`.
- The UI **Load** action prewarms every replica sequentially.
- Direct `:8090` traffic retries transient cold-start 5xx races.
- Evicting the public model evicts every replica.
- `/v1/models` and activity views expose only the public model id.

## 5. Different models on different machines

You do **not** need load balancing for this. Create a normal config per
model. llama-swap holds the fleet catalog and loads the requested model.

```text
aibox1: 27B chat
aibox3: coder model
jetson: ASR/embedding
```

One `:8090` endpoint, one model name per configured model.

## 6. Variants

Variants are additional deployments of the same base model. They can differ
in servers, hardware, context, flags, and launch mode.

```yaml
model_id: qwen3-27b
servers: [aibox1]
device_order: aibox1-CUDA0
variants:
- label: rpc
  servers: [aibox1, aibox3]
  device_order: aibox1-CUDA0,aibox3-CUDA0
  launch_mode: rpc
```

Clients call `qwen3-27b` for the base or `qwen3-27b--rpc` for the variant
(unless a load-balanced public model hides its replicas).

## 7. Serve groups

Serve groups control which models may be loaded together. They are model
groups, not machine groups.

| Setting | Meaning |
|---|---|
| `swap: true` | Loading one member swaps out the others |
| `swap: false` | Members may co-serve |
| `exclusive: true` | Loading a member evicts everything outside the group |
| `persistent: true` | Keep the group warm |

For a load-balanced model, all replicas are expanded into the group and
`swap` is forced off so the replicas co-serve. `exclusive` is preserved.

## 8. When does a model actually load?

llama-swap loads a model when:

- a request first targets it, or
- you press **Load** in the UI / call the load API.

It stays loaded according to its TTL/pin state:

- `ttl: 0` — never auto-unload
- `pinned: true` — keep-warm intent
- `lb_keep_warm: true` — replicas stay loaded
- **Evict** unloads it now

## 9. What clients see

All client traffic goes through `:8090`:

```text
POST http://master:8090/v1/chat/completions
POST http://master:8090/v1/messages
GET  http://master:8090/v1/models
```

The gateway handles auth, model grants, quotas, routing, and replica
selection. llama-swap on `:8080` is internal.

## 10. Common questions

**Can one model use several machines without sharding?**
Yes — load-balanced full replicas.

**Can it use several machines as one larger model?**
Yes — RPC sharding.

**Can I keep everything loaded?**
Yes — `ttl: 0`, `pinned`, `lb_keep_warm`, persistent serve groups.

**Do I need shared storage?**
For remote machines, yes: use `store://` shared storage or enable Model
sync. Local-only models can use a local path.

**Do the machines need the same GPU?**
No. Mixed vendors and VRAM sizes work, but the slowest link/device becomes
the straggler.

**CPU-only machines?**
Supported. Use the `RAM` chip/device. Expect lower throughput.

**What if a model load fails?**
Check `farmctl doctor`, the machine’s agent/engine status, the model path,
and the llama-swap log. For load-balanced cold starts, retry once; the
gateway also retries transient errors itself.

## Related docs

- [models-and-variants.md](models-and-variants.md) — full config schema
- [rpc-model.md](rpc-model.md) — RPC ports and sharding details
- [secure-rpc.md](secure-rpc.md) — mTLS RPC transport
- [topologies.md](topologies.md) — deployment examples
- [access.md](access.md) — keys, grants, quotas
