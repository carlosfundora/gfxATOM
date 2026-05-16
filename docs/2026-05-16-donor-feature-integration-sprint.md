# Donor Feature Integration & Optimization Sprint (ROCm/gfx1030)

## Scope

This sprint covers the extracted one-concern donors already harvested into:

- `/home/local/ai/build/kernels/canonical/harvested/donor_unique_features/`

Extracted donors in scope:

1. `KVCache-Factory`
2. `NautilusQuant`
3. `QAQ-KVCacheQuantization`
4. `RateQuant`
5. `delta-k-quantization`
6. `kv-cache-quantization`
7. `norm-separated-quantization`
8. `rust-llama.cpp`
9. `turbo-quant`
10. `vLLMTurboQuantKVCacheCPU`
11. `vllm_backend`
12. `wobble-quant-cache`

## Business Logic Collision Analysis (Before Integration)

### Collision domain A: KV bit-allocation policy overlap

Competing sources:

- `RateQuant` (rate-distortion)
- `wobble-quant-cache` (variance-aware per-dimension)
- `QAQ-KVCacheQuantization` (quality-adaptive search)
- `delta-k-quantization` (differential compression)

Potential collision:

- Multiple policy engines may independently choose incompatible bit layouts for the same KV block.

Resolution:

- Introduce one policy arbitration layer in `mb-kv-cache`:
  - choose exactly one `kv_policy_family` per request/session
  - enforce canonical policy outputs: `{k_bits, v_bits, group_size, calibration_profile}`

### Collision domain B: recency/freshness vs deterministic transforms

Competing sources:

- `kv-cache-quantization` (freshness-window aging)
- `NautilusQuant` (deterministic transform/LUT)
- `turbo-quant` (Rust low-bit transform primitives)

Potential collision:

- Aging policy can invalidate assumptions used by deterministic transform or decode-time calibration.

Resolution:

- Keep transforms deterministic and stable; apply aging only as an outer-tier eviction/precision downgrade policy.

### Collision domain C: runtime/deployment adapters

Competing sources:

- `vllm_backend` deployment patterns
- `rust-llama.cpp` adapter patterns

Potential collision:

- Runtime wrapper behavior can drift from engine-native scheduling/metrics contracts.

Resolution:

- Treat adapters as edge bindings only; do not let adapter code own quant/KV policy decisions.

## Clear Benefit Map by Extracted Donor

| Donor | Assimilated feature | Benefit | Collision risk | Feature flag |
|---|---|---|---|---|
| KVCache-Factory | Multi-policy KV benchmark harness | Rapid A/B policy comparison | Medium | `GFXATOM_KV_POLICY_FACTORY=1` |
| NautilusQuant | Deterministic transform/LUT quant ideas | Reproducible codec behavior | Medium | `GFXATOM_KV_NAUTILUS=1` |
| QAQ-KVCacheQuantization | Quality-adaptive quant scoring | Better quality under long context | Medium | `GFXATOM_KV_QAQ=1` |
| RateQuant | Rate-distortion bit budgeting | Principled mixed precision | Low | `GFXATOM_KV_RATEQUANT=1` |
| delta-k-quantization | Differential K/V compression | Strong low-bit compression lane | Medium | `GFXATOM_KV_DELTAK=1` |
| kv-cache-quantization | Freshness-window precision aging | Lightweight memory pressure control | Medium | `GFXATOM_KV_AGING=1` |
| norm-separated-quantization | INT4 failure guardrail | Safer low-bit rollout | Low | `GFXATOM_KV_NORM_GUARD=1` |
| rust-llama.cpp | Rust FFI adapter pattern | Cleaner Rust boundary for gguf adapters | Low | `GFXATOM_RUST_LLAMA_ADAPTER=1` |
| turbo-quant | Rust Turbo/QJL/Polar primitives | Fast Rust-native quant core | Low | `GFXATOM_KV_TURBOQUANT=1` |
| vLLMTurboQuantKVCacheCPU | CPU benchmark framing | CPU-audio and no-GPU regression lane | Low | `GFXATOM_KV_CPU_BENCH=1` |
| vllm_backend | Triton-style backend packaging | Cleaner deployment contracts | Low | `GFXATOM_VLLM_BACKEND_COMPAT=1` |
| wobble-quant-cache | Variance-aware allocation heuristic | Better per-dimension bit spend | Medium | `GFXATOM_KV_WOBBLE=1` |

## Fast Integration Sprint Plan

### Wave 1 (highest ROI, lowest risk)

1. `turbo-quant` Rust primitives (`src/turbo.rs`, `src/qjl.rs`, `src/kv.rs`)
2. `RateQuant` policy logic (`rate_distortion.py`, `mixed_precision.py`)
3. `norm-separated-quantization` guardrail logic (`compressors.py`)
4. `vLLMTurboQuantKVCacheCPU` benchmark adapter (`compute_metrics.py`)

Outcome:

- One canonical `kv_policy_family` selector with Rust-first primitives + safety guardrail + benchmark parity.

### Wave 2 (adaptive policies)

1. `delta-k-quantization` differential policy
2. `wobble-quant-cache` variance-aware allocator
3. `QAQ-KVCacheQuantization` quality-adaptive scorer

Outcome:

- Adaptive policy plugins behind strict compatibility contract.

### Wave 3 (tooling/deployment adapters)

1. `KVCache-Factory` comparative harness integration
2. `vllm_backend` deployment adapter docs/templates
3. `rust-llama.cpp` FFI adapter skeleton for gguf path
4. Optional `NautilusQuant` deterministic lane as experimental

Outcome:

- Operational comparability and cleaner runtime packaging.

## Toggle/Configuration Contract

All extracted features are opt-in by default and must remain fail-closed.

- Global master switch: `GFXATOM_DONOR_FEATURES=0|1`
- Per-feature flags from table above.
- Arbitration keys:
  - `GFXATOM_KV_POLICY_FAMILY` = `ratequant|deltak|wobble|qaq|nautilus|baseline`
  - `GFXATOM_KV_POLICY_MODE` = `strict|fallback`

If a policy is unsupported on current engine/model shape:

- log explicit rejection reason
- fall back to baseline policy
- emit telemetry event `kv_policy_rejected`

## Consul/Kong Registration Plan (Governance-Compliant)

Reference:

- `/home/local/ai/docs/consul/consul-port-governance.md`

### Proposed service registration (new engine)

- Inference engine service name: `gfxatom-rust` (engine domain, `400xx`)
- Candidate port: `40013` (currently free in `consul.env`)
- API facade service name: `api-gfxatom` (`93xx`)
- Candidate port: `9314` (currently free in `consul.env`)

### Required pre-registration checks

1. `curl -s http://localhost:9500/v1/catalog/services | jq .`
2. `cat /home/local/ai/consul/consul.env`
3. `ss -ltnp | grep :40013`
4. `ss -ltnp | grep :9314`

### Registration workflow

1. Add JSON service entries under correct domain files in `/home/local/ai/consul/services/`:
   - engine route in mesh/engine mapping using `40013`
   - API route in `api` domain using `9314`
2. Regenerate env mappings:
   - `python3 scripts/generate_consul_env.py`
3. Bind runtime only through `${PORT_*}` variables (no hardcoded ports).
4. Configure Kong using dedicated vars (`PORT_KONG_PROXY`, `PORT_KONG_ADMIN`, `PORT_KONG_MANAGER`) and route `api-gfxatom` upstream.

### Required tags/metadata

- Domain tag: `inference` for engine, `api` for API facade
- Lifecycle tag: `always_on` or `on_demand` (explicitly chosen)
- `meta.project`: `ai` (or tighter module owner once finalized)

## Deliverables to Produce During Implementation

For each integrated donor feature:

1. One feature note file under `gfxATOM-Rust/docs/features/<feature>.md`:
   - source donor and implementation path
   - integration touchpoints
   - enabled/disabled behavior
   - fallback behavior
   - telemetry fields
2. One contract test proving:
   - enable path works
   - disable path bypasses cleanly
   - unsupported runtime rejects and falls back safely
3. One benchmark record:
   - throughput/latency
   - quality proxy (ppl or acceptance proxy)
   - memory/KV footprint

## Immediate Recommendation

Start with Wave 1 only, keep all features gated OFF by default, and enable per-feature canary flags in benchmark harness before any production default changes.
