# Donor Index (gfxATOM-Rust)

Scope: external donor/reference repos cloned under `gfxATOM-Rust/donors/`.

Legend:
- **Claimed focus** = what repo README/docs explicitly claims.
- **Best-at (for us)** = where this donor appears most valuable for fusion work.

| Donor | Claimed focus | Best-at (for gfxATOM-Rust) | Notes |
|---|---|---|---|
| `vllm` | “Easy, fast, and cheap LLM serving” | Baseline serving scheduler + API/runtime parity target | Primary upstream reference for serving behavior and benchmarks. |
| `vllm-omni` | Omni-modality serving across CUDA/ROCm/NPU/XPU | Multimodal runtime patterns, diffusion/audio serving surfaces | Useful for future omni extensions beyond text-only engine path. |
| `vllm-ascend` | vLLM plugin for Ascend hardware | Plugin split architecture, hardware-specific backend separation | Good reference for clean hardware-plugin boundaries. |
| `vllm_backend` | Triton backend for vLLM AsyncEngine | Triton integration model and deployment packaging | Useful if we expose vLLM-compatible inference via Triton. |
| `llm-compressor` | Quantization/transforms (weights/acts/KV/attention) for vLLM deployment | Quant pipeline design and compressed-tensors compatibility ideas | Strong donor for quant tooling flow and artifact formats. |
| `KVCache-Factory` | Unified framework for KV cache compression (PyramidKV/SnapKV/H2O/StreamingLLM) | KV compression method matrix and cross-model experiments | High-value donor for pluggable KV policies and benchmarking. |
| `R-KV` | Decode-time KV compression cluster, “shrink cache keep accuracy” | Compression-aware scheduling/memory pooling integration patterns | Directly relevant to SGLang-side KV compression integration. |
| `vLLMTurboQuantKVCacheCPU` | CPU-only TurboQuant KV cache evaluation framework for vLLM | Benchmarking harness and metrics framing for simulated KV compression | Good reference for comparing baseline vs TurboQuant-style strategies. |
| `delta-k-quantization` | Group-size-robust KV quantization via closed-loop differential encoding | 2-bit K/V compression policy ideas and group-size robustness analysis | Strong donor for low-bit differential KV codecs. |
| `RateQuant` | Optimal mixed-precision KV cache quantization via rate-distortion theory | Bit allocation / mixed-precision policy selection | Excellent fit for policy-layer bit budgeting. |
| `NautilusQuant` | Deterministic orthogonal KV-cache quantization via golden-ratio geometry | Deterministic transform ideas and quantization geometry | Good candidate for reproducible codec experimentation. |
| `norm-separated-quantization` | Training-free fix for KV cache INT4 failures via norm separation | Failure-mode mitigation for INT4-style codecs | Strong donor for “don’t catastrophically break” guardrails. |
| `Higman-sims-quant` | Lattice-RSN / E8-space KV-cache compression framework | Ultra-lossless / lattice-codec experimentation | Useful as a high-end theoretical and codec-space reference. |
| `wobble-quant-cache` | Adaptive KV cache quantization that allocates bits by variance | Per-dimension adaptive bit allocation heuristics | Good fit for dimension-aware mixed precision policies. |
| `QAQ-KVCacheQuantization` | Quality-adaptive KV quantization; claims ~10x compression with minimal loss | Adaptive precision policy ideas for long-context KV | Good research donor for quality-aware bit allocation logic. |
| `turbo-quant` | Rust TurboQuant/PolarQuant/QJL for embeddings + KV compression | Rust-native low-bit vector/KV quant algorithms | Very strong donor for Rust-first quantization core. |
| `kv-cache-quantization` | Adaptive precision KV with freshness window + INT8 aging | Simple recency-based precision scheduling heuristic | Useful as lightweight baseline policy and ablation reference. |
| `tair-kvcache` | Alibaba distributed KVCache system with multi-level caching/memory pooling | Distributed KV metadata, control-plane design, capacity management | Strong systems donor for multi-tier KV orchestration. |
| `quant.cpp` | “SQLite of LLM inference”, GGUF runtime, claims 6.4x KV compression | Minimal C/C++ embedding + compact runtime integration approach | Useful for lean runtime API design and embeddability. |
| `ik_llama.cpp` | llama.cpp fork claiming stronger CPU/hybrid perf + quant variants | CPU/hybrid execution strategy, quant formats, MoE hybrid offload tactics | Useful when optimizing CPU fallback and hybrid placement. |
| `rust-llama.cpp` | Rust bindings over llama.cpp | Rust FFI/bindings patterns for llama.cpp ecosystem | Useful for adapter layer patterns, less for core kernels. |
| `rvllm` | Rust+CUDA GPU and JAX+XLA TPU inference engine | Rust serving architecture and GPU/TPU split design | Good architecture donor for Python-minimized serving path. |
| `llama-swap` | Hot-swap model gateway for OpenAI-compatible backends | Multi-engine routing and model swap control-plane UX | Useful for front-door gateway/routing semantics. |
| `smg` | High-performance model-routing gateway | Worker lifecycle + traffic routing + enterprise gateway controls | Useful for orchestrator features around engine fleet management. |
| `sardeenz` | PoC multi-model packing on GPU(s), built on vLLM + kvcached | Multi-model residency and GPU utilization strategies | Useful for model packing and placement policies. |

## Priority donor set (first-pass assimilation)

1. `RateQuant`, `delta-k-quantization`, `turbo-quant`, `R-KV`
2. `KVCache-Factory`, `QAQ-KVCacheQuantization`, `wobble-quant-cache`, `norm-separated-quantization`
3. `tair-kvcache`, `vllm`, `llm-compressor`, `rvllm`
4. `vLLMTurboQuantKVCacheCPU`, `quant.cpp`, `ik_llama.cpp`, `NautilusQuant`, `Higman-sims-quant`
5. `llama-swap`, `smg`, `sardeenz`, `vllm_backend`, `vllm-omni`, `vllm-ascend`, `rust-llama.cpp`, `kv-cache-quantization`

## Assimilation notes

- Treat README claims as **unverified** until measured in our RDNA2/ROCm test matrix.
- For quant donors, preserve side-by-side parity tests: perplexity, long-context retention, prefix reuse stability, and tokens/sec.
- For systems donors (gateway/control-plane), mine architecture and contract patterns before code transplantation.
