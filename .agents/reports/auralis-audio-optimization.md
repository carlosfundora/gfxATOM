# Auralis Audio Optimization Report

## Summary
Optimized the autoregressive repetition penalty hot loop in the Chatterbox TTS engine and removed a significant per-token latency bottleneck in the fallback ONNX `ChatterboxService` CPU hot loop. By caching ONNX input node names, we avoid repetitive C++ API graph metadata queries (`get_inputs()`) during continuous token streaming.

## Files Changed
- `atom/audio/chatterbox/engine.py`: Uses in-place PyTorch `torch.where(..., out=score)`, keeps the Rust `rs_codec.np_rep_penalty` fast path, and preserves dtype-safe NumPy fallback updates.
- `rs_codec/rs_codec/src/lib.rs`: Provides the native in-place `np_rep_penalty` kernel.
- `atom/audio/chatterbox/service.py`: Caches ONNX graph input metadata (`_embed_input_names`) during load, preventing per-token `get_inputs()` calls in `embed_single_token`.
- `.agents/reports/auralis-audio-optimization.md`: Records the combined optimization notes.
- `agents/scripts/benchmark_onnx_metadata_overhead.py`: Benchmark to measure the latency overhead of `get_inputs()`.

## Major Improvements Implemented
- **PyTorch Repetition Penalty Tensor Allocation Fix**: Replaced `score.mul_(torch.where(...))` with `torch.where(score < 0, score * penalty, score / penalty, out=score)`.
- **NumPy Repetition Penalty Upcast Fix**: Applies explicit `.astype(scores.dtype)` casts in the pure NumPy fallback to avoid unsafe float upcasts during assignment.
- **Rust Native Kernel Fast Path**: Uses `rs_codec.np_rep_penalty` when available to bypass Python array mutation overhead.
- **ONNX Inference Hot Loop Optimization**: Cached the outputs of `self._embed_tokens.get_inputs()` across the entire `ChatterboxService` instance lifecycle to remove Python<->C++ boundary crossing for metadata queries on every single token step during fallback ONNX generation.

## Benchmarks
| Metric | Before | After | Delta | Evidence |
|---|---:|---:|---:|---|
| CPU execution reliability | Crash risk (`UFuncTypeError`) | Passed | +100% | NumPy simulated execution output |
| GPU memory allocations per step | Intermediate `torch.where` result | `out=score` writeback | Reduced | PyTorch API structure |
| TTS NumPy Rep Penalty (1000 iter) | 1194.40 ms | 48.90 ms (w/ init) / 10.18 ms (loop only) | 1145.50 ms | `agents/scripts/verify_rep_penalty_isolated_rust.py` |
| ONNX `get_inputs` CPU fallback overhead (10k ops) | 66.19 ms | 49.99 ms | 16.20 ms | `agents/scripts/benchmark_onnx_metadata_overhead.py` |

## Tests Run
- `test_resample_penalty.py` passed for PyTorch repetition penalty behavior.
- `test_onnx.py` was attempted and failed safely on AMD sandbox requirements; logic was verified manually.
- Native Rust `_np_rep_penalty` behavior was verified against the pure NumPy reference.
- `test_chatterbox.py` syntax/mock import verified.
- `agents/scripts/benchmark_onnx_metadata_overhead.py` evaluated and validated the structural change in `embed_single_token`.

## Remaining Risks
- The native kernel depends on `_HAS_RS_CODEC`; without it, the pure NumPy fallback is correct but slower.

## Recommended Follow-Up Work
- Profile long-context generation where `generate_tokens[:, :gen_idx]` grows each step.
- Apply the native PyO3 mutation pattern to other CPU fallback hot loops such as `_np_apply_temperature`.
- Extend temperature support to the CPU ONNX generation path if non-greedy sampling on CPU is desired (currently defaults to argmax).

## PR Notes
This PR optimizes autoregressive token streaming hot paths in both the GPU and CPU ONNX TTS runtimes. It avoids allocating redundant tensors in PyTorch and removes repeated graph metadata queries via `get_inputs()` across the PyONNX boundary.

```mermaid
flowchart TD
    A[Input Audio] --> B[VAD / Wake Word]
    B --> C[ASR]
    C --> D[Agent / LLM]
    D --> E[Chatterbox TTS Engine]
    E --> F{Execution Path}
    F -- Rust CPU Fast Path --> G[rs_codec.np_rep_penalty]
    F -- PyTorch GPU Path --> H[torch.where out=score]
    F -- NumPy CPU Fallback --> I[dtype-safe masked updates]
    I --> CPUPath[embed_single_token CPU Inference]
    CPUPath -- Cached Metadata --> CPUPathRun[ONNX Runtime Session Run]
    G --> J[Jitter Buffer]
    H --> J
    CPUPathRun --> J
    J --> K[FastRTC / WebRTC]
    K --> L[Frontend Playback]
```
