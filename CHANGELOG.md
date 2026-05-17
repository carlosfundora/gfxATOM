
## Wave 33 Phase 2 Follow-Up: Kernel Improvements Identification

**Date:** 2026-05-17  
**Work:** Identified high-priority kernel improvements from upstream ATOM integration wave  
**Status:** Phase 2 action planning (integration staging)  

### Changes

1. **Created kernel improvements tagging document**
   - File: `docs/kernel-improvements/wave33-phase2-kernel-tagging.md` (moved to manifests)
   - Identified high-priority kernels: FlashInfer CUTLASS, RDNA2 HIP, TurboQuant/RotorQuant
   - Linked to upstream commits: 3b60317, a4da908, 64f7808, d526c3a

2. **Model support additions (no code changes required)**
   - DeepSeek-v4: Text generation with parallel head optimization
   - Kimi K2.5: Eagle3 speculative decoding
   - LFM2/LFM2.5: Audio models (STT/TTS)
   - Fish Speech S2 Pro: TTS audio model
   - VoxCPM2: TTS audio model
   - Note: All models are runtime-driven; gfxATOM-Rust does not enumerate models

3. **Audio layer coordination deferred to next sprint**
   - Documented DEMERZEL boundary (owns orchestration/routing)
   - Upstream ATOM audio improvements are complementary
   - Pattern: improvements flow through ATOM runtime → gfxATOM backend → DEMERZEL routing

### Integration Gates Status

- [x] Upstream ATOM security fix (SafeUnpickler) applied
- [x] gfxATOM-Rust FP8 KV alignment contract added
- [x] Warmup initialization support added to Rust profile
- [x] Full Rust test suite passing (35/35 tests)
- [ ] RDNA2 kernel validation for gfx1030
- [ ] TurboQuant integration test harness
- [ ] Upstream ATOM KV warmup init applied to runtime

### Next Steps

1. **Wave 33A** — Kernel improvement sequencing
   - Verify RDNA2 HIP kernels in canonical collection
   - Tag high-priority kernels (FlashInfer, RDNA2, TurboQuant)
   - Update canonical index with integration targets

2. **Wave 33B** — Audio coordination planning
   - Create task: "Audio layer integration with DEMERZEL"
   - Define coordination boundary and information flow

3. **Phase 3** — Runtime integration
   - KV warmup hook implementation
   - Kernel dispatch routing for RDNA2
   - TurboQuant codec registry integration

