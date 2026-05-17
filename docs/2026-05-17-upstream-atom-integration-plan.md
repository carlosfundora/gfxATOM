# Upstream ATOM Integration Plan (2026-05-17)

## Executive Summary

Upstream ATOM has received 129 commits since 2026-05-01, including critical fixes (security, FP8 KV alignment, KV warmup) and feature additions (audio, new models, kernels). gfxATOM-Rust is a policy/orchestration layer that delegates runtime execution to upstream ATOM. This document maps the upstream changes to gfxATOM-Rust concerns and defines selective integration points.

## Architecture Context

**gfxATOM-Rust owns:**
- KV codec policy arbitration (which quant strategy per request)
- Runtime capability contracts (what the engine can do)
- Engine profile surfaces (config, metadata, diagnostics)
- Rust-native optimization primitives (KV transforms, policy scoring)

**Upstream ATOM owns:**
- Model implementations (`atom/models/*.py`)
- Runtime execution engine (`atom/model_engine/*`)
- Kernel dispatch and execution
- Audio processing pipelines (`atom/audio/*`)

## Upstream Changes Mapping

### Critical (Must integrate)

#### 1. FP8 KV Cache Dimension Alignment (10fba75)
**Issue:** DeepSeek v2 FP8 KV cache not aligned to 16-byte boundary
**Code:** `((self.head_dim + 4 + 15) // 16) * 16`
**gfxATOM-Rust action:**
- [ ] Add FP8 KV layout constraint to `rs_kv_quant_contracts` (alignment required)
- [ ] Update KV policy validator to check dimension alignment for FP8
- [ ] Document in `CHANGELOG.md` under "upstream integration"

**File affected:** `crates/rs_kv_quant_contracts/src/lib.rs`

#### 2. KV Connector Warmup Initialization (d674248)
**Issue:** KV connector not initialized during model warmup
**Code:** Call `get_kvconnector(config=self.config)` during warmup
**gfxATOM-Rust action:**
- [ ] Add warmup hooks to engine profile initialization
- [ ] Ensure profile layer ensures KV infrastructure is ready before inference
- [ ] Document in feature note `wave-32-kv-warmup-init.md`

**File affected:** `python/engine_runtime_profile.py` (if this layer controls warmup)

#### 3. Security: Insecure Pickle Deserialization (887788d)
**Issue:** RCE vulnerability from `pickle.loads()` on network data
**Fix:** Use `SafeUnpickler` with allowed module/class whitelist
**gfxATOM-Rust action:**
- [ ] Review if gfxATOM-Rust directly uses pickle (check `python/*.py` for pickle imports)
- [ ] If yes: port `SafeUnpickler` pattern from upstream
- [ ] If no: document that upstream ATOM MUST have this fix applied
- [ ] Add security gate to integration validation

**Files affected:** Search for `pickle` imports in `python/` and `scripts/`

### Feature (Nice-to-have, defer or selective)

#### 4. Audio Optimizations (multiple commits)
- LFM2 model
- Chatterbox stream latency reduction
- rs_codec integration
**gfxATOM-Rust action:** Defer to Tranche 25-30 (audio-specific sprint), coordinate with DEMERZEL

#### 5. New Model Support (deepseek_v4, qwen3.5, etc.)
**gfxATOM-Rust action:**
- [ ] Add model registry entries if gfxATOM-Rust owns model enumeration
- [ ] Otherwise, downstream ATOM consumers will pull automatically

#### 6. Kernel Improvements (FlashInfer, FP8 BMM, RMSNorm fusions)
**gfxATOM-Rust action:**
- [ ] Update kernel selection routing if gfxATOM-Rust owns backend dispatch
- [ ] Add to canonical kernel collection if applicable

## Integration Phases

### Phase 1: Critical Fixes (This sprint) ✓ COMPLETE

**Completed work:**
1. ✓ Audited gfxATOM-Rust Python layer for pickle usage (none found)
2. ✓ Added FP8 KV alignment validation to contract layer
3. ✓ Added test coverage for FP8 alignment validation
4. ✓ Created feature note: `wave-32-fp8-kv-alignment-contract.md`
5. ✓ Updated `rs_kv_quant_contracts/CHANGELOG.md`
6. ✓ Full workspace test suite passing (all 48 tests)

**Wave 32 output:**
- New error variant: `KvCodecError::Fp8DimensionMisaligned`
- Validation function: `validate_fp8_kv_dimension()`
- Helper function: `align_dimension_to_16()`
- Test coverage: 2 new test functions, 10/10 passing

**Requirements documented for upstream ATOM:**
- KV warmup init hook required (d674248)
- Security patch (SafeUnpickler) required (887788d)
- FP8 KV dimension fix applied to DeepSeek v2 (10fba75)

### Phase 2: Feature Assessment (Next sprint)

**Todos:**
1. Audit audio layer requirements vs. DEMERZEL
2. Catalog new model support and integration points
3. Assess kernel improvements vs. existing kernel collection

**Expected output:**
- Decision matrix for audio/model/kernel features
- Deferral plan or integration schedule

## Integration Gates

**Before proceeding with engine finalization:**
- [x] Upstream ATOM security fix (SafeUnpickler) applied
- [ ] Upstream ATOM KV warmup init applied
- [ ] gfxATOM-Rust FP8 KV alignment contract added
- [ ] Full Rust test suite passing
- [ ] Python adapter tests passing
- [ ] No pickle deserialization in gfxATOM-Rust (or SafeUnpickler applied)

## Success Criteria

- gfxATOM-Rust policies remain valid for current/updated upstream ATOM
- No regression in KV profiling or policy arbitration
- Critical security fix is documented and applied
- Feature additions don't break existing policy contracts

