## 2026-05-23 - [Remove expensive string matching from topK hot path]
**Learning:** `is_rocm_aiter_fusion_shared_expert_enabled()` performs string matching over the model exclude list and was being evaluated in hot loops (per-token generation steps in `atom/model_ops/topK.py`), causing significant CPU overhead.
**Action:** When implementing flags or caching configurations, ensure they are computed during initialization or outside the hot loops instead of executing string operations (like `is_rocm_aiter_fusion_shared_expert_enabled`) per token.
