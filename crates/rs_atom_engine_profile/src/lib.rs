use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum KvStorageTier {
    Hbm,
    CpuRam,
    Lmcache,
    Nvme,
    ObjectStore,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum KvQuantizationMode {
    None,
    Fp8E4M3,
    Fp8E5M2,
    Int8,
    Int4,
    TurboQuant,
    RotorQuant,
    EngineNative,
    Custom(String),
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum KvCompressionMode {
    None,
    CacheGen,
    ZstdSerde,
    LossyPack,
    SparseEviction,
    HybridEvictQuantize,
    EngineNative,
    Custom(String),
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct AttentionKernelCapabilities {
    pub supports_prefix_aware_attention: bool,
    pub supports_multi_tile_decode: bool,
    pub supports_shared_prefix_query_packing: bool,
    pub supports_kv_splitting: bool,
    pub supports_online_softmax_merge: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct AmdKvKernelProfile {
    pub gfx_arch: String,
    pub wavefront_size: u32,
    pub lds_staging_enabled: bool,
    pub split_k_enabled: bool,
    pub paged_kv_traversal_enabled: bool,
    pub branch_packing_enabled: bool,
    pub hsa_queue_priority: String,
    pub svm_page_migration_enabled: bool,
    pub rocprofiler_enabled: bool,
    pub rocm_smi_polling_enabled: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct KvDecodeTelemetryBundle {
    pub request_id: String,
    pub session_id: String,
    pub model_id: String,
    pub engine_id: String,
    pub shape_key: String,
    pub prompt_tokens: u32,
    pub decode_tokens: u32,
    pub prefix_reuse_ratio: f32,
    pub kv_hit_rate: f32,
    pub kv_miss_count: u64,
    pub branch_entropy_mean: Option<f32>,
    pub branch_entropy_max: Option<f32>,
    pub mismatch_depth_mean: Option<f32>,
    pub mismatch_depth_std: Option<f32>,
    pub early_exit_ratio: Option<f32>,
    pub acceptance_rate: Option<f32>,
    pub token_surprisal_mean: Option<f32>,
    pub repeated_token_count: Option<u32>,
    pub calibration_drift_score: Option<f32>,
    pub gpu_memory_used_mb: Option<u64>,
    pub memory_bandwidth_gbps: Option<f32>,
    pub kernel_launch_overhead_ms: Option<f32>,
    pub decode_tps: Option<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct KvObjectRecord {
    pub kv_object_id: String,
    pub cache_key: String,
    pub content_hash: String,
    pub prefix_hash: String,
    pub model_id: String,
    pub engine_id: String,
    pub tokenizer_id: String,
    pub tokenizer_revision: String,
    pub token_start: u32,
    pub token_count: u32,
    pub layer_start: u32,
    pub layer_count: u32,
    pub dtype: String,
    pub compression_mode: KvCompressionMode,
    pub quantization_mode: KvQuantizationMode,
    pub storage_tier: KvStorageTier,
    pub location_uri: String,
    pub reusable_scope: String,
    pub pin_state: String,
    pub eviction_priority: f32,
    pub future_use_score: f32,
    pub metadata_registry: String,
    pub control_plane_store: String,
    pub tensor_store: String,
    pub allow_raw_kv_in_surrealdb: bool,
    pub allow_raw_kv_in_valkey: bool,
    pub created_at_rfc3339: String,
    pub last_used_at_rfc3339: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct PrecomputedContextAsset {
    pub asset_id: String,
    pub asset_type: String,
    pub profile_id: String,
    pub content_hash: String,
    pub token_count: u32,
    pub pinned: bool,
    pub storage_tiers_supported: Vec<KvStorageTier>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct PrecomputedKvAsset {
    pub kv_asset_id: String,
    pub context_asset_id: String,
    pub kv_object_id: String,
    pub quantization_mode: KvQuantizationMode,
    pub compression_mode: KvCompressionMode,
    pub calibration_status: String,
    pub calibration_dataset_id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct KvTransferPlan {
    pub plan_id: String,
    pub source_tier: KvStorageTier,
    pub destination_tier: KvStorageTier,
    pub kv_object_ids: Vec<String>,
    pub disaggregated_prefill_decode: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct KvPrefetchPlan {
    pub plan_id: String,
    pub upcoming_agent_step_ids: Vec<String>,
    pub prefetch_kv_object_ids: Vec<String>,
    pub reason: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct KvEvictionDecision {
    pub decision_id: String,
    pub evict_kv_object_ids: Vec<String>,
    pub preserve_kv_object_ids: Vec<String>,
    pub reason: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct AgentStepGraph {
    pub graph_id: String,
    pub active_step_id: String,
    pub next_likely_step_ids: Vec<String>,
    pub required_kv_object_ids: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ModelRuntimeProfile {
    pub supports_automatic_prefix_caching: bool,
    pub supports_radix_cache: bool,
    pub supports_lmcache_connector: bool,
    pub supports_kv_events: bool,
    pub supports_fp8_kv_cache: bool,
    pub supports_turboquant_kv: bool,
    pub supports_rotorquant_kv: bool,
    pub supports_eagle3: bool,
    pub supports_medusa: bool,
    pub supports_ngram_speculation: bool,
    pub supports_phantom: bool,
    pub supports_phantom_x: bool,
    pub supports_disaggregated_prefill: bool,
    pub supports_disaggregated_decode: bool,
    pub supports_prefix_aware_attention: bool,
    pub supports_content_addressed_cache: bool,
    pub supports_position_independent_cache: bool,
    pub storage_tiers_supported: Vec<KvStorageTier>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct EngineRuntimeProfile {
    pub supports_atom_backend: bool,
    pub supports_atom_attention: bool,
    pub supports_atom_kv_quant: bool,
    pub supports_atom_rocm_telemetry: bool,
    pub supports_atom_fallback: bool,
    pub supports_automatic_prefix_caching: bool,
    pub supports_radix_cache: bool,
    pub supports_lmcache_connector: bool,
    pub supports_kv_events: bool,
    pub supports_fp8_kv_cache: bool,
    pub supports_turboquant_kv: bool,
    pub supports_rotorquant_kv: bool,
    pub supports_eagle3: bool,
    pub supports_medusa: bool,
    pub supports_ngram_speculation: bool,
    pub supports_phantom: bool,
    pub supports_phantom_x: bool,
    pub supports_disaggregated_prefill: bool,
    pub supports_disaggregated_decode: bool,
    pub supports_prefix_aware_attention: bool,
    pub supports_content_addressed_cache: bool,
    pub supports_position_independent_cache: bool,
    pub storage_tiers_supported: Vec<KvStorageTier>,
    pub delegate_backend: Option<String>,
    pub placeholder: Option<String>,
    pub radix_cache_kind: Option<String>,
    pub radix_total_tokens: Option<u64>,
    pub radix_protected_tokens: Option<u64>,
    pub radix_evictable_tokens: Option<u64>,
    pub radix_page_size: Option<u32>,
    pub storage_backend: Option<String>,
    pub storage_supports_stats: Option<bool>,
    pub storage_supports_zero_copy: Option<bool>,
    pub storage_layout_mode: Option<String>,
    pub storage_transfer_mem_type: Option<String>,
    pub attention_kernel_capabilities: Option<AttentionKernelCapabilities>,
    pub amd_kv_kernel_profile: Option<AmdKvKernelProfile>,
    pub adaptive_recommendation: Option<String>,
}

impl Default for ModelRuntimeProfile {
    fn default() -> Self {
        Self {
            supports_automatic_prefix_caching: true,
            supports_radix_cache: true,
            supports_lmcache_connector: false,
            supports_kv_events: true,
            supports_fp8_kv_cache: false,
            supports_turboquant_kv: false,
            supports_rotorquant_kv: false,
            supports_eagle3: false,
            supports_medusa: false,
            supports_ngram_speculation: false,
            supports_phantom: false,
            supports_phantom_x: false,
            supports_disaggregated_prefill: false,
            supports_disaggregated_decode: false,
            supports_prefix_aware_attention: false,
            supports_content_addressed_cache: false,
            supports_position_independent_cache: false,
            storage_tiers_supported: vec![KvStorageTier::Hbm, KvStorageTier::CpuRam],
        }
    }
}

impl Default for EngineRuntimeProfile {
    fn default() -> Self {
        Self {
            supports_atom_backend: true,
            supports_atom_attention: true,
            supports_atom_kv_quant: false,
            supports_atom_rocm_telemetry: true,
            supports_atom_fallback: true,
            supports_automatic_prefix_caching: true,
            supports_radix_cache: true,
            supports_lmcache_connector: false,
            supports_kv_events: true,
            supports_fp8_kv_cache: false,
            supports_turboquant_kv: false,
            supports_rotorquant_kv: false,
            supports_eagle3: false,
            supports_medusa: false,
            supports_ngram_speculation: false,
            supports_phantom: false,
            supports_phantom_x: false,
            supports_disaggregated_prefill: false,
            supports_disaggregated_decode: false,
            supports_prefix_aware_attention: false,
            supports_content_addressed_cache: false,
            supports_position_independent_cache: false,
            storage_tiers_supported: vec![KvStorageTier::Hbm, KvStorageTier::CpuRam],
            delegate_backend: None,
            placeholder: None,
            radix_cache_kind: None,
            radix_total_tokens: None,
            radix_protected_tokens: None,
            radix_evictable_tokens: None,
            radix_page_size: None,
            storage_backend: None,
            storage_supports_stats: None,
            storage_supports_zero_copy: None,
            storage_layout_mode: None,
            storage_transfer_mem_type: None,
            attention_kernel_capabilities: None,
            amd_kv_kernel_profile: None,
            adaptive_recommendation: None,
        }
    }
}

impl EngineRuntimeProfile {
    pub fn with_delegate_backend(mut self, delegate_backend: impl Into<String>) -> Self {
        self.delegate_backend = Some(delegate_backend.into());
        self
    }

    pub fn with_placeholder(mut self, placeholder: impl Into<String>) -> Self {
        self.placeholder = Some(placeholder.into());
        self
    }

    pub fn with_adaptive_recommendation(
        mut self,
        adaptive_recommendation: impl Into<String>,
    ) -> Self {
        self.adaptive_recommendation = Some(adaptive_recommendation.into());
        self
    }

    pub fn with_radix_cache_state(
        mut self,
        radix_cache_kind: impl Into<String>,
        radix_total_tokens: u64,
        radix_protected_tokens: u64,
        radix_evictable_tokens: u64,
        radix_page_size: u32,
    ) -> Self {
        self.radix_cache_kind = Some(radix_cache_kind.into());
        self.radix_total_tokens = Some(radix_total_tokens);
        self.radix_protected_tokens = Some(radix_protected_tokens);
        self.radix_evictable_tokens = Some(radix_evictable_tokens);
        self.radix_page_size = Some(radix_page_size);
        self
    }

    pub fn with_storage_backend_state(
        mut self,
        storage_backend: impl Into<String>,
        storage_supports_stats: bool,
        storage_supports_zero_copy: bool,
        storage_layout_mode: impl Into<String>,
        storage_transfer_mem_type: Option<String>,
    ) -> Self {
        self.storage_backend = Some(storage_backend.into());
        self.storage_supports_stats = Some(storage_supports_stats);
        self.storage_supports_zero_copy = Some(storage_supports_zero_copy);
        self.storage_layout_mode = Some(storage_layout_mode.into());
        self.storage_transfer_mem_type = storage_transfer_mem_type;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_profile_has_expected_flags() {
        let profile = EngineRuntimeProfile::default();
        assert!(profile.supports_atom_backend);
        assert!(profile.supports_atom_attention);
        assert!(profile.supports_automatic_prefix_caching);
        assert!(profile.supports_radix_cache);
        assert!(profile.supports_kv_events);
        assert_eq!(
            profile.storage_tiers_supported,
            vec![KvStorageTier::Hbm, KvStorageTier::CpuRam]
        );
        assert_eq!(profile.delegate_backend, None);
        assert_eq!(profile.placeholder, None);
        assert_eq!(profile.radix_cache_kind, None);
        assert_eq!(profile.storage_backend, None);
    }

    #[test]
    fn delegate_backend_helper_sets_backend() {
        let profile = EngineRuntimeProfile::default().with_delegate_backend("aiter");
        assert_eq!(profile.delegate_backend.as_deref(), Some("aiter"));
    }

    #[test]
    fn placeholder_helper_sets_placeholder() {
        let profile = EngineRuntimeProfile::default().with_placeholder("runtime_profile");
        assert_eq!(profile.placeholder.as_deref(), Some("runtime_profile"));
    }

    #[test]
    fn radix_cache_state_helper_sets_cache_state() {
        let profile = EngineRuntimeProfile::default().with_radix_cache_state(
            "radix", 1024, 256, 768, 16,
        );
        assert_eq!(profile.radix_cache_kind.as_deref(), Some("radix"));
        assert_eq!(profile.radix_total_tokens, Some(1024));
        assert_eq!(profile.radix_protected_tokens, Some(256));
        assert_eq!(profile.radix_evictable_tokens, Some(768));
        assert_eq!(profile.radix_page_size, Some(16));
    }

    #[test]
    fn storage_backend_state_helper_sets_storage_state() {
        let profile = EngineRuntimeProfile::default().with_storage_backend_state(
            "mooncake",
            true,
            true,
            "page_first_direct",
            Some("FILE".to_string()),
        );
        assert_eq!(profile.storage_backend.as_deref(), Some("mooncake"));
        assert_eq!(profile.storage_supports_stats, Some(true));
        assert_eq!(profile.storage_supports_zero_copy, Some(true));
        assert_eq!(
            profile.storage_layout_mode.as_deref(),
            Some("page_first_direct")
        );
        assert_eq!(profile.storage_transfer_mem_type.as_deref(), Some("FILE"));
    }

    #[test]
    fn kv_object_contract_disallows_raw_tensor_writes_to_metadata_stores() {
        let record = KvObjectRecord {
            kv_object_id: "obj-1".into(),
            cache_key: "k".into(),
            content_hash: "h".into(),
            prefix_hash: "p".into(),
            model_id: "m".into(),
            engine_id: "e".into(),
            tokenizer_id: "tok".into(),
            tokenizer_revision: "rev".into(),
            token_start: 0,
            token_count: 1,
            layer_start: 0,
            layer_count: 1,
            dtype: "fp16".into(),
            compression_mode: KvCompressionMode::None,
            quantization_mode: KvQuantizationMode::None,
            storage_tier: KvStorageTier::Hbm,
            location_uri: "engine://kv/obj-1".into(),
            reusable_scope: "persona".into(),
            pin_state: "pinned".into(),
            eviction_priority: 1.0,
            future_use_score: 1.0,
            metadata_registry: "surrealdb".into(),
            control_plane_store: "valkey".into(),
            tensor_store: "engine_native".into(),
            allow_raw_kv_in_surrealdb: false,
            allow_raw_kv_in_valkey: false,
            created_at_rfc3339: "2026-01-01T00:00:00Z".into(),
            last_used_at_rfc3339: "2026-01-01T00:00:00Z".into(),
        };
        assert!(!record.allow_raw_kv_in_surrealdb);
        assert!(!record.allow_raw_kv_in_valkey);
        assert_eq!(record.metadata_registry, "surrealdb");
        assert_eq!(record.control_plane_store, "valkey");
    }
}
