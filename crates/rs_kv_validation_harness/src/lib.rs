use rs_autoquant_policy::{AutoQuantFingerprint, AutoQuantPolicy};
use rs_atom_engine_profile::EngineRuntimeProfile;
use rs_kv_codec_adapters::CodecAdapterRegistry;
use rs_kv_quant_contracts::{normalize_codec_alias, KvCodec, KvPolicyMode, KvQuantPolicy};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ValidationCase {
    pub name: String,
    pub passed: bool,
    pub note: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ValidationReport {
    pub passed: bool,
    pub cases: Vec<ValidationCase>,
}

pub fn run_validation_suite() -> ValidationReport {
    let registry = CodecAdapterRegistry::baseline();
    let mut cases = Vec::new();

    for codec in [KvCodec::Tq4, KvCodec::Tq3, KvCodec::Tq2, KvCodec::Rq3Planar, KvCodec::Rq4Planar, KvCodec::Fp8E4M3] {
        let supported = registry.supports(&codec);
        cases.push(ValidationCase {
            name: format!("baseline_support::{codec:?}"),
            passed: supported,
            note: if supported {
                "baseline adapter present".into()
            } else {
                "missing baseline adapter".into()
            },
        });
    }

    let alias_codec = normalize_codec_alias("atom_fp8").expect("alias must normalize");
    cases.push(ValidationCase {
        name: "alias_normalization::atom_fp8".into(),
        passed: alias_codec == KvCodec::Fp8E4M3,
        note: format!("normalized to {:?}", alias_codec),
    });

    let policy = AutoQuantPolicy::uniform("digest", "model", 2, KvCodec::Tq4, 4, None);
    let round_trip = serde_json::to_string(&policy)
        .ok()
        .and_then(|json| serde_json::from_str::<AutoQuantPolicy>(&json).ok())
        .is_some();
    cases.push(ValidationCase {
        name: "autoquant_round_trip".into(),
        passed: round_trip,
        note: "json policy round-trip".into(),
    });

    let fp = AutoQuantFingerprint {
        gpu_arch: "gfx1030".into(),
        wave_size: 32,
        rocm_version: "7.2".into(),
        triton_version: "3.5".into(),
        python_version: "3.12.0".into(),
        model_family: "qwen".into(),
        n_layers: 32,
        head_dim: 128,
        n_heads: 32,
        n_kv_heads: 8,
        dtype_mode: "fp16".into(),
        codec_set_version: "1".into(),
    };
    cases.push(ValidationCase {
        name: "fingerprint_digest_shape".into(),
        passed: fp.hex_digest().len() == 16,
        note: fp.hex_digest(),
    });

    let kv_policy = KvQuantPolicy::new("model", KvCodec::Tq4, KvPolicyMode::Adaptive);
    cases.push(ValidationCase {
        name: "kv_quant_policy_shape".into(),
        passed: kv_policy.codec == KvCodec::Tq4,
        note: "shared KV quant contract reachable".into(),
    });

    let runtime_profile = EngineRuntimeProfile::default().with_delegate_backend("aiter");
    cases.push(ValidationCase {
        name: "atom_runtime_profile_shape".into(),
        passed: runtime_profile.supports_atom_backend
            && runtime_profile.supports_atom_attention
            && runtime_profile.supports_atom_rocm_telemetry
            && runtime_profile.supports_atom_fallback
            && runtime_profile.delegate_backend.as_deref() == Some("aiter")
            && runtime_profile.placeholder.is_none(),
        note: serde_json::to_string(&runtime_profile).unwrap(),
    });

    cases.push(ValidationCase {
        name: "radix_runtime_profile_defaults".into(),
        passed: runtime_profile.radix_cache_kind.is_none()
            && runtime_profile.radix_total_tokens.is_none()
            && runtime_profile.radix_protected_tokens.is_none()
            && runtime_profile.radix_evictable_tokens.is_none()
            && runtime_profile.radix_page_size.is_none()
            && runtime_profile.supports_automatic_prefix_caching
            && runtime_profile.supports_radix_cache
            && runtime_profile.supports_kv_events
            && !runtime_profile.supports_fp8_kv_cache
            && !runtime_profile.supports_turboquant_kv
            && !runtime_profile.supports_rotorquant_kv
            && !runtime_profile.supports_eagle3
            && !runtime_profile.supports_medusa
            && !runtime_profile.supports_ngram_speculation
            && !runtime_profile.supports_phantom
            && !runtime_profile.supports_phantom_x
            && runtime_profile.storage_tiers_supported.len() == 2
            && runtime_profile.storage_backend.is_none()
            && runtime_profile.storage_supports_stats.is_none()
            && runtime_profile.storage_supports_zero_copy.is_none()
            && runtime_profile.storage_layout_mode.is_none()
            && runtime_profile.storage_transfer_mem_type.is_none()
            && runtime_profile.attention_kernel_capabilities.is_none()
            && runtime_profile.amd_kv_kernel_profile.is_none(),
        note: "radix cache state placeholder is empty until runtime wiring lands".into(),
    });

    let storage_profile = EngineRuntimeProfile::default().with_storage_backend_state(
        "mooncake",
        true,
        true,
        "page_first_direct",
        Some("FILE".to_string()),
    );
    cases.push(ValidationCase {
        name: "storage_runtime_profile_shape".into(),
        passed: storage_profile.storage_backend.as_deref() == Some("mooncake")
            && storage_profile.storage_supports_stats == Some(true)
            && storage_profile.storage_supports_zero_copy == Some(true)
            && storage_profile.storage_layout_mode.as_deref() == Some("page_first_direct")
            && storage_profile.storage_transfer_mem_type.as_deref() == Some("FILE"),
        note: serde_json::to_string(&storage_profile).unwrap(),
    });

    let passed = cases.iter().all(|case| case.passed);
    ValidationReport { passed, cases }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn suite_passes() {
        let report = run_validation_suite();
        assert!(report.passed);
        assert!(!report.cases.is_empty());
    }

    #[test]
    fn report_serializes() {
        let report = run_validation_suite();
        let json = serde_json::to_string_pretty(&report).unwrap();
        assert!(json.contains("baseline_support"));
    }
}
