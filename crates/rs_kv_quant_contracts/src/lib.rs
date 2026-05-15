use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum KvCodec {
    Auto,
    Bf16,
    Fp8E4M3,
    Fp8E5M2,
    Int8,
    Tq4,
    Tq3,
    Tq2,
    Rq3Planar,
    Rq4Planar,
    Rq3Iso,
    Rq4Iso,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum KvPolicyMode {
    Static,
    Adaptive,
    Learned,
    Fallback,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct KvQuantPolicy {
    pub model_id: String,
    pub codec: KvCodec,
    pub mode: KvPolicyMode,
    pub layer_id: Option<u32>,
    pub stage_id: Option<String>,
    pub note: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct KvQuantTelemetry {
    pub prefix_reuse_ratio: Option<u64>,
    pub kv_hit_rate: Option<u64>,
    pub kv_used_bytes: Option<u64>,
    pub kv_capacity_bytes: Option<u64>,
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum KvCodecError {
    #[error("unsupported kv codec alias: {0}")]
    UnsupportedAlias(String),
}

pub fn normalize_codec_alias(alias: &str) -> Result<KvCodec, KvCodecError> {
    match alias.to_ascii_lowercase().as_str() {
        "auto" => Ok(KvCodec::Auto),
        "bf16" | "bfloat16" => Ok(KvCodec::Bf16),
        "fp8_e4m3" | "atom_fp8" => Ok(KvCodec::Fp8E4M3),
        "fp8_e5m2" => Ok(KvCodec::Fp8E5M2),
        "int8" => Ok(KvCodec::Int8),
        "tq4" => Ok(KvCodec::Tq4),
        "tq3" => Ok(KvCodec::Tq3),
        "tq2" => Ok(KvCodec::Tq2),
        "rq3" | "rq3_planar" => Ok(KvCodec::Rq3Planar),
        "rq4" | "rq4_planar" => Ok(KvCodec::Rq4Planar),
        "rq3_iso" => Ok(KvCodec::Rq3Iso),
        "rq4_iso" => Ok(KvCodec::Rq4Iso),
        other => Err(KvCodecError::UnsupportedAlias(other.to_string())),
    }
}

impl KvQuantPolicy {
    pub fn new(model_id: impl Into<String>, codec: KvCodec, mode: KvPolicyMode) -> Self {
        Self {
            model_id: model_id.into(),
            codec,
            mode,
            layer_id: None,
            stage_id: None,
            note: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn aliases_normalize() {
        assert_eq!(normalize_codec_alias("atom_fp8").unwrap(), KvCodec::Fp8E4M3);
        assert_eq!(normalize_codec_alias("rq3").unwrap(), KvCodec::Rq3Planar);
    }

    #[test]
    fn policy_serializes() {
        let p = KvQuantPolicy::new("m", KvCodec::Tq4, KvPolicyMode::Adaptive);
        let s = serde_json::to_string(&p).unwrap();
        assert!(s.contains("tq4"));
    }
}
