use rs_kv_quant_contracts::KvCodec;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::fs;
use std::path::PathBuf;
use tempfile::NamedTempFile;
use thiserror::Error;

pub const POLICY_VERSION: u32 = 1;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
pub enum Stage {
    Prefill,
    Decode,
    Draft,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct CodecChoice {
    pub codec: KvCodec,
    pub bit_width: u8,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub note: Option<String>,
}

impl CodecChoice {
    pub fn new(codec: KvCodec, bit_width: u8, note: impl Into<Option<String>>) -> Self {
        Self {
            codec,
            bit_width,
            note: note.into(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AutoQuantPolicy {
    pub fingerprint_digest: String,
    pub model_family: String,
    pub n_layers: u32,
    #[serde(default)]
    pub layer_codecs: BTreeMap<u32, CodecChoice>,
    #[serde(default)]
    pub stage_overrides: BTreeMap<Stage, BTreeMap<u32, CodecChoice>>,
    pub version: u32,
    pub learner: String,
    pub created_at: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub score: Option<f64>,
}

impl AutoQuantPolicy {
    pub fn uniform(
        fingerprint_digest: impl Into<String>,
        model_family: impl Into<String>,
        n_layers: u32,
        codec: KvCodec,
        bit_width: u8,
        note: impl Into<Option<String>>,
    ) -> Self {
        let choice = CodecChoice::new(codec, bit_width, note);
        let mut layer_codecs = BTreeMap::new();
        for layer in 0..n_layers {
            layer_codecs.insert(layer, choice.clone());
        }
        Self {
            fingerprint_digest: fingerprint_digest.into(),
            model_family: model_family.into(),
            n_layers,
            layer_codecs,
            stage_overrides: BTreeMap::new(),
            version: POLICY_VERSION,
            learner: "uniform".into(),
            created_at: 0.0,
            score: None,
        }
    }

    pub fn codec_for(&self, layer_idx: u32, stage: Option<Stage>) -> CodecChoice {
        if let Some(stage) = stage {
            if let Some(overrides) = self.stage_overrides.get(&stage) {
                if let Some(choice) = overrides.get(&layer_idx) {
                    return choice.clone();
                }
            }
        }
        self.layer_codecs
            .get(&layer_idx)
            .cloned()
            .unwrap_or_else(|| CodecChoice::new(KvCodec::Tq4, 4, Some("autoquant fallback".into())))
    }

    pub fn codec_histogram(&self) -> BTreeMap<String, usize> {
        let mut hist = BTreeMap::new();
        for choice in self.layer_codecs.values() {
            let key = format!("{:?}:{}", choice.codec, choice.bit_width);
            *hist.entry(key).or_insert(0) += 1;
        }
        hist
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct AutoQuantFingerprint {
    pub gpu_arch: String,
    pub wave_size: u32,
    pub rocm_version: String,
    pub triton_version: String,
    pub python_version: String,
    pub model_family: String,
    pub n_layers: u32,
    pub head_dim: u32,
    pub n_heads: u32,
    pub n_kv_heads: u32,
    pub dtype_mode: String,
    pub codec_set_version: String,
}

impl AutoQuantFingerprint {
    pub fn hex_digest(&self) -> String {
        let blob = serde_json::to_vec(self).expect("fingerprint serialization must succeed");
        let digest = Sha256::digest(blob);
        hex::encode(digest)[..16].to_string()
    }
}

#[derive(Debug, Default, Clone, Serialize, Deserialize, PartialEq)]
pub struct SideStats {
    pub sample_count: u64,
    pub dynamic_range: f64,
    pub mean_abs: f64,
    pub rms: f64,
    pub kurtosis: f64,
    pub sparsity: f64,
    pub last_observed_at: f64,
}

#[derive(Debug, Default, Clone, Serialize, Deserialize, PartialEq)]
pub struct AutoQuantObserverSnapshot {
    pub n_layers: u32,
    pub sample_every: u32,
    pub ema_alpha: f64,
    pub sparsity_eps: f64,
    pub total_observations: u64,
    pub layers: BTreeMap<String, SideStats>,
}

#[derive(Debug, Error)]
pub enum AutoQuantError {
    #[error("autoquant cache error: {0}")]
    Cache(String),
    #[error("autoquant policy version mismatch: file={file} code={code}")]
    VersionMismatch { file: u32, code: u32 },
}

pub fn cache_dir() -> PathBuf {
    if let Ok(raw) = std::env::var("SGLANG_AUTOQUANT_DIR") {
        return PathBuf::from(raw);
    }
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".into());
    PathBuf::from(home).join(".cache/sglang/autoquant")
}

pub fn policy_path(fp: &AutoQuantFingerprint) -> PathBuf {
    cache_dir().join(format!("{}.json", fp.hex_digest()))
}

pub fn load_policy(fp: &AutoQuantFingerprint) -> Result<Option<AutoQuantPolicy>, AutoQuantError> {
    let path = policy_path(fp);
    if !path.exists() {
        return Ok(None);
    }
    let text = fs::read_to_string(&path).map_err(|e| AutoQuantError::Cache(e.to_string()))?;
    let policy: AutoQuantPolicy =
        serde_json::from_str(&text).map_err(|e| AutoQuantError::Cache(e.to_string()))?;
    if policy.fingerprint_digest != fp.hex_digest() {
        return Ok(None);
    }
    Ok(Some(policy))
}

pub fn save_policy(
    mut policy: AutoQuantPolicy,
    fingerprint: &AutoQuantFingerprint,
) -> Result<PathBuf, AutoQuantError> {
    if policy.created_at == 0.0 {
        policy.created_at = 1.0;
    }
    let dir = cache_dir();
    fs::create_dir_all(&dir).map_err(|e| AutoQuantError::Cache(e.to_string()))?;
    let target = dir.join(format!("{}.json", fingerprint.hex_digest()));
    let mut tmp =
        NamedTempFile::new_in(&dir).map_err(|e| AutoQuantError::Cache(e.to_string()))?;
    let json = serde_json::to_string_pretty(&policy)
        .map_err(|e| AutoQuantError::Cache(e.to_string()))?;
    std::io::Write::write_all(&mut tmp, json.as_bytes())
        .map_err(|e| AutoQuantError::Cache(e.to_string()))?;
    tmp.persist(&target)
        .map_err(|e| AutoQuantError::Cache(e.error.to_string()))?;
    Ok(target)
}

pub fn delete_policy(fp: &AutoQuantFingerprint) -> Result<bool, AutoQuantError> {
    let path = policy_path(fp);
    if !path.exists() {
        return Ok(false);
    }
    fs::remove_file(&path).map_err(|e| AutoQuantError::Cache(e.to_string()))?;
    Ok(true)
}

pub fn list_policies() -> Result<Vec<PathBuf>, AutoQuantError> {
    let dir = cache_dir();
    if !dir.exists() {
        return Ok(Vec::new());
    }
    let mut out = Vec::new();
    for entry in fs::read_dir(&dir).map_err(|e| AutoQuantError::Cache(e.to_string()))? {
        let entry = entry.map_err(|e| AutoQuantError::Cache(e.to_string()))?;
        let path = entry.path();
        if path.extension().and_then(|s| s.to_str()) == Some("json") {
            out.push(path);
        }
    }
    out.sort();
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn policy_lookup_prefers_layer_codec() {
        let policy = AutoQuantPolicy::uniform("dig", "m", 2, KvCodec::Tq4, 4, None);
        assert_eq!(policy.codec_for(1, None).codec, KvCodec::Tq4);
    }

    #[test]
    fn fingerprint_digest_is_stable() {
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
        assert_eq!(fp.hex_digest().len(), 16);
    }

    #[test]
    fn policy_round_trips() {
        let p = AutoQuantPolicy::uniform("dig", "m", 1, KvCodec::Tq3, 3, None);
        let s = serde_json::to_string(&p).unwrap();
        let d: AutoQuantPolicy = serde_json::from_str(&s).unwrap();
        assert_eq!(d.layer_codecs.len(), 1);
    }
}
