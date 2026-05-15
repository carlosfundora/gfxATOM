use rs_kv_quant_contracts::{normalize_codec_alias, KvCodec, KvCodecError};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct CodecAdapterDescriptor {
    pub codec: KvCodec,
    pub family: String,
    pub backend: String,
    pub supported: bool,
}

pub trait KvCodecAdapter {
    fn descriptor(&self) -> CodecAdapterDescriptor;
}

#[derive(Debug, Clone)]
pub struct BaselineCodecAdapter {
    codec: KvCodec,
    family: String,
}

impl BaselineCodecAdapter {
    pub fn new(codec: KvCodec, family: impl Into<String>) -> Self {
        Self {
            codec,
            family: family.into(),
        }
    }
}

impl KvCodecAdapter for BaselineCodecAdapter {
    fn descriptor(&self) -> CodecAdapterDescriptor {
        CodecAdapterDescriptor {
            codec: self.codec.clone(),
            family: self.family.clone(),
            backend: "baseline".into(),
            supported: true,
        }
    }
}

#[derive(Debug, Default, Clone)]
pub struct CodecAdapterRegistry {
    adapters: BTreeMap<KvCodec, BaselineCodecAdapter>,
}

impl CodecAdapterRegistry {
    pub fn baseline() -> Self {
        let mut adapters = BTreeMap::new();
        for (alias, family) in [
            ("tq4", "turbo"),
            ("tq3", "turbo"),
            ("tq2", "turbo"),
            ("rq3_planar", "rotor"),
            ("rq4_planar", "rotor"),
            ("fp8_e4m3", "fp8"),
        ] {
            if let Ok(codec) = normalize_codec_alias(alias) {
                adapters.insert(codec.clone(), BaselineCodecAdapter::new(codec, family));
            }
        }
        Self { adapters }
    }

    pub fn descriptor_for(&self, codec: &KvCodec) -> Option<CodecAdapterDescriptor> {
        self.adapters.get(codec).map(|adapter| adapter.descriptor())
    }

    pub fn supports(&self, codec: &KvCodec) -> bool {
        self.adapters.contains_key(codec)
    }

    pub fn all_descriptors(&self) -> Vec<CodecAdapterDescriptor> {
        self.adapters.values().map(|adapter| adapter.descriptor()).collect()
    }
}

pub fn normalize_adapter_alias(alias: &str) -> Result<KvCodec, KvCodecError> {
    normalize_codec_alias(alias)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn baseline_registry_exposes_expected_codecs() {
        let registry = CodecAdapterRegistry::baseline();
        assert!(registry.supports(&KvCodec::Tq4));
        assert!(registry.supports(&KvCodec::Rq3Planar));
        assert!(registry.supports(&KvCodec::Fp8E4M3));
    }

    #[test]
    fn descriptor_reports_family() {
        let registry = CodecAdapterRegistry::baseline();
        let desc = registry.descriptor_for(&KvCodec::Tq3).unwrap();
        assert_eq!(desc.family, "turbo");
        assert!(desc.supported);
    }
}
