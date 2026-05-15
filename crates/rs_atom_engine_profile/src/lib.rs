use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct EngineRuntimeProfile {
    pub supports_atom_backend: bool,
    pub supports_atom_attention: bool,
    pub supports_atom_kv_quant: bool,
    pub supports_atom_rocm_telemetry: bool,
    pub supports_atom_fallback: bool,
}

impl Default for EngineRuntimeProfile {
    fn default() -> Self {
        Self {
            supports_atom_backend: true,
            supports_atom_attention: true,
            supports_atom_kv_quant: false,
            supports_atom_rocm_telemetry: true,
            supports_atom_fallback: true,
        }
    }
}

