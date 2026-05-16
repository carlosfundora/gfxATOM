use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct EngineRuntimeProfile {
    pub supports_atom_backend: bool,
    pub supports_atom_attention: bool,
    pub supports_atom_kv_quant: bool,
    pub supports_atom_rocm_telemetry: bool,
    pub supports_atom_fallback: bool,
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
}

impl Default for EngineRuntimeProfile {
    fn default() -> Self {
        Self {
            supports_atom_backend: true,
            supports_atom_attention: true,
            supports_atom_kv_quant: false,
            supports_atom_rocm_telemetry: true,
            supports_atom_fallback: true,
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
        assert!(!profile.supports_atom_kv_quant);
        assert!(profile.supports_atom_rocm_telemetry);
        assert!(profile.supports_atom_fallback);
        assert_eq!(profile.delegate_backend, None);
        assert_eq!(profile.placeholder, None);
        assert_eq!(profile.radix_cache_kind, None);
        assert_eq!(profile.radix_total_tokens, None);
        assert_eq!(profile.radix_protected_tokens, None);
        assert_eq!(profile.radix_evictable_tokens, None);
        assert_eq!(profile.radix_page_size, None);
        assert_eq!(profile.storage_backend, None);
        assert_eq!(profile.storage_supports_stats, None);
        assert_eq!(profile.storage_supports_zero_copy, None);
        assert_eq!(profile.storage_layout_mode, None);
        assert_eq!(profile.storage_transfer_mem_type, None);
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
            "radix",
            1024,
            256,
            768,
            16,
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
}
