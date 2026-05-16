from dataclasses import asdict, dataclass, replace


@dataclass(frozen=True)
class EngineRuntimeProfile:
    supports_atom_backend: bool = True
    supports_atom_attention: bool = True
    supports_atom_kv_quant: bool = False
    supports_atom_rocm_telemetry: bool = True
    supports_atom_fallback: bool = True
    delegate_backend: str | None = None
    placeholder: str | None = None
    radix_cache_kind: str | None = None
    radix_total_tokens: int | None = None
    radix_protected_tokens: int | None = None
    radix_evictable_tokens: int | None = None
    radix_page_size: int | None = None
    storage_backend: str | None = None
    storage_supports_stats: bool | None = None
    storage_supports_zero_copy: bool | None = None
    storage_layout_mode: str | None = None
    storage_transfer_mem_type: str | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    def with_delegate_backend(self, delegate_backend: str | None) -> "EngineRuntimeProfile":
        return replace(self, delegate_backend=delegate_backend)

    def with_placeholder(self, placeholder: str | None) -> "EngineRuntimeProfile":
        return replace(self, placeholder=placeholder)

    def with_radix_cache_state(
        self,
        radix_cache_kind: str | None,
        radix_total_tokens: int | None,
        radix_protected_tokens: int | None,
        radix_evictable_tokens: int | None,
        radix_page_size: int | None,
    ) -> "EngineRuntimeProfile":
        return replace(
            self,
            radix_cache_kind=radix_cache_kind,
            radix_total_tokens=radix_total_tokens,
            radix_protected_tokens=radix_protected_tokens,
            radix_evictable_tokens=radix_evictable_tokens,
            radix_page_size=radix_page_size,
        )

    def with_storage_backend_state(
        self,
        storage_backend: str | None,
        storage_supports_stats: bool | None,
        storage_supports_zero_copy: bool | None,
        storage_layout_mode: str | None,
        storage_transfer_mem_type: str | None = None,
    ) -> "EngineRuntimeProfile":
        return replace(
            self,
            storage_backend=storage_backend,
            storage_supports_stats=storage_supports_stats,
            storage_supports_zero_copy=storage_supports_zero_copy,
            storage_layout_mode=storage_layout_mode,
            storage_transfer_mem_type=storage_transfer_mem_type,
        )
