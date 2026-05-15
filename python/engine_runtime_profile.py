from dataclasses import dataclass


@dataclass(frozen=True)
class EngineRuntimeProfile:
    supports_atom_backend: bool = True
    supports_atom_attention: bool = True
    supports_atom_kv_quant: bool = False
    supports_atom_rocm_telemetry: bool = True
    supports_atom_fallback: bool = True

