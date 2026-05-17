from dataclasses import dataclass

from kv_quant_contracts import KvCodec, normalize_codec_alias


@dataclass(frozen=True)
class CodecAdapterDescriptor:
    codec: KvCodec
    family: str
    backend: str
    supported: bool = True


class CodecAdapterRegistry:
    _baseline = {
        KvCodec.tq1: CodecAdapterDescriptor(KvCodec.tq1, "turbo", "baseline"),
        KvCodec.tq2: CodecAdapterDescriptor(KvCodec.tq2, "turbo", "baseline"),
        KvCodec.tq3: CodecAdapterDescriptor(KvCodec.tq3, "turbo", "baseline"),
        KvCodec.tq4: CodecAdapterDescriptor(KvCodec.tq4, "turbo", "baseline"),
        KvCodec.tq8: CodecAdapterDescriptor(KvCodec.tq8, "turbo", "baseline"),
        KvCodec.rq3_planar: CodecAdapterDescriptor(KvCodec.rq3_planar, "rotor", "baseline"),
        KvCodec.rq4_planar: CodecAdapterDescriptor(KvCodec.rq4_planar, "rotor", "baseline"),
        KvCodec.fp8_e4m3: CodecAdapterDescriptor(KvCodec.fp8_e4m3, "fp8", "baseline"),
    }

    def descriptor_for(self, codec: KvCodec) -> CodecAdapterDescriptor | None:
        return self._baseline.get(codec)

    def supports(self, codec: KvCodec) -> bool:
        return codec in self._baseline

    def all_descriptors(self) -> list[CodecAdapterDescriptor]:
        return list(self._baseline.values())


def normalize_adapter_alias(alias: str) -> KvCodec:
    return normalize_codec_alias(alias)

