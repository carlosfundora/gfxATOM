from dataclasses import dataclass
from enum import Enum


class KvCodec(str, Enum):
    auto = "auto"
    bf16 = "bf16"
    fp8_e4m3 = "fp8_e4m3"
    fp8_e5m2 = "fp8_e5m2"
    int8 = "int8"
    tq4 = "tq4"
    tq3 = "tq3"
    tq2 = "tq2"
    rq3_planar = "rq3_planar"
    rq4_planar = "rq4_planar"
    rq3_iso = "rq3_iso"
    rq4_iso = "rq4_iso"


class KvPolicyMode(str, Enum):
    static = "static"
    adaptive = "adaptive"
    learned = "learned"
    fallback = "fallback"


@dataclass(frozen=True)
class KvQuantPolicy:
    model_id: str
    codec: KvCodec
    mode: KvPolicyMode
    layer_id: int | None = None
    stage_id: str | None = None
    note: str | None = None


def normalize_codec_alias(alias: str) -> KvCodec:
    value = alias.lower()
    if value in {"bf16", "bfloat16"}:
        return KvCodec.bf16
    if value in {"fp8_e4m3", "atom_fp8"}:
        return KvCodec.fp8_e4m3
    if value == "fp8_e5m2":
        return KvCodec.fp8_e5m2
    if value == "int8":
        return KvCodec.int8
    if value in {"rq3", "rq3_planar"}:
        return KvCodec.rq3_planar
    if value in {"rq4", "rq4_planar"}:
        return KvCodec.rq4_planar
    if value == "rq3_iso":
        return KvCodec.rq3_iso
    if value == "rq4_iso":
        return KvCodec.rq4_iso
    if value in KvCodec.__members__:
        return KvCodec[value]
    raise ValueError(f"unsupported kv codec alias: {alias}")

