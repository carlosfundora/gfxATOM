import importlib.util
import re
from pathlib import Path


def _load_python_profile_class():
    source = (
        Path(__file__).resolve().parents[1] / "python" / "engine_runtime_profile.py"
    )
    spec = importlib.util.spec_from_file_location("engine_runtime_profile", source)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.EngineRuntimeProfile


def _extract_rust_profile_fields() -> list[str]:
    source = (
        Path(__file__).resolve().parents[1]
        / "crates"
        / "rs_atom_engine_profile"
        / "src"
        / "lib.rs"
    )
    text = source.read_text()
    struct_match = re.search(
        r"pub struct EngineRuntimeProfile\s*\{(?P<body>.*?)\n\}",
        text,
        flags=re.DOTALL,
    )
    assert struct_match is not None, "EngineRuntimeProfile struct not found in Rust source"
    body = struct_match.group("body")
    return re.findall(r"pub\s+([a-zA-Z0-9_]+)\s*:", body)


def test_python_and_rust_engine_runtime_profile_fields_match():
    engine_runtime_profile = _load_python_profile_class()
    python_fields = list(engine_runtime_profile.__dataclass_fields__.keys())
    rust_fields = _extract_rust_profile_fields()

    assert python_fields == rust_fields


def test_python_runtime_profile_adaptive_recommendation_helper():
    engine_runtime_profile = _load_python_profile_class()
    profile = engine_runtime_profile().with_adaptive_recommendation(
        {"family": "ratequant", "score": 0.82, "reason": "high_prefix_reuse_high_kv_hit"}
    )
    payload = profile.to_dict()
    assert payload["adaptive_recommendation"] is not None
    assert payload["adaptive_recommendation"]["family"] == "ratequant"
