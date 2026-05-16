from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from kv_policy_arbiter import (  # noqa: E402
    KvPolicyFamily,
    build_wave1_policy_profile,
    resolve_policy,
)


def test_ratequant_requires_global_and_feature_flags():
    decision = resolve_policy(
        "ratequant",
        donor_features_enabled=False,
        strict_mode=True,
        feature_overrides={"GFXATOM_KV_RATEQUANT": True},
    )
    assert decision.selected_family == KvPolicyFamily.baseline
    assert decision.accepted is False
    assert decision.rejection_reason == "donor_features_disabled"


def test_ratequant_accepts_when_enabled():
    decision = resolve_policy(
        "ratequant",
        donor_features_enabled=True,
        strict_mode=True,
        feature_overrides={"GFXATOM_KV_RATEQUANT": True},
    )
    assert decision.selected_family == KvPolicyFamily.ratequant
    assert decision.accepted is True
    assert decision.rejection_reason is None


def test_unknown_family_fails_closed():
    decision = resolve_policy(
        "not-a-policy",
        donor_features_enabled=True,
        strict_mode=True,
    )
    assert decision.selected_family == KvPolicyFamily.baseline
    assert decision.accepted is False
    assert decision.rejection_reason is not None


def test_wave1_profile_includes_ratequant_and_guardrail():
    profile = build_wave1_policy_profile(
        "ratequant",
        donor_features_enabled=True,
        strict_mode=True,
        feature_overrides={"GFXATOM_KV_RATEQUANT": True},
        sensitivities=[1.0, 8.0],
        total_budget=8.0,
        total_bits_per_value=4,
        outlier_ratio=0.2,
    )
    assert profile.decision.selected_family == KvPolicyFamily.ratequant
    assert profile.ratequant is not None
    assert profile.turboquant is not None
    assert profile.guardrail is not None
    assert profile.guardrail.accepted is False
