from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from wave1_donor_adapters import (  # noqa: E402
    cpu_bench_summary_from_requests,
    norm_separated_guardrail,
    ratequant_optimal_bit_allocation,
    turboquant_profile,
)


def test_ratequant_allocation_biases_high_sensitivity():
    profile = ratequant_optimal_bit_allocation([1.0, 16.0], total_budget=6.0)
    assert len(profile.bit_allocations) == 2
    assert profile.bit_allocations[1] > profile.bit_allocations[0]


def test_norm_guardrail_rejects_low_bit_high_outlier():
    guard = norm_separated_guardrail(requested_bits=4, outlier_ratio=0.2)
    assert guard.accepted is False
    assert guard.recommended_bits == 8
    assert guard.reason == "low_bit_outlier_risk"


def test_turboquant_profile_split_is_consistent():
    profile = turboquant_profile(vector_dim=128, total_bits_per_value=4)
    assert profile.polar_bits_per_value == 3
    assert profile.qjl_bits_per_value == 1
    assert profile.qjl_projections > 0


def test_cpu_bench_summary_aggregates_records():
    summary = cpu_bench_summary_from_requests(
        [
            {"e2e_ms": 10.0, "tokens_per_second": 120.0},
            {"e2e_ms": 30.0, "tokens_per_second": 80.0},
        ]
    )
    assert summary.num_requests == 2
    assert summary.avg_e2e_ms == 20.0
    assert summary.p95_e2e_ms == 10.0
    assert summary.avg_tokens_per_second == 100.0
