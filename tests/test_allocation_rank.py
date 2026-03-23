from __future__ import annotations

from scripts.allocation_rank import rank_report


def test_rank_report_prefers_large_high_hint_tensors() -> None:
    report = {
        "summary": {"num_tensors": 2, "total_numel": 0, "total_bytes": 0, "float_bytes": 0, "weighted_zero_frac": 0.0, "weighted_finite_frac": 1.0},
        "all_tensors": [
            {
                "name": "tok_emb.weight",
                "shape": [1024, 512],
                "dtype": "torch.float32",
                "nbytes": 1000,
                "zero_frac": 0.0,
            },
            {
                "name": "misc.small",
                "shape": [4],
                "dtype": "torch.float32",
                "nbytes": 100,
                "zero_frac": 0.0,
            },
        ],
    }
    ranked = rank_report(report)
    assert ranked["ranked_tensors"][0]["name"] == "tok_emb.weight"
    assert ranked["ranked_tensors"][0]["priority_score"] > ranked["ranked_tensors"][1]["priority_score"]


def test_rank_report_penalizes_high_zero_fraction() -> None:
    report = {
        "summary": {"num_tensors": 2, "total_numel": 0, "total_bytes": 0, "float_bytes": 0, "weighted_zero_frac": 0.0, "weighted_finite_frac": 1.0},
        "all_tensors": [
            {
                "name": "mlp.proj.weight",
                "shape": [8, 8],
                "dtype": "torch.float32",
                "nbytes": 1000,
                "zero_frac": 0.9,
            },
            {
                "name": "mlp.proj.weight_copy",
                "shape": [8, 8],
                "dtype": "torch.float32",
                "nbytes": 1000,
                "zero_frac": 0.0,
            },
        ],
    }
    ranked = rank_report(report)
    assert ranked["ranked_tensors"][0]["name"] == "mlp.proj.weight_copy"
