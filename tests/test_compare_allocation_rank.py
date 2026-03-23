from __future__ import annotations

from scripts.compare_allocation_rank import compare_ranked


def test_compare_ranked_emits_priority_and_action_deltas() -> None:
    baseline_ranked = {
        "summary": {"total_bytes": 1010},
        "ranked_tensors": [
            {
                "name": "tok_emb.weight",
                "nbytes": 1000,
                "zero_frac": 0.0,
                "priority_score": 100.0,
                "recommended_action": "protect_precision_or_precondition",
            }
        ],
    }
    candidate_ranked = {
        "summary": {"total_bytes": 2010},
        "ranked_tensors": [
            {
                "name": "tok_emb.weight",
                "nbytes": 1200,
                "zero_frac": 0.0,
                "priority_score": 140.0,
                "recommended_action": "protect_precision_or_precondition",
            },
            {
                "name": "mlp.proj.weight",
                "nbytes": 800,
                "zero_frac": 0.0,
                "priority_score": 20.0,
                "recommended_action": "kernel_or_fusion_candidate",
            },
        ],
    }

    payload = compare_ranked(baseline_ranked, candidate_ranked, top_k=4)
    assert payload["baseline"]["summary"]["total_bytes"] == 1010
    assert payload["candidate"]["summary"]["total_bytes"] == 2010
    assert payload["diff"][0]["name"] == "tok_emb.weight"
    assert payload["diff"][0]["priority_delta"] == 40.0
    assert payload["diff"][0]["candidate_action"] == "protect_precision_or_precondition"
