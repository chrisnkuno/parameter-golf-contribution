from __future__ import annotations

from scripts.checkpoint_analysis import _render_summary_markdown


def test_render_summary_markdown_includes_actions_and_deltas() -> None:
    baseline_ranked = {
        "summary": {"total_bytes": 1000},
        "ranked_tensors": [
            {
                "name": "tok_emb.weight",
                "nbytes": 1000,
                "priority_score": 100.0,
                "recommended_action": "protect_precision_or_precondition",
            }
        ],
    }
    candidate_ranked = {
        "summary": {"total_bytes": 1200},
        "ranked_tensors": [
            {
                "name": "tok_emb.weight",
                "nbytes": 1200,
                "priority_score": 130.0,
                "recommended_action": "protect_precision_or_precondition",
            }
        ],
    }
    diff_payload = {
        "diff": [
            {
                "name": "tok_emb.weight",
                "priority_delta": 30.0,
                "bytes_delta": 200,
                "baseline_action": "protect_precision_or_precondition",
                "candidate_action": "protect_precision_or_precondition",
            }
        ]
    }

    text = _render_summary_markdown(
        baseline_path="baseline.pt",
        baseline_ranked=baseline_ranked,
        candidate_path="candidate.pt",
        candidate_ranked=candidate_ranked,
        diff_payload=diff_payload,
    )
    assert "`baseline.pt`" in text
    assert "`candidate.pt`" in text
    assert "Largest Allocation Deltas" in text
    assert "protect_precision_or_precondition" in text
