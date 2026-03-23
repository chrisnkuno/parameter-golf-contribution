from __future__ import annotations

import torch

from core.tensor_report import render_html, state_dict_report, tensor_stats


def test_tensor_stats_reports_shape_bytes_and_histogram() -> None:
    tensor = torch.tensor([[0.0, 1.0], [2.0, 0.0]], dtype=torch.float32)
    stats = tensor_stats("w", tensor, bins=4)
    assert stats["name"] == "w"
    assert stats["shape"] == [2, 2]
    assert stats["numel"] == 4
    assert stats["nbytes"] == 16
    assert stats["zero_frac"] == 0.5
    assert sum(stats["histogram"]["counts"]) == 4


def test_state_dict_report_sorts_largest_tensors_first() -> None:
    state_dict = {
        "small": torch.ones(2, dtype=torch.float32),
        "large": torch.ones(8, dtype=torch.float32),
    }
    report = state_dict_report(state_dict, top_k=2)
    assert report["summary"]["num_tensors"] == 2
    assert report["summary"]["total_bytes"] == 40
    assert report["top_tensors_by_nbytes"][0]["name"] == "large"
    assert report["top_tensors_by_nbytes"][1]["name"] == "small"


def test_render_html_contains_tensor_names() -> None:
    report = state_dict_report({"tok_emb.weight": torch.randn(4, 4)}, top_k=1)
    html = render_html(report, title="Demo")
    assert "<h1>Demo</h1>" in html
    assert "tok_emb.weight" in html
    assert "Histogram" in html
