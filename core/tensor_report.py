from __future__ import annotations

from html import escape
from typing import Any

import numpy as np
import torch
from torch import Tensor


def _float_view(tensor: Tensor) -> Tensor:
    if tensor.is_floating_point():
        return tensor.detach().to(torch.float32, copy=False).cpu()
    if tensor.dtype == torch.bool:
        return tensor.detach().to(torch.int32).to(torch.float32).cpu()
    return tensor.detach().to(torch.float32).cpu()


def tensor_stats(name: str, tensor: Tensor, bins: int = 32) -> dict[str, Any]:
    t = tensor.detach().cpu()
    flat = _float_view(t).reshape(-1)
    numel = int(t.numel())
    nbytes = int(t.numel() * t.element_size())
    finite_mask = torch.isfinite(flat) if flat.numel() else torch.empty(0, dtype=torch.bool)
    finite_flat = flat[finite_mask]
    zero_frac = float((flat == 0).float().mean().item()) if flat.numel() else 0.0
    finite_frac = float(finite_mask.float().mean().item()) if flat.numel() else 1.0

    if finite_flat.numel():
        mean = float(finite_flat.mean().item())
        std = float(finite_flat.std(unbiased=False).item())
        min_value = float(finite_flat.min().item())
        max_value = float(finite_flat.max().item())
        abs_mean = float(finite_flat.abs().mean().item())
        l2_norm = float(torch.linalg.vector_norm(finite_flat).item())
        hist_counts, hist_edges = np.histogram(finite_flat.numpy(), bins=bins)
        hist = {
            "counts": hist_counts.astype(int).tolist(),
            "edges": hist_edges.astype(float).tolist(),
        }
    else:
        mean = std = min_value = max_value = abs_mean = l2_norm = 0.0
        hist = {"counts": [], "edges": []}

    return {
        "name": name,
        "shape": list(t.shape),
        "dtype": str(t.dtype),
        "numel": numel,
        "nbytes": nbytes,
        "mean": mean,
        "std": std,
        "min": min_value,
        "max": max_value,
        "abs_mean": abs_mean,
        "l2_norm": l2_norm,
        "zero_frac": zero_frac,
        "finite_frac": finite_frac,
        "histogram": hist,
    }


def state_dict_report(
    state_dict: dict[str, Tensor],
    *,
    bins: int = 32,
    top_k: int = 32,
) -> dict[str, Any]:
    tensors = [tensor_stats(name, tensor, bins=bins) for name, tensor in state_dict.items()]
    tensors.sort(key=lambda item: item["nbytes"], reverse=True)
    total_bytes = int(sum(item["nbytes"] for item in tensors))
    total_numel = int(sum(item["numel"] for item in tensors))
    float_bytes = int(sum(item["nbytes"] for item in tensors if "float" in item["dtype"]))
    zero_weighted_sum = sum(item["zero_frac"] * item["numel"] for item in tensors)
    finite_weighted_sum = sum(item["finite_frac"] * item["numel"] for item in tensors)
    return {
        "summary": {
            "num_tensors": len(tensors),
            "total_numel": total_numel,
            "total_bytes": total_bytes,
            "float_bytes": float_bytes,
            "weighted_zero_frac": zero_weighted_sum / max(total_numel, 1),
            "weighted_finite_frac": finite_weighted_sum / max(total_numel, 1),
        },
        "top_tensors_by_nbytes": tensors[:top_k],
        "all_tensors": tensors,
    }


def _svg_histogram(counts: list[int], width: int = 240, height: int = 48) -> str:
    if not counts:
        return ""
    max_count = max(counts) or 1
    bar_width = max(width / len(counts), 1.0)
    rects: list[str] = []
    for idx, count in enumerate(counts):
        bar_height = height * (count / max_count)
        x = idx * bar_width
        y = height - bar_height
        rects.append(
            f'<rect x="{x:.2f}" y="{y:.2f}" width="{bar_width - 1:.2f}" '
            f'height="{bar_height:.2f}" fill="#3b82f6" />'
        )
    return (
        f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" '
        f'xmlns="http://www.w3.org/2000/svg">{"".join(rects)}</svg>'
    )


def render_html(report: dict[str, Any], *, title: str = "Tensor Report") -> str:
    rows: list[str] = []
    for item in report["top_tensors_by_nbytes"]:
        rows.append(
            "<tr>"
            f"<td><code>{escape(item['name'])}</code></td>"
            f"<td>{escape(str(item['shape']))}</td>"
            f"<td>{escape(item['dtype'])}</td>"
            f"<td>{item['nbytes']}</td>"
            f"<td>{item['min']:.4g}</td>"
            f"<td>{item['max']:.4g}</td>"
            f"<td>{item['std']:.4g}</td>"
            f"<td>{item['zero_frac']:.4f}</td>"
            f"<td>{_svg_histogram(item['histogram']['counts'])}</td>"
            "</tr>"
        )
    summary = report["summary"]
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>{escape(title)}</title>
  <style>
    body {{ font-family: ui-monospace, SFMono-Regular, Menlo, monospace; margin: 24px; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border: 1px solid #ddd; padding: 6px 8px; vertical-align: top; }}
    th {{ background: #f3f4f6; text-align: left; }}
    code {{ white-space: nowrap; }}
    .summary {{ margin-bottom: 24px; }}
  </style>
</head>
<body>
  <h1>{escape(title)}</h1>
  <div class="summary">
    <p>num_tensors={summary['num_tensors']} total_numel={summary['total_numel']}
    total_bytes={summary['total_bytes']} float_bytes={summary['float_bytes']}
    weighted_zero_frac={summary['weighted_zero_frac']:.6f}
    weighted_finite_frac={summary['weighted_finite_frac']:.6f}</p>
  </div>
  <table>
    <thead>
      <tr>
        <th>Name</th>
        <th>Shape</th>
        <th>DType</th>
        <th>Bytes</th>
        <th>Min</th>
        <th>Max</th>
        <th>Std</th>
        <th>Zero Frac</th>
        <th>Histogram</th>
      </tr>
    </thead>
    <tbody>
      {"".join(rows)}
    </tbody>
  </table>
</body>
</html>
"""

