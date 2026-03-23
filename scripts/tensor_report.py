from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from core.tensor_report import render_html, state_dict_report


def _extract_state_dict(obj: Any) -> dict[str, Tensor]:
    if isinstance(obj, dict) and all(isinstance(v, Tensor) for v in obj.values()):
        return obj
    if isinstance(obj, dict):
        for key in ("state_dict", "model", "model_state_dict"):
            value = obj.get(key)
            if isinstance(value, dict) and all(isinstance(v, Tensor) for v in value.values()):
                return value
    raise TypeError("Could not extract a tensor-only state_dict from the provided object")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a tensor report for a checkpoint or state dict.")
    parser.add_argument("--input", required=True, help="Path to a .pt/.pth state dict or checkpoint")
    parser.add_argument("--output-json", required=True, help="Path to write the JSON summary")
    parser.add_argument("--output-html", help="Optional path to write a self-contained HTML report")
    parser.add_argument("--top-k", type=int, default=32, help="Number of largest tensors to include prominently")
    parser.add_argument("--bins", type=int, default=32, help="Histogram bins per tensor")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    obj = torch.load(input_path, map_location="cpu")
    state_dict = _extract_state_dict(obj)
    report = state_dict_report(state_dict, bins=args.bins, top_k=args.top_k)
    Path(args.output_json).write_text(json.dumps(report, indent=2))
    if args.output_html:
        Path(args.output_html).write_text(render_html(report, title=f"Tensor Report: {input_path.name}"))
    print(
        f"wrote_tensor_report tensors={report['summary']['num_tensors']} "
        f"total_bytes={report['summary']['total_bytes']} json={args.output_json} "
        f"html={args.output_html or '-'}"
    )


if __name__ == "__main__":
    main()
