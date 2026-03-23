from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.compare_allocation_rank import compare_ranked


def _extract_state_dict(obj: Any) -> dict[str, Any]:
    import torch

    if isinstance(obj, dict) and all(isinstance(v, torch.Tensor) for v in obj.values()):
        return obj
    if isinstance(obj, dict):
        for key in ("state_dict", "model", "model_state_dict"):
            value = obj.get(key)
            if isinstance(value, dict) and all(isinstance(v, torch.Tensor) for v in value.values()):
                return value
    raise TypeError("Could not extract a tensor-only state_dict from the provided object")


def _load_report(path: str, top_k: int) -> dict[str, Any]:
    import torch

    from core.tensor_report import state_dict_report

    obj = torch.load(path, map_location="cpu")
    state_dict = _extract_state_dict(obj)
    return state_dict_report(state_dict, top_k=max(top_k, 64))


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2))


def _render_summary_markdown(
    *,
    baseline_path: str,
    baseline_ranked: dict[str, Any],
    candidate_path: str | None = None,
    candidate_ranked: dict[str, Any] | None = None,
    diff_payload: dict[str, Any] | None = None,
) -> str:
    baseline_summary = baseline_ranked["summary"]
    lines = [
        "# Checkpoint Analysis",
        "",
        f"- baseline: `{Path(baseline_path).name}`",
        f"- baseline_total_bytes: `{baseline_summary['total_bytes']}`",
        "",
        "## Baseline Top Allocation Targets",
    ]
    for item in baseline_ranked["ranked_tensors"][:10]:
        lines.append(
            f"- `{item['name']}` bytes={item['nbytes']} "
            f"priority={item['priority_score']:.2f} "
            f"action={item['recommended_action']}"
        )
    if candidate_path and candidate_ranked and diff_payload:
        candidate_summary = candidate_ranked["summary"]
        lines.extend(
            [
                "",
                f"- candidate: `{Path(candidate_path).name}`",
                f"- candidate_total_bytes: `{candidate_summary['total_bytes']}`",
                "",
                "## Largest Allocation Deltas",
            ]
        )
        for item in diff_payload["diff"][:10]:
            lines.append(
                f"- `{item['name']}` priority_delta={item['priority_delta']:.2f} "
                f"bytes_delta={item['bytes_delta']} "
                f"baseline_action={item['baseline_action']} "
                f"candidate_action={item['candidate_action']}"
            )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a full tensor/allocation analysis bundle")
    parser.add_argument("--baseline", required=True, help="Baseline checkpoint path")
    parser.add_argument("--candidate", help="Optional candidate checkpoint path for diffing")
    parser.add_argument("--output-dir", required=True, help="Directory to write analysis artifacts into")
    parser.add_argument("--top-k", type=int, default=64, help="How many tensors to keep in ranked outputs")
    return parser.parse_args()


def main() -> None:
    from core.tensor_report import render_html
    from scripts.allocation_rank import rank_report

    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    baseline_report = _load_report(args.baseline, args.top_k)
    baseline_ranked = rank_report(baseline_report)
    _write_json(out_dir / "baseline_tensor_report.json", baseline_report)
    _write_json(out_dir / "baseline_allocation_rank.json", baseline_ranked)
    (out_dir / "baseline_tensor_report.html").write_text(
        render_html(baseline_report, title=f"Tensor Report: {Path(args.baseline).name}")
    )

    if args.candidate:
        candidate_report = _load_report(args.candidate, args.top_k)
        candidate_ranked = rank_report(candidate_report)
        _write_json(out_dir / "candidate_tensor_report.json", candidate_report)
        _write_json(out_dir / "candidate_allocation_rank.json", candidate_ranked)
        (out_dir / "candidate_tensor_report.html").write_text(
            render_html(candidate_report, title=f"Tensor Report: {Path(args.candidate).name}")
        )
        diff_payload = compare_ranked(baseline_ranked, candidate_ranked, top_k=args.top_k)
        _write_json(out_dir / "allocation_diff.json", diff_payload)
        (out_dir / "summary.md").write_text(
            _render_summary_markdown(
                baseline_path=args.baseline,
                baseline_ranked=baseline_ranked,
                candidate_path=args.candidate,
                candidate_ranked=candidate_ranked,
                diff_payload=diff_payload,
            )
        )
    else:
        (out_dir / "summary.md").write_text(
            _render_summary_markdown(baseline_path=args.baseline, baseline_ranked=baseline_ranked)
        )

    manifest = {
        "baseline": args.baseline,
        "candidate": args.candidate,
        "top_k": args.top_k,
        "files": sorted(path.name for path in out_dir.iterdir()),
    }
    _write_json(out_dir / "manifest.json", manifest)
    print(f"wrote_checkpoint_analysis output_dir={out_dir}")


if __name__ == "__main__":
    main()
