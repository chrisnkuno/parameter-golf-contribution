from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


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


def _ranked_map(path: str, top_k: int) -> tuple[dict[str, Any], dict[str, Any]]:
    import torch

    from core.tensor_report import state_dict_report
    from scripts.allocation_rank import rank_report

    obj = torch.load(path, map_location="cpu")
    state_dict = _extract_state_dict(obj)
    report = state_dict_report(state_dict, top_k=max(top_k, 64))
    ranked = rank_report(report)
    ranked_map = {item["name"]: item for item in ranked["ranked_tensors"]}
    return ranked, ranked_map


def compare_ranked(
    baseline_ranked: dict[str, Any],
    candidate_ranked: dict[str, Any],
    *,
    top_k: int,
) -> dict[str, Any]:
    baseline_map = {item["name"]: item for item in baseline_ranked["ranked_tensors"]}
    candidate_map = {item["name"]: item for item in candidate_ranked["ranked_tensors"]}
    names = sorted(set(baseline_map) | set(candidate_map))
    rows = []
    for name in names:
        base = baseline_map.get(name)
        cand = candidate_map.get(name)
        base_score = float(base["priority_score"]) if base else 0.0
        cand_score = float(cand["priority_score"]) if cand else 0.0
        base_bytes = int(base["nbytes"]) if base else 0
        cand_bytes = int(cand["nbytes"]) if cand else 0
        rows.append(
            {
                "name": name,
                "baseline_priority": base_score,
                "candidate_priority": cand_score,
                "priority_delta": cand_score - base_score,
                "baseline_bytes": base_bytes,
                "candidate_bytes": cand_bytes,
                "bytes_delta": cand_bytes - base_bytes,
                "baseline_zero_frac": float(base["zero_frac"]) if base else None,
                "candidate_zero_frac": float(cand["zero_frac"]) if cand else None,
                "baseline_action": base["recommended_action"] if base else None,
                "candidate_action": cand["recommended_action"] if cand else None,
            }
        )
    rows.sort(key=lambda item: abs(item["priority_delta"]), reverse=True)

    return {
        "baseline": {
            "summary": baseline_ranked["summary"],
        },
        "candidate": {
            "summary": candidate_ranked["summary"],
        },
        "diff": rows[:top_k],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare allocation ranking across two checkpoints")
    parser.add_argument("--baseline", required=True, help="Reference checkpoint path")
    parser.add_argument("--candidate", required=True, help="Candidate checkpoint path")
    parser.add_argument("--output", required=True, help="JSON output path")
    parser.add_argument("--top-k", type=int, default=64, help="How many tensors to keep from each side")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    baseline_ranked, _ = _ranked_map(args.baseline, args.top_k)
    candidate_ranked, _ = _ranked_map(args.candidate, args.top_k)
    payload = compare_ranked(baseline_ranked, candidate_ranked, top_k=args.top_k)
    payload["baseline"]["path"] = args.baseline
    payload["candidate"]["path"] = args.candidate
    Path(args.output).write_text(json.dumps(payload, indent=2))
    print(
        f"wrote_allocation_diff rows={len(payload['diff'])} "
        f"baseline_bytes={payload['baseline']['summary']['total_bytes']} "
        f"candidate_bytes={payload['candidate']['summary']['total_bytes']} "
        f"output={args.output}"
    )


if __name__ == "__main__":
    main()
