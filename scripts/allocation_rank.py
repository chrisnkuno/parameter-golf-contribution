from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


DEFAULT_TRAFFIC_HINTS: dict[str, float] = {
    "tok_emb.weight": 3.0,
    "lm_head.weight": 3.0,
    "attn_norm": 4.0,
    "mlp_norm": 4.0,
    "attn.c_q.weight": 4.0,
    "attn.c_k.weight": 4.0,
    "attn.c_v.weight": 4.0,
    "attn.proj.weight": 4.0,
    "mlp.fc.weight": 4.0,
    "mlp.proj.weight": 4.0,
    "attn_scale": 3.0,
    "mlp_scale": 3.0,
    "resid_mix": 3.0,
}


DEFAULT_SENSITIVITY_HINTS: dict[str, float] = {
    "tok_emb.weight": 3.0,
    "lm_head.weight": 3.0,
    "attn.proj.weight": 2.5,
    "mlp.proj.weight": 2.0,
    "mlp.fc.weight": 1.5,
    "attn.c_q.weight": 1.5,
    "attn.c_k.weight": 1.25,
    "attn.c_v.weight": 1.25,
    "attn_scale": 2.0,
    "mlp_scale": 2.0,
    "resid_mix": 2.0,
}


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


def _lookup_hint(name: str, table: dict[str, float], default: float) -> float:
    for pattern, value in table.items():
        if pattern in name:
            return value
    return default


def _residency(name: str) -> str:
    if any(pattern in name for pattern in ("running_", "momentum_buffer", "avg_state")):
        return "persistent_state"
    if any(pattern in name for pattern in ("attn_scale", "mlp_scale", "resid_mix", "skip_weight", "q_gain")):
        return "persistent_control"
    return "persistent_weight"


def _role(name: str) -> str:
    if "tok_emb.weight" in name or "lm_head" in name:
        return "embedding_io"
    if "attn." in name:
        return "attention"
    if "mlp." in name:
        return "mlp"
    if any(pattern in name for pattern in ("attn_scale", "mlp_scale", "resid_mix", "skip_weight", "q_gain")):
        return "control"
    return "other"


def _recommended_action(
    *,
    role: str,
    traffic: float,
    sensitivity: float,
    nbytes: int,
) -> str:
    if role == "embedding_io":
        return "protect_precision_or_precondition"
    if traffic >= 4.0 and role in {"attention", "mlp"}:
        return "kernel_or_fusion_candidate"
    if sensitivity >= 2.0 and nbytes >= 1_000_000:
        return "precondition_or_mixed_precision"
    if nbytes >= 1_000_000:
        return "quant_layout_candidate"
    if role == "control":
        return "keep_precise"
    return "inspect"


def rank_report(report: dict[str, Any]) -> dict[str, Any]:
    ranked = []
    for item in report["all_tensors"]:
        traffic = _lookup_hint(item["name"], DEFAULT_TRAFFIC_HINTS, 1.0)
        sensitivity = _lookup_hint(item["name"], DEFAULT_SENSITIVITY_HINTS, 1.0)
        zero_penalty = max(1.0 - float(item["zero_frac"]), 0.05)
        priority = float(item["nbytes"]) * traffic * sensitivity * zero_penalty
        role = _role(item["name"])
        ranked.append(
            {
                "name": item["name"],
                "shape": item["shape"],
                "dtype": item["dtype"],
                "nbytes": item["nbytes"],
                "traffic_hint": traffic,
                "sensitivity_hint": sensitivity,
                "zero_frac": item["zero_frac"],
                "residency": _residency(item["name"]),
                "role": role,
                "recommended_action": _recommended_action(
                    role=role,
                    traffic=traffic,
                    sensitivity=sensitivity,
                    nbytes=int(item["nbytes"]),
                ),
                "priority_score": priority,
            }
        )
    ranked.sort(key=lambda x: x["priority_score"], reverse=True)
    return {
        "summary": report["summary"],
        "ranked_tensors": ranked,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rank tensors by allocation priority")
    parser.add_argument("--input", required=True, help="Path to a checkpoint or state dict")
    parser.add_argument("--output", required=True, help="Path to write JSON results")
    parser.add_argument("--top-k", type=int, default=64, help="Number of top tensors to keep in output")
    return parser.parse_args()


def main() -> None:
    import torch

    from core.tensor_report import state_dict_report

    args = parse_args()
    obj = torch.load(args.input, map_location="cpu")
    state_dict = _extract_state_dict(obj)
    report = state_dict_report(state_dict, top_k=max(args.top_k, 64))
    ranked = rank_report(report)
    ranked["ranked_tensors"] = ranked["ranked_tensors"][: args.top_k]
    Path(args.output).write_text(json.dumps(ranked, indent=2))
    print(
        f"wrote_allocation_rank top_k={len(ranked['ranked_tensors'])} "
        f"total_bytes={ranked['summary']['total_bytes']} output={args.output}"
    )


if __name__ == "__main__":
    main()
