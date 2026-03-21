from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


TRAIN_STEP_RE = re.compile(r"step:(?P<step>\d+)/(?P<iters>\d+) train_loss:.* step_avg:(?P<step_ms>\d+(?:\.\d+)?)ms")
TRAIN_BATCH_TOKENS_RE = re.compile(r"train_batch_tokens:(?P<train_batch_tokens>\d+)")
REPO_ROOT = Path(__file__).resolve().parents[1]

ANCHOR_PRESETS = {
    "winner_2026_03_19": {
        "source": "records/track_10min_16mb/2026-03-19_SlidingWindow_FP16Emb_10L_MuonWD_OvertoneInit/submission.json",
    },
    "naive_2026_03_17": {
        "source": "records/track_10min_16mb/2026-03-17_NaiveBaseline/train.log",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Estimate 8xH100 train timing from 5090 logs using a calibrated anchor."
    )
    parser.add_argument("--anchor-log", type=Path, required=True, help="5090 log for the reference config.")
    parser.add_argument("--target-log", type=Path, required=True, help="5090 log for the target config.")
    parser.add_argument(
        "--anchor-preset",
        choices=sorted(ANCHOR_PRESETS),
        default="winner_2026_03_19",
        help="Known 8xH100 reference for the anchor config.",
    )
    parser.add_argument(
        "--anchor-8xh100-ms",
        type=float,
        default=None,
        help="Override the preset 8xH100 ms/step for the anchor config.",
    )
    parser.add_argument(
        "--tail",
        type=int,
        default=50,
        help="Average this many final train step_avg entries from each log.",
    )
    parser.add_argument(
        "--budget-seconds",
        type=float,
        default=600.0,
        help="Wallclock budget to convert into estimated steps.",
    )
    parser.add_argument(
        "--safety-factor",
        type=float,
        default=0.90,
        help="Fraction of the budget treated as safely usable. 0.90 means target <= 540s under a 600s cap.",
    )
    return parser.parse_args()


def extract_train_step_ms(log_path: Path, tail: int) -> tuple[float, int, int | None]:
    text = log_path.read_text(encoding="utf-8")
    matches = list(TRAIN_STEP_RE.finditer(text))
    if not matches:
        raise ValueError(f"No train step_avg lines found in {log_path}")
    tail_matches = matches[-tail:]
    avg_ms = sum(float(match.group("step_ms")) for match in tail_matches) / len(tail_matches)
    final_step = int(matches[-1].group("step"))
    batch_match = TRAIN_BATCH_TOKENS_RE.search(text)
    train_batch_tokens = int(batch_match.group("train_batch_tokens")) if batch_match else None
    return avg_ms, final_step, train_batch_tokens


def format_tokens(value: float) -> str:
    if value >= 1_000_000_000:
        return f"{value / 1_000_000_000:.3f}B"
    if value >= 1_000_000:
        return f"{value / 1_000_000:.3f}M"
    return f"{value:.0f}"


def resolve_anchor_8xh100_ms(preset_name: str, tail: int) -> tuple[float, Path]:
    source = REPO_ROOT / ANCHOR_PRESETS[preset_name]["source"]
    if source.suffix == ".json":
        payload = json.loads(source.read_text(encoding="utf-8"))
        seed_results = payload.get("seed_results", {})
        ms_values = [float(run["ms_per_step"]) for run in seed_results.values()]
        if not ms_values:
            raise ValueError(f"No seed_results.ms_per_step values found in {source}")
        return sum(ms_values) / len(ms_values), source
    if source.suffix == ".log":
        ms_per_step, _, _ = extract_train_step_ms(source, tail)
        return ms_per_step, source
    raise ValueError(f"Unsupported preset source type for {source}")


def main() -> None:
    args = parse_args()
    preset = ANCHOR_PRESETS[args.anchor_preset]
    resolved_anchor_8xh100_ms, preset_source = resolve_anchor_8xh100_ms(args.anchor_preset, args.tail)
    anchor_8xh100_ms = args.anchor_8xh100_ms or resolved_anchor_8xh100_ms

    anchor_5090_ms, _, anchor_batch_tokens = extract_train_step_ms(args.anchor_log, args.tail)
    target_5090_ms, target_steps, target_batch_tokens = extract_train_step_ms(args.target_log, args.tail)

    calibration_factor = anchor_5090_ms / anchor_8xh100_ms
    est_target_8xh100_ms = target_5090_ms / calibration_factor
    est_steps_in_budget = (args.budget_seconds * 1000.0) / est_target_8xh100_ms
    safe_budget_seconds = args.budget_seconds * args.safety_factor
    est_steps_in_safe_budget = (safe_budget_seconds * 1000.0) / est_target_8xh100_ms
    est_target_runtime_seconds = (target_steps * est_target_8xh100_ms) / 1000.0

    print(f"anchor_preset: {args.anchor_preset}")
    print(f"anchor_source: {preset['source']}")
    print(f"anchor_source_path: {preset_source}")
    print(f"anchor_8xh100_ms_per_step: {anchor_8xh100_ms:.4f}")
    print(f"anchor_5090_ms_per_step: {anchor_5090_ms:.4f}")
    print(f"calibration_factor_5090_to_8xh100: {calibration_factor:.6f}")
    print()
    print(f"target_5090_ms_per_step: {target_5090_ms:.4f}")
    print(f"estimated_8xh100_ms_per_step: {est_target_8xh100_ms:.4f}")
    print(f"estimated_steps_in_{args.budget_seconds:.0f}s: {est_steps_in_budget:.1f}")
    print(f"estimated_steps_in_safe_budget_{safe_budget_seconds:.0f}s: {est_steps_in_safe_budget:.1f}")
    print(f"estimated_8xh100_runtime_for_{target_steps}_steps: {est_target_runtime_seconds:.2f}s")
    if target_batch_tokens is not None:
        print(f"target_train_batch_tokens: {target_batch_tokens}")
        print(
            "estimated_tokens_in_budget: "
            f"{format_tokens(est_steps_in_budget * target_batch_tokens)}"
        )
        print(
            "estimated_tokens_in_safe_budget: "
            f"{format_tokens(est_steps_in_safe_budget * target_batch_tokens)}"
        )
    if anchor_batch_tokens is not None and target_batch_tokens is not None and anchor_batch_tokens != target_batch_tokens:
        print(
            "warning: anchor and target TRAIN_BATCH_TOKENS differ; "
            "the calibration may be less reliable."
        )


if __name__ == "__main__":
    main()
