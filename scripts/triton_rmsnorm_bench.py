from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from core.triton_harness import BenchmarkSpec, assert_triton_close, do_bench, has_triton
from core.triton_rmsnorm import has_triton_rmsnorm, rmsnorm_reference, triton_rmsnorm


def parse_shape(text: str) -> tuple[int, ...]:
    return tuple(int(part) for part in text.lower().split("x") if part)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reference/correctness/benchmark harness for Triton RMSNorm")
    parser.add_argument("--shape", action="append", default=[], help="Tensor shape like 2048x512")
    parser.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--bench", action="store_true", help="Run Triton benchmark if available")
    parser.add_argument("--json", help="Optional output path for JSON results")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    shapes = [parse_shape(text) for text in args.shape] or [(256, 512), (2048, 512), (4096, 64)]
    dtype = getattr(torch, args.dtype)
    spec = BenchmarkSpec()
    results: list[dict[str, Any]] = []

    for shape in shapes:
        x = torch.randn(*shape, device="cuda" if torch.cuda.is_available() else "cpu", dtype=dtype)
        ref = rmsnorm_reference(x, eps=args.eps)
        item: dict[str, Any] = {
            "shape": list(shape),
            "dtype": str(dtype).replace("torch.", ""),
            "triton_available": has_triton() and has_triton_rmsnorm(),
        }
        if x.is_cuda and has_triton_rmsnorm():
            out = triton_rmsnorm(x, eps=args.eps)
            assert_triton_close(out.float(), ref.float(), atol=5e-3, rtol=5e-3)
            item["correct"] = True
            if args.bench:
                ref_ms = do_bench(lambda: rmsnorm_reference(x, eps=args.eps), warmup=spec.warmup, rep=spec.rep)
                tri_ms = do_bench(lambda: triton_rmsnorm(x, eps=args.eps), warmup=spec.warmup, rep=spec.rep)
                item["bench_ms"] = {
                    "reference": list(ref_ms),
                    "triton": list(tri_ms),
                }
        else:
            item["correct"] = None
        results.append(item)

    payload = {
        "kernel": "rmsnorm",
        "results": results,
    }
    text = json.dumps(payload, indent=2)
    if args.json:
        Path(args.json).write_text(text)
    print(text)


if __name__ == "__main__":
    main()
