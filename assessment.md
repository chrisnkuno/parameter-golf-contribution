# Repo Assessment

## What Is Working

- `core/metric_core.py` and `core/schedule_core.py` are the strongest parts of the repo.
  - Metric accounting is explicit.
  - Eval scheduling is small, testable, and close to the benchmark contract.
- `core/quant_core.py` and `core/artifact_core.py` are moving in the right direction.
  - Post-quant roundtrip BPB is treated as the real objective.
  - Artifact packing is modular enough to evolve.
- The new low-level layer is now viable:
  - tensor reports
  - allocation ranking
  - allocation diffing
  - Triton harness
  - first real kernel target (`RMSNorm`)

## What We Did Wrong

1. We explored architecture novelty before locking benchmark alignment.
   - XSA, ZeroS, local mixers, and shared-depth variants were worth trying, but the repo's own records show larger gains from eval, quantization, and wider MLP regimes first.
2. We relied on reduced-data development runs for too long.
   - One-shard runs are useful for bootstrap, not for leaderboard conclusions.
3. We treated the `16 MB` cap too late at first.
   - The correct objective is the best legal post-quant checkpoint, not the best dense or over-cap checkpoint.
4. Too much experiment surface lives inside `train_gpt.py`.
   - Training, eval, TTT, architecture branches, quant/export, checkpoint frontier, and model averaging all share one file.
5. Low-level observability arrived late.
   - Without tensor-level inspection and allocation ranking, we were reasoning from aggregate scores alone.

## Current Architecture View

The `11L + XSA4` branch is coherent, but it is not yet proven to be the best mainline for leaderboard chasing.

What the repo and record history still suggest more strongly:

- `10-11` layers
- wider MLPs
- stronger eval geometry
- better post-quant policies
- low-bit-aware late training

So `11L + XSA4` should be treated as a good branch, not a settled thesis.

## How To Keep Lowering BPB

1. Use full-data, fixed-eval discipline for all serious comparisons.
2. Rank only by the constrained objective:
   - lowest post-quant exact BPB
   - under the legal byte cap
3. Return to proven levers:
   - wider MLP
   - mixed lower precision
   - stronger eval
   - quant-aware late training
4. Spend artifact effort on payload quality, not metadata.
5. Spend systems effort on a few hot kernels, not broad novelty.

## Low-Level Research Program

The current best low-level sequence is:

1. run `scripts/checkpoint_analysis.py` on the best legal and best raw checkpoints
2. inspect which tensors dominate bytes and allocation priority
3. benchmark Triton `RMSNorm`
4. integrate Triton `RMSNorm` behind a feature flag
5. implement fused `ReLU^2` MLP glue
6. implement rowwise quant/dequant kernels

## Data Handling View

What is good:

- deterministic shard handling
- explicit validation loading
- rigorous metric accounting

What is weak:

- little overlap/prefetch instrumentation
- checkpoint frontier writes are still a systems afterthought
- host/device data movement is not profiled tightly enough

Highest-value data-handling work:

- async checkpoint frontier writes
- clearer separation of hot tensors vs streamed tensors
- H100 profiler traces tied back to real shapes

## Immediate Priorities

1. Finish validating the new checkpoint analysis bundle on real checkpoints.
2. Benchmark Triton `RMSNorm` on actual repo shapes.
3. Integrate Triton `RMSNorm` only in eager/inference mode.
4. Start a wider-MLP + stronger low-bit branch once the observability loop is working.
