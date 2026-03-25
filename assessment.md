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

Running Baseline on 2×RTX 5090
Here's what the repo does and what you need to adapt for 2×5090.

How Distributed Training Works
In train_gpt.py main(), the key constraint is:

python
if 8 % world_size != 0:
    raise ValueError(f"WORLD_SIZE={world_size} must divide 8 so grad_accum_steps stays integral")
grad_accum_steps = 8 // world_size
So valid world sizes are: 1, 2, 4, 8. For 2×5090 → nproc_per_node=2, grad_accum_steps=4 (same effective batch size as 8xH100, just 4 microsteps per GPU instead of 1).

Training uses standard torchrun + DDP. No code edits needed.

Baseline Command for 2×5090
bash
# 1. Download data (one-time)
python3 data/cached_challenge_fineweb.py --variant sp1024
# 2. Run baseline
RUN_ID=baseline_2x5090 \
DATA_PATH=./data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
MAX_WALLCLOCK_SECONDS=600 \
VAL_LOSS_EVERY=200 \
TRAIN_LOG_EVERY=50 \
NCCL_IB_DISABLE=1 \
torchrun --standalone --nproc_per_node=2 train_gpt.py
That's it — no model config changes needed. The 8xH100 baseline config defaults are:

VOCAB_SIZE=1024 NUM_LAYERS=9 MODEL_DIM=512 NUM_HEADS=8 NUM_KV_HEADS=4 MLP_MULT=2
TRAIN_BATCH_TOKENS=524288 TRAIN_SEQ_LEN=1024 TIE_EMBEDDINGS=1
What to Expect vs 8×H100
8×H100 baseline	2×5090
grad_accum_steps	1	4
step_avg	~43.5 ms	slower (4 microsteps + weaker interconnect)
Peak mem/GPU	~10.2 GB	similar or lower (smaller batch per rank)
10-min val_bpb	1.2244	will see fewer steps → higher BPB
Score (leaderboard)	this IS the baseline	not a valid leaderboard entry (requires 8xH100)
Key caveat: The challenge rules require final leaderboard submissions to run on 8×H100 SXM under 10 min. The 2×5090 run is for local iteration/research only — not a valid record submission.

Tips for 5090
Flash Attention is already enabled (enable_flash_sdp(True)) — the 5090 with Blackwell architecture needs PyTorch ≥ 2.7 or a build with FA3 support. Check: python -c "import torch; print(torch.__version__)".

torch.compile is on by default (COMPILE_MODEL=1). First run will be slow (~10 min compile overhead). For quick smoke tests: COMPILE_MODEL=0.

Longer uncapped runs: The max_wallclock_seconds=600 cap means the 2×5090 will stop at fewer steps. To explore quality: MAX_WALLCLOCK_SECONDS=0 (unlimited, runs all 20,000 iters).

Memory: Each rank gets train_batch_tokens / (world_size * grad_accum_steps) = 524288 / (2*4) = 65536 tokens = 64 sequences of 1024 — well within 24GB VRAM.