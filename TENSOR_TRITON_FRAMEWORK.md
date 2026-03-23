# Tensor + Triton Framework

This document defines the local workflow for low-level tensor inspection and future Triton kernel work in this repo.

## Goals

We want a framework that is:

- observable: inspect tensor distributions, byte concentration, sparsity, and numerical range
- verifiable: compare every candidate kernel against a trusted PyTorch reference
- benchmarkable: measure wall time on real challenge shapes, not toy shapes
- hardware-aware: reason about memory traffic, layout, and launch geometry

## Immediate Local Tooling

### Tensor report

Use:

```bash
python scripts/tensor_report.py \
  --input path/to/checkpoint.pt \
  --output-json /tmp/report.json \
  --output-html /tmp/report.html
```

Allocation ranking:

```bash
python scripts/allocation_rank.py \
  --input path/to/checkpoint.pt \
  --output /tmp/allocation_rank.json
```

This reports:

- per-tensor bytes
- shape and dtype
- min / max / std
- zero fraction
- histogram shape

Start with:

- `tok_emb.weight`
- attention projections
- MLP weights
- quantization scales / passthrough tensors

The point is to make byte-heavy or numerically awkward tensors obvious before changing quantization or kernel layout.
Use `ALLOCATION_POLICY.md` and `scripts/allocation_rank.py` to convert that inspection into a prioritized optimization list.

### Triton harness

Use:

```bash
python scripts/triton_rmsnorm_bench.py
```

Current local files:

- `core/triton_harness.py`
- `core/triton_rmsnorm.py`
- `scripts/triton_rmsnorm_bench.py`
- `tests/test_triton_harness.py`

These define:

- kernel case descriptors
- benchmark defaults
- Triton availability checks
- a first real kernel target: forward `RMSNorm`

## Triton Verification Loop

For each candidate kernel:

1. define a pure PyTorch reference
2. run fixed-shape correctness tests
3. run edge-shape correctness tests
4. benchmark on real model shapes
5. profile with hardware tools only after correctness is locked

## Invariants

Every Triton candidate should satisfy:

- exact output shape match
- dtype contract is explicit
- `torch.testing.assert_close(...)` against reference
- no NaN / inf creation on representative inputs
- deterministic benchmark inputs
- benchmark run on the actual tensor shapes used by `train_gpt.py`

Suggested tolerances:

- exact equality for indexing / masking / layout kernels
- strict `rtol` / `atol` for numerical kernels, recorded per kernel

## Debug / Validation Stack

Use official Triton / PyTorch / NVIDIA tooling:

- Triton interpreter mode: `TRITON_INTERPRET=1`
- Triton compile-time checks: `tl.static_assert`
- Triton runtime checks: `tl.device_assert`
- Triton benchmarking helpers: `triton.testing.do_bench`, `triton.testing.perf_report`
- PyTorch profiler for operator + CUDA trace inspection
- Nsight Compute CLI for kernel-level profiling

## Candidate Kernel Order

The likely high-value order for this repo is:

1. per-row quantization / dequantization helpers
2. RMSNorm / residual-scale style elementwise kernels
3. ReLU^2 MLP activation + projection glue
4. artifact-side packing helpers if memory traffic dominates

The current starting point is `RMSNorm`.

Do not start with full attention. The verification surface is too large.

## Benchmark Rules

Benchmarks should always record:

- input shape
- dtype
- device
- warmup count
- timed iteration count
- median / p20 / p80 latency when possible

Use challenge-relevant shapes taken from:

- `TRAIN_SEQ_LEN`
- `TRAIN_BATCH_TOKENS`
- `MODEL_DIM`
- `NUM_HEADS`
- `NUM_KV_HEADS`
- selected checkpoint tensors

## Recommended Next Steps

1. Run `scripts/tensor_report.py` on:
   - current legal checkpoint
   - current best raw checkpoint
2. Run `scripts/allocation_rank.py` on the same checkpoints
3. identify the top byte-heavy tensors and numerically extreme tensors
4. choose one small Triton candidate with a clean PyTorch reference
5. build correctness tests before optimization
6. only then add performance profiling and autotuning

## Source Guidance

This workflow is based on official documentation:

- Triton tutorials and benchmarking patterns:
  - https://triton-lang.org/main/getting-started/tutorials/
- Triton compile-time assertions:
  - https://triton-lang.org/main/python-api/generated/triton.language.static_assert.html
- Triton runtime assertions:
  - https://triton-lang.org/main/python-api/generated/triton.language.device_assert.html
- PyTorch profiler:
  - https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html
- NVIDIA Nsight Compute CLI:
  - https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html
