# Allocation Policy

This document defines how we decide where bytes, tensors, and compute budget should go in this repo.

The goal is not abstract elegance. The goal is to improve:

- final post-quant `val_bpb`
- training throughput
- legal artifact size

under the actual challenge constraints.

## Core Principle

We do **not** allocate by parameter count alone.

We allocate by:

`priority = bytes × traffic × sensitivity`

where:

- `bytes`: how much memory or artifact budget the tensor consumes
- `traffic`: how often it is read/written in the hot path
- `sensitivity`: how much BPB or stability degrades if we change precision, layout, or recomputation policy

This is the main rule behind:

- what stays high precision
- what gets quantized more aggressively
- what deserves a fused kernel
- what should be recomputed
- what should be streamed asynchronously

## Allocation Axes

Every tensor or buffer should be classified on 5 axes.

### 1. Residency

- `persistent`
  - model weights
  - optimizer state
  - quantization metadata
  - artifact payload
- `ephemeral`
  - activations
  - normalization buffers
  - q/k/v intermediates
  - MLP hidden activations
- `streamed`
  - input batches
  - checkpoint writes
  - eval/export artifacts

### 2. Traffic

Questions:

- how many reads per step?
- how many writes per step?
- does this tensor participate in bandwidth-bound operations?

High-traffic tensors are kernel/fusion candidates before they are compression candidates.

### 3. Sensitivity

Questions:

- does lower precision hurt final post-quant BPB?
- does reordering / preconditioning help or hurt?
- is the tensor numerically outlier-heavy?

Sensitivity is empirical, not assumed.

### 4. Liveness

Questions:

- how long must the tensor stay alive?
- can it be recomputed cheaply?
- does it need to survive only one sublayer, one block, or the whole step?

Short-lived tensors are recomputation candidates.

### 5. Hardware Fit

Questions:

- is the last dimension contiguous?
- does shape align with tensor-core-friendly tiles?
- is the memory access coalesced?
- does the current layout match the intended kernel?

Poor hardware fit means the tensor likely needs a different layout or a different kernel path.

## Practical Classes

### Class A: persistent + high sensitivity

Examples:

- tied embedding / logits path
- large weights that materially affect post-quant BPB
- critical quantization metadata

Default action:

- keep higher precision if justified
- use careful row/group quantization
- consider rotations/smoothing before lowering precision

### Class B: persistent + low sensitivity

Examples:

- some optimizer state
- some gradient buffers
- some checkpoint staging buffers

Default action:

- lower precision first
- compress or stream
- move off hot path where possible

### Class C: ephemeral + high traffic

Examples:

- RMSNorm inputs/outputs
- q/k/v intermediates
- ReLU^2 hidden activations
- residual-scale / mix buffers

Default action:

- fuse
- make in-place if safe
- benchmark with Triton or optimized kernels

### Class D: streamed state

Examples:

- dataloader batches
- checkpoint frontier writes
- export/eval artifacts

Default action:

- optimize transfer overlap
- compress or stream asynchronously
- avoid blocking training

## What Recent Research Suggests

### Tempo: activation memory should be allocated explicitly

Tempo shows that transformer-specific memory reductions can improve throughput, not just footprint. In-place normalization/activation handling and targeted recomputation can increase usable batch size and total training speed.

Implication for this repo:

- treat activations as first-class allocation objects
- prioritize in-place/fused norm and activation work
- use targeted recomputation instead of broad checkpointing

Source:
- https://arxiv.org/abs/2210.10246

### FlashAttention-3: attention allocation should be tile-local and asynchronous

FA3 emphasizes Hopper-specific overlap of data movement and compute, warp specialization, and low-precision paths.

Implication for this repo:

- do not think of attention data as one large materialized matrix
- allocate attention work around tiles/blocks
- first ensure backend use is good before custom kernels

Source:
- https://arxiv.org/abs/2407.08608

### FP8-LM: training-state allocation can be lower precision if scaling is correct

FP8-LM shows gradients and communication paths can use lower precision without catastrophic failure when scaling is handled well.

Implication for this repo:

- training-state allocation should be treated separately from weight allocation
- optimizer and gradient storage are candidates for lower precision earlier than core weights

Source:
- https://arxiv.org/abs/2310.18313

### SmoothQuant: outliers determine low-bit allocation

SmoothQuant moves quantization difficulty from activations to weights via an equivalent transformation.

Implication for this repo:

- the real question is often not “int8 or not”
- it is “where are the outliers and can we relocate or smooth them?”

Source:
- https://arxiv.org/abs/2211.10438

### QuaRot: rotations change what can be allocated cheaply

QuaRot shows structured rotations can remove outliers while preserving function, enabling lower-bit inference.

Implication for this repo:

- allocation policy should allow preconditioning to change which tensors can be low precision
- “sensitivity” can be changed by transformations, not just measured passively

Source:
- https://arxiv.org/abs/2404.00456

### QQQ: low-bit success depends on kernel/layout co-design

QQQ reports that specialized low-bit GEMM kernels and quantization layout choices drive practical speed gains.

Implication for this repo:

- quantization allocation must be group/row aware
- serialization policy alone is not enough

Source:
- https://arxiv.org/abs/2406.09904

### AQLM: extreme compression may require codebook-style allocation

AQLM shows that very low-bit compression can be competitive when weights are represented through structured additive/codebook allocations.

Implication for this repo:

- if per-row direct quantization saturates, allocation should evolve toward structured codebooks
- this is more invasive and belongs later in the roadmap

Source:
- https://arxiv.org/abs/2401.06118

### DataStates-LLM: checkpoint data is streamed state, not hot training state

DataStates-LLM shows asynchronous checkpointing can reduce training interruption.

Implication for this repo:

- checkpoint frontier generation should be treated as a streaming pipeline
- do not let checkpoint persistence dominate step time

Source:
- https://arxiv.org/abs/2406.10707

### AtP*: analysis infrastructure helps decide allocation faster

AtP* is not a systems paper, but it supports the broader point that cheap, local analysis can guide what to optimize before expensive interventions.

Implication for this repo:

- tensor visualization / attribution-style prefilters should be part of the allocation workflow

Source:
- https://arxiv.org/abs/2403.00745

## Allocation Decisions For This Repo

### Weights

Allocate weights by:

1. byte concentration
2. post-quant sensitivity
3. outlier structure
4. kernel friendliness

Current rule:

- high-byte + high-sensitivity tensors are the first place to spend extra precision or preconditioning
- high-byte + low-sensitivity tensors are the first place to push lower bitwidth

### Activations

Allocate activations by:

1. lifetime
2. traffic
3. rematerialization cost

Current rule:

- short-lived high-traffic activations should be fused or recomputed
- do not preserve them just because they are convenient

### Optimizer State / Gradients

Allocate training state by:

1. stability requirement
2. communication / storage burden
3. hardware-native low-precision opportunities

Current rule:

- treat optimizer state as separate from model quality-sensitive weights

### Checkpoints / Artifacts

Allocate checkpoint and artifact data by:

1. whether it is official submission payload
2. whether it is intermediate frontier state
3. whether it blocks training

Current rule:

- official payload is quality-sensitive and budget-constrained
- frontier checkpoints are streamed state and should be off the hot path

## Operational Workflow

1. Run `scripts/tensor_report.py` on a checkpoint.
2. Rank tensors by allocation priority.
3. Mark each tensor:
   - persistent / ephemeral / streamed
   - high / medium / low sensitivity
4. Choose the optimization type:
   - fuse
   - recompute
   - quantize differently
   - precondition
   - stream asynchronously
5. Validate numerics.
6. Benchmark on challenge-relevant shapes.

## Immediate Targets

1. `RMSNorm`
2. `ReLU^2` MLP activation path
3. rowwise quant/dequant layout and kernels
4. checkpoint frontier streaming

## Anti-Patterns

Do not:

- optimize tensors only by size
- optimize kernels only by microbenchmark without real shapes
- treat all low precision opportunities as equivalent
- spend time on metadata when payload quality dominates
- start with full custom attention before smaller kernels are verified

## Companion Tooling

- `scripts/tensor_report.py`
- `core/tensor_report.py`
- `core/triton_harness.py`
- `core/triton_rmsnorm.py`
- `scripts/triton_rmsnorm_bench.py`
- `LOW_LEVEL_OPTIMIZATION_BACKLOG.md`

