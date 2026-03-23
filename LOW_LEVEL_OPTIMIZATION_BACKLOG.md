# Low-Level Optimization Backlog

This backlog focuses on low-level improvements that are plausible for this repo, ordered by expected value for the current Parameter Golf setup.

The intended use is:

1. inspect tensors with `scripts/tensor_report.py`
2. pick one operator or memory path
3. add a PyTorch reference
4. add Triton or systems optimization
5. verify numerics and benchmark on challenge-relevant shapes

## Priority 0: What This Repo Actually Needs

The current bottlenecks are not generic.

- artifact size remains the main constraint on the best checkpoints
- training throughput still matters because the challenge is time-bounded
- small-kernel and memory-traffic overheads matter because the model is compact

So the highest-value low-level work is:

- faster/fused small transformer operators
- better quantization kernels / data layouts
- memory-footprint reductions that let us keep larger effective batch or longer training

## Priority 1: Immediate Kernel Targets

### 1. RMSNorm Triton forward/backward

Why:
- RMSNorm is used repeatedly in `train_gpt.py`
- the correctness surface is clean
- it is a natural first Triton target

Repo target:
- `core/triton_rmsnorm.py`
- later integrate behind a feature flag in `train_gpt.py`

Research signal:
- Transformer-specific memory/throughput work emphasizes normalization layers as high-value targets for memory and throughput tuning.
- Source: Tempo, https://arxiv.org/abs/2210.10246

### 2. Fused ReLU^2 MLP glue

Why:
- this repo uses ReLU^2 MLP, not GELU/SwiGLU
- the current path likely pays extra memory traffic for:
  - linear
  - relu
  - square
  - projection

Goal:
- fuse activation-side work around `relu(x).square()`
- reduce reads/writes for the hidden activation

Research signal:
- Tempo shows in-place/fused activation handling can materially reduce memory footprint and improve throughput in transformers.
- Source: https://arxiv.org/abs/2210.10246

### 3. Quantize / dequantize kernels for rowwise int8 and mixed precision

Why:
- this repo’s artifact bottleneck is quantized payload quality and size
- rowwise quantization, scale application, and dequant paths are core to the final objective

Goal:
- faster rowwise stats / scaling
- fused dequant + matmul staging where possible
- more hardware-friendly packed layouts

Research signal:
- modern low-bit systems get speed not just from quantization policy, but from dedicated GEMM / dequant kernels
- Source: QQQ, https://arxiv.org/abs/2406.09904

## Priority 2: Hardware-Aware Attention Work

### 4. FlashAttention-3 style Hopper path

Why:
- if we train or evaluate on H100/Hopper, attention kernels can still be improved substantially
- FA3 specifically targets Hopper asynchrony, TMA, warp specialization, and FP8 paths

What to borrow conceptually:
- overlap data movement and compute
- warp specialization
- low-precision block processing where numerically safe

Reality check:
- do not reimplement full FA3 from scratch first
- first ensure we are using the best available attention backend before writing custom kernels

Research signal:
- FlashAttention-3 reports large H100 speedups and much better hardware utilization
- Source: https://arxiv.org/abs/2407.08608

### 5. Attention-adjacent fused softmax / masking / score paths

Why:
- full attention is too large a first Triton target
- but score-side and normalization-side helpers can still matter

Goal:
- benchmark whether small fused score utilities help at this model scale

Research signal:
- FA3 and Tempo both emphasize memory traffic reduction around the attention block
- Sources:
  - https://arxiv.org/abs/2407.08608
  - https://arxiv.org/abs/2210.10246

## Priority 3: Quantization-Driven Low-Level Work

### 6. Rotation/preconditioning-aware kernels

Why:
- we already saw signal that quantization quality, not metadata, dominates artifact outcomes
- rotations can reduce outliers and make low-bit kernels easier

Goal:
- structured Hadamard/sign/permutation preconditioning
- kernels that apply or absorb the transform cheaply

Research signal:
- QuaRot shows rotation can remove outliers while preserving function, enabling much lower precision
- Source: https://arxiv.org/abs/2404.00456

### 7. W4A8 / mixed low-bit GEMM path

Why:
- the repo is still mostly on an int8-centric artifact path
- mixed lower precision is one of the few ways to improve both speed and bytes if accuracy survives

Goal:
- prototype a low-bit matmul path for selected tensors
- likely for export/eval before training

Research signal:
- QQQ reports strong speedups from dedicated W4A8 kernels
- Source: https://arxiv.org/abs/2406.09904

### 8. Extreme codebook/additive quantization path

Why:
- if plain per-row quantization saturates, codebook-style methods may offer better bytes/quality tradeoffs

Caution:
- implementation complexity is much higher
- likely not the first Triton task

Research signal:
- AQLM shows strong extreme compression regimes with practical kernels
- Source: https://arxiv.org/abs/2401.06118

## Priority 4: Memory and Training Systems

### 9. In-place / rematerialized normalization and activation paths

Why:
- if memory is the real limiter for larger batches or longer contexts, in-place derivations help

What to adapt:
- in-place LayerNorm / normalization ideas
- in-place activation storage reductions

Research signal:
- Tempo reports better throughput from transformer-specific memory reductions, not just lower footprint
- Source: https://arxiv.org/abs/2210.10246

### 10. Selective recomputation inside attention/MLP sublayers

Why:
- broad checkpointing is often too expensive
- sublayer-specific recomputation is more targeted

Goal:
- drop only the expensive intermediate tensors that are cheap to recompute

Research signal:
- Tempo’s sub-layer dropout recomputation is exactly this kind of optimization
- Source: https://arxiv.org/abs/2210.10246

### 11. Asynchronous checkpointing / frontier export pipeline

Why:
- this repo relies heavily on checkpoint frontiers for legal model selection
- copying and evaluating checkpoints can become the bottleneck in longer runs

Goal:
- decouple checkpoint persistence from the hot training loop
- keep frontier generation cheap enough to use more frequently

Research signal:
- DataStates-LLM shows large wins from lazy asynchronous checkpointing for LLM workflows
- Source: https://arxiv.org/abs/2406.10707

## Priority 5: Interpretability / Visualization Infrastructure

These do not directly speed training, but they can accelerate optimization decisions.

### 12. Tensor visualizer / histogram / byte concentration views

Already started:
- `scripts/tensor_report.py`

Next:
- compare legal vs over-cap checkpoints
- compare quantized payload vs passthrough tensors
- surface outlier-heavy layers automatically

### 13. Fast attribution / patching prefilters for tensor debugging

Why:
- useful for deciding where approximation or precision changes are safe

Research signal:
- AtP* shows scalable attribution patching can be used as a fast filter before expensive verification
- Source: https://arxiv.org/abs/2403.00745

## Recommended Order For This Repo

1. finish the tensor visualizer and run it on real checkpoints
2. complete Triton RMSNorm correctness + benchmark
3. implement fused ReLU^2 MLP glue
4. build rowwise quant/dequant kernels and packed layouts
5. only then move to stronger Hopper-specific attention work
6. after that, revisit rotation-aware quantization and low-bit GEMM

## What Not To Do First

- do not start with full custom attention
- do not start with a giant FP8 training rewrite
- do not start with a fully custom quantization codec before kernel baselines exist
- do not optimize microbenchmarks on shapes that do not occur in `train_gpt.py`

## Sources

- Tempo: Accelerating Transformer-Based Models for Training
  - https://arxiv.org/abs/2210.10246
- FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision
  - https://arxiv.org/abs/2407.08608
- QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs
  - https://arxiv.org/abs/2404.00456
- QQQ: Quality Quattuor-Bit Quantization for Large Language Models
  - https://arxiv.org/abs/2406.09904
- AQLM: Extreme Compression of Large Language Models via Additive Quantization
  - https://arxiv.org/abs/2401.06118
- DataStates-LLM: Lazy Asynchronous Checkpointing for Large Language Models
  - https://arxiv.org/abs/2406.10707
- AtP*: An efficient and scalable method for localizing LLM behaviour to components
  - https://arxiv.org/abs/2403.00745
