# 11L + XSA4 Plan

## Objective

Push the `11L + XSA4` branch toward a record-capable submission under the actual challenge constraints:

- train in `<= 600s` on `8xH100`
- eval in `<= 600s`
- artifact `< 16,000,000` bytes
- optimize final post-quant, post-eval `val_bpb`

This file is the execution plan for the current branch. It is intentionally narrower than the broader strategy docs.

## Current Locked Baseline

What we keep:

- architecture: `11L`
- tail attention variant: `XSA_TAIL_LAYERS=4`
- artifact path: `packed_zstd`
- large-float keep: `tok_emb.weight`
- metric contract: final comparisons use `final_int8_zlib_roundtrip_exact`
- invariants: accounting, architecture layout, quantization, and artifact roundtrip stay enforced

What we do not keep as the main branch right now:

- local-mixer prefix variants
- ZeroS middle-trunk variants
- shared-depth variants
- broader XSA depths beyond `4`

Reason:

- `11L + XSA4` survived the longer proxy better than the neighboring architecture ideas.
- local mixer looked good on short runs and failed on longer ones.
- ZeroS was slower and worse in this stack.

## Layer 1: Modeling

This layer determines the quality ceiling of the model.

Scope:

- architecture
- model size allocation
- hyperparameters
- training recipe

Current modeling thesis:

- freeze the macro-architecture at `11L + XSA4`
- improve the training stack around it instead of reopening architecture search immediately

### Modeling Priorities

1. Stabilize `11L + XSA4`

- test EMA / SWA style stabilizers
- test tighter warmdown and longer low-LR tails
- test small schedule changes before changing width or depth

2. Retune the training recipe for this exact stack

- `TRAIN_SEQ_LEN`
- `TRAIN_BATCH_TOKENS`
- `WARMDOWN_ITERS`
- `MUON_MOMENTUM`
- learning-rate split across token / matrix / scalar params

3. Reallocate capacity only if the stable stack is saturated

- only reopen width or layer-count changes if the stabilized `11L + XSA4` line stalls
- do not reopen local mixer or ZeroS unless a later systems or eval result changes the picture materially

### Modeling Rules

- use the `200-step` proxy only to reject obvious losers
- require a `400-step` rerun before promoting any architecture or training change
- rank by post-quant `final_int8_zlib_roundtrip_exact val_bpb`, not train loss

### Modeling Experiment Order

1. `11L + XSA4 + EMA`
2. `11L + XSA4 + SWA`
3. warmdown sweep
4. sequence-length / batch-token sweep
5. Muon / LR split sweep

## Layer 2: Systems

This layer determines how much useful training we can buy in 10 minutes.

Scope:

- FA3
- Triton
- custom CUDA kernels
- memory layout
- fused ops
- distributed efficiency

Current systems thesis:

- prefer proven throughput wins before heroic kernel work
- only write custom kernels for measured hotspots

### Systems Priorities

1. Keep the fast attention path locked

- prefer FlashAttention / FA3-class paths when compatible
- verify they stay active on the actual training stack

2. Profile the real bottlenecks

- step time
- optimizer time
- data/input stalls
- compile overhead
- distributed sync overhead

3. Only then do targeted low-level work

- Triton or custom kernels only for hotspots with clear payoff
- no speculative kernel projects without a measured bottleneck

### Systems Rules

- every systems change must report `ms/step`
- every systems change must preserve final metric behavior
- if a system optimization changes numerics, it is a modeling change and must be retested as such

### Systems Experiment Order

1. lock attention backend and profile
2. tune compile/runtime settings
3. improve distributed efficiency
4. only then evaluate Triton / custom kernels for proven hotspots

## Layer 3: Artifact Economics

This layer determines how much model quality we can fit inside `16 MB`.

Scope:

- quantization
- packing
- code-size minimization
- custom serialization
- dependency minimization

Current artifact thesis:

- keep the current packed artifact path
- stop spending time on generic save-format tweaks
- focus on quantized payload quality

### Current Artifact Policy

- default format: `packed_zstd`
- keep `tok_emb.weight` in float passthrough
- keep the accounting / roundtrip invariants

### Artifact Priorities

1. Preserve the wins already found

- `packed_zstd`
- `tok_emb.weight` float keep

2. Improve payload quality, not metadata

- selective preconditioning only if it survives stronger checkpoints
- mixed precision only where it clearly buys BPB per byte

3. Avoid dead ends

- no more scale-codec work
- no more generic serializer tinkering
- no broad fp16 sweeps over MLP weights

### Artifact Rules

- measure compressed bytes on the actual final artifact
- compare post-quant BPB, not dense BPB
- keep artifact roundtrip invariant-tested

## Layer 4: Benchmark Alignment

This layer determines whether the whole stack is aimed at the real target.

Scope:

- data packing
- sequence length choices
- evaluation-aware tuning
- compression-centric objectives

Current benchmark-alignment thesis:

- optimize the actual submission path, not a proxy detached from the leaderboard rules
- use proxies only as staged filters

### Benchmark Priorities

1. Keep the metric contract fixed

- final comparison metric: `final_int8_zlib_roundtrip_exact`
- preserve `loss_sum`, `token_count`, `byte_count` accounting
- do not silently change tokenizer or validation assets

2. Use staged evaluation correctly

- short proxy: reject losers
- longer proxy: validate survivors
- fuller eval path: decide promotion

3. Tune sequence length and training recipe against final post-quant behavior

- not against dense train loss alone
- not against short-run pre-quant results alone

### Benchmark Rules

- no promotion from a short-run win alone
- no claim of improvement without post-quant comparison
- if eval behavior changes, rerun the raw-accounting checks

## Execution Order

### Phase A: Lock The Branch

- keep `11L + XSA4`
- keep `packed_zstd`
- keep `tok_emb.weight` float keep
- stop reopening local mixer and ZeroS for now

### Phase B: Train-Recipe Improvement

- EMA / SWA branch
- warmdown branch
- seq-len / batch-token branch
- Muon / LR split branch

### Phase C: Systems Throughput

- profile current stack
- recover easy throughput wins
- only escalate to Triton / custom kernels if profiling justifies it

### Phase D: Artifact Polish

- preserve current artifact stack
- only pursue payload-quality improvements with positive measured BPB-per-byte returns

### Phase E: Full Alignment

- rerun the best candidate through the fuller eval path
- check train/eval budget feasibility
- check final artifact size

## Promotion Gates

A candidate graduates only if it clears all of:

1. better post-quant `final_int8_zlib_roundtrip_exact val_bpb`
2. no budget violation on projected `8xH100` timing
3. artifact remains `< 16 MB`
4. no invariant break in tests

## Immediate Next Steps

1. `11L + XSA4 + EMA`
2. `11L + XSA4 + SWA`
3. warmdown sweep around the best stabilizer
4. sequence-length and batch-token sweep on the stabilized stack
5. only after that, revisit systems-heavy work

## Anti-Goals

For this branch, do not spend time on:

- more local-mixer exploration
- more ZeroS exploration
- reopening broad architecture search
- generic serializer tweaking
- speculative kernel work without a measured hotspot

The branch is now simple:

- keep `11L + XSA4`
- make it train better
- make it run faster
- keep the artifact efficient
- evaluate it the same way the leaderboard will judge it
