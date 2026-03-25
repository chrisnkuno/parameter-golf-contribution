# 3x5090 SOTA Chase Plan

This file defines the practical plan for chasing near-SOTA performance on `3xRTX 5090` under compute constraints.

It is **not** a claim that `3x5090` matches the official record environment. The record contract remains `<= 600s` on `8xH100 SXM`.

## What The Open PR Frontier Says

Recent open PRs suggest the current transferable frontier is driven by stacks like:

- value residual + gated attention + TTT  
  Source: [PR #490](https://github.com/openai/parameter-golf/pull/490)
- trigram hash + value residual + stronger post-quant stack + TTT  
  Source: [PR #486](https://github.com/openai/parameter-golf/pull/486)
- SwiGLU / activation upgrades and EMA-style stabilization  
  Source: [PR #505](https://github.com/openai/parameter-golf/pull/505)
- long unlimited-compute training with pure int6 + legal TTT  
  Source: [PR #612](https://github.com/openai/parameter-golf/pull/612)
- long unlimited-compute GEPA / GPTQ-lite / TTT stack  
  Source: [PR #628](https://github.com/openai/parameter-golf/pull/628)

The strongest pattern is:

1. stronger modeling features
2. stronger low-bit policy
3. legal score-first TTT
4. longer training than the 10-minute record budget when allowed

## What We Implemented Locally

The local mainline now supports the closest high-EV architectural subset of that frontier:

- `USE_ATTENTION_GATE=1`
- `USE_VALUE_RESIDUAL=1`
- `HASH_NGRAM_ORDER=2|3`
- `HASH_VOCAB_SIZE`
- `HASH_EMBED_DIM`
- `MLP_ACTIVATION=relu2|leaky_relu2|swiglu`

These features are configurable in [train_gpt.py](/home/chris/param-golf/parameter-golf/train_gpt.py) and are intended to approximate the most reusable open-PR architecture ideas without depending on the exact private stacks from each submission.

## 3x5090 Role

`3x5090` should be treated as:

- a full-data recipe-search platform
- a legal-checkpoint frontier platform
- a pre-`8xH100` unlimited-compute / non-record platform

It should **not** be treated as a timing-equivalent replacement for `8xH100 SXM`.

## Wall-Clock Targets

Based on the open non-record PRs:

- [PR #612](https://github.com/openai/parameter-golf/pull/612): `12k` steps on `4xA100-40GB` in `~100 min`
- [PR #628](https://github.com/openai/parameter-golf/pull/628): `20k` steps on `4xA100-40GB` in `~2.8h`

That implies the realistic `3x5090` target should be:

- short screening: `500-1000` steps
- medium recipe runs: `2k-4k` steps
- serious frontier runs: `8k-15k` steps

Expected wall-clock should be budgeted in broad bands, not exact promises:

- smoke / architecture reject: `10-20 min`
- recipe run: `30-90 min`
- serious full-data run: `2-6 h`

## Run Order

### Phase 1: Architecture Screening

Keep:

- `11L`
- `XSA_TAIL_LAYERS=4`

Sweep:

- `USE_ATTENTION_GATE=1`
- `USE_VALUE_RESIDUAL=1`
- `HASH_NGRAM_ORDER=2`
- `HASH_NGRAM_ORDER=3`
- `MLP_ACTIVATION=swiglu`
- `MLP_ACTIVATION=leaky_relu2`

Use:

- full challenge shards
- checkpoint frontier saving
- no TTT during screening

### Phase 2: Strong Recipe Runs

Promote only the best `1-2` architecture settings and sweep:

- `ITERATIONS=2000, 4000`
- longer warmdown
- EMA / SWA
- artifact policy

### Phase 3: Frontier Runs

For the best architecture + recipe:

- `8k-15k` steps
- legal checkpoint selection
- pure int6 / GPTQ-lite / QAT work
- legal score-first TTT

## Low-Level Layer

Use the existing low-level plan in parallel:

- tensor report
- allocation ranking
- allocation diffing
- Triton RMSNorm benchmarking

The order remains:

1. inspect legal vs raw checkpoints
2. identify hot tensors
3. benchmark Triton RMSNorm
4. then move to fused `ReLU^2` / MLP glue
5. then quant/dequant kernels

## Immediate Starting Configuration

Recommended first `3x5090` stack:

```bash
NUM_LAYERS=11
XSA_TAIL_LAYERS=4
USE_ATTENTION_GATE=1
USE_VALUE_RESIDUAL=1
HASH_NGRAM_ORDER=2
HASH_VOCAB_SIZE=12288
HASH_EMBED_DIM=128
MLP_ACTIVATION=swiglu
TRAIN_SEQ_LEN=256
TRAIN_BATCH_TOKENS=524288
WARMDOWN_ITERS=200
SAVE_DENSE_CHECKPOINT_EVERY=100
RUN_TTT_EVAL=0
```

If that is unstable or too large, back off in this order:

1. keep `USE_VALUE_RESIDUAL=1`, drop hash
2. keep hash, revert `MLP_ACTIVATION=relu2`
3. keep only one of value residual or gated attention

## Success Criteria

On `3x5090`, success means:

- better full-data post-quant BPB than the current local control
- checkpoint frontier under the legal byte cap
- enough recipe stability to justify later low-bit and TTT work
