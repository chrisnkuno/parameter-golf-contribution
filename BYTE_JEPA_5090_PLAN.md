# Byte JEPA 5090-First Plan

This note explains how the standalone byte-level JEPA prototype in [scripts/byte_jepa_v5.py](/home/chris/param-golf/parameter-golf/scripts/byte_jepa_v5.py) should be treated for Parameter Golf.

## What It Is

The prototype is a byte-level JEPA-style architecture with:

- decode-near / predict-far training
- CE on chunk `t+1`
- JEPA alignment on chunk `t+2`
- EMA target encoder
- invariant checks for:
  - shape contracts
  - horizon alignment
  - target encoder freeze
  - EMA semantics
  - gradient flow

## Why It Is Worth Trying

This is a credible architecture direction because:

- Parameter Golf is tokenizer-agnostic and scored in BPB on the fixed FineWeb validation set.
- The prototype has moved beyond a toy setup: it has explicit horizon structure, target-network semantics, and invariant testing.
- JEPA/GEPA-style latent-prediction ideas are already appearing in the public challenge discussion and PR ecosystem.

## What It Is Not Yet

It is not yet a challenge-ready submission path.

It still needs:

1. porting to the official FineWeb pipeline
2. legal artifact accounting under the `16,000,000`-byte cap
3. post-quant evaluation
4. official-style evaluation geometry
5. real throughput measurements against the challenge budget

## 5090-First Development Role

On a single `RTX 5090`, the correct goal is not to replicate the `8xH100` submission environment exactly.

The correct goals are:

- architecture ablations
- invariant testing
- throughput profiling
- memory profiling
- post-quant experiments
- deciding whether JEPA actually helps on the real task

## Required Migration Steps

### Phase 1: Local Proxy Validation

Use the standalone script only to answer:

- does the JEPA objective stabilize?
- what chunk sizes and horizons are viable?
- what is the throughput / VRAM profile?
- does the model collapse?

### Phase 2: Official-Repo Port

Port only the winning JEPA variant into the official repo flow:

- FineWeb shard loader
- official metric accounting
- legal eval path
- artifact export path

### Phase 3: Real Comparison

The first official comparison should be:

- same byte/chunk model without JEPA
- same byte/chunk model with JEPA
- same quantization / artifact path
- same official eval

Only then decide whether JEPA is helping.

### Phase 4: Artifact + Eval Layer

If the JEPA variant wins, the next work is:

- quantization-aware export
- legal checkpoint frontier selection
- stronger eval geometry

## 5090 Protocol

For the single-`5090` stage, every run should record:

- proxy BPB
- CE loss
- JEPA loss
- latent std
- tokens/sec
- step time
- max VRAM
- compressed checkpoint size once ported

## Bottom Line

The JEPA prototype is worth trying in Parameter Golf, but only as a staged research branch.

The first milestone is not “beat the leaderboard”.

The first milestone is:

- port it into the official repo environment
- compare JEPA vs no-JEPA under the real metric
- and determine whether it improves legal post-quant BPB enough to justify further work
