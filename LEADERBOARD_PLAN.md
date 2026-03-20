# Leaderboard Plan

## Objective

Become the top public submission on the Parameter Golf leaderboard.

Current public SOTA in this repo snapshot is `1.1748 BPB` from `2026-03-19`:
- `records/track_10min_16mb/2026-03-19_SlidingWindow_FP16Emb_10L_MuonWD_OvertoneInit/README.md`

Challenge acceptance requires beating SOTA by at least `0.005 nats`, not `0.005 BPB`:
- `README.md`

At the current frontier, `0.005 nats ~= 0.0030 BPB`, so the practical target is roughly:
- `target <= 1.1718 BPB`

## Core Thesis

The repo evidence says the fastest route to `#1` is not a brand-new macro-architecture.

The highest expected-value path is:
1. Exploit evaluation harder.
2. Preserve or improve post-quant quality.
3. Only then retune training and architecture around the better evaluator.

Why:
- Sliding-window eval alone delivered about `-0.0319 BPB`.
- LoRA-TTT ablations show document isolation and striding produced most of the gain.
- The current top run uses only about `160s` of the allowed `600s` eval budget.
- The current top artifact is about `15.35 MB`, leaving about `650 KB` of size slack.

## What The Repo Already Proved

### 1. Evaluation is the largest lever

From `records/track_10min_16mb/2026-03-19_SlidingWindowEval/README.md`:
- Baseline post-quant: `1.2244 BPB`
- Sliding-window post-quant: `1.1925 BPB`
- Gain: `-0.0319 BPB`

From `records/track_10min_16mb/2026-03-17_LoRA_TTT/README.md`:
- Baseline: `1.2278 BPB`
- `+ doc-isolated`: `1.2168 BPB`
- `+ stride`: `1.1941 BPB`
- `+ LoRA TTT`: `1.1910 BPB`

Implication:
- Document boundaries matter.
- Richer eval context matters.
- Adaptation matters, but less than eval geometry.

### 2. Post-quantization quality matters almost as much as model quality

From `records/track_10min_16mb/2026-03-18_FP16Embed_WD3600/README.md`:
- Keeping `tok_emb.weight` in fp16 reduced quantization damage from about `0.007 BPB` to about `0.0005 BPB`.

From `records/track_10min_16mb/2026-03-19_WarmdownQuantization/README.md`:
- The main bottleneck on 8xH100 was quantization quality, not just pre-quant loss.
- Very long warmdown improved final post-quant BPB substantially.

Implication:
- Optimize the exported model, not just the training loss curve.

### 3. Longer context is valuable even when it reduces step count

From `records/track_10min_16mb/2026-03-19_TrainingOptSeq4096/README.md`:
- `TRAIN_SEQ_LEN=4096` still beat shorter-context runs despite fewer steps.

Implication:
- The model is context-starved.
- Context budget is often worth more than raw optimizer-step count.

### 4. The current SOTA is still structurally conservative

The current winner stacks:
- 10 layers
- sliding-window eval
- fp16 tied embedding export
- Muon decoupled weight decay
- spectral embedding init
- phase-scheduled residual mixing

But it still:
- recomputes heavily overlapping eval windows
- does not combine sliding eval with document isolation
- leaves large eval-time budget unused
- uses a simple export path: int8 + fp16 embedding + zlib

## Mathematical Framing

Let final score be:

`S = Eval(PostQuant(Train(theta))))`

This challenge is not won by minimizing only:

`L_train(theta)` or `L_prequant(theta)`

It is won by minimizing:

`BPB_final = BPB(eval_protocol, quantized_export(theta))`

So decompose gains into three buckets:

1. `Delta_train`
Improvement in the trained dense model before export.

2. `Delta_quant`
Improvement from reducing export damage.

3. `Delta_eval`
Improvement from better scoring context and permitted eval-time compute.

Repo evidence suggests:
- `Delta_eval` is the biggest bucket.
- `Delta_quant` is the second biggest bucket.
- `Delta_train` is still useful, but should be optimized against the improved evaluator.

## Concrete Step-By-Step Plan

### Phase 0: Lock the reference point

1. Reproduce the current top run exactly.
   Use:
   - `records/track_10min_16mb/2026-03-19_SlidingWindow_FP16Emb_10L_MuonWD_OvertoneInit/train_gpt.py`

2. Record these metrics separately for every run:
   - flat-stream pre-quant BPB
   - flat-stream post-quant BPB
   - final leaderboard BPB
   - train wallclock
   - eval wallclock
   - artifact bytes

3. Acceptance gate for reproduction:
   - `val_bpb` within about `+-0.0015`
   - train time under `600s`
   - total artifact under `16,000,000` bytes

Reason:
- If reproduction is noisy, every later conclusion is unreliable.

### Phase 1: Build the highest-EV evaluator

4. Implement document-isolated sliding-window evaluation.

   Required behavior:
   - detect BOS document boundaries
   - never let context cross document boundaries
   - still score every token exactly once
   - preserve byte-count accounting exactly

   Motivation:
   - The LoRA-TTT ablation suggests doc isolation alone was worth about `0.011 BPB` on baseline-like eval.

5. Replace overlap recomputation with KV-cache or streaming-state eval.

   Current issue:
   - The public SOTA evaluator rebuilds 1024-token windows every 64 tokens.
   - That throws away repeated prefix work.

   Goal:
   - Make `stride=32` and possibly `stride=16` affordable within the 10-minute eval cap.

6. Add an eval sweep harness over:
   - `eval_seq_len in {1024, 1408, 1536, 2048, 3072}`
   - `stride in {64, 32, 16}`
   - `doc_isolated in {0, 1}`

7. Choose the best evaluator under the hard constraint:
   - total eval time < `600s`

Success criterion for Phase 1:
- Beat the current winner using the same trained weights, or get within about `0.001-0.002 BPB` of the required record margin.

This is the most likely direct path to `#1`.

### Phase 2: Add controlled adaptation at eval time

8. Start with minimal adaptation that preserves fast eval:
   - head-only adaptation
   - top-layer-only adaptation
   - tiny-rank LoRA (`r in {4, 8}`)

9. Sweep:
   - adapter target set: `{lm_head}`, `{lm_head + q}`, `{lm_head + q + v}`
   - LoRA rank: `{4, 8}`
   - chunk size: `{64, 128, 256}`
   - learning rate: small logarithmic sweep

10. Only keep variants that clear both:
   - no leakage across documents
   - eval wallclock still under `600s`

Reason:
- Once doc-isolated sliding is in place, adaptation may only need to buy `0.001-0.003 BPB` to cross the record threshold.

### Phase 3: Retune training against the better evaluator

11. Retune training sequence length first:
   - `TRAIN_SEQ_LEN in {1024, 1536, 2048}`

12. Retune update frequency vs context:
   - `TRAIN_BATCH_TOKENS in {524288, 393216}`

13. Sweep optimization schedule:
   - `WARMDOWN_ITERS in {2500, 6000, 20000}`
   - `MUON_MOMENTUM in {0.95, 0.98, 0.99}`
   - `MUON_BACKEND_STEPS in {5, 7}`
   - embed/scalar/matrix learning rates around the current winner

14. Optimize against final evaluated BPB, not flat pre-quant metrics.

Reason:
- The repo already shows that a training recipe that looks only modestly better under flat eval can become much stronger under richer evaluation.

### Phase 4: Spend remaining artifact bytes rationally

15. Treat export precision as a knapsack problem.

For each tensor group, estimate:

`marginal_value = score_drop_if_precision_reduced / bytes_saved`

Then allocate bytes to the highest-value tensors first.

Likely ordering:
1. tied embedding
2. first and last layers
3. attention output projections
4. middle layers
5. small control tensors

16. Use the current slack of roughly `650 KB` deliberately.

High-probability uses:
- preserve more high-sensitivity tensors in fp16
- make precision asymmetric by layer position
- keep more scales or metadata where it actually helps

Avoid:
- blindly adding an 11th layer before precision/value analysis says it fits

### Phase 5: Low-level systems optimization only where it buys score

17. Optimize the training hot path if step count becomes the bottleneck.

Primary candidates:
- Muon update path
- repeated weight casts in `CastedLinear`
- unnecessary allocations in eval

18. Objective:
- reduce 10-layer step time from roughly `56 ms` toward `52 ms`

That would buy extra optimizer steps without changing the model family.

19. Optimize evaluator throughput after correctness is frozen.

Primary candidates:
- cache reuse
- static-shape batching
- fewer Python loops
- precomputed document windows

## Prioritized Experiment Queue

Run these in order.

### Tier 1: Immediate highest EV

1. Current winner + document-isolated sliding eval
2. Current winner + cache-based sliding eval + stride sweep
3. Current winner + doc-isolated cache-based eval + `eval_seq_len` sweep

### Tier 2: Near-term likely record

4. Best Tier 1 evaluator + small-rank head-only adaptation
5. Best Tier 1 evaluator + `lm_head + q + v` LoRA
6. Best Tier 1 evaluator + moderate long-eval context with NTK scaling

### Tier 3: Training retune once evaluator is strong

7. `TRAIN_SEQ_LEN=1536 or 2048` with the best evaluator
8. momentum/warmdown/backend-step sweep
9. export-precision reallocation sweep

### Tier 4: Higher-risk moonshots

10. custom tokenizer work
11. deeper or partially shared model variants
12. more exotic compression formats

## Success Metrics

For every experiment, log:
- `prequant_flat_bpb`
- `postquant_flat_bpb`
- `final_bpb`
- `delta_eval = postquant_flat_bpb - final_bpb`
- `delta_quant = postquant_flat_bpb - prequant_flat_bpb`
- train time
- eval time
- artifact bytes

Decision rule:
- Prefer experiments that improve `final_bpb` directly.
- If two runs tie on `final_bpb`, prefer the one with more eval-time headroom and more artifact slack.

## Risk Management

### Main risks

1. Overfitting to a buggy evaluator
   - Fix with exact token-count and byte-count invariants.

2. Spending too much time on training when eval still has unused budget
   - Fix by exhausting evaluator improvements first.

3. Chasing pre-quant loss improvements that disappear after export
   - Fix by always measuring post-quant and final BPB.

4. Burning artifact slack on low-value tensors
   - Fix with sensitivity-per-byte analysis.

5. Running out of eval budget after adding adaptation
   - Fix by first making sliding eval structurally efficient.

## Best Path To #1

If the goal is to maximize probability of taking the top spot quickly, the best sequence is:

1. Reproduce the current 10-layer winner.
2. Add document-isolated sliding eval.
3. Make sliding eval cache-efficient.
4. Sweep `stride` and `eval_seq_len` under the 600s eval budget.
5. Add lightweight eval-time adaptation only if needed.
6. Retune training against the improved evaluator.

This is the most rigorous path because it follows the empirical gradient already visible in the repo:
- evaluation geometry produced the largest historical gains
- quantization-aware export produced the next largest gains
- training and architecture refinements mattered after those two were handled

