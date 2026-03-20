# Rigorous Math-Driven Approach

## Objective

Produce a submission that is accepted as a new public record on the Parameter Golf leaderboard by optimizing the actual acceptance objective under the exact challenge constraints.

This document is intentionally low-level. It treats the challenge as a constrained optimization problem with noisy measurements, hard runtime limits, and a strict artifact budget. The goal is not to "find a better model" in the abstract. The goal is to maximize the probability of a verified record submission.

## 1. Challenge As A Constrained Optimization Problem

Let:

- `theta` be the trained dense parameters before export
- `Q(.)` be the export and quantization operator
- `E(.)` be the allowed evaluation procedure
- `B(.)` be the measured validation score in bits per byte
- `T_train(.)` be training wallclock on 8xH100
- `T_eval(.)` be evaluation wallclock on 8xH100
- `A(.)` be artifact size in bytes

The leaderboard objective is:

`minimize B(E(Q(theta)))`

subject to:

- `T_train(theta) <= 600 s`
- `T_eval(E, Q(theta)) <= 600 s`
- `A(Q(theta)) < 16,000,000 bytes`

Acceptance for a new record additionally requires:

- improvement over current SOTA of at least `0.005 nats`
- enough repeated runs to establish the required significance

So the real optimization target is:

`maximize P(accepted record | measured score, variance, runtime, size)`

This changes how decisions must be made:

- A change that improves dense training loss but worsens post-quant score is bad.
- A change that improves post-quant score but violates eval runtime is unusable.
- A change that gives a tiny mean gain with large variance may fail acceptance.
- A change that improves score by spending more of the allowed eval compute is valid and high-value.

## 2. Unit Discipline

Every experiment must keep units explicit.

Primary units:

- score: `BPB`
- acceptance margin: `nats` and `BPB`
- artifact size: `bytes`
- training throughput: `ms/step`, `steps/s`, `tokens/s`
- eval throughput: `tokens/s`, `seconds`
- compute budget: `seconds`

Useful conversions:

- repo `val_loss` is mean token cross-entropy in natural log units, i.e. `nats / token`
- repo `val_bpb` is computed from accumulated loss and accumulated byte count
- exact code-path identity is:

`val_loss = loss_sum / token_count`

`val_bpb = (loss_sum / ln(2)) / byte_count`

`val_bpb = val_loss / ln(2) * (token_count / byte_count)`

This means the acceptance-margin conversion depends on the actual `token_count / byte_count` ratio of the scored validation corpus and evaluator, not just on `1 / ln(2)`.

If the acceptance threshold is interpreted through the repo's natural-log score:

`Delta_BPB = Delta_nats_per_token / ln(2) * (token_count / byte_count)`

At the current frontier, the observed ratio is about `token_count / byte_count ~= 0.41`, so:

`0.005 nats/token -> about 0.005 / ln(2) * 0.41 ~= 0.0030 BPB`

That matches the practical target used in the leaderboard plan.

Operational rule:

- Never compare results with mixed units.
- Never claim "record margin" in BPB unless the exact evaluator and byte accounting used for submission define the conversion.
- Always record both the raw score delta and the acceptance-margin interpretation.
- When in doubt, compute the margin from raw `loss_sum`, `token_count`, and `byte_count`, not from rounded printed summaries.

### Metric-Lock Requirements

The metric implementation itself must be treated as frozen infrastructure.

From the repo code, byte accounting is not a trivial token-length lookup. It depends on:

- `base_bytes_lut[target_token]`
- whether the target token carries a leading space
- whether the previous token is a boundary token

So a rigorous run must preserve:

- exact validation shard set
- exact tokenizer model
- exact LUT generation logic
- exact boundary-token semantics
- exact global accumulation order: accumulate `loss_sum`, `token_count`, `byte_count`, then divide once at the end

Operationally:

- never compare runs that silently changed tokenizer assets or validation data
- hash or otherwise pin the validation shards and tokenizer file
- never average per-batch BPB values; only compute final BPB from global sums
- if tokenizer or dataset changes are ever considered, treat metric re-validation as a first-class project with its own proof burden

### BPB As Operational Code Length

The cleanest way to think about `val_bpb` is as code length, not as an opaque benchmark score.

From source coding and arithmetic coding:

- an ideal entropy coder turns model probabilities into code lengths
- for a token sequence `t_1, ..., t_n` with coding distribution `p`, the code length is approximately

`L_code(t_1:n; p) ~= sum_i -log_2 p(t_i | t_<i)`

up to small finite-precision coding overhead

This is exactly why the repo metric takes the form:

`loss_sum = sum_i -ln p(t_i | t_<i)`

`val_bpb = loss_sum / (ln 2 * byte_count)`

So `val_bpb` is not merely correlated with compression. It is a normalized compression length under the evaluation code.

If `q` denotes the true validation-stream distribution under the fixed evaluator contract, then the expected BPB of model `p` can be written as:

`E[val_bpb(p)] = H(q, p) / E_q[bytes_per_token]`

where `H(q, p)` is token cross-entropy in bits per token.

Using the standard cross-entropy decomposition:

`H(q, p) = H(q) + D_KL(q || p)`

so:

`E[val_bpb(p)] = (H(q) + D_KL(q || p)) / E_q[bytes_per_token]`

This is the core mathematical interpretation of the leaderboard metric:

- `H(q) / E_q[bytes_per_token]` is the irreducible entropy floor in bits per raw byte
- `D_KL(q || p) / E_q[bytes_per_token]` is the model's excess code length per raw byte
- every real BPB improvement is a reduction in excess codelength per byte

This also clarifies tokenization:

- tokenization changes the symbol alphabet used by the coder
- it is not a free compression gain by itself
- it only helps if it lowers cross-entropy enough, improves effective context usage, or improves the evaluator's legal coding geometry

For this repo specifically, byte accounting is the anchor that makes the metric roughly tokenizer-agnostic. That is why `byte_count` must be treated as first-class state rather than a post-hoc conversion.

### Two Nested Compression Problems

This challenge contains two mathematically distinct compression problems:

1. Lossless predictive compression of the validation stream

`minimize val_bpb = code_length(validation | model) / raw_validation_bytes`

2. Lossy compression of the model artifact

`compress theta into < 16,000,000 bytes while preserving predictive quality`

The first problem is evaluated directly by the leaderboard.
The second is a hard feasibility constraint on the submitted compressor.

This is why the project must never talk vaguely about "compression" without saying which one:

- BPB is about coding the validation data
- artifact size is about coding the model parameters and metadata

## 3. Decompose Final Score Into Controllable Terms

Final score is:

`B_final = B(E(Q(theta)))`

For engineering purposes, decompose:

`B_final = B_dense + Delta_quant + Delta_eval`

where:

- `B_dense` is dense-model score under a fixed base evaluator
- `Delta_quant = B(Q(theta)) - B(theta)` under a matched evaluator
- `Delta_eval = B(E(Q(theta))) - B(base_eval(Q(theta)))`

We care about marginal improvement:

`Delta_total = Delta_train + Delta_quant_improvement + Delta_eval_improvement`

This decomposition forces correct attribution:

- Training changes affect `Delta_train` first.
- Export changes affect `Delta_quant`.
- Eval protocol changes affect `Delta_eval`.

Without this separation, the project will overfit to the wrong bottleneck.

## 4. Hard Constraint Accounting

Treat every run as consuming three budgets:

1. Training time budget
2. Evaluation time budget
3. Artifact byte budget

Define slack variables:

- `S_train = 600 - T_train`
- `S_eval = 600 - T_eval`
- `S_artifact = 16,000,000 - A`

These are not cosmetic metrics. They are optimization resources.

Interpretation:

- Positive `S_eval` means unused test-time compute that can be converted into score.
- Positive `S_artifact` means unused precision capacity that can be reallocated to sensitive tensors.
- Very small `S_train` means the configuration is fragile to noise and machine variance.

Operational thresholds:

- avoid treating any result with `S_train < 20 s` as robust
- avoid treating any result with `S_eval < 20 s` as robust
- maintain enough artifact slack to absorb metadata and serialization noise
- time the full submission path, not a stripped microbenchmark
- include compile, load, decompression, and evaluator setup costs if they occur in the real run path

A mathematically clean result that sits on the edge of the wallclock limit is not a reliable submission.

## 5. Noise Model And Statistical Decision Rule

Leaderboard acceptance is not based on a single lucky run. Therefore every optimization must be judged using variance-aware estimates.

For configuration `c`, let repeated measured scores be:

`x_1, x_2, ..., x_n`

Compute:

- sample mean: `mu_hat`
- sample standard deviation: `s`
- standard error: `SE = s / sqrt(n)`

For two configs `a` and `b`, compare:

`d_i = x_i^(a) - x_i^(b)`

and estimate:

- paired mean difference `d_hat`
- paired standard deviation `s_d`
- paired standard error `SE_d = s_d / sqrt(n)`

Decision rule:

- rank variants by lower confidence bound on improvement, not by raw mean alone
- only escalate expensive confirmation runs for variants whose conservative bound is still positive

Practical proxy:

`LCB = d_hat - z * SE_d`

with `z` chosen according to how conservative we want to be during exploration.

This avoids the common failure mode of chasing tiny gains smaller than noise.

For final record claims, the bar is higher than internal exploration:

- use a fixed seed set for candidate vs reference whenever possible
- report enough runs to support the challenge's `p < 0.01` acceptance requirement
- do not promote a candidate to "submission-ready" based only on a loose exploratory confidence bound

For a final one-sided record check in natural-log units, define:

- `m_sota` = current SOTA score in the matched natural-log metric
- `delta_req = 0.005`
- `y_1, ..., y_n` = candidate repeated scores in the same metric
- `ybar` = candidate sample mean
- `s_y` = candidate sample standard deviation

Then a conservative acceptance-style condition is:

`ybar + t_(0.99, n-1) * s_y / sqrt(n) < m_sota - delta_req`

Equivalently, the lower confidence bound on the improvement exceeds `0.005 nats`.

This is the right way to think about "record margin":

- not as a single best run
- not as a BPB delta copied from rounded tables
- but as a one-sided confidence statement under the exact submission metric

## 6. First Principles For Experimental Ranking

For every candidate change, estimate:

- expected score gain: `E[Delta_BPB]`
- runtime cost: `Delta_T_train`, `Delta_T_eval`
- size cost: `Delta_A`
- variance impact: `Delta_sigma`
- implementation complexity and correctness risk

Then compute a crude efficiency metric:

`ROI_eval = -Delta_BPB / Delta_T_eval`

`ROI_train = -Delta_BPB / Delta_T_train`

`ROI_size = -Delta_BPB / Delta_A`

Interpretation:

- higher `ROI_eval` means score gained per second of eval budget spent
- higher `ROI_size` means score gained per byte of artifact budget spent

This allows principled prioritization:

- first spend the budget dimension with the highest marginal return
- do not spend scarce bytes where eval seconds are cheaper
- do not spend training seconds on architecture changes if export changes buy more score per unit budget

## 7. Evaluation Must Be Treated As An Algorithm

Evaluation is not a passive measurement layer. Under challenge rules, it is an allowed optimization surface.

Define the evaluation protocol by:

- context geometry
- token coverage rule
- document boundary rule
- state reuse rule
- adaptation rule, if any

For a token position `t`, the score contribution is:

`ell_t = -log p(x_t | context_t)`

Final score is:

`B = (sum_t ell_t) / (bytes * ln(2))`

Therefore:

- increasing effective context for scored tokens can directly reduce `ell_t`
- changing context semantics can change the score materially, so evaluator contracts must be explicit
- redundant recomputation is pure systems waste that can be converted into better context geometry

Required evaluator invariants:

- every target token counted exactly once
- byte accounting matches the official metric convention exactly
- no training data or hidden side information is consulted at eval time unless it is explicitly paid for inside the artifact budget
- any adaptation uses only past tokens that are legal under the chosen evaluation semantics

For this specific approach, we additionally prefer:

- document-local causal context
- explicit state reset rules
- evaluator behavior that is easy to audit and reproduce

## 8. Sliding Evaluation Should Be Optimized Analytically

Suppose sliding eval uses:

- window length `L`
- stride `s`

For naive recomputation, the approximate number of forward passes is:

`N_pass ~= ceil(N_tokens / s)`

and approximate total token-compute is:

`C_naive ~= L * N_pass ~= L * N_tokens / s`

Relative overhead versus ideal one-pass streaming is roughly:

`overhead ~= L / s`

Examples:

- `L=1024, s=64` gives about `16x` token recomputation
- `L=1024, s=32` gives about `32x`
- `L=1536, s=16` gives about `96x`

This means:

- naive sliding eval quickly becomes dominated by redundant prefix work
- any KV-cache or recurrent-state reuse that approaches one-pass compute changes the feasible operating point

Conclusion:

- before sweeping smaller strides aggressively, fix evaluator asymptotics
- otherwise we will incorrectly conclude that the best score geometry is "too expensive"

## 9. Document Isolation Is A High-Rigor Design Choice

Let documents be `D_1, D_2, ..., D_m`.

For token `x_t` in document `D_j`, valid context is:

`context_t subseteq prefix(D_j, t)`

not:

`prefix(flat_stream, t)`

Important distinction:

- the challenge rules allow aggressive evaluation and do not forbid flat-stream evaluation
- this document recommends document isolation because it gives cleaner semantics, cleaner adaptation boundaries, and a lower-risk measurement story

Why this is the rigorous choice for this approach:

- it removes ambiguity about whether unrelated documents are being used as free side context
- it makes any eval-time adaptation obviously causal within a document
- it gives more interpretable per-document diagnostics
- it reduces the chance of a high-scoring but hard-to-defend submission

So document isolation is not a universal validity requirement of the challenge. It is the preferred evaluation contract for this math-first strategy.

Low-level implementation requirements:

- detect BOS boundaries exactly
- create per-document scoring intervals
- reset KV cache or recurrent state at each boundary
- ensure the first scored tokens of a document only see legal prefix tokens from that document

Validation checks:

- total number of scored target tokens matches the baseline evaluator
- total bytes scored matches the official accounting
- per-document token counts sum exactly to the corpus total

### Side-Information Constraint

Evaluation is allowed to be clever, but it is not allowed to smuggle in extra information.

Therefore:

- no reading training documents at eval time unless those bits are included in the artifact
- no hidden retrieval tables derived from the training set unless serialized into the artifact budget
- no external caches, offline statistics, or helper models unless they are part of the submitted package and fit the rules

This should be stated explicitly because "better evaluator" is only legitimate if it stays inside the information budget.

## 10. Quantization Must Be Modeled As Distortion Allocation

Export is a lossy compression problem under a hard size cap.

Partition tensors into groups `g = 1...G`.

For each group:

- byte cost at chosen precision: `A_g`
- score penalty induced by that precision: `L_g`

We want to allocate precision to minimize:

`sum_g L_g`

subject to:

`sum_g A_g < 16,000,000`

The correct framing is a knapsack or marginal-allocation problem.

For a precision downgrade on tensor group `g`, define:

`rho_g = Delta_score_g / bytes_saved_g`

Interpretation:

- low `rho_g`: cheap bytes to save, good candidates for aggressive compression
- high `rho_g`: expensive bytes to save, preserve these at higher precision

This is strictly better than making export decisions by layer folklore.

Likely high-sensitivity groups include:

- tied embedding / LM head
- early and late transformer blocks
- attention output projections
- tensors repeatedly reused in both forward and output pathways

But sensitivity must be measured, not assumed.

### Rate-Distortion / Lagrangian Form

The artifact side should be treated as a discrete rate-distortion problem.

For tensor group `g`, let:

- `z_g` be its chosen quantized representation
- `R_g(z_g)` be its true serialized byte cost in the submitted artifact
- `D_g(z_g)` be its induced score damage proxy

Then the mathematically correct constrained problem is:

`minimize sum_g D_g(z_g)`

subject to:

`sum_g R_g(z_g) < 16,000,000`

The standard Lagrangian relaxation is:

`minimize sum_g [D_g(z_g) + lambda * R_g(z_g)]`

This gives the correct local optimality rule for bit allocation:

`-Delta D_g / Delta R_g ~= lambda`

Interpretation:

- extra bytes should be spent where they buy the largest distortion reduction
- bytes should be removed where they cost the least distortion
- at a well-tuned operating point, marginal distortion reduction per extra byte is approximately equalized across the active tensor groups

This is the low-level mathematical replacement for rules of thumb like "keep embeddings high precision" or "middle layers matter less." Those may be true, but they should emerge from measured marginal distortion-per-byte curves.

### Real Byte Cost Means Post-Serialization Byte Cost

Because the repo exports quantized tensors and then compresses them again, usually with `zlib`, the byte-rate side is not equal to nominal bitwidth times parameter count.

In practice:

`R_g != (#params_g * nominal_bits_g / 8)`

because final bytes also depend on:

- entropy of the quantized symbol stream
- repeated patterns and runs that the lossless compressor can exploit
- scale tables and other side metadata
- framing / container overhead

Therefore the only trustworthy byte-cost number is:

- bytes measured on the actual exported artifact
- or measured deltas from swapping one tensor group at a time inside the actual export path

This is a critical point. A tensor can be expensive in nominal bits but cheap after `zlib`, or vice versa.

### Second-Order Distortion Proxies

The searched compression literature gives a clean mathematical hierarchy for choosing `D_g`.

At the most rigorous level:

- OBC / OBQ formulate layer-wise compression as minimizing output reconstruction error under a compression constraint, and show that OBS-style formulas are exact for the quadratic layer loss they use.
- GPTQ scales this same second-order idea to large GPT models using approximate Hessian information and blockwise updates.
- HAWQ-V2 argues that average Hessian trace is a better layer sensitivity metric than just the top Hessian eigenvalue, and uses a Pareto-frontier view for mixed-precision selection.
- APTQ extends Hessian-trace sensitivity to attention-aware mixed-precision quantization for LLMs.

For this project, that implies the following priority order for sensitivity measurement:

1. best: direct measured final-score damage from actual export changes
2. next best: post-quant eval damage under a matched evaluator
3. fallback proxy: Hessian-aware or output-error-aware sensitivity
4. worst proxy: raw weight magnitude or folklore about layer importance

If we are tight on experimental budget, Hessian-aware proxies are the right mathematical shortcut.

## 11. Training Must Be Judged By Post-Quant Final Score

Let:

- `B_pre` be dense score before export
- `B_post` be score after export under the chosen evaluator

If a training change gives:

- `Delta B_pre < 0`
- but `Delta B_post >= 0`

then the change is not useful for the leaderboard.

Therefore training experiments must report:

- dense/base-eval score
- post-quant/base-eval score
- post-quant/final-eval score

This supports three diagnostics:

1. Better training that survives export
2. Better training that collapses under export
3. Better training that only pays off under richer evaluation

The third case is especially important in this challenge because context utilization is a major axis of gain.

## 12. Throughput Model For Training

Training quality depends on the number of optimizer updates and tokens processed under the 600-second cap.

Let:

- `tau` be measured `ms/step`
- `N_steps = floor(600,000 / tau)`
- `Toks_step` be tokens per optimizer step

Then total train tokens are:

`Toks_total = N_steps * Toks_step`

Any systems optimization that reduces `tau` increases `N_steps` linearly:

`Delta N_steps ~= -600,000 * Delta_tau / tau^2`

This lets us value low-level speedups quantitatively.

Example interpretation:

- shaving `4 ms` from a `56 ms/step` run is not "small"
- it can buy hundreds of extra updates inside the same fixed wallclock budget

So low-level code work is justified only when the induced increase in `N_steps` or `Toks_total` maps to measurable final-score gain.

## 13. Required Experimental Table For Every Run

Every run should log the following fields in one structured row:

- git commit
- config identifier
- random seed
- train wallclock `T_train`
- eval wallclock `T_eval`
- artifact size `A`
- slack values `S_train`, `S_eval`, `S_artifact`
- dense flat-stream score
- post-quant flat-stream score
- post-quant final-eval score
- quantization damage: `B_post_flat - B_dense_flat`
- eval gain: `B_post_final - B_post_flat`
- final score delta vs reference
- measured step time
- measured eval tokens/s
- raw `loss_sum`
- raw `token_count`
- raw `byte_count`

Derived columns:

- score gain per eval second
- score gain per artifact byte
- confidence interval width across seeds
- implied `token_count / byte_count`
- implied acceptance-margin conversion from `nats/token` to `BPB`

If an experiment does not produce this table row, it is incomplete.

## 14. Gating Rules

We need strict promotion rules between stages.

### Gate 0: Metric Lock

Before trusting any comparison, verify:

- validation shards are unchanged
- tokenizer model and byte-accounting LUTs are unchanged
- printed `val_bpb` matches recomputation from raw `loss_sum`, `token_count`, and `byte_count`

### Gate A: Correctness

Before comparing scores, verify:

- artifact loads correctly
- score accounting matches expected token and byte counts
- evaluator semantics match the intended comparison contract
- runtime and artifact budgets are satisfied

### Gate B: Practical significance

Promote a change only if:

- mean improvement exceeds measurement noise by a conservative margin
- or the change creates enough budget slack to enable the next optimization stage

### Gate C: Acceptance path

Invest in multi-seed confirmation only when:

- projected mean score beats the current target with margin
- runtime is comfortably under cap
- no known correctness caveats remain

### Gate D: Submission Buffer

Because leaderboard acceptance is public and chronological, do not target the bare minimum improvement.

Require:

- a score target that exceeds the theoretical acceptance floor by a practical buffer
- enough runtime slack to survive reproduction variance
- enough statistical margin that a modest SOTA move does not immediately invalidate the submission effort

## 15. Recommended Optimization Order

A rigorous budget-aware order is:

1. Reproduce the current reference exactly
2. Freeze a trusted measurement harness
3. Improve evaluator correctness and asymptotics
4. Spend unused eval seconds for score
5. Reduce export distortion using byte-allocation analysis
6. Retune training against the improved evaluator
7. Only then explore architectural changes

Reason:

- early architectural exploration without a trustworthy evaluator wastes search effort
- export and eval gains may dominate model changes at the current frontier
- the project should always optimize the active bottleneck

## 16. Failure Modes To Avoid

Common invalid approaches:

- optimizing pre-quant loss as the primary target
- comparing single-seed deltas smaller than observed noise
- mixing score units
- using eval methods whose context semantics are ambiguous or hard to defend
- spending artifact bytes without measured sensitivity analysis
- relying on wallclock results with no safety margin under the cap
- conflating better evaluation with better training

Each of these can produce attractive-looking but non-submittable results.

## 17. Minimal Working Checklist

Before claiming a serious candidate run, confirm:

- reference reproduction is within tolerance
- all metrics are recorded in consistent units
- dense, post-quant, and final-eval scores are separated
- for this strategy, evaluator is document-isolated and exact in token accounting
- eval state reuse is implemented if sliding geometry is used aggressively
- artifact allocation decisions are justified by measured marginal value
- runtime slack is sufficient to survive variance
- enough seeds are available to estimate variance honestly

## 18. Final Principle

This challenge should be approached like numerical optimization under constraints, not like open-ended model tinkering.

The winning workflow is:

- define the true objective exactly
- decompose the score into measurable components
- quantify every budget
- estimate variance before trusting small gains
- spend bytes, seconds, and implementation complexity only where their marginal return is highest

If we do that rigorously, the path to a verified record becomes much clearer:

- use evaluation as a legal compute budget
- treat export as a constrained distortion-allocation problem
- retune training only after the score pipeline is mathematically nailed down

## Appendix: Mathematical References

Primary sources and near-primary references used to tighten the math in this document:

- Claude Shannon, *A Mathematical Theory of Communication* (1948): source coding theorem foundation.
  https://www.mpi.nl/publications/item2383162/mathematical-theory-communication

- Grégoire Delétang et al., *Language Modeling Is Compression* (2023/2024): explicit predictor-compressor equivalence, arithmetic coding view, and tokenization discussion.
  https://arxiv.org/abs/2309.10668

- Alistair Moffat, Radford Neal, Ian Witten, *Arithmetic Coding Revisited* (1998): practical finite-precision arithmetic coding perspective.
  https://researchcommons.waikato.ac.nz/items/ef7c7d25-0857-448f-a02e-0895747df2bc

- Elias Frantar, Sidak Pal Singh, Dan Alistarh, *Optimal Brain Compression* (2022/2023): exact quadratic layer-loss framing and OBS-based quantization/pruning math.
  https://arxiv.org/abs/2208.11580

- Elias Frantar et al., *GPTQ* (2022/2023): approximate second-order quantization scaled to large GPT models.
  https://arxiv.org/abs/2210.17323

- Zhen Dong et al., *HAWQ-V2* (2019/2020): Hessian-trace sensitivity and Pareto-front mixed-precision selection.
  https://arxiv.org/abs/1911.03852

- Ziyi Guan et al., *APTQ* (2024): attention-aware Hessian-trace mixed-precision quantization for LLMs.
  https://arxiv.org/abs/2402.14866
