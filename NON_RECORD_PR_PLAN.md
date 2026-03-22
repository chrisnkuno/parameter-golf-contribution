# Non-Record PR Plan

This document prepares a non-record submission and compute-grant request for the current `parameter-golf` branch.

The branch we are carrying forward is:

- `11L + XSA4`
- `TRAIN_SEQ_LEN=256`
- `TRAIN_BATCH_TOKENS=524288`
- `WARMDOWN_ITERS=200`
- `packed_zstd` artifact path
- `tok_emb.weight` float keep only when it is worth the bytes
- checkpoint-frontier training via `SAVE_DENSE_CHECKPOINT_EVERY`

The working goal is:

- push below `1.4 BPB`
- stay under the decimal `16,000,000` byte cap
- if the run does not satisfy the `10 minute on 8xH100` record constraints, package it as a strong non-record unlimited-compute submission

## Why A Non-Record PR

The challenge supports two different ambitions:

- record-track runs that train in under `10 minutes` on `8xH100`
- non-record unlimited-compute runs that are still interesting, reproducible, and under the `16 MB` artifact cap

Our current line is well-suited for the non-record track because:

- it is a real architectural branch, not a trivial hyperparameter retune
- it has already shown strong proxy improvement on the `5090`
- it is still under active artifact-economics work, so final quality may benefit from more compute even if the record-time constraint is not yet locked

This is aligned with the challenge text:

- the organizers explicitly encourage unique and interesting non-record submissions
- unlimited-compute non-record submissions are accepted
- compute grants are intended to help participants explore serious technical directions, not only record-ready ones

## Current Technical Thesis

The thesis for this branch is:

1. `11L + XSA4` is the best architecture line we have kept after ruling out local mixer and ZeroS in the tested regime.
2. The biggest training-side gain so far came from moving to the challenge-like global batch regime:
   - `TRAIN_BATCH_TOKENS=524288`
   - `TRAIN_SEQ_LEN=256`
3. The bottleneck is no longer obvious architecture instability.
4. The bottleneck is the joint optimization problem:
   - lower post-quant `val_bpb`
   - while still staying under `16,000,000` total bytes

That makes this branch a good candidate for more compute because longer runs alone improve BPB, but artifact size then becomes the limiting factor. More compute lets us map the frontier more cleanly and choose the best legal checkpoint.

## Why We Need More Compute

The argument for more compute should be:

- We are no longer in a random-search phase.
- We have a stable, measured architecture and training regime.
- The remaining work is a frontier-selection problem across checkpoints and export policies.

Concretely:

- `600`-step proxy checkpoints got close to the byte cap while remaining legal.
- `800`- and `1000`-step checkpoints improved score further but exceeded the cap.
- This means the branch is promising, but final quality depends on:
  - longer training
  - denser checkpoint selection
  - stronger post-training export selection

That is exactly the type of work that benefits from more GPU time.

## Proposed Compute-Grant Framing

Use language like this:

> I am working on a parameter-golf branch built around an `11-layer` transformer with `XSA` in the final `4` layers, trained at the challenge-like global batch regime (`524288` tokens/step, `seq_len=256`). The branch is already producing strong post-quant results on a `5090` proxy, but the main open problem is selecting the best checkpoint/export pair under the strict `16,000,000` byte artifact cap.  
>
> The requested compute would be used for a reproducible unlimited-compute non-record submission and for mapping the checkpoint frontier under the actual `8xH100` environment. The work is focused and measured rather than exploratory brute force: longer runs, saved checkpoint frontiers, post-quant artifact audits, and legal checkpoint selection.  
>
> This branch explores a concrete architectural idea (`11L + XSA4`) plus submission-specific artifact economics, which fits the challenge’s emphasis on parameter-efficient modeling, compression-aware design, and creative but rigorous engineering.

Short justification bullets:

- novel architecture ingredient: `XSA` in the tail only
- challenge-aligned training regime: large global batch, tied embeddings, aggressive post-quant accounting
- measured artifact-economics pipeline: packed artifact, `zstd`, checkpoint frontier selection
- intended output: non-record unlimited-compute submission plus reproducible logs and code

## What The Non-Record PR Should Contain

The PR should add exactly one new folder under:

- `records/track_non_record_16mb/<DATE>_<RUN_NAME>/`

Suggested folder shape:

- `README.md`
- `submission.json`
- `train.log`
- `train_gpt.py`
- `requirements.txt`
- optionally small helper scripts if the run depends on them and they are actually used in the record folder

Do not rely on the repo root at review time. The record folder must be self-contained enough to run.

## Recommended Folder Naming

Use a descriptive folder name such as:

- `records/track_non_record_16mb/2026-03-22_11L_XSA4_Seq256_Batch524K_Frontier`

If the final run is clearly unlimited-compute, include that in the README, not necessarily the folder name.

## README Structure

The record README should include:

1. What the submission is

- non-record unlimited-compute track
- still under the `16,000,000` byte artifact cap
- not claiming record-track `10 minute / 8xH100` compliance unless we actually verify it

2. Core idea

- `11L + XSA4`
- why XSA is used only in the tail
- why `seq256 / batch524288 / wd200` was kept
- how checkpoint-frontier selection is used to stay under the byte cap

3. Exact configuration

- architecture knobs
- optimizer knobs
- batch / sequence length
- warmdown
- model averaging choice if any
- artifact format and quantization settings

4. Exact command

- the full command used on the target machine

5. Results

- best dense metric
- best post-quant metric
- exact printed `final_int8_zlib_roundtrip_exact`
- total submission bytes
- wallclock and hardware used

6. What was learned

- what worked
- what failed
- why this is interesting even if it is non-record

## submission.json Fields

The `submission.json` should include at least:

- `author`
- `github_id`
- `name`
- `blurb`
- `date`
- `track`
- `val_loss`
- `val_bpb`
- `pre_quant_val_loss`
- `pre_quant_val_bpb`
- `step_stop`
- `wallclock_seconds`
- `bytes_total`
- `bytes_model_int8_zlib` or equivalent final artifact bytes
- `bytes_code`

Use:

- `track: "non-record-unlimited-compute-16mb"`

if the run is intentionally outside the main record-time constraint.

## H100 Deployment Checklist

After deployment on H100:

1. Verify the environment

- CUDA visible
- FlashAttention path active if expected
- tokenizer and dataset present

2. Run a short smoke

- confirm exact branch settings
- confirm `SAVE_DENSE_CHECKPOINT_EVERY` works

3. Run the long frontier job

- save checkpoints every `100` steps
- keep validation off during the body if the goal is frontier generation

4. Sweep the saved checkpoints afterward

- evaluate post-quant BPB
- compute exact final artifact bytes
- keep only candidates under `16,000,000`

5. Select the best legal checkpoint

- lowest `final_int8_zlib_roundtrip_exact val_bpb`
- subject to the byte cap

6. Package the record folder from that selected checkpoint only

## Practical Submission Rule

Do not package “the last checkpoint” by default.

Package:

- the best legal checkpoint on the frontier

That is the key operational rule for this branch because score improves with more steps, but artifact size also grows.

## Minimum Acceptance Standard For Our PR

Even though it is non-record, the submission should still clear a high bar:

- fully reproducible
- technically interesting
- artifact under `16,000,000`
- clean explanation of the approach
- clear logs and exact metrics

If the run is not under the byte cap, do not open the PR yet.

## Immediate Next Actions

1. Finish the current frontier run with saved checkpoints.
2. Sweep checkpoint exports to identify the best legal candidate.
3. Deploy on H100.
4. Repeat the frontier workflow on H100.
5. Package the best legal H100 result into a non-record record folder.
6. Use this document’s compute justification language in the compute-grant request.
