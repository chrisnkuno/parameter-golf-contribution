This record captures a non-record 16MB submission built from the current root `train_gpt.py`, validated on a single `H100 80GB`.

This run is not presented as a main-leaderboard record because the challenge record track requires reproducible `8xH100` evidence and statistical significance over the current SOTA. It is instead a legal, reproducible non-record result that packages the best under-cap checkpoint from an H100 checkpoint frontier.

Configuration:
- Track: `non-record`, still under the decimal `16,000,000` byte artifact cap
- Layout: `VOCAB_SIZE=1024 NUM_LAYERS=11 MODEL_DIM=512 NUM_HEADS=8 NUM_KV_HEADS=4 MLP_MULT=2`
- Attention: `XSA_TAIL_LAYERS=4`
- Tied output/input embeddings: `TIE_EMBEDDINGS=1`
- Batching: `TRAIN_BATCH_TOKENS=524288 TRAIN_SEQ_LEN=256`
- Validation/export mode during training: `VAL_LOSS_EVERY=0 RUN_TTT_EVAL=0`
- Frontier saving: `SAVE_DENSE_CHECKPOINT_EVERY=25`
- Artifact path: `packed_zstd`

Training command:
```bash
RUN_ID=11l_xsa4_h100_scored \
NUM_LAYERS=11 \
XSA_TAIL_LAYERS=4 \
TRAIN_SEQ_LEN=256 \
EVAL_SEQ_LEN=256 \
EVAL_STRIDE=256 \
EVAL_BATCH_SEQS=32 \
TRAIN_BATCH_TOKENS=524288 \
WARMDOWN_ITERS=200 \
ITERATIONS=650 \
TRAIN_LOG_EVERY=50 \
VAL_LOSS_EVERY=0 \
RUN_TTT_EVAL=0 \
SAVE_DENSE_CHECKPOINT_EVERY=25 \
MAX_WALLCLOCK_SECONDS=0 \
python train_gpt.py
```

Checkpoint selection:
- The raw final export at `step650` scored better before the legal-size filter but exceeded the artifact cap.
- We therefore selected from the saved frontier checkpoints after training.
- The chosen legal checkpoint is:
  - `logs/checkpoints/11l_xsa4_h100_scored_step00650.pt`
  - evaluated with `--no-default-large-keeps`

Post-training selection command:
```bash
python scripts/eval_quant_candidate.py \
  --state-dict-path logs/checkpoints/11l_xsa4_h100_scored_step00650.pt \
  --no-default-large-keeps \
  --val-batch-size 524288 \
  --num-layers 11 \
  --xsa-tail-layers 4 \
  --train-seq-len 256
```

Key metrics:
- Raw final run at `step650`:
  - `val_loss:2.4650`
  - `val_bpb:1.4599`
  - final roundtrip exact: `val_bpb:1.46031761`
  - total submission size int8+zlib: `16291660` bytes
- Best legal checkpoint selection:
  - `step650 + no_default_large_keeps`
  - `val_loss:2.46718130`
  - `val_bpb:1.46120374`
  - compressed model bytes: `15907290`
  - code bytes: `76313`
  - total bytes: `15983603`

Performance notes:
- Single-H100 train step average at `650` steps: about `388.36ms`
- Peak memory: `13451 MiB allocated`, `13496 MiB reserved`
- Dense checkpoint frontier was saved every `25` steps through `650`

Included files:
- `train_gpt.py` (exact code snapshot used for the run)
- `train.log` (exact H100 training log)
- `submission.json` (metadata)
- `requirements.txt` (runtime packages)

What to do next for better scores:
- Run the same `11L + XSA4` stack on `8xH100` and repeat the legal frontier selection under the real record-track budget.
- Keep the `step650 + no_keep` policy as the baseline legal selector, then test whether a slightly later checkpoint remains under cap on `8xH100`.
- Push artifact economics further. The current bottleneck is still payload compression, not metadata. The next promising branch remains stronger post-training quantization or precision allocation, not more serializer work.
- Re-run the selected checkpoint frontier with multiple seeds if the goal shifts from a non-record package to a statistically defensible record attempt.
