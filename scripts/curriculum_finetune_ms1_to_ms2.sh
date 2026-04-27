#!/bin/bash
set -e

# Two experiments comparing curriculum finetuning vs. direct training at ms=2.
#
# Run A — Baseline: train ms=2 from scratch for 16K steps.
# Run B — Finetune: pretrain ms=1 for 16K steps, then finetune ms=2 for 16K steps
#          with warm-start weights but a fresh optimizer and fresh LR schedule.
#
# Each experiment is a single wandb run (two phases of Run B resume the same run).

DATA_DIR=/mnt/scratch/data/finemath-10bt
STEPS=16000

# ── Run A: ms=2 from scratch ─────────────────────────────────────────────────
RUN_A_OUTPUT=/mnt/scratch/checkpoints/zip2zip_1b_ms2_scratch_16k
RUN_A_WANDB_ID=${RUN_A_WANDB_ID:-ms2_scratch_$(date +%s)}

# echo "========================================"
# echo "Run A: ms=2 from scratch (16K steps)"
# echo "========================================"
# torchrun --nproc_per_node=4 -m zip2zip_core.train \
#     --data_dir "$DATA_DIR" \
#     --output_dir "$RUN_A_OUTPUT" \
#     --max_subtokens 2 \
#     --max_active_codebook_size 4096 \
#     --steps "$STEPS" \
#     --save_freq "$STEPS" \
#     --wandb \
#     --wandb_project "zip2zip-core-GB200" \
#     --wandb_name "zip2zip_1b_ms2_scratch" \
#     --wandb_id "$RUN_A_WANDB_ID" --wandb_resume allow \
#     --no_remap_codebook --hyper_causal_mask

# echo "Run A complete."

# ── Run B phase 1: ms=1 pretrain for 16K steps ───────────────────────────────
RUN_B_OUTPUT=/mnt/scratch/checkpoints/zip2zip_1b_ms1_then_ms2_16k+16k
RUN_B_WANDB_ID=${RUN_B_WANDB_ID:-ms1_finetune_ms2_$(date +%s)}

# echo "========================================"
# echo "Run B phase 1: ms=1 pretrain (16K steps)"
# echo "========================================"
# torchrun --nproc_per_node=4 -m zip2zip_core.train \
#     --data_dir "$DATA_DIR" \
#     --output_dir "$RUN_B_OUTPUT" \
#     --max_subtokens 1 \
#     --max_active_codebook_size 4096 \
#     --steps "$STEPS" \
#     --save_freq "$STEPS" \
#     --wandb \
#     --wandb_project "zip2zip-core-GB200" \
#     --wandb_name "zip2zip_1b_ms1_then_ms2" \
#     --wandb_id "$RUN_B_WANDB_ID" --wandb_resume allow \
#     --no_remap_codebook --hyper_causal_mask

echo "========================================"
echo "Run B phase 2: finetune ms=2 (16K steps, warm-start from ms=1 ckpt)"
echo "  - loads ms=1 weights (warm start)"
echo "  - fresh optimizer (no ms=1 optimizer state)"
echo "  - fresh LR schedule from step 0"
echo "========================================"
torchrun --nproc_per_node=4 -m zip2zip_core.train \
    --data_dir "$DATA_DIR" \
    --output_dir "$RUN_B_OUTPUT" \
    --max_subtokens 2 \
    --max_active_codebook_size 4096 \
    --steps "$STEPS" \
    --resume_from "${RUN_B_OUTPUT}/step_${STEPS}" \
    --reset_step \
    --save_freq "$STEPS" \
    --wandb \
    --wandb_project "zip2zip-core-GB200" \
    --wandb_name "zip2zip_1b_ms1_then_ms2" \
    --no_remap_codebook --hyper_causal_mask

echo "All done!"
