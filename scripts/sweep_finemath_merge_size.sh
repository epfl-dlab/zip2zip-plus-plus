#!/bin/bash
set -e

DATA_DIR=/mnt/scratch/data/finemath-10bt
OUTPUT_BASE=/mnt/scratch/checkpoints
STEPS=6000

for ms in 6 5 4 3 2 1; do
    echo "========================================"
    echo "Training with max_subtokens=${ms}"
    echo "========================================"
    torchrun --nproc_per_node=4 -m zip2zip_core.train \
        --data_dir "$DATA_DIR" \
        --output_dir "${OUTPUT_BASE}/zip2zip_1b_finemath_10bt_ms${ms}" \
        --max_subtokens "$ms" \
        --max_active_codebook_size 4096 \
        --steps "$STEPS" \
        --wandb \
        --wandb_project "zip2zip-core-GB200" \
        --wandb_name "zip2zip_1b_ms${ms}"
done
