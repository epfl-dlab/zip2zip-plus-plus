#!/bin/bash
set -e

# Ablation sweep: codebook masking strategies (1B model, ms=6, 6K steps each)
# 3 setups:
#   1) no flags (remap codebook, no hyper causal mask)
#   2) --no_remap_codebook only
#   3) --no_remap_codebook --hyper_causal_mask

DATA_DIR=/mnt/scratch/data/finemath-10bt
OUTPUT_BASE=/mnt/scratch/checkpoints
STEPS=6000
MODEL=1B
MS=6

echo "========================================"
echo "Setup 1: remap codebook, no hyper causal mask (default)"
echo "========================================"
torchrun --nproc_per_node=4 -m zip2zip_core.train \
    --data_dir "$DATA_DIR" \
    --output_dir "${OUTPUT_BASE}/zip2zip_${MODEL}_ablate_cb_remap" \
    --model_config "$MODEL" \
    --mode lm \
    --max_subtokens "$MS" \
    --max_active_codebook_size 4096 \
    --steps "$STEPS" \
    --wandb \
    --wandb_project "zip2zip-core-GB200" \
    --wandb_name "zip2zip_${MODEL}_cb_remap" \
    --wandb_group "ablate_codebook_masking" \
    --wandb_tags lm "$MODEL" codebook-ablation remap

echo "========================================"
echo "Setup 2: no remap codebook, no hyper causal mask"
echo "========================================"
torchrun --nproc_per_node=4 -m zip2zip_core.train \
    --data_dir "$DATA_DIR" \
    --output_dir "${OUTPUT_BASE}/zip2zip_${MODEL}_ablate_cb_noremap" \
    --model_config "$MODEL" \
    --mode lm \
    --max_subtokens "$MS" \
    --max_active_codebook_size 4096 \
    --steps "$STEPS" \
    --wandb \
    --wandb_project "zip2zip-core-GB200" \
    --wandb_name "zip2zip_${MODEL}_cb_noremap" \
    --wandb_group "ablate_codebook_masking" \
    --wandb_tags lm "$MODEL" codebook-ablation noremap \
    --no_remap_codebook

echo "========================================"
echo "Setup 3: no remap codebook + hyper causal mask"
echo "========================================"
torchrun --nproc_per_node=4 -m zip2zip_core.train \
    --data_dir "$DATA_DIR" \
    --output_dir "${OUTPUT_BASE}/zip2zip_${MODEL}_ablate_cb_noremap_hcm" \
    --model_config "$MODEL" \
    --mode lm \
    --max_subtokens "$MS" \
    --max_active_codebook_size 4096 \
    --steps "$STEPS" \
    --wandb \
    --wandb_project "zip2zip-core-GB200" \
    --wandb_name "zip2zip_${MODEL}_cb_noremap_hcm" \
    --wandb_group "ablate_codebook_masking" \
    --wandb_tags lm "$MODEL" codebook-ablation noremap hyper-causal-mask \
    --no_remap_codebook --hyper_causal_mask

echo "Codebook masking ablation complete!"
