#!/bin/bash
set -e

DATA_DIR=/mnt/scratch/data/zip2zip-plus-mixture-20b
OUTPUT_BASE=/mnt/scratch/checkpoints/ft
HF_MODEL=meta-llama/Llama-3.2-1B

echo "========================================"
echo "Finetuning Llama 3.2 1B with max_subtokens=2"
echo "========================================"
torchrun --nproc_per_node=4 -m zip2zip_core.train \
    --data_dir "$DATA_DIR" \
    --output_dir "${OUTPUT_BASE}/zip2zip_1b_llama32_finetune_ms1" \
    --model_config 1B_llama3.2 \
    --init_from_hf "$HF_MODEL" \
    --max_subtokens 1 \
    --max_active_codebook_size 4096 \
    --lr 5e-5 \
    --min_lr 5e-6 \
    --wandb \
    --wandb_project "zip2zip-core-GB200" \
    --no_remap_codebook --hyper_causal_mask \
    --gradient_accumulation_steps 4 \
    --max_tokens 1_000_000_000 \
    --wandb_name "zip2zip_1b_llama32_finetune_ms1"
