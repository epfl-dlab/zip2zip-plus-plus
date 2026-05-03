#!/bin/bash
set -e

DATA_DIR=/mnt/scratch/data/llaza-20B
OUTPUT_BASE=/mnt/scratch/checkpoints/llaza
HF_MODEL=meta-llama/Llama-3.2-1B-Instruct

echo "========================================"
echo "Finetuning Llama 3.2 1B Instruct with max_subtokens=2"
echo "========================================"
torchrun --nproc_per_node=4 -m zip2zip_core.train \
    --data_dir "$DATA_DIR" \
    --output_dir "${OUTPUT_BASE}/llaza_1b_llama32_instruct_ms2_4bt" \
    --model_config 1B_llama3.2 \
    --init_from_hf "$HF_MODEL" \
    --max_subtokens 2 \
    --max_active_codebook_size 4096 \
    --lr 5e-5 \
    --min_lr 5e-6 \
    --wandb \
    --no_remap_codebook --hyper_causal_mask \
    --gradient_accumulation_steps 4 \
    --max_tokens 4_000_000_000 \
    --wandb_name "llaza_1b_llama32_instruct_ms2_4bt"
