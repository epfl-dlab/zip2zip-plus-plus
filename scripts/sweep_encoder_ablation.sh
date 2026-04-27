#!/bin/bash
set -e

# Ablation sweep: hyper encoder architecture (1B model, ms=4, 6K steps each)
# Base model dim=2048. Default encoder: dim=512, n_heads=8, n_layers=2, intermediate_size=2048
# Each ablation changes one axis while keeping others at default.

DATA_DIR=/mnt/scratch/data/finemath-10bt
OUTPUT_BASE=/mnt/scratch/checkpoints
STEPS=6000
MODEL=1B
MS=2

run_ablation() {
    local name=$1
    shift
    echo "========================================"
    echo "Encoder ablation: ${name}"
    echo "========================================"
    torchrun --nproc_per_node=4 -m zip2zip_core.train \
        --data_dir "$DATA_DIR" \
        --output_dir "${OUTPUT_BASE}/zip2zip_${MODEL}_encoder_ablation_${name}" \
        --model_config "$MODEL" \
        --mode lm \
        --max_subtokens "$MS" \
        --max_active_codebook_size 4096 \
        --steps "$STEPS" \
        --wandb \
        --wandb_project "zip2zip-core-GB200" \
        --wandb_name "zip2zip_${MODEL}_enc_${name}" \
        --wandb_group "sweep_encoder_ablation" \
        --wandb_tags lm "$MODEL" encoder-ablation "$name" \
        --local_batch_size 4 --gradient_accumulation_steps 4 \
        --no_remap_codebook --hyper_causal_mask \
        "$@"
}

# --- Ablate encoder dim (+ proportional intermediate_size) ---
# Default is dim=512, intermediate=2048. Sweep: same as base model → very small.
run_ablation "dim2048" --encoder_n_heads 32 --encoder_dim 2048 --encoder_intermediate_size 8192
run_ablation "dim1024" --encoder_n_heads 16 --encoder_dim 1024 --encoder_intermediate_size 4096
# run_ablation "dim512"  # baseline (default)

# --- Ablate encoder heads (scale dim to keep head_dim=64) ---
run_ablation "heads4"  --encoder_n_heads 4 --encoder_dim 256 --encoder_intermediate_size 1024
run_ablation "heads2"  --encoder_n_heads 2 --encoder_dim 128 --encoder_intermediate_size 512

# --- Ablate encoder layers ---
run_ablation "layers1" --encoder_n_layers 1

echo "Encoder ablation sweep complete!"
