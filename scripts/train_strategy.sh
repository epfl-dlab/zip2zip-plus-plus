#!/bin/bash
set -e

DATA_DIR=/mnt/scratch/data/finemath-10bt
OUTPUT_BASE=/mnt/scratch/checkpoints/strategy_ablation
STEPS=6000
HF_MODEL=meta-llama/Llama-3.2-1B
NGPU=1
COMMON="--data_dir $DATA_DIR --model_config 1B_llama3.2 --init_from_hf $HF_MODEL --max_subtokens 2 --max_active_codebook_size 4096 --steps $STEPS --lr 5e-5 --min_lr 5e-6 --wandb --wandb_project zip2zip-core-GB200 --encoder_dim 2048"

declare -a EXPERIMENTS=(
    # "freeze_res|--freeze_decoder|Freeze decoder + residual"
    # "freeze_nores|--freeze_decoder --no_encoder_residual|Freeze decoder + no residual"
    # "fullft_res||Full finetune + residual"
    # "fullft_nores|--no_encoder_residual|Full finetune + no residual"
    # "lora_res|--lora_rank 16|LoRA + residual"
    "lora_nores|--lora_rank 16 --no_encoder_residual|LoRA + no residual"
)

for exp in "${EXPERIMENTS[@]}"; do
    IFS='|' read -r name flags desc <<< "$exp"
    echo "========================================"
    echo "$desc ($name)"
    echo "========================================"
    torchrun --nproc_per_node=$NGPU -m zip2zip_core.train \
        $COMMON \
        --output_dir "${OUTPUT_BASE}/${name}" \
        --wandb_name "strategy_${name}" \
        $flags
done
