#!/bin/bash
set -e

DATA_DIR=/mnt/scratch/data/finemath-10bt
OUTPUT_BASE=/mnt/scratch/checkpoints
STEPS=6000

for model_size in 20M 50M 150M 400M 3B; do
  for ms in 4 2; do
      echo "========================================"
      echo "Training COMPRESS mode with model_config=${model_size}, max_subtokens=${ms}"
      echo "========================================"
      torchrun --nproc_per_node=4 -m zip2zip_core.train \
          --data_dir "$DATA_DIR" \
          --output_dir "${OUTPUT_BASE}/zip2zip_${model_size}_finemath_10bt_compress_ms${ms}" \
          --model_config "$model_size" \
          --mode compress \
          --max_subtokens "$ms" \
          --max_active_codebook_size 4096 \
          --steps "$STEPS" \
          --wandb \
          --wandb_project "zip2zip-core-GB200" \
          --wandb_name "zip2zip_${model_size}_compress_ms${ms}" \
          --wandb_group "sweep_compress_model_size_ms" \
          --wandb_tags compress "${model_size}" "ms${ms}" finemath-10bt \
          --no_remap_codebook --hyper_causal_mask
  done
done
