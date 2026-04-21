#!/bin/bash
set -e

DATA_DIR=/mnt/scratch/data/finemath-10bt
EVAL_STEPS=${EVAL_STEPS:-100}

# Usage:
#   bash scripts/eval_lm.sh /path/to/checkpoint
#   MODEL_CONFIG=400M MAX_SUBTOKENS=2 bash scripts/eval_lm.sh /path/to/checkpoint
CHECKPOINT=${1:?Usage: bash scripts/eval_lm.sh <checkpoint_dir>}
MODEL_CONFIG=${MODEL_CONFIG:-1B}
MAX_SUBTOKENS=${MAX_SUBTOKENS:-4}

echo "========================================"
echo "Evaluating checkpoint: ${CHECKPOINT}"
echo "  model_config=${MODEL_CONFIG}, max_subtokens=${MAX_SUBTOKENS}"
echo "========================================"
torchrun --nproc_per_node=1 -m zip2zip_core.train \
    --data_dir "$DATA_DIR" \
    --output_dir /tmp/eval_dummy \
    --model_config "$MODEL_CONFIG" \
    --max_subtokens "$MAX_SUBTOKENS" \
    --max_active_codebook_size 4096 \
    --steps "$EVAL_STEPS" \
    --resume_from "$CHECKPOINT" \
    --eval \
    --no_remap_codebook --hyper_causal_mask
