#!/bin/bash
set -e

DATA_DIR=${DATA_DIR:-/mnt/scratch/data/finemath-10bt}
EVAL_STEPS=${EVAL_STEPS:-100}

# Usage:
#   bash scripts/eval_lm.sh /path/to/checkpoint
#   MODEL_CONFIG=400M MAX_SUBTOKENS=2 bash scripts/eval_lm.sh /path/to/checkpoint
#   HYPER_ENCODER_TYPE=hierarchical bash scripts/eval_lm.sh /path/to/checkpoint
#   ONLINE_CODEBOOK_MASK=1 bash scripts/eval_lm.sh /path/to/v0.6.5-checkpoint
CHECKPOINT=${1:?Usage: bash scripts/eval_lm.sh <checkpoint_dir>}
MODEL_CONFIG=${MODEL_CONFIG:-1B}
MAX_SUBTOKENS=${MAX_SUBTOKENS:-4}
HYPER_ENCODER_TYPE=${HYPER_ENCODER_TYPE:-flat}
LOCAL_BATCH_SIZE=${LOCAL_BATCH_SIZE:-8}
ONLINE_CODEBOOK_MASK=${ONLINE_CODEBOOK_MASK:-}
ONLINECB_FLAG=""
if [ -n "$ONLINE_CODEBOOK_MASK" ] && [ "$ONLINE_CODEBOOK_MASK" != "0" ]; then
    ONLINECB_FLAG="--online_codebook_mask"
fi

echo "========================================"
echo "Evaluating checkpoint: ${CHECKPOINT}"
echo "  data_dir=${DATA_DIR}"
echo "  model_config=${MODEL_CONFIG}, max_subtokens=${MAX_SUBTOKENS}, hyper_encoder_type=${HYPER_ENCODER_TYPE}"
echo "  online_codebook_mask=${ONLINE_CODEBOOK_MASK:-0} flag='${ONLINECB_FLAG}'"
echo "========================================"
torchrun --nproc_per_node=1 --master_port=${MASTER_PORT:-29500} -m zip2zip_core.train \
    --data_dir "$DATA_DIR" \
    --output_dir /tmp/eval_dummy \
    --model_config "$MODEL_CONFIG" \
    --max_subtokens "$MAX_SUBTOKENS" \
    --hyper_encoder_type "$HYPER_ENCODER_TYPE" \
    --max_active_codebook_size 4096 \
    --local_batch_size "$LOCAL_BATCH_SIZE" \
    --steps "$EVAL_STEPS" \
    --resume_from "$CHECKPOINT" \
    --eval \
    --no_remap_codebook --hyper_causal_mask \
    $ONLINECB_FLAG
