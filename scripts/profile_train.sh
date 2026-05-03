#!/bin/bash
# Profile a few training steps to measure hyper_encoder vs decoder time.
# Usage:
#   bash scripts/profile_train.sh /path/to/tokens [model_config] [max_subtokens] [profile_steps]
#
# Example:
#   bash scripts/profile_train.sh /scratch/fineweb-tokens debugmodel 2 5
#   bash scripts/profile_train.sh /scratch/fineweb-tokens 1B 4 3
#
# Outputs:
#   - Console table sorted by device_time_total (look for hyper_encoder / decoder / output_logits)
#   - Chrome/TensorBoard traces in $OUTPUT_DIR/profiler_traces/
#     Open with: chrome://tracing or tensorboard --logdir $OUTPUT_DIR/profiler_traces

set -euo pipefail

DATA_DIR=${1:?Usage: $0 <data_dir> [model_config=debugmodel] [max_subtokens=2] [profile_steps=5]}
MODEL_CONFIG=${2:-debugmodel}
MAX_SUBTOKENS=${3:-2}
PROFILE_STEPS=${4:-5}

OUTPUT_DIR=${OUTPUT_DIR:-/tmp/zip2zip-profile}
NGPU=${NGPU:-1}

echo "=== zip2zip profiling ==="
echo "model=$MODEL_CONFIG  max_subtokens=$MAX_SUBTOKENS  profile_steps=$PROFILE_STEPS  ngpu=$NGPU"
echo "output_dir=$OUTPUT_DIR"

torchrun --nproc_per_node=$NGPU -m zip2zip_core.train \
    --data_dir "$DATA_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --model_config "$MODEL_CONFIG" \
    --max_subtokens "$MAX_SUBTOKENS" \
    --local_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --steps 100 \
    --log_freq 1 \
    --save_freq 9999 --hyper_encoder_type flat --no_remap_codebook --hyper_causal_mask --no_encoder_residual --encoder_dim 2048
