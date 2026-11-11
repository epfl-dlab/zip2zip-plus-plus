#!/bin/bash
# Inference-Time Pareto Sweep on Phi-3.5-mini-instruct (zip2zip++ step_8000)
# Supports Inference-Time Pareto Sweep Q2: Accuracy vs. Compression trade-off.
# Evaluates narrowed merge sizes (M in {1, 2, 3, 4}) and codebook capacities (K in {0, 256, 512, 1024, 2048, 4096})
# without any retraining.
#
# Usage (Run:AI single GPU):
#   runai-rcp-prod submit --name eval-phi35-pareto-sweep \
#     --image ghcr.io/jkminder/dlab-runai-images/pytorch:master \
#     --gpu 1 --cpu 8 --memory 64Gi \
#     --pvc dlab-scratch:/mnt --large-shm --node-pools h100,default \
#     --environment WANDB_API_KEY="" \
#     -- bash ${SCRATCH_DIR:-/mnt/scratch}/code/zip2zip-core/scripts/run_pareto_sweep_phi35.sh
#
set -euo pipefail

Z2Z_SCRATCH=${Z2Z_SCRATCH:-${SCRATCH_DIR:-/mnt/scratch}}
PROJECT_DIR=${PROJECT_DIR:-$Z2Z_SCRATCH/code/zip2zip-core}
CKPT_DIR=${CKPT_DIR:-$Z2Z_SCRATCH/zip2zip-outputs/andrea-z2z-phi35-4B-sharedS-zeroinit-1BData-v0.6.4-Zip2zipCore/step_8000}
TOKENIZER=${TOKENIZER:-microsoft/Phi-3.5-mini-instruct}
LOG_DIR=$Z2Z_SCRATCH/logs/eval
mkdir -p "$LOG_DIR"

if [ ! -f "$CKPT_DIR/model.pt" ] || [ ! -f "$CKPT_DIR/meta.pt" ]; then
    echo "FATAL: Checkpoint $CKPT_DIR is incomplete or missing." >&2
    exit 1
fi

echo "======================================================================"
echo "Starting Inference-Time Pareto Sweep"
echo "Checkpoint: $CKPT_DIR"
echo "Tokenizer:  $TOKENIZER"
echo "======================================================================"

# Sweep points:
# 1. Merge size narrowing (fixed K=4096): M in {1, 2, 3} (M=4 is default)
# 2. Codebook capacity narrowing (fixed M=4): K in {0, 256, 512, 1024, 2048} (K=4096 is default)
SWEEP_POINTS=(
    "1 4096"
    "2 4096"
    "3 4096"
    "4 0"
    "4 256"
    "4 512"
    "4 1024"
    "4 2048"
)

for point in "${SWEEP_POINTS[@]}"; do
    read -r M K <<< "$point"
    echo ""
    echo "======================================================================"
    echo ">>> Running Sweep Point: M=$M, K=$K"
    echo "======================================================================"
    
    # Run multiple-choice benchmarks + GSM8K + TriviaQA
    EVAL_MAX_SUBTOKENS="$M" \
    EVAL_MAX_CODEBOOK_SIZE="$K" \
    DISABLE_DIGIT_IDS=1 \
    CKPT_DIR="$CKPT_DIR" \
    TOKENIZER="$TOKENIZER" \
    PRESET=default \
    WANDB=0 \
        bash "$PROJECT_DIR/scripts/eval_ckpt_rcp.sh"

    # Run WikiText perplexity
    EVAL_MAX_SUBTOKENS="$M" \
    EVAL_MAX_CODEBOOK_SIZE="$K" \
    DISABLE_DIGIT_IDS=1 \
    CKPT_DIR="$CKPT_DIR" \
    TOKENIZER="$TOKENIZER" \
    PRESET=perplexity \
    TASKS=wikitext \
    WANDB=0 \
        bash "$PROJECT_DIR/scripts/eval_ckpt_rcp.sh"

    echo ">>> Completed Sweep Point: M=$M, K=$K"
done

echo ""
echo "======================================================================"
echo "All Pareto sweep points completed successfully!"
echo "======================================================================"
