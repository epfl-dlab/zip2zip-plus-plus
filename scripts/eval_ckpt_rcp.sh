#!/bin/bash
# Evaluate a zip2zip training checkpoint (model.pt + meta.pt) hosted on
# HuggingFace, using eval_harness.py and the Zip2ZipLM adapter.
#
# ── Usage (Run:AI) ──────────────────────────────────────────────────────
#
#   # MC benchmarks (from-scratch model, no chat template):
#   runai submit --name eval-llaza-ms4-default \
#     --image ghcr.io/jkminder/dlab-runai-images/pytorch:master \
#     --gpu 1 --cpu 8 --memory 64Gi \
#     --pvc dlab-scratch:/mnt --large-shm --node-pools default \
#     --environment HF_REPO=epfl-dlab/candidate-Llaza-MS4-flat-20BT-v1 \
#     --environment PRESET=default_base \
#     -- bash /dlabscratch1/<your-username>/code/zip2zip-core/scripts/eval_ckpt_rcp.sh
#
#   # Perplexity:
#   runai submit --name eval-llaza-ms4-ppl \
#     --image ghcr.io/jkminder/dlab-runai-images/pytorch:master \
#     --gpu 1 --cpu 8 --memory 64Gi \
#     --pvc dlab-scratch:/mnt --large-shm --node-pools default \
#     --environment HF_REPO=epfl-dlab/candidate-Llaza-MS4-flat-20BT-v1 \
#     --environment PRESET=perplexity \
#     -- bash /dlabscratch1/<your-username>/code/zip2zip-core/scripts/eval_ckpt_rcp.sh
#
# ── Environment variables ───────────────────────────────────────────────
#
#   HF_REPO=...     (required) HF repo with model.pt + meta.pt
#   PRESET=...      Eval preset (default: default_base)
#   LIMIT=20        Per-task sample limit for smoke tests
#   TASKS=gsm8k     Comma-separated task override (default: preset's tasks)
#   SCRATCH=...     Your PVC scratch dir (default: /dlabscratch1/gentilin)
#
set -euo pipefail

SCRATCH=${SCRATCH:-/dlabscratch1/gentilin}
export HF_HOME=$SCRATCH/.cache/huggingface
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

PROJECT_DIR=$SCRATCH/code/zip2zip-core
LOG_DIR=$SCRATCH/logs/eval
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# ---------- config ----------
HF_REPO=${HF_REPO:?HF_REPO is required}
PRESET=${PRESET:-default_base}
LIMIT=${LIMIT:-}
TASKS=${TASKS:-}

MODEL_SHORT=$(echo "$HF_REPO" | sed 's|.*/||')
TASKS_SUFFIX=${TASKS:+_${TASKS//,/-}}

LOGFILE="$LOG_DIR/eval_${MODEL_SHORT}_${PRESET}${TASKS_SUFFIX}_${TIMESTAMP}.log"
OUTPUT_JSON="$LOG_DIR/results_${MODEL_SHORT}_${PRESET}${TASKS_SUFFIX}_${TIMESTAMP}.json"

# ---------- venv ----------
VENV_DIR=$SCRATCH/.venvs/lm-eval
if [ ! -f "$VENV_DIR/bin/activate" ]; then
    echo "[eval_ckpt] Creating venv at $VENV_DIR..."
    python -m venv --system-site-packages "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"
pip install --quiet "lm-eval==0.4.9" "huggingface_hub"

# ---------- run ----------
cd "$PROJECT_DIR"

LIMIT_ARG=""
if [ -n "$LIMIT" ]; then
    LIMIT_ARG="--limit $LIMIT"
fi
TASKS_ARG=""
if [ -n "$TASKS" ]; then
    TASKS_ARG="--tasks $TASKS"
fi

{
echo "=== zip2zip checkpoint evaluation ==="
echo "  HF_REPO:     $HF_REPO"
echo "  PRESET:      $PRESET"
echo "  TASKS:       ${TASKS:-<preset default>}"
echo "  LIMIT:       ${LIMIT:-<full>}"
echo "  LOGFILE:     $LOGFILE"
echo "  OUTPUT_JSON: $OUTPUT_JSON"
echo "======================================="

python scripts/eval_harness.py \
    --hf_repo "$HF_REPO" \
    --preset "$PRESET" \
    --no_wandb \
    --resume_wandb_id none \
    --output_path "$OUTPUT_JSON" \
    $LIMIT_ARG \
    $TASKS_ARG

} 2>&1 | tee "$LOGFILE"
