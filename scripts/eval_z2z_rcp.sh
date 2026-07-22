#!/bin/bash
# Evaluate the released zip2zip HF model on the EPFL RCP cluster.
#
# Uses the zip2zip pip package (ext/zip2zip) which can load the HF adapter
# format (adapter_model.safetensors + zip2zip_encoders.safetensors).
#
# ── Usage (Run:AI training jobs — fire and forget) ───────────────────────
#
#   # Smoke test (20 samples):
#   runai submit --name eval-z2z-smoke \
#     --image ghcr.io/jkminder/dlab-runai-images/pytorch:master \
#     --gpu 1 --cpu 8 --memory 64Gi \
#     --pvc dlab-scratch:/mnt --large-shm \
#     --node-pools default \
#     --environment LIMIT=20 \
#     -- bash /dlabscratch1/<your-username>/code/zip2zip-core/scripts/eval_z2z_rcp.sh
#
#   # Full eval:
#   runai submit --name eval-z2z-full \
#     --image ghcr.io/jkminder/dlab-runai-images/pytorch:master \
#     --gpu 1 --cpu 8 --memory 64Gi \
#     --pvc dlab-scratch:/mnt --large-shm \
#     --node-pools default \
#     -- bash /dlabscratch1/<your-username>/code/zip2zip-core/scripts/eval_z2z_rcp.sh
#
# ── Environment variables ────────────────────────────────────────────────
#
#   HF_MODEL=...    HF model repo (default: epfl-dlab/zip2zip-Phi-3.5-mini-instruct-v0.1)
#   PRESET=default  Eval preset from eval_presets.yaml (default: default)
#   LIMIT=20        Per-task sample limit for smoke tests
#   LIMIT=          Full evaluation (default)
#   TASKS=gsm8k     Comma-separated task override (default: preset's task list)
#   Z2Z_SCRATCH=...     Your PVC scratch dir (default: /dlabscratch1/gentilin)
#
set -euo pipefail

Z2Z_SCRATCH=${Z2Z_SCRATCH:-/dlabscratch1/gentilin}
export HF_HOME=$Z2Z_SCRATCH/.cache/huggingface
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

PROJECT_DIR=$Z2Z_SCRATCH/code/zip2zip-core
LOG_DIR=$Z2Z_SCRATCH/logs/eval
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# ---------- venv ----------
VENV_DIR=$Z2Z_SCRATCH/.venvs/lm-eval
if [ ! -f "$VENV_DIR/bin/activate" ]; then
    echo "[eval_z2z] Creating venv at $VENV_DIR..."
    python -m venv --system-site-packages "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"
pip install --quiet "lm-eval==0.4.9" "zip2zip-compression>=0.3.3"

# ---------- install ext/zip2zip if missing ----------
cd "$PROJECT_DIR"
if [ ! -f ext/zip2zip/src/zip2zip/model.py ]; then
    echo "[eval_z2z] Initializing git submodules..."
    git submodule update --init --recursive
fi
python -c "import zip2zip" 2>/dev/null \
    || pip install --quiet -e ext/zip2zip

# ---------- config ----------
HF_MODEL=${HF_MODEL:-epfl-dlab/zip2zip-Phi-3.5-mini-instruct-v0.1}
PRESET=${PRESET:-default}
LIMIT=${LIMIT:-}
TASKS=${TASKS:-}

LIMIT_ARG=""
if [ -n "$LIMIT" ]; then
    LIMIT_ARG="--limit $LIMIT"
fi

TASKS_ARG=""
if [ -n "$TASKS" ]; then
    TASKS_ARG="--tasks $TASKS"
fi

MODEL_SHORT=$(basename "$HF_MODEL")
TASKS_SUFFIX=${TASKS:+_${TASKS//,/-}}
LOGFILE="$LOG_DIR/eval_${MODEL_SHORT}_${PRESET}${TASKS_SUFFIX}_${TIMESTAMP}.log"
OUTPUT_JSON="$LOG_DIR/results_${MODEL_SHORT}_${PRESET}${TASKS_SUFFIX}_${TIMESTAMP}.json"

{
echo "=== zip2zip HF model evaluation ==="
echo "  HF_MODEL:   $HF_MODEL"
echo "  PRESET:     $PRESET"
echo "  LIMIT:      ${LIMIT:-<full>}"
echo "  TASKS:      ${TASKS:-<preset default>}"
echo "  LOGFILE:    $LOGFILE"
echo "  OUTPUT_JSON: $OUTPUT_JSON"
echo "====================================="

python scripts/eval_hf_model.py \
    --model "$HF_MODEL" \
    --preset "$PRESET" \
    --output_path "$OUTPUT_JSON" \
    $LIMIT_ARG $TASKS_ARG

} 2>&1 | tee "$LOGFILE"
