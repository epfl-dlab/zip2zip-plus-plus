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
#     -- bash /dlabscratch1/gentilin/code/zip2zip-core/scripts/eval_z2z_rcp.sh
#
#   # Full eval:
#   runai submit --name eval-z2z-full \
#     --image ghcr.io/jkminder/dlab-runai-images/pytorch:master \
#     --gpu 1 --cpu 8 --memory 64Gi \
#     --pvc dlab-scratch:/mnt --large-shm \
#     --node-pools default \
#     -- bash /dlabscratch1/gentilin/code/zip2zip-core/scripts/eval_z2z_rcp.sh
#
# ── Environment variables ────────────────────────────────────────────────
#
#   HF_MODEL=...    HF model repo (default: epfl-dlab/zip2zip-Phi-3.5-mini-instruct-v0.1)
#   LIMIT=20        Per-task sample limit for smoke tests
#   LIMIT=          Full evaluation (default)
#
set -euo pipefail

SCRATCH=/dlabscratch1/gentilin
export HF_HOME=$SCRATCH/.cache/huggingface
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

PROJECT_DIR=$SCRATCH/code/zip2zip-core
LOG_DIR=$SCRATCH/logs/eval
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# ---------- venv ----------
VENV_DIR=$SCRATCH/.venvs/lm-eval
if [ ! -f "$VENV_DIR/bin/activate" ]; then
    echo "[eval_z2z] Creating venv at $VENV_DIR..."
    python -m venv --system-site-packages "$VENV_DIR"
    source "$VENV_DIR/bin/activate"
    pip install --quiet lm-eval "zip2zip-compression>=0.3.3"
else
    source "$VENV_DIR/bin/activate"
fi

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
LIMIT=${LIMIT:-}

LIMIT_ARG=""
if [ -n "$LIMIT" ]; then
    LIMIT_ARG="--limit $LIMIT"
fi

LOGFILE="$LOG_DIR/eval_z2z_${TIMESTAMP}.log"
OUTPUT_JSON="$LOG_DIR/results_z2z_${TIMESTAMP}.json"

{
echo "=== zip2zip HF model evaluation ==="
echo "  HF_MODEL:   $HF_MODEL"
echo "  LIMIT:      ${LIMIT:-<full>}"
echo "  LOGFILE:    $LOGFILE"
echo "  OUTPUT_JSON: $OUTPUT_JSON"
echo "====================================="

python scripts/eval_hf_model.py \
    --model "$HF_MODEL" \
    --tasks "arc_challenge,arc_easy,hellaswag,openbookqa,piqa,winogrande,gsm8k" \
    --num_fewshot 2 \
    --output_path "$OUTPUT_JSON" \
    $LIMIT_ARG

} 2>&1 | tee "$LOGFILE"
