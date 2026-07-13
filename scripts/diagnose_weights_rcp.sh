#!/bin/bash
# Run scripts/diagnose_ckpt_weights.py on the RCP cluster (Run:AI). CPU-only work
# (state-dict comparison), ~10-20 min incl. loading two full state dicts.
#
#   runai submit --name diagnose-phi35-weights \
#     --image ghcr.io/jkminder/dlab-runai-images/pytorch:master \
#     --gpu 1 --cpu 8 --memory 64Gi \
#     --pvc dlab-scratch:/mnt --large-shm --node-pools default \
#     --environment CKPT_DIR='/dlabscratch1/gentilin/zip2zip-outputs/<run>/step_8000' \
#     -- bash /dlabscratch1/gentilin/code/zip2zip-core/scripts/diagnose_weights_rcp.sh
#
# Env vars:
#   CKPT_DIR=...   (required) step_N checkpoint dir
#   HF_MODEL=...   default microsoft/Phi-3.5-mini-instruct
#   RUN_DIR=...    run dir for cross-step frozen check (default: dirname of CKPT_DIR)
#   REPAIR_DIR=... where to write the repaired ckpt on mismatch
#                  (default: ${CKPT_DIR}_repaired)
set -euo pipefail

SCRATCH=${SCRATCH:-/dlabscratch1/gentilin}
export HF_HOME=$SCRATCH/.cache/huggingface
export PYTHONUNBUFFERED=1

PROJECT_DIR=$SCRATCH/code/zip2zip-core
LOG_DIR=$SCRATCH/logs/eval
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

CKPT_DIR=${CKPT_DIR:?Must set CKPT_DIR}
HF_MODEL=${HF_MODEL:-microsoft/Phi-3.5-mini-instruct}
RUN_DIR=${RUN_DIR:-$(dirname "$CKPT_DIR")}
REPAIR_DIR=${REPAIR_DIR:-${CKPT_DIR}_repaired}

CKPT_SHORT=$(echo "$(basename "$(dirname "$CKPT_DIR")")_$(basename "$CKPT_DIR")" | tr -d '()')
LOGFILE="$LOG_DIR/weights_${CKPT_SHORT}_${TIMESTAMP}.log"

VENV_DIR=$SCRATCH/.venvs/lm-eval
if [ ! -f "$VENV_DIR/bin/activate" ]; then
    python -m venv --system-site-packages "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"
pip install --quiet safetensors huggingface_hub

cd "$PROJECT_DIR"

{
echo "=== zip2zip checkpoint weight-integrity check ==="
echo "  CKPT_DIR:   $CKPT_DIR"
echo "  HF_MODEL:   $HF_MODEL"
echo "  RUN_DIR:    $RUN_DIR"
echo "  REPAIR_DIR: $REPAIR_DIR"
echo "  LOGFILE:    $LOGFILE"
echo "=================================================="

python scripts/diagnose_ckpt_weights.py \
    --ckpt_dir "$CKPT_DIR" \
    --hf_model "$HF_MODEL" \
    --compare_steps "$RUN_DIR" \
    --write_repaired "$REPAIR_DIR"

} 2>&1 | tee "$LOGFILE"
