#!/bin/bash
# Evaluate a zip2zip checkpoint on the EPFL RCP cluster using eval_harness.py.
#
# ── Usage (Run:AI training jobs — fire and forget) ───────────────────────
#
#   # Smoke test (20 samples, no W&B):
#   runai submit --name eval-z2z-smoke \
#     --image ghcr.io/jkminder/dlab-runai-images/pytorch:master \
#     --gpu 1 --cpu 8 --memory 64Gi \
#     --pvc dlab-scratch:/mnt --large-shm \
#     --node-pools default \
#     --environment PRESET=smoke \
#     -- bash /dlabscratch1/gentilin/code/zip2zip-core/scripts/eval_z2z_rcp.sh
#
#   # Full eval (W&B enabled, matches paper Table 3):
#   runai submit --name eval-z2z-full \
#     --image ghcr.io/jkminder/dlab-runai-images/pytorch:master \
#     --gpu 1 --cpu 8 --memory 64Gi \
#     --pvc dlab-scratch:/mnt --large-shm \
#     --node-pools default \
#     --environment PRESET=default \
#     -- bash /dlabscratch1/gentilin/code/zip2zip-core/scripts/eval_z2z_rcp.sh
#
# ── Environment variables ────────────────────────────────────────────────
#
#   PRESET=smoke|default     Eval preset (default: smoke)
#   HF_REPO=...              HF model repo (default: epfl-dlab/zip2zip-Phi-3.5-mini-instruct-v0.1)
#   HF_REVISION=...          HF revision/branch (default: none)
#   RESUME_WANDB_ID=...      W&B run ID to resume, or 'none' for new run (default: none)
#   WANDB_PROJECT=...        Override W&B project (default: from project.py)
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
    python -c "import zip2zip_compression" 2>/dev/null \
        || pip install --quiet "zip2zip-compression>=0.3.3"
fi

# ---------- submodules (torchtitan is loaded via sys.path) ----------
cd "$PROJECT_DIR"
if [ ! -f ext/torchtitan/torchtitan/__init__.py ]; then
    echo "[eval_z2z] Initializing git submodules..."
    git submodule update --init --recursive
fi

# ---------- config ----------
HF_REPO=${HF_REPO:-epfl-dlab/zip2zip-Phi-3.5-mini-instruct-v0.1}
HF_REVISION=${HF_REVISION:-}
PRESET=${PRESET:-smoke}
RESUME_WANDB_ID=${RESUME_WANDB_ID:-none}

HF_REV_ARG=""
if [ -n "$HF_REVISION" ]; then
    HF_REV_ARG="--hf_revision $HF_REVISION"
fi

WANDB_PROJECT_ARG=""
if [ -n "${WANDB_PROJECT:-}" ]; then
    WANDB_PROJECT_ARG="--wandb_project $WANDB_PROJECT"
fi

LOGFILE="$LOG_DIR/eval_z2z_${PRESET}_${TIMESTAMP}.log"
OUTPUT_JSON="$LOG_DIR/results_z2z_${PRESET}_${TIMESTAMP}.json"

{
echo "=== zip2zip checkpoint evaluation ==="
echo "  HF_REPO:         $HF_REPO"
echo "  HF_REVISION:     ${HF_REVISION:-<default>}"
echo "  PRESET:          $PRESET"
echo "  RESUME_WANDB_ID: $RESUME_WANDB_ID"
echo "  LOGFILE:         $LOGFILE"
echo "  OUTPUT_JSON:     $OUTPUT_JSON"
echo "======================================="

python scripts/eval_harness.py \
    --preset "$PRESET" \
    --hf_repo "$HF_REPO" \
    $HF_REV_ARG \
    --resume_wandb_id "$RESUME_WANDB_ID" \
    --output_path "$OUTPUT_JSON" \
    $WANDB_PROJECT_ARG

} 2>&1 | tee "$LOGFILE"
