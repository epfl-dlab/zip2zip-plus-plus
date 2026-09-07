#!/bin/bash
# Validate the eval benchmark configuration by running standard lm-eval on
# microsoft/Phi-3.5-mini-instruct and comparing against the paper's Table 3
# (2506.01084v2, "Base" row for Phi-3.5-4B).
#
# This does NOT use eval_harness.py or the zip2zip adapter — it runs stock
# lm-eval on a standard HF model to confirm the task/fewshot settings are
# correct before trusting zip2zip checkpoint evals.
#
# ── On the EPFL RCP cluster (Run:AI) ─────────────────────────────────────
#
#   # Smoke test (20 samples per task, ~5 min):
#   runai submit \
#     --name eval-phi35-smoke \
#     --image ghcr.io/jkminder/dlab-runai-images/pytorch:master \
#     --gpu 1 --cpu 8 --memory 64Gi \
#     --pvc dlab-scratch:/mnt --large-shm \
#     --node-pools default \
#     --environment LIMIT=20 \
#     -- bash /dlabscratch1/<your-username>/code/zip2zip-core/scripts/validate_phi35_rcp.sh
#
#   # Full eval (no LIMIT, ~1-2 hours on A100):
#   runai submit \
#     --name eval-phi35-full \
#     --image ghcr.io/jkminder/dlab-runai-images/pytorch:master \
#     --gpu 1 --cpu 8 --memory 64Gi \
#     --pvc dlab-scratch:/mnt --large-shm \
#     --node-pools default \
#     -- bash /dlabscratch1/<your-username>/code/zip2zip-core/scripts/validate_phi35_rcp.sh
#
# ── Environment variables (override via --environment or export) ─────────
#
#   LIMIT=20        Run only 20 samples per task (quick smoke test)
#   LIMIT=          Full evaluation (default)
#   HF_HOME=...     HuggingFace cache dir (default: /dlabscratch1/gentilin/.cache/huggingface)
#   Z2Z_SCRATCH=...     Your PVC scratch dir (default: /dlabscratch1/gentilin)
#
# ── Expected results (paper Table 3, Phi-3.5-4B Base, 2-shot) ───────────
#
#   ARC-c   ARC-e   HS      OBQA    PIQA    WG      GSM8K
#   0.60    0.83    0.66    0.46    0.79    0.75    0.82
#
set -euo pipefail

# ---------- paths ----------
Z2Z_SCRATCH=${Z2Z_SCRATCH:-/dlabscratch1/gentilin}
export HF_HOME=${HF_HOME:-$Z2Z_SCRATCH/.cache/huggingface}
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
PRESET=${PRESET:-default}
LIMIT=${LIMIT:-}
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

LOG_DIR=$Z2Z_SCRATCH/logs/eval
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOGFILE="$LOG_DIR/validate_phi35_${PRESET}_${TIMESTAMP}.log"

# ---------- persistent venv on scratch (survives across jobs) ----------
VENV_DIR=$Z2Z_SCRATCH/.venvs/lm-eval
if [ ! -f "$VENV_DIR/bin/activate" ]; then
    echo "[validate] Creating venv at $VENV_DIR (first run only)..."
    python -m venv --system-site-packages "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"
pip install --quiet "lm-eval==0.4.9"

# ---------- config from eval_presets.yaml (single source of truth) ----------
# MODEL=<hf id> evaluates any off-the-shelf HF model with the same preset
# (paper "Original" rows); the default keeps the historical Phi validation.
MODEL=${MODEL:-microsoft/Phi-3.5-mini-instruct}
# SentencePiece tokenizers (Phi) need the slow tokenizer; byte-level BPE
# families (Llama-3) ship no slow tokenizer, so only Phi gets the flag.
case "$MODEL" in
    microsoft/Phi-*) TOKENIZER_ARGS=${TOKENIZER_ARGS:-,use_fast_tokenizer=false} ;;
    *)               TOKENIZER_ARGS=${TOKENIZER_ARGS:-} ;;
esac
MODEL_STEM=phi35
if [ "$MODEL" != "microsoft/Phi-3.5-mini-instruct" ]; then
    MODEL_STEM=$(basename "$MODEL")
fi

eval "$(python3 "$SCRIPT_DIR/load_preset.py" "$PRESET")"

# The expected-results block above has paper Table 3 numbers only for the
# original 7 tasks; triviaqa (added to the presets in 2026-08) has no reference
# row, so keep the validation scope fixed.
TASKS=${TASKS//,triviaqa/}
TASKS=${TASKS//triviaqa,/}

LIMIT_ARG=""
if [ -n "$LIMIT" ]; then
    LIMIT_ARG="--limit $LIMIT"
fi

# ---------- lm-eval results output ----------
RESULTS_DIR="$LOG_DIR/results_${MODEL_STEM}_${PRESET}_${TIMESTAMP}"
mkdir -p "$RESULTS_DIR"

{
echo "=== stock lm-eval benchmark run ($MODEL_STEM) ==="
echo "  MODEL:       $MODEL"
echo "  TASKS:       $TASKS"
echo "  NUM_FEWSHOT: $NUM_FEWSHOT"
echo "  LIMIT:       ${LIMIT:-<full>}"
echo "  HF_HOME:     $HF_HOME"
echo "  LOGFILE:     $LOGFILE"
echo "  RESULTS_DIR: $RESULTS_DIR"
echo "  VENV:        $VENV_DIR"
echo "==================================================="

python -m lm_eval \
    --model hf \
    --model_args "pretrained=$MODEL,max_length=$MAX_LENGTH${TOKENIZER_ARGS}" \
    --tasks "$TASKS" \
    ${NUM_FEWSHOT:+--num_fewshot "$NUM_FEWSHOT"} \
    --device cuda \
    --batch_size "$BATCH_SIZE" \
    --output_path "$RESULTS_DIR" \
    --include_path "$SCRIPT_DIR/lm_eval_tasks" \
    ${CHAT_TEMPLATE_FLAG:-} \
    ${MULTITURN_FLAG:-} \
    $LIMIT_ARG

echo ""
echo "=== Expected (paper Table 3, Phi-3.5-4B Base, 2-shot) ==="
echo "  ARC-c  ARC-e  HS    OBQA  PIQA  WG    GSM8K"
echo "  0.60   0.83   0.66  0.46  0.79  0.75  0.82"
echo "========================================================="
echo ""
echo "Results JSON saved to: $RESULTS_DIR"
echo "Full log saved to:     $LOGFILE"
} 2>&1 | tee "$LOGFILE"
