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
#   # Local checkpoint not pushed to HF (use CKPT_DIR instead of HF_REPO),
#   # e.g. a Phi-3.5 finetune — TOKENIZER must match the training tokenizer,
#   # the eval_harness.py default (Llama-3-8B) is wrong for non-Llama runs:
#   runai submit --name eval-phi35-repro-mc \
#     --image ghcr.io/jkminder/dlab-runai-images/pytorch:master \
#     --gpu 1 --cpu 8 --memory 64Gi \
#     --pvc dlab-scratch:/mnt --large-shm --node-pools default \
#     --environment CKPT_DIR=/dlabscratch1/<your-username>/zip2zip-outputs/<run>/step_8000 \
#     --environment TOKENIZER=microsoft/Phi-3.5-mini-instruct \
#     --environment PRESET=default \
#     -- bash /dlabscratch1/<your-username>/code/zip2zip-core/scripts/eval_ckpt_rcp.sh
#
# ── Environment variables ───────────────────────────────────────────────
#
#   HF_REPO=...     HF repo with model.pt + meta.pt (mutually exclusive w/ CKPT_DIR)
#   CKPT_DIR=...    Local step_N checkpoint dir (mutually exclusive w/ HF_REPO)
#   TOKENIZER=...   Must match the tokenizer the checkpoint was trained with
#                   (default: meta-llama/Meta-Llama-3-8B — the Llaza default)
#   PRESET=...      Eval preset (default: default_base)
#   LIMIT=20        Per-task sample limit for smoke tests
#   TASKS=gsm8k     Comma-separated task override (default: preset's tasks)
#   SCRATCH=...     Your PVC scratch dir (default: /dlabscratch1/gentilin)
#   WANDB=1         Log results/samples/compression to W&B (default: off, JSON+log
#                   only). Requires WANDB_API_KEY in the job env. Entity is
#                   hardcoded to epfl-dlab in src/zip2zip_core/project.py.
#   WANDB_NAME=...  W&B run name; eval_harness.py prepends "eval-" automatically
#                   (default: auto from checkpoint/repo name)
#   WANDB_PROJECT=. W&B project (default: project.py's WANDB_PROJECT, i.e. llaza)
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
HF_REPO=${HF_REPO:-}
CKPT_DIR=${CKPT_DIR:-}
if [ -z "$HF_REPO" ] && [ -z "$CKPT_DIR" ]; then
    echo "Must set HF_REPO (HF hub repo) or CKPT_DIR (local checkpoint dir)." >&2
    exit 1
fi
if [ -n "$HF_REPO" ] && [ -n "$CKPT_DIR" ]; then
    echo "Set only one of HF_REPO or CKPT_DIR, not both." >&2
    exit 1
fi
TOKENIZER=${TOKENIZER:-meta-llama/Meta-Llama-3-8B}
PRESET=${PRESET:-default_base}
LIMIT=${LIMIT:-}
TASKS=${TASKS:-}
WANDB=${WANDB:-0}
WANDB_NAME=${WANDB_NAME:-}
WANDB_PROJECT=${WANDB_PROJECT:-}

if [ -n "$CKPT_DIR" ]; then
    MODEL_SHORT=$(echo "$(basename "$(dirname "$CKPT_DIR")")_$(basename "$CKPT_DIR")" | tr -d '()')
else
    MODEL_SHORT=$(echo "$HF_REPO" | sed 's|.*/||')
fi
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
# W&B: off by default (--no_wandb). WANDB=1 logs results + per-sample tables +
# compression ratios; run names get an "eval-" prefix inside eval_harness.py.
WANDB_ARGS="--no_wandb"
if [ "$WANDB" != "0" ]; then
    WANDB_ARGS=""
    if [ -n "$WANDB_NAME" ]; then
        WANDB_ARGS="--wandb_name $WANDB_NAME"
    fi
    if [ -n "$WANDB_PROJECT" ]; then
        WANDB_ARGS="$WANDB_ARGS --wandb_project $WANDB_PROJECT"
    fi
fi

{
echo "=== zip2zip checkpoint evaluation ==="
echo "  HF_REPO:     ${HF_REPO:-<none>}"
echo "  CKPT_DIR:    ${CKPT_DIR:-<none>}"
echo "  TOKENIZER:   $TOKENIZER"
echo "  PRESET:      $PRESET"
echo "  TASKS:       ${TASKS:-<preset default>}"
echo "  LIMIT:       ${LIMIT:-<full>}"
echo "  WANDB:       $WANDB${WANDB_NAME:+ (name: eval-$WANDB_NAME)}${WANDB_PROJECT:+ (project: $WANDB_PROJECT)}"
echo "  LOGFILE:     $LOGFILE"
echo "  OUTPUT_JSON: $OUTPUT_JSON"
echo "======================================="

if [ -n "$CKPT_DIR" ]; then
    python scripts/eval_harness.py \
        --ckpt_dir "$CKPT_DIR" \
        --tokenizer "$TOKENIZER" \
        --preset "$PRESET" \
        --resume_wandb_id none \
        --output_path "$OUTPUT_JSON" \
        $WANDB_ARGS \
        $LIMIT_ARG \
        $TASKS_ARG
else
    python scripts/eval_harness.py \
        --hf_repo "$HF_REPO" \
        --tokenizer "$TOKENIZER" \
        --preset "$PRESET" \
        --resume_wandb_id none \
        --output_path "$OUTPUT_JSON" \
        $WANDB_ARGS \
        $LIMIT_ARG \
        $TASKS_ARG
fi

} 2>&1 | tee "$LOGFILE"
