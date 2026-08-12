#!/bin/bash
# Run scripts/diagnose_ckpt_replay.py on the RCP cluster (Run:AI).
#
# Replays training conditions through the eval loading+forward path on actual
# training shards, to separate "checkpoint damaged" from "eval path/regime broken".
# Fast: a handful of forward passes, ~10-15 min wall-clock including model load.
#
#   runai submit --name diagnose-phi35-repro \
#     --image ghcr.io/jkminder/dlab-runai-images/pytorch:master \
#     --gpu 1 --cpu 8 --memory 64Gi \
#     --pvc dlab-scratch:/mnt --large-shm --node-pools default \
#     --environment CKPT_DIR='/dlabscratch1/gentilin/zip2zip-outputs/<run>/step_8000' \
#     -- bash /dlabscratch1/gentilin/code/zip2zip-core/scripts/diagnose_ckpt_rcp.sh
#
# Env vars:
#   CKPT_DIR=...    (required) step_N checkpoint dir
#   DATA_DIR=...    training shards (default: .../phi-1B-sft-8shards-eosfix)
#   TOKENIZER=...   default microsoft/Phi-3.5-mini-instruct
#   N_CHUNKS=8      number of 4096-base-token chunks to score
#   START_FRAC=0.5  where in shard 0 to sample
#   OUTPUT_JSON=... optional dual-denominator aggregate artifact
#   REPLAY_ONLY=1   stop after the train-style replay/denominator diagnostic
set -euo pipefail

Z2Z_SCRATCH=${Z2Z_SCRATCH:-/dlabscratch1/gentilin}
export HF_HOME=$Z2Z_SCRATCH/.cache/huggingface
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

PROJECT_DIR=$Z2Z_SCRATCH/code/zip2zip-core
LOG_DIR=$Z2Z_SCRATCH/logs/eval
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

CKPT_DIR=${CKPT_DIR:?Must set CKPT_DIR}
DATA_DIR=${DATA_DIR:-$Z2Z_SCRATCH/datasets/phi-1B-sft-8shards-eosfix}
TOKENIZER=${TOKENIZER:-microsoft/Phi-3.5-mini-instruct}
N_CHUNKS=${N_CHUNKS:-8}
START_FRAC=${START_FRAC:-0.5}
OUTPUT_JSON=${OUTPUT_JSON:-}
REPLAY_ONLY=${REPLAY_ONLY:-0}

CKPT_SHORT=$(echo "$(basename "$(dirname "$CKPT_DIR")")_$(basename "$CKPT_DIR")" | tr -d '()')
LOGFILE="$LOG_DIR/diagnose_${CKPT_SHORT}_${TIMESTAMP}.log"

# Same venv as eval_ckpt_rcp.sh (torch from system site-packages + lm-eval).
VENV_DIR=$Z2Z_SCRATCH/.venvs/lm-eval
if [ ! -f "$VENV_DIR/bin/activate" ]; then
    python -m venv --system-site-packages "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"
pip install --quiet "lm-eval==0.4.9" "zip2zip-compression>=0.3.3"

cd "$PROJECT_DIR"

{
echo "=== zip2zip checkpoint diagnosis (training-replay) ==="
echo "  CKPT_DIR:   $CKPT_DIR"
echo "  DATA_DIR:   $DATA_DIR"
echo "  TOKENIZER:  $TOKENIZER"
echo "  N_CHUNKS:   $N_CHUNKS  START_FRAC: $START_FRAC"
echo "  OUTPUT_JSON: ${OUTPUT_JSON:-<none>}"
echo "  REPLAY_ONLY: $REPLAY_ONLY"
echo "  LOGFILE:    $LOGFILE"
echo "======================================================="

DIAG_ARGS=(
    scripts/diagnose_ckpt_replay.py
    --ckpt_dir "$CKPT_DIR"
    --data_dir "$DATA_DIR"
    --tokenizer "$TOKENIZER"
    --n_chunks "$N_CHUNKS"
    --start_frac "$START_FRAC"
)
if [ -n "$OUTPUT_JSON" ]; then
    DIAG_ARGS+=(--output_json "$OUTPUT_JSON")
fi
if [ "$REPLAY_ONLY" != "0" ]; then
    DIAG_ARGS+=(--replay_only)
fi
python "${DIAG_ARGS[@]}"

} 2>&1 | tee "$LOGFILE"
