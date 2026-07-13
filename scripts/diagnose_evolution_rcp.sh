#!/bin/bash
# Run scripts/diagnose_ckpt_evolution.py on the RCP cluster (Run:AI), ~30 min.
#
#   runai submit --name diagnose-phi35-evolution \
#     --image ghcr.io/jkminder/dlab-runai-images/pytorch:master \
#     --gpu 1 --cpu 8 --memory 64Gi \
#     --pvc dlab-scratch:/mnt --large-shm --node-pools default \
#     --environment RUN_DIR='/dlabscratch1/gentilin/zip2zip-outputs/<run dir>' \
#     -- bash /dlabscratch1/gentilin/code/zip2zip-core/scripts/diagnose_evolution_rcp.sh
#
# Env vars:
#   RUN_DIR=...   (required) run dir containing step_* checkpoints
#   DATA_DIR=...  training shards (default: $SCRATCH/datasets/phi-1B-sft-8shards)
#   TOKENIZER=... default microsoft/Phi-3.5-mini-instruct
#   REPLAY_STEPS=500,1000,8000
set -euo pipefail

SCRATCH=${SCRATCH:-/dlabscratch1/gentilin}
export HF_HOME=$SCRATCH/.cache/huggingface
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

PROJECT_DIR=$SCRATCH/code/zip2zip-core
LOG_DIR=$SCRATCH/logs/eval
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

RUN_DIR=${RUN_DIR:?Must set RUN_DIR}
DATA_DIR=${DATA_DIR:-$SCRATCH/datasets/phi-1B-sft-8shards}
TOKENIZER=${TOKENIZER:-microsoft/Phi-3.5-mini-instruct}
REPLAY_STEPS=${REPLAY_STEPS:-500,1000,8000}

RUN_SHORT=$(basename "$RUN_DIR" | tr -d '()')
LOGFILE="$LOG_DIR/evolution_${RUN_SHORT}_${TIMESTAMP}.log"

VENV_DIR=$SCRATCH/.venvs/lm-eval
if [ ! -f "$VENV_DIR/bin/activate" ]; then
    python -m venv --system-site-packages "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"
pip install --quiet "lm-eval==0.4.9" "zip2zip-compression>=0.3.3"

cd "$PROJECT_DIR"

{
echo "=== zip2zip checkpoint evolution diagnosis (stale-save test) ==="
echo "  RUN_DIR:      $RUN_DIR"
echo "  DATA_DIR:     $DATA_DIR"
echo "  REPLAY_STEPS: $REPLAY_STEPS"
echo "  LOGFILE:      $LOGFILE"
echo "================================================================"

python scripts/diagnose_ckpt_evolution.py \
    --run_dir "$RUN_DIR" \
    --data_dir "$DATA_DIR" \
    --tokenizer "$TOKENIZER" \
    --replay_steps "$REPLAY_STEPS"

} 2>&1 | tee "$LOGFILE"
