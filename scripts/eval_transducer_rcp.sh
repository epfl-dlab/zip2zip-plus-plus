#!/bin/bash
# Sequence-level evaluation of a transducer sweep point on the RCP cluster:
# scores every checkpoint of one MODEL_CONFIG, both directions, on held-out data
# and logs the curve back into each run's own W&B run. See scripts/eval_transducer.py.
#
# Single GPU, single process -- this is inference over a few dozen samples, not
# a training job.
#
# ── Usage (Run:AI) ──────────────────────────────────────────────────────
#
#   runai submit --name eval-phi-1b \
#     --image ghcr.io/jkminder/dlab-runai-images/pytorch:master \
#     --gpu 1 --cpu 16 --memory 128Gi \
#     --pvc dlab-scratch:/mnt --large-shm --node-pools h100 \
#     --environment MODEL_CONFIG=Phi-1B \
#     --environment WANDB_API_KEY=<your-key> \
#     -- "bash /dlabscratch1/xinma/code/zip2zip-core/scripts/eval_transducer_rcp.sh"
#
#   Add the relaxed (base-space) metric and the implementation self-check:
#     --environment RELAXED=1 --environment VERIFY_GENERATION=8
#   RELAXED needs real autoregressive generation and the model has no KV cache,
#   so every generated token replays the whole prefix. Run strict first, then
#   turn it on for the checkpoints that showed a non-zero strict rate.
#
set -euo pipefail

Z2Z_SCRATCH=${Z2Z_SCRATCH:-/dlabscratch1/xinma}
export HF_HOME=${HF_HOME:-$Z2Z_SCRATCH/.cache/huggingface}
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

PROJECT_DIR=${PROJECT_DIR:-$Z2Z_SCRATCH/code/zip2zip-core}
OUTPUT_BASE=${OUTPUT_BASE:-$Z2Z_SCRATCH/zip2zip-outputs}
LOG_DIR=$Z2Z_SCRATCH/logs/eval
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

cd "$PROJECT_DIR"
PYTHON=$PROJECT_DIR/.venv/bin/python
[ -x "$PYTHON" ] || { echo "FATAL: no venv at $PYTHON (run uv sync first)"; exit 1; }

MODEL_CONFIG=${MODEL_CONFIG:-Phi-1B}
DIRECTIONS=${DIRECTIONS:-"compress decompress"}
MAX_SUBTOKENS=${MAX_SUBTOKENS:-4}
N_SAMPLES=${N_SAMPLES:-64}
BATCH_SIZE=${BATCH_SIZE:-8}
EVAL_OFFSET=${EVAL_OFFSET:-1000000000}
WANDB=${WANDB:-1}

EXTRA=""
# STEPS=3000,6000 scores a subset of the checkpoints. Worth reaching for with
# RELAXED=1: that path decodes every strict-FAILING sample, so it is slowest
# exactly where strict is 0 -- which is also where it carries the most
# information (a valid compression that is not canonical LZW).
if [ -n "${STEPS:-}" ]; then
    EXTRA="$EXTRA --steps $STEPS"
fi
if [ -n "${RELAXED:-}" ] && [ "${RELAXED}" != "0" ]; then
    EXTRA="$EXTRA --relaxed"
fi
if [ -n "${VERIFY_GENERATION:-}" ] && [ "${VERIFY_GENERATION}" != "0" ]; then
    EXTRA="$EXTRA --verify_generation ${VERIFY_GENERATION}"
fi
if [ -n "$WANDB" ] && [ "$WANDB" != "0" ]; then
    EXTRA="$EXTRA --wandb"
    if [ -z "${WANDB_API_KEY:-}" ]; then
        echo "FATAL: --wandb requested but WANDB_API_KEY is unset; the job would"
        echo "       die in wandb.init() exactly like the first sweep attempt."
        exit 1
    fi
fi

LOGFILE="$LOG_DIR/eval_${MODEL_CONFIG}_${TIMESTAMP}.log"

{
echo "=== transducer sequence-level eval: $MODEL_CONFIG ==="
echo "directions=$DIRECTIONS  n_samples=$N_SAMPLES  batch=$BATCH_SIZE  extra='$EXTRA'"

for DIRECTION in $DIRECTIONS; do
    # Run dirs are zip2zip_<config>_<direction>_ms<N>_scratch, sometimes with a
    # suffix (a re-run appends _r1). Take the newest match rather than guessing.
    PATTERN="${OUTPUT_BASE}/zip2zip_${MODEL_CONFIG}_${DIRECTION}_ms${MAX_SUBTOKENS}_scratch*"
    RUN_DIR=$(ls -dt $PATTERN 2>/dev/null | head -1 || true)
    if [ -z "$RUN_DIR" ]; then
        echo "SKIP $DIRECTION: no run dir matching $PATTERN"
        continue
    fi
    echo "--- $DIRECTION -> $RUN_DIR"
    "$PYTHON" scripts/eval_transducer.py \
        --run_dir "$RUN_DIR" \
        --n_samples "$N_SAMPLES" \
        --batch_size "$BATCH_SIZE" \
        --eval_offset "$EVAL_OFFSET" \
        --device cuda \
        $EXTRA
done

echo "=== done: $MODEL_CONFIG ==="
} 2>&1 | tee "$LOGFILE"
