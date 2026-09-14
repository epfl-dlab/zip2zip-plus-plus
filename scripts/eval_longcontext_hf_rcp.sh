#!/bin/bash
# Run the untouched Hugging Face Phi-3.5 model on one RULER task/length.
# This wrapper keeps Run:AI's command line simple and mirrors the checkpoint
# launcher's shared lm-eval environment and cache locations.

set -euo pipefail

Z2Z_SCRATCH=${Z2Z_SCRATCH:-/dlabscratch1/xinma}
PROJECT_DIR=${PROJECT_DIR:-$Z2Z_SCRATCH/zip2zip-core}
MODEL=${MODEL:-microsoft/Phi-3.5-mini-instruct}
TOKENIZER=${TOKENIZER:-$MODEL}
PRESET=${PRESET:-longcontext_ruler}
TASKS=${TASKS:-ruler}
RULER_LENGTHS=${RULER_LENGTHS:-4096}
LIMIT=${LIMIT:-}

export HF_HOME=${HF_HOME:-$Z2Z_SCRATCH/.cache/huggingface}
export NLTK_DATA=${NLTK_DATA:-$Z2Z_SCRATCH/.cache/nltk}
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1

mkdir -p "$NLTK_DATA" "$Z2Z_SCRATCH/logs/eval"

VENV_DIR=$Z2Z_SCRATCH/.venvs/lm-eval
if [ ! -x "$VENV_DIR/bin/python" ]; then
    echo "Missing shared lm-eval environment: $VENV_DIR/bin/python" >&2
    echo "Run one checkpoint eval first so scripts/eval_ckpt_rcp.sh creates it." >&2
    exit 1
fi

TASK_TAG=$(printf '%s' "$TASKS" | tr -cs '[:alnum:]._-' '_')
INSTANCE_RAW=${RUNAI_JOB_NAME:-local}-${HOSTNAME:-host}-$$
INSTANCE_TAG=$(printf '%s' "$INSTANCE_RAW" | tr -cs '[:alnum:]._-' '_')
TIMESTAMP=${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}
STEM="phi35-original_ruler-${RULER_LENGTHS}_${TASK_TAG}_limit${LIMIT:-full}_${TIMESTAMP}_${INSTANCE_TAG}"
LOGFILE="$Z2Z_SCRATCH/logs/eval/eval_${STEM}.log"
OUTPUT_JSON="$Z2Z_SCRATCH/logs/eval/results_${STEM}.json"

limit_args=()
if [ -n "$LIMIT" ]; then
    limit_args=(--limit "$LIMIT")
fi

cd "$PROJECT_DIR"

{
echo "=== untouched Phi-3.5 RULER evaluation ==="
echo "  MODEL:        $MODEL"
echo "  TOKENIZER:    $TOKENIZER"
echo "  TASKS:        $TASKS"
echo "  RULER_LENGTH: $RULER_LENGTHS"
echo "  LIMIT:        ${LIMIT:-<full>}"
echo "  LOGFILE:      $LOGFILE"
echo "  OUTPUT_JSON:  $OUTPUT_JSON"
echo "==========================================="

"$VENV_DIR/bin/python" scripts/eval_longcontext_hf.py \
    --preset "$PRESET" \
    --model "$MODEL" \
    --tokenizer "$TOKENIZER" \
    --tasks "$TASKS" \
    --ruler_lengths "$RULER_LENGTHS" \
    --output_path "$OUTPUT_JSON" \
    "${limit_args[@]}"
} 2>&1 | tee "$LOGFILE"
