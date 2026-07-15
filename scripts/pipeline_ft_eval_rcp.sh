#!/bin/bash
# Finetune -> eval pipeline for the RCP cluster (Run:AI). ONE sequential job,
# ONE W&B run containing: training curves, smoke-eval curves per checkpoint
# (own x-axis 'smoke/step'), the full final eval incl. all per-sample tables
# (GSM8K generations visible in W&B), and notes pointing at every RCP artifact.
#
#   runai submit --name ft-<run-name> \
#     --image ghcr.io/jkminder/dlab-runai-images/pytorch:master \
#     --gpu 4 --cpu 16 --memory 128Gi \
#     --pvc dlab-scratch:/mnt --large-shm --node-pools default \
#     --environment RUN_NAME=andrea-z2z-phi35-4B-eosfix-v0.2 \
#     --environment DATA_DIR=/dlabscratch1/gentilin/datasets/phi-1B-sft-8shards-eosfix \
#     --environment WANDB_API_KEY=$WANDB_API_KEY \
#     -- bash /dlabscratch1/gentilin/code/zip2zip-core/scripts/pipeline_ft_eval_rcp.sh
#
# Phases (all logged to one pipeline log, job aborts on the first failure):
#   0. preflight   hard-fail on: existing output dir (no silent "(1)" suffixes),
#                  missing data/masks, missing WANDB_API_KEY
#   1. train       scripts/finetune_phi35_rcp.sh (its own env/venv), 4 GPUs,
#                  W&B run created under a pre-chosen id = one run for everything
#   2. smoke       every SMOKE_EVERY-th checkpoint: arc_easy,hellaswag,winogrande,
#                  gsm8k_boxed @ LIMIT=SMOKE_LIMIT -> logged into the same W&B run
#                  at x = smoke/step (1 GPU)
#   3. final eval  last checkpoint: full 'default' preset (with all sample tables
#                  -> W&B) + wikitext perplexity (logged under final/)
#   4. notes       append final-ckpt path, RCP log paths, and a copy-paste-ready
#                  runai command for the remaining perplexity corpora that logs
#                  into this same W&B run (eval_ckpt_rcp.sh RESUME_WANDB_ID)
#
# Env vars:
#   RUN_NAME=...      (required) also the W&B run name — make it exhaustive
#   DATA_DIR=...      (default: $SCRATCH/datasets/phi-1B-sft-8shards-eosfix)
#   STEPS=8000        total steps (forwarded to the finetune launcher)
#   SMOKE_EVERY=1000  smoke-eval cadence in steps
#   SMOKE_LIMIT=200   per-task sample limit for smoke evals
#   SMOKE_TASKS=arc_easy,hellaswag,winogrande,gsm8k_boxed
#   TOKENIZER=microsoft/Phi-3.5-mini-instruct
#   WANDB_PROJECT=zip2zip-core
#   SKIP_FULL_PPL=0   set 1 to also skip the wikitext perplexity in phase 3
#   FINAL_LIMIT=      per-task sample limit for the FINAL eval (default: full).
#                     Only for pipeline rehearsals — never for real numbers.
# Anything else the finetune launcher reads (LR, SEQ_LEN, ...) passes through.
#
# REHEARSE THE PIPELINE FIRST (~30 min end-to-end, exercises every phase):
#   ... --environment RUN_NAME=pipeline-rehearsal-1 \
#       --environment STEPS=20 --environment SAVE_FREQ=10 --environment LOG_FREQ=1 \
#       --environment SMOKE_EVERY=10 --environment SMOKE_LIMIT=8 \
#       --environment FINAL_LIMIT=8 ...
set -euo pipefail

SCRATCH=${SCRATCH:-/dlabscratch1/gentilin}
export HF_HOME=${HF_HOME:-$SCRATCH/.cache/huggingface}
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

PROJECT_DIR=$SCRATCH/code/zip2zip-core
RUN_NAME=${RUN_NAME:?Must set RUN_NAME (also becomes the W&B run name)}
export DATA_DIR=${DATA_DIR:-$SCRATCH/datasets/phi-1B-sft-8shards-eosfix}
export STEPS=${STEPS:-8000}
SMOKE_EVERY=${SMOKE_EVERY:-1000}
SMOKE_LIMIT=${SMOKE_LIMIT:-200}
SMOKE_TASKS=${SMOKE_TASKS:-arc_easy,hellaswag,winogrande,gsm8k_boxed}
TOKENIZER=${TOKENIZER:-microsoft/Phi-3.5-mini-instruct}
export WANDB_PROJECT=${WANDB_PROJECT:-zip2zip-core}
SKIP_FULL_PPL=${SKIP_FULL_PPL:-0}
FINAL_LIMIT=${FINAL_LIMIT:-}

OUTPUT_BASE=${OUTPUT_BASE:-$SCRATCH/zip2zip-outputs}
OUTPUT_DIR=$OUTPUT_BASE/$RUN_NAME
EVAL_LOG_DIR=$SCRATCH/logs/eval
PIPE_LOG_DIR=$SCRATCH/logs/pipeline
mkdir -p "$EVAL_LOG_DIR" "$PIPE_LOG_DIR"
TS=$(date +%Y%m%d_%H%M%S)
PIPELINE_LOG="$PIPE_LOG_DIR/pipeline_${RUN_NAME}_${TS}.log"

# One pre-chosen W&B id ties train + evals into a single run.
export WANDB_ID=$(python - <<'PY'
import secrets, string
print("".join(secrets.choice(string.ascii_lowercase + string.digits) for _ in range(8)))
PY
)
export WANDB=1
export RUN_NAME

{
echo "=== zip2zip finetune->eval pipeline ==="
echo "  RUN_NAME:    $RUN_NAME"
echo "  OUTPUT_DIR:  $OUTPUT_DIR"
echo "  DATA_DIR:    $DATA_DIR"
echo "  STEPS:       $STEPS  SMOKE_EVERY: $SMOKE_EVERY  SMOKE_LIMIT: $SMOKE_LIMIT"
echo "  SMOKE_TASKS: $SMOKE_TASKS"
echo "  W&B:         project=$WANDB_PROJECT run_id=$WANDB_ID name=$RUN_NAME"
echo "  PIPELINE_LOG: $PIPELINE_LOG"
echo "  git commit:  $(git -C "$PROJECT_DIR" rev-parse --short HEAD)"
echo "========================================"

# ---------- phase 0: preflight (fail fast, before any GPU work) ----------
if [ -e "$OUTPUT_DIR" ]; then
    echo "FATAL: output dir already exists: $OUTPUT_DIR"
    echo "Pick a fresh RUN_NAME — this pipeline never reuses or suffixes dirs."
    exit 1
fi
if [ ! -d "$DATA_DIR" ] || ! ls "$DATA_DIR"/shard_*.npy >/dev/null 2>&1; then
    echo "FATAL: DATA_DIR missing or has no shard_*.npy: $DATA_DIR"
    exit 1
fi
if ! ls "$DATA_DIR"/mask_*.npy >/dev/null 2>&1; then
    echo "FATAL: DATA_DIR has no mask_*.npy (EOS-in-loss datasets always have masks): $DATA_DIR"
    exit 1
fi
if [ -z "${WANDB_API_KEY:-}" ]; then
    echo "FATAL: WANDB_API_KEY not set — the pipeline's whole point is one logged W&B run."
    exit 1
fi
if [ ! -f "$PROJECT_DIR/scripts/lm_eval_tasks/gsm8k_boxed.yaml" ]; then
    echo "FATAL: gsm8k_boxed.yaml missing — pull the latest finetuning-andrea."
    exit 1
fi

# ---------- phase 1: train (the launcher owns its env/venv/flags) ----------
echo "=== phase 1/4: training ==="
bash "$PROJECT_DIR/scripts/finetune_phi35_rcp.sh"

FINAL_CKPT="$OUTPUT_DIR/step_$STEPS"
if [ ! -f "$FINAL_CKPT/model.pt" ]; then
    echo "FATAL: training finished but $FINAL_CKPT/model.pt is missing."
    echo "If a '(1)'-suffixed sibling of $OUTPUT_DIR exists, something pre-created"
    echo "the output dir and train.py's collision logic moved the run there."
    exit 1
fi

# ---------- eval env (same venv as eval_ckpt_rcp.sh) ----------
VENV_DIR=$SCRATCH/.venvs/lm-eval
if [ ! -f "$VENV_DIR/bin/activate" ]; then
    python -m venv --system-site-packages "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"
pip install --quiet "lm-eval==0.4.9" "zip2zip-compression>=0.3.3" wandb

cd "$PROJECT_DIR"

# ---------- phase 2: smoke evals on intermediate checkpoints ----------
echo "=== phase 2/4: smoke evals (every $SMOKE_EVERY steps, limit $SMOKE_LIMIT) ==="
step=$SMOKE_EVERY
while [ "$step" -le "$STEPS" ]; do
    CKPT="$OUTPUT_DIR/step_$step"
    if [ -f "$CKPT/model.pt" ]; then
        echo "--- smoke eval: step_$step ---"
        SMOKE_JSON="$EVAL_LOG_DIR/results_${RUN_NAME}_step${step}_smoke_${TS}.json"
        python scripts/eval_harness.py \
            --ckpt_dir "$CKPT" \
            --tokenizer "$TOKENIZER" \
            --preset default \
            --tasks "$SMOKE_TASKS" \
            --limit "$SMOKE_LIMIT" \
            --no_wandb --resume_wandb_id none \
            --output_path "$SMOKE_JSON"
        python scripts/log_results_to_wandb.py \
            --json "$SMOKE_JSON" \
            --resume_id "$WANDB_ID" --project "$WANDB_PROJECT" \
            --prefix smoke --step "$step"
    else
        echo "--- smoke eval: step_$step skipped (no checkpoint) ---"
    fi
    step=$((step + SMOKE_EVERY))
done

# ---------- phase 3: full eval on the final checkpoint ----------
echo "=== phase 3/4: full eval on $FINAL_CKPT ==="
MC_JSON="$EVAL_LOG_DIR/results_${RUN_NAME}_step${STEPS}_default_${TS}.json"
MC_LOG="$EVAL_LOG_DIR/eval_${RUN_NAME}_step${STEPS}_default_${TS}.log"
python scripts/eval_harness.py \
    --ckpt_dir "$FINAL_CKPT" \
    --tokenizer "$TOKENIZER" \
    --preset default \
    --resume_wandb_id "$WANDB_ID" --wandb_project "$WANDB_PROJECT" \
    ${FINAL_LIMIT:+--limit "$FINAL_LIMIT"} \
    --output_path "$MC_JSON" 2>&1 | tee "$MC_LOG"

# lm-eval's logger above writes the MC metrics as top-level summary keys and
# uploads the per-sample tables; ALSO log them under final/ so every final
# metric sits in one consistent W&B section next to final/wikitext/*.
python scripts/log_results_to_wandb.py \
    --json "$MC_JSON" \
    --resume_id "$WANDB_ID" --project "$WANDB_PROJECT" \
    --prefix final --step "$STEPS"

PPL_JSON=""
if [ "$SKIP_FULL_PPL" = "0" ]; then
    PPL_JSON="$EVAL_LOG_DIR/results_${RUN_NAME}_step${STEPS}_wikitext_${TS}.json"
    python scripts/eval_harness.py \
        --ckpt_dir "$FINAL_CKPT" \
        --tokenizer "$TOKENIZER" \
        --preset perplexity \
        --tasks wikitext \
        ${FINAL_LIMIT:+--limit "$FINAL_LIMIT"} \
        --no_wandb --resume_wandb_id none \
        --output_path "$PPL_JSON"
    python scripts/log_results_to_wandb.py \
        --json "$PPL_JSON" \
        --resume_id "$WANDB_ID" --project "$WANDB_PROJECT" \
        --prefix final --step "$STEPS"
fi

# ---------- phase 4: pin every artifact path in the W&B run notes ----------
echo "=== phase 4/4: append artifact paths to W&B notes ==="
# Run:AI job name for the follow-up perplexity job: lowercase, no dots, <50 chars.
PPL_JOB_NAME=$(printf 'ppl-%s' "$RUN_NAME" | tr '[:upper:]' '[:lower:]' \
    | sed -e 's/[^a-z0-9-]/-/g' | cut -c1-49 | sed -e 's/-*$//')
if [ -n "$PPL_JSON" ]; then
    PPL_TASKS="zip2zip_pile,zip2zip_mc4,zip2zip_dc4"
    PPL_INTRO="To complete the FULL 4-corpora perplexity (adds Pile/mC4/dC4, ~15h, 1 GPU),"
else
    PPL_TASKS="wikitext,zip2zip_pile,zip2zip_mc4,zip2zip_dc4"
    PPL_INTRO="To run the FULL 4-corpora perplexity (wikitext+Pile+mC4+dC4, 1 GPU),"
fi
NOTES="pipeline: $(basename "$0") @ commit $(git rev-parse --short HEAD)
final checkpoint: $FINAL_CKPT
pipeline log (RCP): $PIPELINE_LOG
final MC results: $MC_JSON
final MC eval log (all samples printed): $MC_LOG${PPL_JSON:+
wikitext ppl results: $PPL_JSON}
smoke results: $EVAL_LOG_DIR/results_${RUN_NAME}_step*_smoke_${TS}.json
$PPL_INTRO
paste in a shell where WANDB_API_KEY is set — the results land in THIS W&B run
under final/, next to the other evals:

runai submit --name $PPL_JOB_NAME \\
  --image ghcr.io/jkminder/dlab-runai-images/pytorch:master \\
  --gpu 1 --cpu 8 --memory 64Gi \\
  --pvc dlab-scratch:/mnt --large-shm --node-pools default \\
  --environment CKPT_DIR='$FINAL_CKPT' \\
  --environment TOKENIZER=$TOKENIZER \\
  --environment PRESET=perplexity \\
  --environment TASKS=$PPL_TASKS \\
  --environment RESUME_WANDB_ID=$WANDB_ID \\
  --environment WANDB_STEP=$STEPS \\
  --environment WANDB_PROJECT=$WANDB_PROJECT \\
  --environment WANDB_API_KEY=\$WANDB_API_KEY \\
  -- bash $PROJECT_DIR/scripts/eval_ckpt_rcp.sh"
python scripts/log_results_to_wandb.py \
    --resume_id "$WANDB_ID" --project "$WANDB_PROJECT" \
    --append_notes "$NOTES"

echo "=== pipeline complete: W&B run $WANDB_PROJECT/$WANDB_ID ($RUN_NAME) ==="
} 2>&1 | tee "$PIPELINE_LOG"
