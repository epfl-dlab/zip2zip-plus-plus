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
#   Z2Z_SCRATCH=...     Your PVC scratch dir (default: /dlabscratch1/gentilin)
#   WANDB=1         Log results/samples/compression to W&B (default: off, JSON+log
#                   only). Requires WANDB_API_KEY in the job env. Entity is
#                   hardcoded to epfl-dlab in src/zip2zip_core/project.py.
#   WANDB_NAME=...  W&B run name; eval_harness.py prepends "eval-" automatically
#                   (default: auto from checkpoint/repo name)
#   WANDB_PROJECT=. W&B project (default: project.py's WANDB_PROJECT, i.e. llaza)
#   RESUME_WANDB_ID=... Existing W&B run id: log the results INTO that run
#                   (via log_results_to_wandb.py) instead of creating a separate
#                   eval run. Used by the ready-made perplexity command that
#                   pipeline_ft_eval_rcp.sh appends to its W&B notes, so the
#                   remaining corpora land next to the run's other evals.
#                   Also appends the results JSON + log paths to the run notes,
#                   and (perplexity preset) replaces the notes' [pending]
#                   command block. Forces --no_wandb on the harness; requires
#                   WANDB_API_KEY.
#   WANDB_PREFIX=final  With RESUME_WANDB_ID: metrics namespaced <prefix>/<task>/<metric>
#   WANDB_STEP=...  With RESUME_WANDB_ID: checkpoint step, logged on the
#                   '<prefix>/step' x-axis so it lines up with the pipeline's
#                   final evals (e.g. final/wikitext/*)
#
set -euo pipefail

Z2Z_SCRATCH=${Z2Z_SCRATCH:-/dlabscratch1/gentilin}
export HF_HOME=$Z2Z_SCRATCH/.cache/huggingface
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

PROJECT_DIR=$Z2Z_SCRATCH/code/zip2zip-core
LOG_DIR=$Z2Z_SCRATCH/logs/eval
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
# EVAL_MODE=base for MAX_CODEBOOK_SIZE=0 control checkpoints (base-mode-only).
EVAL_MODE=${EVAL_MODE:-}
# DISABLE_DIGIT_IDS=1: diagnostic — digits never LZW-merge into hypertokens.
# "0" is normalized to off ("" ) so it cannot half-trigger the :+ passthrough.
DISABLE_DIGIT_IDS=${DISABLE_DIGIT_IDS:-}
[ "$DISABLE_DIGIT_IDS" = "0" ] && DISABLE_DIGIT_IDS=""
# DISABLE_MATHSYM_IDS=1: triage — also protect math operators/symbols.
DISABLE_MATHSYM_IDS=${DISABLE_MATHSYM_IDS:-}
[ "$DISABLE_MATHSYM_IDS" = "0" ] && DISABLE_MATHSYM_IDS=""
# NO_ONLINE_CODEBOOK_MASK=1: score a v0.6.5+ checkpoint with the LEGACY k<=t mask
# instead of the exact decoder-time mask it trained with. This is the setting for
# comparing against v0.1-v0.6.4, whose numbers were all produced with the legacy
# mask. "0" normalizes to off so it cannot half-trigger the :+ passthrough.
NO_ONLINE_CODEBOOK_MASK=${NO_ONLINE_CODEBOOK_MASK:-}
[ "$NO_ONLINE_CODEBOOK_MASK" = "0" ] && NO_ONLINE_CODEBOOK_MASK=""
WANDB=${WANDB:-0}
WANDB_NAME=${WANDB_NAME:-}
WANDB_PROJECT=${WANDB_PROJECT:-}
RESUME_WANDB_ID=${RESUME_WANDB_ID:-}
WANDB_PREFIX=${WANDB_PREFIX:-final}
WANDB_STEP=${WANDB_STEP:-}

if [ -n "$RESUME_WANDB_ID" ] && [ -z "${WANDB_API_KEY:-}" ]; then
    echo "RESUME_WANDB_ID is set but WANDB_API_KEY is empty — cannot log into the run." >&2
    exit 1
fi


if [ -n "$CKPT_DIR" ]; then
    MODEL_SHORT=$(echo "$(basename "$(dirname "$CKPT_DIR")")_$(basename "$CKPT_DIR")" | tr -d '()')
else
    MODEL_SHORT=$(echo "$HF_REPO" | sed 's|.*/||')
fi
TASKS_SUFFIX=${TASKS:+_${TASKS//,/-}}

LOGFILE="$LOG_DIR/eval_${MODEL_SHORT}_${PRESET}${TASKS_SUFFIX}_${TIMESTAMP}.log"
OUTPUT_JSON="$LOG_DIR/results_${MODEL_SHORT}_${PRESET}${TASKS_SUFFIX}_${TIMESTAMP}.json"

# ---------- venv ----------
VENV_DIR=$Z2Z_SCRATCH/.venvs/lm-eval
if [ ! -f "$VENV_DIR/bin/activate" ]; then
    echo "[eval_ckpt] Creating venv at $VENV_DIR..."
    python -m venv --system-site-packages "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"
pip install --quiet "lm-eval==0.4.9" "huggingface_hub" wandb

# Verify the resume target EXISTS before spending an hour of GPU. The upload runs
# LAST, so a wrong id used to surface only after the whole eval had finished — and
# Run:AI then retried the job, re-burning the GPU on every attempt (this cost ~6
# GPU-hours on 2026-07-26). The id must come from the pipeline log that actually
# contains training steps: a crashed pipeline that Run:AI retried leaves several
# short logs, each with its OWN freshly-generated run_id that was never created.
if [ -n "$RESUME_WANDB_ID" ]; then
    python - "$RESUME_WANDB_ID" "${WANDB_PROJECT:-zip2zip-core}" "${WANDB_ENTITY:-epfl-dlab}" <<'PY' || exit 1
import sys
import wandb
run_id, project, entity = sys.argv[1:4]
try:
    wandb.Api().run(f"{entity}/{project}/{run_id}")
except Exception as exc:
    sys.exit(
        f"[eval_ckpt] FATAL: RESUME_WANDB_ID={run_id} does not exist in "
        f"{entity}/{project} ({type(exc).__name__}). Refusing to run the eval: it "
        f"would take an hour and then fail on resume. Get the id with\n"
        f"  grep -l 'step=' $Z2Z_SCRATCH/logs/pipeline/pipeline_<RUN_NAME>_*.log\n"
        f"then read 'run_id=' from THAT log, or copy it from the W&B URL."
    )
print(f"[eval_ckpt] W&B resume target {entity}/{project}/{run_id} exists")
PY
fi

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
# RESUME_WANDB_ID instead keeps the harness at --no_wandb and backfills the
# results JSON into the existing run after the eval (see below).
WANDB_ARGS="--no_wandb"
if [ -z "$RESUME_WANDB_ID" ] && [ "$WANDB" != "0" ]; then
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
echo "  RESUME_ID:   ${RESUME_WANDB_ID:-<none>}${RESUME_WANDB_ID:+ (prefix: $WANDB_PREFIX${WANDB_STEP:+, step: $WANDB_STEP})}"
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
        ${EVAL_MODE:+--eval_mode "$EVAL_MODE"} \
        ${DISABLE_DIGIT_IDS:+--disable_digit_ids} \
        ${DISABLE_MATHSYM_IDS:+--disable_mathsym_ids} \
        ${NO_ONLINE_CODEBOOK_MASK:+--no_online_codebook_mask} \
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
        ${EVAL_MODE:+--eval_mode "$EVAL_MODE"} \
        ${DISABLE_DIGIT_IDS:+--disable_digit_ids} \
        ${DISABLE_MATHSYM_IDS:+--disable_mathsym_ids} \
        ${NO_ONLINE_CODEBOOK_MASK:+--no_online_codebook_mask} \
        $WANDB_ARGS \
        $LIMIT_ARG \
        $TASKS_ARG
fi

# Same audit as the pipeline: the results JSON must record the flags we asked
# for — a silently dropped flag fails loudly instead of producing numbers from
# the wrong distribution.
if [ -n "${EVAL_MODE:-}" ] || [ -n "${DISABLE_DIGIT_IDS:-}" ] || [ -n "${DISABLE_MATHSYM_IDS:-}" ]; then
    python - "$OUTPUT_JSON" "${EVAL_MODE:-}" "${DISABLE_DIGIT_IDS:-}" "${DISABLE_MATHSYM_IDS:-}" <<'PY'
import json, sys
path, want_mode, want_digits, want_syms = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
args = json.load(open(path))["args"]
if want_mode:
    got = args["eval_mode"]
    assert got == want_mode, f"eval-mode audit FAILED: {path} recorded {got!r}, expected {want_mode!r}"
if want_digits:
    got = args.get("disable_digit_ids")
    assert got is True, f"digit-flag audit FAILED: {path} recorded disable_digit_ids={got!r}, expected True"
if want_syms:
    got = args.get("disable_mathsym_ids")
    assert got is True, f"mathsym-flag audit FAILED: {path} recorded disable_mathsym_ids={got!r}, expected True"
print(f"[eval_ckpt] eval-flags audit OK: {path}")
PY
fi

if [ -n "$RESUME_WANDB_ID" ]; then
    echo "=== logging results into existing W&B run $RESUME_WANDB_ID under $WANDB_PREFIX/ ==="
    # A perplexity run completes the pipeline's follow-up: drop the [pending]
    # command block from the run notes (no-op if the notes don't have one).
    RESOLVE_PENDING=""
    if [ "$PRESET" = "perplexity" ]; then
        RESOLVE_PENDING="--resolve_pending"
    fi
    python scripts/log_results_to_wandb.py \
        --json "$OUTPUT_JSON" \
        --resume_id "$RESUME_WANDB_ID" \
        ${WANDB_PROJECT:+--project "$WANDB_PROJECT"} \
        --prefix "$WANDB_PREFIX" \
        ${WANDB_STEP:+--step "$WANDB_STEP"} \
        $RESOLVE_PENDING \
        --append_notes "$PRESET${TASKS:+ [$TASKS]} eval results: $OUTPUT_JSON
$PRESET eval log (RCP): $LOGFILE"
fi

} 2>&1 | tee "$LOGFILE"
