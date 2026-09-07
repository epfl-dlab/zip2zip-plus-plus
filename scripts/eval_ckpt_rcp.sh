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
#   PRESET=...      Eval preset (default: default_base). PRESET=perplexity_subset
#                   is the quick perplexity: full wikitext + pinned 1000-doc
#                   subsets of Pile/mC4/dC4, logged under subset_ppl/ (see
#                   WANDB_PREFIX below). PRESET=postsft is the paper's Table 2
#                   generation set (math500, humaneval_instruct, ifeval; each
#                   task keeps its own few-shot). humaneval_instruct executes
#                   model-written code, so this script exports
#                   HF_ALLOW_CODE_EVAL=1 for that preset — set
#                   HF_ALLOW_CODE_EVAL=0 to refuse and fail fast instead.
#   LIMIT=20        Per-task sample limit for smoke tests. Refused together with
#                   PRESET=perplexity_subset + RESUME_WANDB_ID under the default
#                   prefix: a partial subset score must not enter the
#                   comparable subset_ppl/ series.
#   TASKS=gsm8k     Comma-separated task override (default: preset's tasks)
#   LEGACY_UNTRIMMED_STOPS=1  Do not cut generated text at stop strings, as all
#                   evals before 2026-08 did. Only for bit-exact reproduction of
#                   those results (pin the old task list via TASKS too — the
#                   presets gained triviaqa); wrong for text-scored tasks.
#   LEGACY_STRIPPED_GENERATION=1  Decode generations standalone, as all evals
#                   before 2026-09 did: SentencePiece tokenizers (Phi) then drop
#                   one leading space, which breaks HumanEval indentation. Only
#                   for bit-exact reproduction of old generation samples.
#   Z2Z_SCRATCH=...     Your PVC scratch dir (default: /dlabscratch1/gentilin)
#   WANDB=1         Log results/samples/compression to W&B (default: off, JSON+log
#                   only). Requires WANDB_API_KEY in the job env.
#   WANDB_ENTITY=. W&B entity (default: epfl-dlab)
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
#   WANDB_PREFIX=final  With RESUME_WANDB_ID: metrics namespaced <prefix>/<task>/<metric>.
#                   Default: final — except PRESET=perplexity_subset, which
#                   defaults to subset_ppl and refuses final (quick pinned-subset
#                   numbers must never share the full-run section).
#   WANDB_STEP=...  With RESUME_WANDB_ID: checkpoint step, logged on the
#                   '<prefix>/step' x-axis so it lines up with the pipeline's
#                   final evals (e.g. final/wikitext/*)
#   EVAL_PATHS_ONLY=1  Print collision-safe log/result paths and exit (no eval).
#
set -euo pipefail

Z2Z_SCRATCH=${Z2Z_SCRATCH:-/dlabscratch1/gentilin}
export HF_HOME=$Z2Z_SCRATCH/.cache/huggingface
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

PROJECT_DIR=${PROJECT_DIR:-$Z2Z_SCRATCH/code/zip2zip-core}
LOG_DIR=$Z2Z_SCRATCH/logs/eval
mkdir -p "$LOG_DIR"
TIMESTAMP=${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}

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
# The postsft tasks (math500, humaneval_instruct, ifeval) need extra packages
# that go into their own --target directory prepended to PYTHONPATH (see the venv block below); any preset/task list
# naming them selects it.
POSTSFT_DEPS=""
case "$PRESET,${TASKS:-}" in
    postsft,*|*humaneval*|*math500*|*ifeval*) POSTSFT_DEPS=1 ;;
esac
# humaneval_instruct runs the model's code through HF evaluate's code_eval, which
# refuses unless HF_ALLOW_CODE_EVAL=1; eval_harness.py mirrors the same variable
# into lm-eval's confirm_run_unsafe_code. Only the postsft preset (or an explicit
# humaneval task list) turns it on — no other preset ever executes generated text.
case "$PRESET,${TASKS:-}" in
    postsft,*|*humaneval*) export HF_ALLOW_CODE_EVAL=${HF_ALLOW_CODE_EVAL:-1} ;;
esac
# EVAL_MODE=base for MAX_CODEBOOK_SIZE=0 control checkpoints (base-mode-only).
EVAL_MODE=${EVAL_MODE:-}
# EVAL_MAX_SUBTOKENS: eval-only LZW merge-size override. Unset follows meta.pt.
# "0" normalizes to off so it cannot half-trigger the :+ passthrough.
EVAL_MAX_SUBTOKENS=${EVAL_MAX_SUBTOKENS:-}
[ "$EVAL_MAX_SUBTOKENS" = "0" ] && EVAL_MAX_SUBTOKENS=""
# DISABLE_DIGIT_IDS=1: diagnostic — digits never LZW-merge into hypertokens.
# "0" is normalized to off ("" ) so it cannot half-trigger the :+ passthrough.
DISABLE_DIGIT_IDS=${DISABLE_DIGIT_IDS:-}
[ "$DISABLE_DIGIT_IDS" = "0" ] && DISABLE_DIGIT_IDS=""
# DISABLE_MATHSYM_IDS=1: triage — also protect math operators/symbols.
DISABLE_MATHSYM_IDS=${DISABLE_MATHSYM_IDS:-}
[ "$DISABLE_MATHSYM_IDS" = "0" ] && DISABLE_MATHSYM_IDS=""
# NO_ONLINE_CODEBOOK_MASK=1: score a v0.6.5 checkpoint with the LEGACY k<=t mask
# instead of the exact decoder-time mask it trained with. This is the setting for
# comparing against v0.1-v0.6.4, whose numbers were all produced with the legacy
# mask. "0" normalizes to off so it cannot half-trigger the :+ passthrough.
NO_ONLINE_CODEBOOK_MASK=${NO_ONLINE_CODEBOOK_MASK:-}
[ "$NO_ONLINE_CODEBOOK_MASK" = "0" ] && NO_ONLINE_CODEBOOK_MASK=""
# LEGACY_UNTRIMMED_STOPS=1: reproduce pre-2026-08 generation returns bit-exactly
# (no stop-string cut). "0" normalizes to off like the flags above.
LEGACY_UNTRIMMED_STOPS=${LEGACY_UNTRIMMED_STOPS:-}
[ "$LEGACY_UNTRIMMED_STOPS" = "0" ] && LEGACY_UNTRIMMED_STOPS=""
# LEGACY_STRIPPED_GENERATION=1: decode generations standalone (pre-2026-09), which
# loses one leading space on SentencePiece tokenizers. "0" normalizes to off.
LEGACY_STRIPPED_GENERATION=${LEGACY_STRIPPED_GENERATION:-}
[ "$LEGACY_STRIPPED_GENERATION" = "0" ] && LEGACY_STRIPPED_GENERATION=""
WANDB=${WANDB:-0}
WANDB_NAME=${WANDB_NAME:-}
export WANDB_PROJECT=${WANDB_PROJECT:-llaza}
export WANDB_ENTITY=${WANDB_ENTITY:-epfl-dlab}
RESUME_WANDB_ID=${RESUME_WANDB_ID:-}
# Subset-perplexity runs default to their own W&B section: quick pinned-subset
# numbers must never sit in the same panels as full-corpus numbers under
# final/. (The task names differ too — zip2zip_pile_sub1k vs zip2zip_pile —
# so even a forced shared prefix cannot merge panels, but final/ is refused
# outright to keep that section full-run-only.)
if [ "$PRESET" = "perplexity_subset" ]; then
    WANDB_PREFIX=${WANDB_PREFIX:-subset_ppl}
    if [ "$WANDB_PREFIX" = "final" ]; then
        echo "PRESET=perplexity_subset with WANDB_PREFIX=final: refusing —" \
             "subset numbers must not land in the full-run final/ section." >&2
        exit 1
    fi
    # LIMIT truncates the pinned subset to its first N docs (lm-eval applies
    # --limit after process_docs), so a LIMIT smoke run logged into the
    # subset_ppl/ series would silently poison the very comparability the
    # pinned subset exists for. Smoke-test under a different prefix, or
    # without RESUME_WANDB_ID.
    if [ -n "$LIMIT" ] && [ -n "$RESUME_WANDB_ID" ] \
        && [ "$WANDB_PREFIX" = "subset_ppl" ]; then
        echo "PRESET=perplexity_subset with LIMIT=$LIMIT and RESUME_WANDB_ID:" \
             "refusing — a partial subset score under subset_ppl/ is not" \
             "comparable with the pinned-subset series. Drop LIMIT, or set" \
             "WANDB_PREFIX to a scratch prefix." >&2
        exit 1
    fi
else
    WANDB_PREFIX=${WANDB_PREFIX:-final}
fi
WANDB_STEP=${WANDB_STEP:-}

if { [ -n "$RESUME_WANDB_ID" ] || [ "$WANDB" != "0" ]; } \
    && [ -z "${WANDB_API_KEY:-}" ]; then
    echo "W&B logging is enabled but WANDB_API_KEY is empty." >&2
    exit 1
fi


if [ -n "$CKPT_DIR" ]; then
    MODEL_SHORT=$(echo "$(basename "$(dirname "$CKPT_DIR")")_$(basename "$CKPT_DIR")" | tr -d '()')
else
    MODEL_SHORT=$(echo "$HF_REPO" | sed 's|.*/||')
fi
TASKS_SUFFIX=${TASKS:+_${TASKS//,/-}}

# Include the requested evaluation regime in the human-readable tag. "auto"
# means the adapter chooses compressed/base from the checkpoint. A unique
# pod/process suffix prevents same-second jobs (including identical retries)
# from overwriting one another.
MODE_TAG=$(printf '%s' "${EVAL_MODE:-auto}" | tr -cs '[:alnum:]._-' '_')
VARIANT_TAG="mode-${MODE_TAG}"
[ -n "$DISABLE_DIGIT_IDS" ] && VARIANT_TAG="${VARIANT_TAG}-nodigits"
[ -n "$DISABLE_MATHSYM_IDS" ] && VARIANT_TAG="${VARIANT_TAG}-nomathsym"
[ -n "$NO_ONLINE_CODEBOOK_MASK" ] && VARIANT_TAG="${VARIANT_TAG}-legacycbmask"
[ -n "$LEGACY_UNTRIMMED_STOPS" ] && VARIANT_TAG="${VARIANT_TAG}-untrimmedstops"
[ -n "$LEGACY_STRIPPED_GENERATION" ] && VARIANT_TAG="${VARIANT_TAG}-strippedgen"
if [ -n "$LIMIT" ]; then
    LIMIT_TAG=$(printf '%s' "$LIMIT" | tr -cs '[:alnum:]._-' '_')
    VARIANT_TAG="${VARIANT_TAG}-limit${LIMIT_TAG}"
fi
INSTANCE_RAW="${RUNAI_JOB_NAME:-local}-${HOSTNAME:-host}-$$"
INSTANCE_TAG=$(printf '%s' "$INSTANCE_RAW" | tr -cs '[:alnum:]._-' '_')
ARTIFACT_STEM="${MODEL_SHORT}_${PRESET}${TASKS_SUFFIX}_${VARIANT_TAG}_${TIMESTAMP}_${INSTANCE_TAG}"

LOGFILE="$LOG_DIR/eval_${ARTIFACT_STEM}.log"
OUTPUT_JSON="$LOG_DIR/results_${ARTIFACT_STEM}.json"

if [ "${EVAL_PATHS_ONLY:-0}" != "0" ]; then
    echo "LOGFILE=$LOGFILE"
    echo "OUTPUT_JSON=$OUTPUT_JSON"
    exit 0
fi

# ---------- venv ----------
VENV_DIR=$Z2Z_SCRATCH/.venvs/lm-eval
if [ ! -f "$VENV_DIR/bin/activate" ]; then
    echo "[eval_ckpt] Creating venv at $VENV_DIR..."
    python -m venv --system-site-packages "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"
pip install --quiet \
    "lm-eval==0.4.9" \
    "huggingface_hub" \
    "zip2zip-compression>=0.3.3" \
    wandb
if [ -n "$POSTSFT_DEPS" ]; then
    # The postsft tasks need packages the SHARED venv must not get: lm-eval 0.4.9's
    # minerva_math.utils (which math500 delegates to) asserts
    # antlr4-python3-runtime==4.11 at import time, and 4.11 breaks omegaconf
    # ("Could not deserialize ATN"), which eval_hf_model.py / eval_z2z_rcp.sh
    # import from that same venv (measured 2026-09-06). A second venv is not an
    # option either: the shared one carries a torch nightly (varlen attention)
    # a fresh --system-site-packages venv does not see. So the extras live in
    # their own directory and are prepended to PYTHONPATH for THIS process only
    # — antlr4 4.11 shadows the venv's 4.9.3 here and nowhere else. langdetect +
    # immutabledict are ifeval's. Pinned so a re-created dir scores identically
    # (4.11.0 = lm-eval's own [math] extra).
    POSTSFT_EXTRAS_DIR=$Z2Z_SCRATCH/.venvs/lm-eval-postsft-extras
    pip install --quiet --upgrade --target "$POSTSFT_EXTRAS_DIR" \
        "langdetect==1.0.9" \
        "immutabledict==4.2.1" \
        "math-verify==0.7.0" \
        "antlr4-python3-runtime==4.11.0"
    export PYTHONPATH="$POSTSFT_EXTRAS_DIR${PYTHONPATH:+:$PYTHONPATH}"
    echo "[eval_ckpt] postsft extras on PYTHONPATH: $POSTSFT_EXTRAS_DIR"
fi

# Verify the resume target EXISTS before spending an hour of GPU. The upload runs
# LAST, so a wrong id used to surface only after the whole eval had finished — and
# Run:AI then retried the job, re-burning the GPU on every attempt (this cost ~6
# GPU-hours on 2026-07-26). The id must come from the pipeline log that actually
# contains training steps: a crashed pipeline that Run:AI retried leaves several
# short logs, each with its OWN freshly-generated run_id that was never created.
if [ -n "$RESUME_WANDB_ID" ]; then
    python - "$RESUME_WANDB_ID" "$WANDB_PROJECT" "$WANDB_ENTITY" <<'PY' || exit 1
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
echo "  VARIANT:     $VARIANT_TAG"
echo "  EVAL_MS:     ${EVAL_MAX_SUBTOKENS:-<checkpoint>}"
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
        ${EVAL_MAX_SUBTOKENS:+--eval_max_subtokens "$EVAL_MAX_SUBTOKENS"} \
        ${DISABLE_DIGIT_IDS:+--disable_digit_ids} \
        ${DISABLE_MATHSYM_IDS:+--disable_mathsym_ids} \
        ${NO_ONLINE_CODEBOOK_MASK:+--no_online_codebook_mask} \
        ${LEGACY_UNTRIMMED_STOPS:+--legacy_untrimmed_stops} \
        ${LEGACY_STRIPPED_GENERATION:+--legacy_stripped_generation} \
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
        ${EVAL_MAX_SUBTOKENS:+--eval_max_subtokens "$EVAL_MAX_SUBTOKENS"} \
        ${DISABLE_DIGIT_IDS:+--disable_digit_ids} \
        ${DISABLE_MATHSYM_IDS:+--disable_mathsym_ids} \
        ${NO_ONLINE_CODEBOOK_MASK:+--no_online_codebook_mask} \
        ${LEGACY_UNTRIMMED_STOPS:+--legacy_untrimmed_stops} \
        ${LEGACY_STRIPPED_GENERATION:+--legacy_stripped_generation} \
        $WANDB_ARGS \
        $LIMIT_ARG \
        $TASKS_ARG
fi

# Same audit as the pipeline: the results JSON must record the flags we asked
# for — a silently dropped flag fails loudly instead of producing numbers from
# the wrong distribution.
if [ -n "${EVAL_MODE:-}" ] || [ -n "${EVAL_MAX_SUBTOKENS:-}" ] || [ -n "${DISABLE_DIGIT_IDS:-}" ] || [ -n "${DISABLE_MATHSYM_IDS:-}" ]; then
    python - "$OUTPUT_JSON" "${EVAL_MODE:-}" "${EVAL_MAX_SUBTOKENS:-}" "${DISABLE_DIGIT_IDS:-}" "${DISABLE_MATHSYM_IDS:-}" <<'PY'
import json, sys
path, want_mode, want_ms, want_digits, want_syms = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5]
args = json.load(open(path))["args"]
if want_mode:
    got = args["eval_mode"]
    assert got == want_mode, f"eval-mode audit FAILED: {path} recorded {got!r}, expected {want_mode!r}"
if want_ms:
    got = args.get("eval_max_subtokens")
    assert got == int(want_ms), f"eval-max-subtokens audit FAILED: {path} recorded {got!r}, expected {want_ms}"
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
    # perplexity_subset deliberately does NOT match: a quick subset score
    # leaves the full 4-corpora run pending.
    RESOLVE_PENDING=""
    if [ "$PRESET" = "perplexity" ]; then
        RESOLVE_PENDING="--resolve_pending"
    fi
    python scripts/log_results_to_wandb.py \
        --json "$OUTPUT_JSON" \
        --resume_id "$RESUME_WANDB_ID" \
        --entity "$WANDB_ENTITY" \
        --project "$WANDB_PROJECT" \
        --prefix "$WANDB_PREFIX" \
        ${WANDB_STEP:+--step "$WANDB_STEP"} \
        $RESOLVE_PENDING \
        --append_notes "$PRESET${TASKS:+ [$TASKS]} eval results: $OUTPUT_JSON
$PRESET eval log (RCP): $LOGFILE"
fi

} 2>&1 | tee "$LOGFILE"
