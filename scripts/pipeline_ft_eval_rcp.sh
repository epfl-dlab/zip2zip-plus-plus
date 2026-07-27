#!/bin/bash
# Finetune -> eval pipeline for the RCP cluster (Run:AI). ONE sequential job,
# With WANDB=1 (default), one W&B run contains training curves, smoke-eval
# curves, final metrics/sample tables, and artifact notes. With WANDB=0 the
# same train/eval/audit phases run without an API key or network uploads; JSON,
# checkpoint, and log artifacts remain on the PVC.
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
#                  missing data/masks, and (only with WANDB=1) missing API key
#   1. train       scripts/finetune_phi35_rcp.sh (its own env/venv), 4 GPUs,
#                  optionally creates the shared W&B run
#   2. smoke       every SMOKE_EVERY-th checkpoint: arc_easy,hellaswag,winogrande,
#                  gsm8k_boxed @ LIMIT=SMOKE_LIMIT -> JSON always, plus W&B
#                  smoke/step curves when enabled (1 GPU)
#   3. final eval  last checkpoint: full 'default' preset (with all sample tables
#                  in JSON and optionally W&B) + wikitext perplexity
#   4. artifacts   WANDB=1: append paths and the remaining-perplexity command to
#                  W&B notes. WANDB=0: print the same local artifact paths.
#
# Env vars:
#   RUN_NAME=...      required output name; also the W&B name when enabled
#   RECIPE=v0.6.4     (recommended) the named recipe: fills in every recipe env
#                     var you did not set yourself. THIS IS THE CURRENT STANDARD
#                     RECIPE — see `python scripts/recipes.py --list`. Explicit
#                     env vars still win, so RECIPE=v0.6.4 ENCODER_N_LAYERS=4 is
#                     a clean single-variable experiment on top of it.
#   DATA_DIR=...      (default: $Z2Z_SCRATCH/datasets/phi-1B-sft-8shards-eosfix)
#   STEPS=8000        total steps (forwarded to the finetune launcher)
#   SMOKE_EVERY=1000  smoke-eval cadence in steps
#   SMOKE_LIMIT=200   per-task sample limit for smoke evals
#   SMOKE_TASKS=arc_easy,hellaswag,winogrande,gsm8k_boxed
#   TOKENIZER=microsoft/Phi-3.5-mini-instruct
#   WANDB_PROJECT=zip2zip-core
#   WANDB_ENTITY=epfl-dlab  W&B entity used consistently by train/eval/backfill
#   WANDB=1         one W&B train/eval run (default); requires WANDB_API_KEY
#   WANDB=0         fully offline: no API-key check and no W&B logging/uploads
#   SKIP_SMOKE=0      set 1 to skip intermediate smoke evals
#   SKIP_FULL_PPL=0   set 1 to also skip the wikitext perplexity in phase 3
#   EVAL_MODE=        force eval_harness --eval_mode for all evals; set to
#                     'base' for MAX_CODEBOOK_SIZE=0 control runs (their
#                     checkpoints are base-mode-only). Empty = preset default.
#   DISABLE_DIGIT_IDS= set 1 to keep digits out of LZW merges in BOTH training
#                     (finetune launcher) and every eval — one lever so the
#                     model trains and evals in the same digit-protected
#                     distribution. Leave empty to disable (never set to 0).
#                     CANONICAL since v0.4-digitsafe: pass 1 on new finetunes.
#   UNTIED_HYPER_ENCODER= set 1 for a separate output-role hyper-encoder
#                     (matches the released model). Inherited by the train
#                     launcher only; evals auto-configure from the checkpoint
#                     meta.pt (no eval passthrough), so no "0" ambiguity.
#   SHARE_HYPER_ENCODER_WEIGHTS= with UNTIED_HYPER_ENCODER=1, re-encode lm_head
#                     rows for the output role but reuse hyper_encoder weights
#                     instead of training a second hyper_output module.
#   HYPER_ENCODER_TYPE= flat by default; set hierarchical for the left-fold
#                     composer experiment. Eval restores it from meta.pt.
#   ENCODER_N_LAYERS= hyper-encoder depth (default 2 = v0.5). Set 4 for the
#                     v0.6.1 deeper-encoder recipe (negative result — keep 2).
#                     Inherited by the train launcher; evals auto-configure
#                     from meta.pt like the untied flag.
#   BASE_TOKEN_POSITIONS= set 1 for uncompressed-stream RoPE positions
#                     (v0.6.2 recipe). Inherited by the train launcher; evals
#                     auto-configure from meta.pt like the untied flag.
#   TWO_AXIS_ROPE=    set 1 to split complex RoPE pairs between base-stream
#                     and compressed-token positions (v0.7 negative result).
#                     Requires
#                     BASE_TOKEN_POSITIONS=1; evals auto-configure from meta.pt.
#   GATED_COMPRESSED_ROPE= v0.7.1 sets the single named profile
#                     lowfreq_all_layers: a zero-initialized learned compressed-
#                     position delta. Mutually exclusive with TWO_AXIS_ROPE;
#                     eval restores the resolved geometry from meta.pt.
#   GATED_ROPE_START_LAYER= first decoder layer using the learned delta
#                     (v0.7.1: 0, so all 32 Phi-3.5 layers are learnable).
#   GATED_ROPE_START_PAIR= first complex frequency pair using the delta
#                     (v0.7.1: 32, so only low-frequency pairs 32-47).
#   BASE_VIEW_REPLAY_PROB= fraction of training microbatches replayed as
#                     ordinary base-token views (v0.6.6: 0.25; v0.7.1: 0).
#                     Training-only; recorded in meta.pt and deterministic.
#   TOKEN_TYPE_LOSS_WEIGHT= auxiliary base-vs-hyper type loss weight
#                     (v0.6.3 recipe, 0.05). 0/off = v0.6.2 behavior. Inherited
#                     by the train launcher; evals auto-configure from meta.pt.
#   ZERO_INIT_ENCODER_OUTPUT= set 1 to start the hyper-encoder at exactly zero
#                     (v0.6.4 recipe — fixes a silent init no-op). Training-only
#                     (init), so evals need nothing.
#   NO_ENCODER_RESIDUAL= set 1 to remove the first-token residual from the
#                     hyper-encoder. Eval/inference restore this from meta.pt.
#   ONLINE_CODEBOOK_MASK= set 1 to train with decoder-time codebook availability
#                     instead of the k<=t approximation (v0.6.5 candidate).
#                     The checkpoint records it so compressed offline eval
#                     automatically uses the matching mask; generation is
#                     already incremental.
#   FINAL_LIMIT=      per-task sample limit for the FINAL eval (default: full).
#                     Only for pipeline rehearsals — never for real numbers.
#   SEED=42           training seed forwarded to train.py.
#   NUM_WORKERS=1     dataloader workers; set 0 for exact resumable matched runs
# Anything else the finetune launcher reads (LR, SEQ_LEN, ...) passes through.
#
# REHEARSE THE PIPELINE FIRST (~30 min end-to-end, exercises every phase):
#   ... --environment RUN_NAME=pipeline-rehearsal-1 \
#       --environment STEPS=20 --environment SAVE_FREQ=10 --environment LOG_FREQ=1 \
#       --environment SMOKE_EVERY=10 --environment SMOKE_LIMIT=8 \
#       --environment FINAL_LIMIT=8 ...
set -euo pipefail

Z2Z_SCRATCH=${Z2Z_SCRATCH:-/dlabscratch1/gentilin}
export HF_HOME=${HF_HOME:-$Z2Z_SCRATCH/.cache/huggingface}
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

PROJECT_DIR=${PROJECT_DIR:-$Z2Z_SCRATCH/code/zip2zip-core}
RUN_NAME=${RUN_NAME:?Must set RUN_NAME (output name and optional W&B run name)}

# Resolve a named recipe HERE, before anything below reads these variables: the
# digit-protection preflight and every eval phase consume them in THIS process,
# and the finetune launcher (a child) inherits them. An explicit env var always
# wins; resolving is idempotent, so the launcher re-resolving is harmless.
RECIPE=${RECIPE:-}
if [ -n "$RECIPE" ]; then
    # recipes.py is stdlib-only, so any interpreter works. Prefer one on PATH
    # (the job container has it) and fall back to the repo venv, which is the
    # only python present in some shells.
    _py=$(command -v python3 || command -v python || echo "$PROJECT_DIR/.venv/bin/python")
    [ -x "$_py" ] || { echo "FATAL: no python found to resolve RECIPE=$RECIPE" >&2; exit 1; }
    _recipe_env=$("$_py" "$PROJECT_DIR/scripts/recipes.py" "$RECIPE") || exit 1
    eval "$_recipe_env"
    unset _recipe_env _py
fi

# Resolve the gated profile in this parent process as well as in the finetune
# child. The parent owns the post-training eval audit, so it must retain the
# same effective geometry that the child records in meta.pt. Manual `1` keeps
# the generic all-pair default; v0.7.1's named profile selects Phi pairs 32-47.
case "${GATED_COMPRESSED_ROPE:-}" in
    ""|0|1)
        export GATED_ROPE_START_LAYER=${GATED_ROPE_START_LAYER:-0}
        export GATED_ROPE_START_PAIR=${GATED_ROPE_START_PAIR:-0}
        ;;
    lowfreq_all_layers)
        export GATED_ROPE_START_LAYER=${GATED_ROPE_START_LAYER:-0}
        export GATED_ROPE_START_PAIR=${GATED_ROPE_START_PAIR:-32}
        ;;
    *)
        echo "FATAL: unknown GATED_COMPRESSED_ROPE profile '${GATED_COMPRESSED_ROPE}' (expected 0, 1, or lowfreq_all_layers)" >&2
        exit 1
        ;;
esac

export RECIPE
export DATA_DIR=${DATA_DIR:-$Z2Z_SCRATCH/datasets/phi-1B-sft-8shards-eosfix}
export STEPS=${STEPS:-8000}
export SEED=${SEED:-42}
SMOKE_EVERY=${SMOKE_EVERY:-1000}
SMOKE_LIMIT=${SMOKE_LIMIT:-200}
SMOKE_TASKS=${SMOKE_TASKS:-arc_easy,hellaswag,winogrande,gsm8k_boxed}
TOKENIZER=${TOKENIZER:-microsoft/Phi-3.5-mini-instruct}
export WANDB_PROJECT=${WANDB_PROJECT:-zip2zip-core}
export WANDB_ENTITY=${WANDB_ENTITY:-epfl-dlab}
SKIP_SMOKE=${SKIP_SMOKE:-0}
SKIP_FULL_PPL=${SKIP_FULL_PPL:-0}
FINAL_LIMIT=${FINAL_LIMIT:-}

OUTPUT_BASE=${OUTPUT_BASE:-$Z2Z_SCRATCH/zip2zip-outputs}
OUTPUT_DIR=$OUTPUT_BASE/$RUN_NAME
EVAL_LOG_DIR=$Z2Z_SCRATCH/logs/eval
PIPE_LOG_DIR=$Z2Z_SCRATCH/logs/pipeline
mkdir -p "$EVAL_LOG_DIR" "$PIPE_LOG_DIR"
TS=$(date +%Y%m%d_%H%M%S)
PIPELINE_LOG="$PIPE_LOG_DIR/pipeline_${RUN_NAME}_${TS}.log"

# One pre-chosen W&B id ties train + evals into a single run. Offline runs keep
# it empty and never call a W&B logger/uploader.
export WANDB=${WANDB:-1}
WANDB_ID=${WANDB_ID:-}
if [ "$WANDB" = "0" ]; then
    WANDB_ID=""
elif [ -z "${WANDB_API_KEY:-}" ]; then
    echo "FATAL: WANDB=1 requires WANDB_API_KEY for the shared train/eval run."
    echo "Set WANDB=0 to run the complete pipeline offline."
    exit 1
elif [ -z "$WANDB_ID" ]; then
    _wandb_py=$(command -v python3 || command -v python || echo "$PROJECT_DIR/.venv/bin/python")
    [ -x "$_wandb_py" ] || {
        echo "FATAL: no Python interpreter found to generate WANDB_ID." >&2
        exit 1
    }
    WANDB_ID=$("$_wandb_py" - <<'PY'
import secrets, string
print("".join(secrets.choice(string.ascii_lowercase + string.digits) for _ in range(8)))
PY
)
    unset _wandb_py
fi
export WANDB_ID
export RUN_NAME

{
echo "=== zip2zip finetune->eval pipeline ==="
echo "  RUN_NAME:    $RUN_NAME"
echo "  RECIPE:      ${RECIPE:-<none, explicit env only>}"
echo "  OUTPUT_DIR:  $OUTPUT_DIR"
echo "  DATA_DIR:    $DATA_DIR"
echo "  STEPS:       $STEPS  SEED: $SEED  SMOKE_EVERY: $SMOKE_EVERY  SMOKE_LIMIT: $SMOKE_LIMIT"
echo "  SMOKE_TASKS: $SMOKE_TASKS"
echo "  SKIP_SMOKE:  $SKIP_SMOKE"
if [ "$WANDB" = "0" ]; then
    echo "  W&B:         disabled (offline JSON/log artifacts only)"
else
    echo "  W&B:         entity=$WANDB_ENTITY project=$WANDB_PROJECT run_id=$WANDB_ID name=$RUN_NAME"
fi
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
if [ ! -f "$PROJECT_DIR/scripts/lm_eval_tasks/gsm8k_boxed.yaml" ]; then
    echo "FATAL: gsm8k_boxed.yaml missing — pull the latest finetuning-andrea."
    exit 1
fi
if [ "${MAX_CODEBOOK_SIZE:-4096}" = "0" ] && [ "${EVAL_MODE:-}" != "base" ]; then
    echo "FATAL: MAX_CODEBOOK_SIZE=0 trains an uncompressed control checkpoint,"
    echo "which is base-mode-only at eval — set EVAL_MODE=base."
    exit 1
fi
if [ "${DISABLE_DIGIT_IDS:-}" = "0" ]; then
    echo "FATAL: DISABLE_DIGIT_IDS=0 is ambiguous — the training launcher would"
    echo "treat it as OFF while the eval passthrough would treat it as ON,"
    echo "splitting train/eval distributions. Leave it unset/empty to disable."
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
VENV_DIR=$Z2Z_SCRATCH/.venvs/lm-eval
if [ ! -f "$VENV_DIR/bin/activate" ]; then
    python -m venv --system-site-packages "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"
EVAL_DEPS=("lm-eval==0.4.9" "zip2zip-compression>=0.3.3")
if [ "$WANDB" != "0" ]; then
    EVAL_DEPS+=(wandb)
fi
pip install --quiet "${EVAL_DEPS[@]}"

cd "$PROJECT_DIR"

EVAL_WANDB_ARGS=(--no_wandb --resume_wandb_id none)
if [ "$WANDB" != "0" ]; then
    EVAL_WANDB_ARGS=(
        --resume_wandb_id "$WANDB_ID"
        --wandb_project "$WANDB_PROJECT"
    )
fi

# Audit: the results JSON must record the eval flags we asked for. Catches
# plumbing regressions where a control run (EVAL_MODE=base) or a digit-protected
# run (DISABLE_DIGIT_IDS=1) silently falls back to defaults. The online-mask
# check also proves that the evaluator recovered the training behavior from
# meta.pt. No-op when none of these env vars is set.
audit_eval_mode() {
    [ -z "${EVAL_MODE:-}" ] \
        && [ -z "${DISABLE_DIGIT_IDS:-}" ] \
        && [ -z "${ONLINE_CODEBOOK_MASK:-}" ] \
        && [ -z "${TWO_AXIS_ROPE:-}" ] \
        && [ -z "${GATED_COMPRESSED_ROPE:-}" ] \
        && return 0
    python - "$1" "${EVAL_MODE:-}" "${DISABLE_DIGIT_IDS:-}" \
        "${ONLINE_CODEBOOK_MASK:-}" "${TWO_AXIS_ROPE:-}" \
        "${GATED_COMPRESSED_ROPE:-}" "${GATED_ROPE_START_LAYER:-0}" \
        "${GATED_ROPE_START_PAIR:-0}" <<'PY'
import json, sys
(
    path,
    want_mode,
    want_digits,
    want_online_raw,
    want_two_axis_raw,
    want_gated_raw,
    want_start_raw,
    want_pair_raw,
) = sys.argv[1:9]
args = json.load(open(path))["args"]
if want_mode:
    got = args["eval_mode"]
    assert got == want_mode, f"eval-mode audit FAILED: {path} recorded {got!r}, expected {want_mode!r}"
if want_digits:
    got = args.get("disable_digit_ids")
    assert got is True, f"digit-flag audit FAILED: {path} recorded disable_digit_ids={got!r}, expected True"
if want_online_raw:
    want_online = want_online_raw != "0"
    got = args.get("online_codebook_mask")
    assert got is want_online, (
        f"online-mask audit FAILED: {path} recorded "
        f"online_codebook_mask={got!r}, expected {want_online!r}"
    )
    got_active = args.get("online_codebook_mask_active")
    want_active = want_online and args.get("eval_mode") == "compressed"
    assert got_active is want_active, (
        f"online-mask audit FAILED: {path} recorded "
        f"online_codebook_mask_active={got_active!r}, expected {want_active!r}"
    )
if want_two_axis_raw:
    want_two_axis = want_two_axis_raw != "0"
    got = args.get("two_axis_rope")
    assert got is want_two_axis, (
        f"two-axis RoPE audit FAILED: {path} recorded "
        f"two_axis_rope={got!r}, expected {want_two_axis!r}"
    )
if want_gated_raw:
    want_gated = want_gated_raw != "0"
    got = args.get("gated_compressed_rope")
    assert got is want_gated, (
        f"gated-RoPE audit FAILED: {path} recorded "
        f"gated_compressed_rope={got!r}, expected {want_gated!r}"
    )
    got_start = args.get("gated_rope_start_layer")
    want_start = int(want_start_raw)
    assert got_start == want_start, (
        f"gated-RoPE audit FAILED: {path} recorded "
        f"gated_rope_start_layer={got_start!r}, expected {want_start!r}"
    )
    got_pair = args.get("gated_rope_start_pair")
    want_pair = int(want_pair_raw)
    assert got_pair == want_pair, (
        f"gated-RoPE audit FAILED: {path} recorded "
        f"gated_rope_start_pair={got_pair!r}, expected {want_pair!r}"
    )
print(f"[pipeline] eval-flags audit OK: {path}")
PY
}

# ---------- phase 2: smoke evals on intermediate checkpoints ----------
if [ "$SKIP_SMOKE" = "1" ]; then
    echo "=== phase 2/4: smoke evals skipped (SKIP_SMOKE=1) ==="
else
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
                ${EVAL_MODE:+--eval_mode "$EVAL_MODE"} \
                ${DISABLE_DIGIT_IDS:+--disable_digit_ids} \
                --no_wandb --resume_wandb_id none \
                --output_path "$SMOKE_JSON"
            audit_eval_mode "$SMOKE_JSON"
            if [ "$WANDB" != "0" ]; then
                python scripts/log_results_to_wandb.py \
                    --json "$SMOKE_JSON" \
                    --resume_id "$WANDB_ID" --entity "$WANDB_ENTITY" \
                    --project "$WANDB_PROJECT" \
                    --prefix smoke --step "$step"
            fi
        else
            echo "--- smoke eval: step_$step skipped (no checkpoint) ---"
        fi
        step=$((step + SMOKE_EVERY))
    done
fi

# ---------- phase 3: full eval on the final checkpoint ----------
echo "=== phase 3/4: full eval on $FINAL_CKPT ==="
MC_JSON="$EVAL_LOG_DIR/results_${RUN_NAME}_step${STEPS}_default_${TS}.json"
MC_LOG="$EVAL_LOG_DIR/eval_${RUN_NAME}_step${STEPS}_default_${TS}.log"
python scripts/eval_harness.py \
    --ckpt_dir "$FINAL_CKPT" \
    --tokenizer "$TOKENIZER" \
    --preset default \
    "${EVAL_WANDB_ARGS[@]}" \
    ${FINAL_LIMIT:+--limit "$FINAL_LIMIT"} \
    ${EVAL_MODE:+--eval_mode "$EVAL_MODE"} \
    ${DISABLE_DIGIT_IDS:+--disable_digit_ids} \
    --output_path "$MC_JSON" 2>&1 | tee "$MC_LOG"
audit_eval_mode "$MC_JSON"

# With W&B enabled, also log under final/ so every final metric sits in one
# consistent section next to final/wikitext/*. Offline runs keep the same JSON.
if [ "$WANDB" != "0" ]; then
    python scripts/log_results_to_wandb.py \
        --json "$MC_JSON" \
        --resume_id "$WANDB_ID" --entity "$WANDB_ENTITY" \
        --project "$WANDB_PROJECT" \
        --prefix final --step "$STEPS"
fi

PPL_JSON=""
if [ "$SKIP_FULL_PPL" = "0" ]; then
    PPL_JSON="$EVAL_LOG_DIR/results_${RUN_NAME}_step${STEPS}_wikitext_${TS}.json"
    python scripts/eval_harness.py \
        --ckpt_dir "$FINAL_CKPT" \
        --tokenizer "$TOKENIZER" \
        --preset perplexity \
        --tasks wikitext \
        ${FINAL_LIMIT:+--limit "$FINAL_LIMIT"} \
        ${EVAL_MODE:+--eval_mode "$EVAL_MODE"} \
        ${DISABLE_DIGIT_IDS:+--disable_digit_ids} \
        --no_wandb --resume_wandb_id none \
        --output_path "$PPL_JSON"
    audit_eval_mode "$PPL_JSON"
    if [ "$WANDB" != "0" ]; then
        python scripts/log_results_to_wandb.py \
            --json "$PPL_JSON" \
            --resume_id "$WANDB_ID" --entity "$WANDB_ENTITY" \
            --project "$WANDB_PROJECT" \
            --prefix final --step "$STEPS"
    fi
fi

# ---------- phase 4: report/pin artifacts ----------
if [ "$WANDB" = "0" ]; then
    echo "=== phase 4/4: offline artifact summary ==="
    echo "  final checkpoint: $FINAL_CKPT"
    echo "  pipeline log:     $PIPELINE_LOG"
    echo "  training log:     $Z2Z_SCRATCH/logs/train/finetune_${RUN_NAME}_*.log"
    echo "  final MC JSON:    $MC_JSON"
    echo "  final MC log:     $MC_LOG"
    if [ -n "$PPL_JSON" ]; then
        echo "  WikiText JSON:    $PPL_JSON"
    fi
    echo "  smoke JSONs:      $EVAL_LOG_DIR/results_${RUN_NAME}_step*_smoke_${TS}.json"
    echo "=== pipeline complete offline: $RUN_NAME ==="
else
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
[pending] $PPL_INTRO
paste in a shell where WANDB_API_KEY is set — the results land in THIS W&B run
under final/, and this block gets replaced by the result paths once the job runs:

runai submit --name $PPL_JOB_NAME \\
  --image ghcr.io/jkminder/dlab-runai-images/pytorch:master \\
  --gpu 1 --cpu 8 --memory 64Gi \\
  --pvc dlab-scratch:/mnt --large-shm --node-pools default \\
  --environment CKPT_DIR='$FINAL_CKPT' \\
  --environment TOKENIZER=$TOKENIZER \\
  --environment PRESET=perplexity \\
  --environment TASKS=$PPL_TASKS \\${DISABLE_DIGIT_IDS:+
  --environment DISABLE_DIGIT_IDS=1 \\}${EVAL_MODE:+
  --environment EVAL_MODE=$EVAL_MODE \\}
  --environment RESUME_WANDB_ID=$WANDB_ID \\
  --environment WANDB_STEP=$STEPS \\
  --environment WANDB_ENTITY=$WANDB_ENTITY \\
  --environment WANDB_PROJECT=$WANDB_PROJECT \\
  --environment WANDB_API_KEY=\$WANDB_API_KEY \\
  -- bash $PROJECT_DIR/scripts/eval_ckpt_rcp.sh
[/pending]"
python scripts/log_results_to_wandb.py \
    --resume_id "$WANDB_ID" --entity "$WANDB_ENTITY" \
    --project "$WANDB_PROJECT" \
    --append_notes "$NOTES"

echo "=== pipeline complete: W&B run $WANDB_ENTITY/$WANDB_PROJECT/$WANDB_ID ($RUN_NAME) ==="
fi
} 2>&1 | tee "$PIPELINE_LOG"
