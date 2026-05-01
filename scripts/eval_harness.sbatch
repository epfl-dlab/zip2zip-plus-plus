#!/bin/bash
#SBATCH --job-name=z2z-eval-harness
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --time=2:00:00
#SBATCH --environment=/capstor/store/cscs/swissai/a0101/env.toml
#SBATCH --output=%x_%j.out
#SBATCH --account=a0101

# Run lm-eval-harness on a single Zip2Zip checkpoint.
#
# Submit with:
#   sbatch --export=ALL,CHECKPOINT=/path/to/step_6000 scripts/eval_harness.sbatch
#   sbatch --export=ALL,CHECKPOINT=/path/to/ckpt,TASKS=arc_challenge,wikitext,LIMIT=200 \\
#       scripts/eval_harness.sbatch
#   sbatch --export=ALL,CHECKPOINT=/path/to/ckpt,EVAL_MODE=base scripts/eval_harness.sbatch
#
# Generation tasks (gsm8k/humaneval/mbpp/ifeval):
#   sbatch --export=ALL,CHECKPOINT=/path/to/ckpt,TASKS=gsm8k,humaneval,mbpp,ifeval \\
#       scripts/eval_harness.sbatch
# HF_ALLOW_CODE_EVAL=1 is set below so humaneval/mbpp pass@1 metrics run.
# ifeval needs `langdetect` + `nltk` — pip install if missing in the env.
#
# Or as a sub-shell:
#   CHECKPOINT=/path/to/ckpt TASKS=arc_easy LIMIT=10 sbatch \\
#       --export=ALL scripts/eval_harness.sbatch

set -euo pipefail

# ---------- paths ----------
export SCRATCH=/iopsstor/scratch/cscs/nathanrchn
export HF_HOME=$SCRATCH/.cache/huggingface
PROJECT_DIR=/capstor/scratch/cscs/nathanrchn/zip2zip-core
export PYTHON=$PROJECT_DIR/.venv/bin/python

# ---------- eval config (override via environment) ----------
CHECKPOINT=${CHECKPOINT:?Must set CHECKPOINT=/path/to/step_N or .../latest}
TASKS=${TASKS:-arc_challenge,arc_easy,hellaswag,openbookqa,piqa,winogrande,commonsense_qa,medqa_4options,wikitext,zip2zip_pile,zip2zip_mc4,zip2zip_dc4}
TOKENIZER=${TOKENIZER:-meta-llama/Llama-3.1-8B}
BATCH_SIZE=${BATCH_SIZE:-1}
# MAX_LENGTH = max base-token context per scoring call. Defaults to 4096 to
# match the 4k training window (the model's RoPE supports up to 8192, but it
# was only trained at 4k). Old MAX_SEQ_LEN env var still works.
MAX_LENGTH=${MAX_LENGTH:-${MAX_SEQ_LEN:-4096}}
DTYPE=${DTYPE:-bfloat16}
EVAL_MODE=${EVAL_MODE:-compressed}
NUM_FEWSHOT=${NUM_FEWSHOT:-0}
SEED=${SEED:-1234}
LIMIT=${LIMIT:-}
INCLUDE_PATH=${INCLUDE_PATH:-$PROJECT_DIR/scripts/lm_eval_tasks}

# Output: results/<ckpt-parent>_<step>.json next to the checkpoint by default
CKPT_NAME="$(basename "$(dirname "$CHECKPOINT")")_$(basename "$CHECKPOINT")"
OUTPUT=${OUTPUT:-$PROJECT_DIR/results/eval_${CKPT_NAME}_${SLURM_JOB_ID:-local}.json}
mkdir -p "$(dirname "$OUTPUT")"

# ---------- environment for HF datasets / single-GPU torch ----------
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
# Required for lm-eval's `code_eval` metric (humaneval / mbpp pass@1).
export HF_ALLOW_CODE_EVAL=${HF_ALLOW_CODE_EVAL:-1}

LIMIT_ARG=""
if [ -n "$LIMIT" ]; then
    LIMIT_ARG="--limit $LIMIT"
fi

NO_HCM_ARG=""
if [ "${NO_HYPER_CAUSAL_MASK:-0}" = "1" ]; then
    NO_HCM_ARG="--no_hyper_causal_mask"
fi

echo "=== zip2zip-core lm-eval-harness ==="
echo "  CHECKPOINT:   $CHECKPOINT"
echo "  TASKS:        $TASKS"
echo "  EVAL_MODE:    $EVAL_MODE"
echo "  BATCH_SIZE:   $BATCH_SIZE"
echo "  MAX_LENGTH:   $MAX_LENGTH"
echo "  NUM_FEWSHOT:  $NUM_FEWSHOT"
echo "  LIMIT:        ${LIMIT:-<none>}"
echo "  OUTPUT:       $OUTPUT"
echo "  INCLUDE_PATH: $INCLUDE_PATH"
echo "===================================="

srun --export=ALL "$PYTHON" "$PROJECT_DIR/scripts/eval_harness.py" \
    --ckpt_dir "$CHECKPOINT" \
    --tokenizer "$TOKENIZER" \
    --tasks "$TASKS" \
    --num_fewshot "$NUM_FEWSHOT" \
    --batch_size "$BATCH_SIZE" \
    --max_length "$MAX_LENGTH" \
    --device cuda \
    --dtype "$DTYPE" \
    --eval_mode "$EVAL_MODE" \
    --seed "$SEED" \
    --include_path "$INCLUDE_PATH" \
    --output_path "$OUTPUT" \
    $LIMIT_ARG \
    $NO_HCM_ARG
