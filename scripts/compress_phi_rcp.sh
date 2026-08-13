#!/bin/bash
# Train ONE Phi-shaped model on the LZW transducer task (--mode compress) on the
# RCP cluster (Run:AI, single node). This is the Phi counterpart of the Llama
# sweep in scripts/sweep_finemath_merge_size_compress.sh, which produced the
# compression loss/accuracy grokking figure (plot/plot_compress_loss_grokking.py).
#
# One job = one model size x one DIRECTION. The two directions are trained as
# SEPARATE models (DIRECTION=compress / decompress), so each run's `loss` is that
# direction's loss with nothing blended in; the sweep is therefore 4 sizes x 2
# directions = 8 jobs. DIRECTION=both is the old single-model 50/50 mix, kept
# only to reproduce the published Llama figure.
#
# Defaults reproduce the Llama sweep's hyperparameters EXACTLY -- everything the
# old script left at train.py defaults is left at train.py defaults here too
# (seq_len 4096, lr 3e-4 -> 3e-5, warmup 500, wd 0.1, beta2 0.95, 6000 steps,
# 262,144 tokens/step) so the only variables are the tokenizer and the model
# shape. Do not "improve" them without also re-running the Llama baseline.
#
# ── Usage (Run:AI) ──────────────────────────────────────────────────────
#
#   # Smoke test (1 GPU, 20 steps, local Phi-tokenized SFT shards, no W&B):
#   runai submit --name compress-phi-smoke \
#     --image ghcr.io/jkminder/dlab-runai-images/pytorch:master \
#     --gpu 1 --cpu 16 --memory 128Gi \
#     --pvc dlab-scratch:/mnt --large-shm --node-pools h100 \
#     --environment MODEL_CONFIG=Phi-20M --environment DIRECTION=compress \
#     --environment STEPS=20 \
#     --environment DATA_DIR=/dlabscratch1/xinma/datasets/phi-1B-sft-8shards-eosfix \
#     --environment SEQ_LEN=512 --environment LOCAL_BATCH_SIZE=2 \
#     --environment GRAD_ACCUM=1 \
#     --environment DEBUG_FIRST_STEPS=2 --environment WANDB=0 \
#     -- "bash /dlabscratch1/xinma/code/zip2zip-core/scripts/compress_phi_rcp.sh"
#
#   # One sweep point (4 GPUs, full 6000 steps). Submit once per direction:
#   runai submit --name compress-phi-150m-c \
#     --image ghcr.io/jkminder/dlab-runai-images/pytorch:master \
#     --gpu 4 --cpu 32 --memory 256Gi \
#     --pvc dlab-scratch:/mnt --large-shm --node-pools h100 \
#     --environment MODEL_CONFIG=Phi-150M --environment DIRECTION=compress \
#     --environment WANDB_API_KEY=<your-key> \
#     --environment WANDB_ENTITY=<your-personal-wandb-username> \
#     -- "bash /dlabscratch1/xinma/code/zip2zip-core/scripts/compress_phi_rcp.sh"
#
#   # The pretrained-init control (3.8B, the only Phi size with released
#   # weights). NOT a sweep point -- it answers a different question, so it
#   # belongs in its own panel:
#   runai submit --name compress-phi35-pretrained \
#     ... --gpu 4 ... \
#     --environment MODEL_CONFIG=Phi3.5-mini \
#     --environment INIT_FROM_HF=microsoft/Phi-3.5-mini-instruct \
#     --environment LR=3e-5 --environment MIN_LR=3e-6 \
#     -- "bash /dlabscratch1/xinma/code/zip2zip-core/scripts/compress_phi_rcp.sh"
#   Two caveats for that one: (a) a pretrained decoder wants a smaller LR than a
#   from-scratch run, hence the override above; (b) the Phi3.5-mini config uses
#   unscaled RoPE while the official checkpoint was trained with LongRoPE
#   short/long factors (see configs.py) -- a pre-existing baseline discrepancy
#   that must be stated in the figure caption, not silently inherited.
#
# NOTE (DLAB image): pass the command string directly after `--` (the image
# entrypoint wraps it in bash -c itself; adding your own `bash -c` silently no-ops).
#
set -euo pipefail

Z2Z_SCRATCH=${Z2Z_SCRATCH:-/dlabscratch1/xinma}
export HF_HOME=${HF_HOME:-$Z2Z_SCRATCH/.cache/huggingface}
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PROJECT_DIR=${PROJECT_DIR:-$Z2Z_SCRATCH/code/zip2zip-core}
LOG_DIR=$Z2Z_SCRATCH/logs/train
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# ---------- venv (uv-managed, same bootstrap as the other RCP scripts) ----------
cd "$PROJECT_DIR"
if [ ! -x .venv/bin/python ]; then
    if ! command -v uv >/dev/null 2>&1; then
        echo "[compress_phi] Installing uv..."
        pip install --quiet --user uv
        export PATH="$HOME/.local/bin:$PATH"
    fi
    echo "[compress_phi] Creating venv with uv sync (torch nightly cu130)..."
    uv sync --frozen
fi
PYTHON=$PROJECT_DIR/.venv/bin/python

# ---------- config ----------
NUM_GPUS=${NUM_GPUS:-$(nvidia-smi -L | wc -l | tr -d ' ')}

MODEL_CONFIG=${MODEL_CONFIG:-Phi-20M}
TOKENIZER=${TOKENIZER:-microsoft/Phi-3.5-mini-instruct}
# Phi-tokenized finemath, produced by scripts/pretokenize_finemath_phi_rcp.sh.
DATA_DIR=${DATA_DIR:-$Z2Z_SCRATCH/datasets/finemath-10bt-phi35}
OUTPUT_BASE=${OUTPUT_BASE:-$Z2Z_SCRATCH/zip2zip-outputs}

# From scratch by default: the sweep's whole point is that the LZW algorithm has
# to be learned, not transferred. Set INIT_FROM_HF for the pretrained control.
INIT_FROM_HF=${INIT_FROM_HF:-}
INIT_TAG=scratch
INIT_FLAG=""
if [ -n "$INIT_FROM_HF" ]; then
    INIT_FLAG="--init_from_hf $INIT_FROM_HF"
    INIT_TAG=pretrained
fi

# One model per direction: DIRECTION=compress learns base -> compressed,
# DIRECTION=decompress learns compressed -> base. Each run's `loss` is then that
# direction's loss outright, with nothing blended into it. DIRECTION=both is the
# historical 50/50 mix and exists only to reproduce the published Llama sweep.
DIRECTION=${DIRECTION:-compress}
case "$DIRECTION" in
    compress|decompress|both) ;;
    *) echo "FATAL: DIRECTION must be compress, decompress, or both (got '$DIRECTION')"; exit 1 ;;
esac

MAX_SUBTOKENS=${MAX_SUBTOKENS:-4}
MAX_CODEBOOK_SIZE=${MAX_CODEBOOK_SIZE:-4096}
MAX_ACTIVE_CODEBOOK_SIZE=${MAX_ACTIVE_CODEBOOK_SIZE:-4096}

STEPS=${STEPS:-6000}
SEED=${SEED:-42}
SEQ_LEN=${SEQ_LEN:-4096}
LOCAL_BATCH_SIZE=${LOCAL_BATCH_SIZE:-8}
# The Llama sweep ran 4 GPUs x bs 8 x 4096 x accum 2 = 262,144 tokens/step.
# Derive accum from the actual GPU count / batch size so that budget survives a
# different node shape (Phi-1B may need LOCAL_BATCH_SIZE=2 to fit) instead of
# silently changing the recipe.
TARGET_TOKENS_PER_STEP=${TARGET_TOKENS_PER_STEP:-262144}
GRAD_ACCUM=${GRAD_ACCUM:-$(( TARGET_TOKENS_PER_STEP / (LOCAL_BATCH_SIZE * SEQ_LEN * NUM_GPUS) ))}
[ "$GRAD_ACCUM" -lt 1 ] && GRAD_ACCUM=1
TOKENS_PER_STEP=$(( LOCAL_BATCH_SIZE * SEQ_LEN * NUM_GPUS * GRAD_ACCUM ))
# train.py's loop ignores --steps once --max_tokens is set, so keep them in sync.
MAX_TOKENS=${MAX_TOKENS:-$(( STEPS * TOKENS_PER_STEP ))}
NUM_WORKERS=${NUM_WORKERS:-1}

# train.py defaults, spelled out because this recipe must stay comparable to the
# Llama sweep that produced the published figure.
LR=${LR:-3e-4}
MIN_LR=${MIN_LR:-3e-5}
WARMUP_STEPS=${WARMUP_STEPS:-500}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.1}
ADAM_BETA2=${ADAM_BETA2:-0.95}
SAVE_FREQ=${SAVE_FREQ:-1000}
LOG_FREQ=${LOG_FREQ:-10}
DEBUG_FIRST_STEPS=${DEBUG_FIRST_STEPS:-0}
ACTIVATION_CHECKPOINT=${ACTIVATION_CHECKPOINT:-}

RUN_NAME=${RUN_NAME:-zip2zip_${MODEL_CONFIG}_${DIRECTION}_ms${MAX_SUBTOKENS}_${INIT_TAG}}
OUTPUT_DIR=${OUTPUT_DIR:-${OUTPUT_BASE}/${RUN_NAME}}

WANDB=${WANDB:-1}
# Its own project, NOT the zip2zip-core / llaza one. A transducer run's `loss`
# is the LZW objective, not language-model NLL, and its `compression` column is a
# property of the sample rather than of the model: dropping these runs into the
# LM project puts two incomparable meanings on the same panel. The Llama sweep
# that produced the published figure did the same (it logged to
# "zip2zip-core-GB200"). eval_harness / lm-eval metrics never apply here either.
WANDB_PROJECT=${WANDB_PROJECT:-zip2zip-compression-modeling}
WANDB_RUN_NAME=${WANDB_RUN_NAME:-$RUN_NAME}
WANDB_GROUP=${WANDB_GROUP:-sweep_compress_phi}
# Until epfl-dlab org access is granted, set this to your own personal W&B
# username: train.py otherwise falls back to a hardcoded "epfl-dlab" entity
# (src/zip2zip_core/project.py) and the run fails if you are not a member.
WANDB_ENTITY=${WANDB_ENTITY:-}

RESUME_FROM=${RESUME_FROM:-}

# ---------- optional flags ----------
WANDB_FLAG=""
if [ -n "$WANDB" ] && [ "$WANDB" != "0" ]; then
    WANDB_FLAG="--wandb --wandb_name $WANDB_RUN_NAME --wandb_project $WANDB_PROJECT --wandb_group $WANDB_GROUP"
    WANDB_FLAG="$WANDB_FLAG --wandb_tags $DIRECTION $MODEL_CONFIG ms${MAX_SUBTOKENS} $INIT_TAG phi35-finemath"
    if [ -n "$WANDB_ENTITY" ]; then
        WANDB_FLAG="$WANDB_FLAG --wandb_entity $WANDB_ENTITY"
    else
        echo "WARNING: WANDB_ENTITY is not set — train.py will default to the" \
             "hardcoded 'epfl-dlab' entity and the run will fail unless you're" \
             "already a member. Set WANDB_ENTITY=<your-personal-wandb-username>."
    fi
fi

RESUME_FLAG=""
if [ -n "$RESUME_FROM" ]; then
    RESUME_FLAG="--resume_from $RESUME_FROM"
fi

AC_FLAG=""
if [ -n "$ACTIVATION_CHECKPOINT" ] && [ "$ACTIVATION_CHECKPOINT" != "0" ]; then
    AC_FLAG="--activation_checkpoint"
fi

DEBUG_FLAG=""
if [ "$DEBUG_FIRST_STEPS" -gt 0 ]; then
    DEBUG_FLAG="--debug_first_steps $DEBUG_FIRST_STEPS"
fi

LOGFILE="$LOG_DIR/compress_${RUN_NAME}_${TIMESTAMP}.log"

{
echo "=== zip2zip-core LZW TRANSDUCER sweep point (Phi family, RCP) ==="
echo "MODEL=$MODEL_CONFIG  DIRECTION=$DIRECTION  INIT=${INIT_FROM_HF:-<from scratch>}  TOKENIZER=$TOKENIZER"
echo "MAX_SUBTOKENS=$MAX_SUBTOKENS  CODEBOOK=$MAX_CODEBOOK_SIZE (active $MAX_ACTIVE_CODEBOOK_SIZE)"
echo "NUM_GPUS=$NUM_GPUS  SEQ_LEN=$SEQ_LEN  BS=$LOCAL_BATCH_SIZE  ACCUM=$GRAD_ACCUM  TOKENS/STEP=$TOKENS_PER_STEP"
echo "STEPS=$STEPS  MAX_TOKENS=$MAX_TOKENS  LR=$LR->$MIN_LR warmup=$WARMUP_STEPS wd=$WEIGHT_DECAY beta2=$ADAM_BETA2"
echo "DATA_DIR=$DATA_DIR"
echo "OUTPUT_DIR=$OUTPUT_DIR"
if [ "$TOKENS_PER_STEP" -ne 262144 ]; then
    echo "WARNING: tokens/step != 262144 — not the batch size the Llama sweep used;" \
         "its curves are no longer directly comparable."
fi

# --no_remap_codebook + --hyper_causal_mask are the Llama sweep's flags, not
# incidental: remapping renumbers hyper ids per sample, and the causal mask is
# what stops the model from reading codebook rows that the decoder has not
# installed yet at that position. The transducer task is meaningless without it.
"$PYTHON" -m torch.distributed.run --standalone --nproc_per_node="$NUM_GPUS" \
    -m zip2zip_core.train \
    --data_dir "$DATA_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --model_config "$MODEL_CONFIG" \
    --tokenizer "$TOKENIZER" \
    --mode compress \
    --direction "$DIRECTION" \
    --max_subtokens "$MAX_SUBTOKENS" \
    --max_codebook_size "$MAX_CODEBOOK_SIZE" \
    --max_active_codebook_size "$MAX_ACTIVE_CODEBOOK_SIZE" \
    --no_remap_codebook \
    --hyper_causal_mask \
    --steps "$STEPS" \
    --seed "$SEED" \
    --max_tokens "$MAX_TOKENS" \
    --seq_len "$SEQ_LEN" \
    --local_batch_size "$LOCAL_BATCH_SIZE" \
    --gradient_accumulation_steps "$GRAD_ACCUM" \
    --num_workers "$NUM_WORKERS" \
    --lr "$LR" \
    --min_lr "$MIN_LR" \
    --warmup_steps "$WARMUP_STEPS" \
    --weight_decay "$WEIGHT_DECAY" \
    --adam_beta2 "$ADAM_BETA2" \
    --save_freq "$SAVE_FREQ" \
    --log_freq "$LOG_FREQ" \
    --no_hf_repo \
    $INIT_FLAG $WANDB_FLAG $RESUME_FLAG $AC_FLAG $DEBUG_FLAG

echo "=== done: $RUN_NAME ==="
} 2>&1 | tee "$LOGFILE"
