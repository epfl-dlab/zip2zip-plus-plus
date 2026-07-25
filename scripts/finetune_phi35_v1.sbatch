#!/bin/bash
#SBATCH --job-name=z2z-phi35-sft
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=64
#SBATCH --time=8:00:00
#SBATCH --output=%x_%j.out
#SBATCH --account=a0101

# --environment goes on the srun line (not #SBATCH) so the launcher runs on the
# host and only the task runs in the container — needed for multi-node OFI/NCCL,
# and avoids the slingshot-plugin-in-container crash. Works for 1 node too.
ENV_TOML=/capstor/store/cscs/swissai/a0101/env.toml

set -euo pipefail

# ============================================================================
# Stage 2: SFT (instruction-tune) the from-scratch Phi-3.5-mini zip2zip model
# on the Phi-tokenized zip2zip-1B SFT data.
#   - Resume from a Stage-1 from-scratch checkpoint (RESUME_FROM) + RESET_STEP.
#   - NO --init_from_hf  => we continue OUR from-scratch model, not a HF model.
#   - Same zip2zip component config as Stage 1 (flat, enc_dim 512, ms 3, cb 4096).
#   - Data: produced by scripts/create_sft_dataset_zip2zip1B_phi.py
# ============================================================================

# ---------- paths ----------
export SCRATCH=/iopsstor/scratch/cscs/mxx
export HF_HOME=$SCRATCH/.cache/huggingface
PROJECT_DIR=/capstor/scratch/cscs/mxx/zip2zip-core

# Use the uv venv (has torch nightly with varlen_attn + all deps)
export PYTHON=$PROJECT_DIR/.venv/bin/python

# export WANDB_API_KEY=<your-wandb-api-key>

# ---------- finetuning config (override via environment) ----------
DATA_DIR=${DATA_DIR:-/capstor/store/cscs/swissai/a0101/mxx/zip2zip-data/phi-1B-sft-8shards}
OUTPUT_BASE=${OUTPUT_BASE:-/capstor/scratch/cscs/mxx/zip2zip-outputs}
RUN_NAME=${RUN_NAME:-candidate-Phi35-MS3-flat-sft-1bt}
OUTPUT_DIR=${OUTPUT_DIR:-${OUTPUT_BASE}/${RUN_NAME}}

TOKENIZER=${TOKENIZER:-microsoft/Phi-3.5-mini-instruct}
MODEL_CONFIG=${MODEL_CONFIG:-Phi3.5-mini}
MAX_SUBTOKENS=${MAX_SUBTOKENS:-3}
MAX_ACTIVE_CODEBOOK_SIZE=${MAX_ACTIVE_CODEBOOK_SIZE:-4096}

# Stage-1 checkpoint to continue from (REQUIRED). RESET_STEP restarts the LR schedule.
RESUME_FROM=${RESUME_FROM:-}
RESET_STEP=${RESET_STEP:-1}

MAX_TOKENS=${MAX_TOKENS:-1000000000}
STEPS=${STEPS:-19000}
SEQ_LEN=${SEQ_LEN:-4096}
LOCAL_BATCH_SIZE=${LOCAL_BATCH_SIZE:-1}
GRAD_ACCUM=${GRAD_ACCUM:-8}
NUM_WORKERS=${NUM_WORKERS:-1}

LR=${LR:-5e-5}
MIN_LR=${MIN_LR:-5e-6}
HYPER_LR=${HYPER_LR:-}
MIN_HYPER_LR=${MIN_HYPER_LR:-}
WARMUP_STEPS=${WARMUP_STEPS:-500}
SAVE_FREQ=${SAVE_FREQ:-1000}
LOG_FREQ=${LOG_FREQ:-10}

ENCODER_DIM=${ENCODER_DIM:-512}
ENCODER_INTERMEDIATE_SIZE=${ENCODER_INTERMEDIATE_SIZE:-2048}
HYPER_ENCODER_TYPE=${HYPER_ENCODER_TYPE:-flat}

WANDB=${WANDB:-1}
WANDB_PROJECT=${WANDB_PROJECT:-candidate-Phi35-1B}
WANDB_RUN_NAME=${WANDB_RUN_NAME:-$RUN_NAME}
WANDB_GROUP=${WANDB_GROUP:-}

FREEZE_DECODER=${FREEZE_DECODER:-}
LORA_RANK=${LORA_RANK:-}
LORA_ALPHA=${LORA_ALPHA:-1.0}
NO_ENCODER_RESIDUAL=${NO_ENCODER_RESIDUAL:-}
DEBUG_FIRST_STEPS=${DEBUG_FIRST_STEPS:-0}
# FSDP2 shards params/grads/optimizer (see zip2zip_core.train). Activation
# checkpointing optional; set ACTIVATION_CHECKPOINT=1 for extra headroom.
ACTIVATION_CHECKPOINT=${ACTIVATION_CHECKPOINT:-}

if [ -z "$RESUME_FROM" ]; then
    echo "ERROR: RESUME_FROM is required (path to a Stage-1 from-scratch checkpoint)." >&2
    exit 1
fi

HYPER_LR_FLAG=""
if [ -n "$HYPER_LR" ]; then
    HYPER_LR_FLAG="--hyper_lr $HYPER_LR"
    if [ -n "$MIN_HYPER_LR" ]; then
        HYPER_LR_FLAG="$HYPER_LR_FLAG --min_hyper_lr $MIN_HYPER_LR"
    fi
fi

mkdir -p "$OUTPUT_DIR"

# ---------- NCCL / network ----------
export NCCL_DEBUG=WARN
export NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3
export FI_CXI_DISABLE_HOST_REGISTER=1
export FI_MR_CACHE_MONITOR=userfaultfd
export FI_CXI_DEFAULT_CQ_SIZE=262144
export FI_CXI_DEFAULT_TX_SIZE=256

HEAD_NODE=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export MASTER_ADDR=$HEAD_NODE
export MASTER_PORT=29500
export WORLD_SIZE=$((SLURM_NNODES * 4))

echo "=== zip2zip-core Phi-3.5-mini SFT (stage 2) ==="
echo "NODES=$SLURM_NNODES  WORLD_SIZE=$WORLD_SIZE  MODEL=$MODEL_CONFIG  MAX_SUBTOKENS=$MAX_SUBTOKENS"
echo "RESUME_FROM=$RESUME_FROM  RESET_STEP=$RESET_STEP  TOKENIZER=$TOKENIZER"
echo "DATA_DIR=$DATA_DIR"
echo "OUTPUT_DIR=$OUTPUT_DIR  RUN_NAME=$RUN_NAME  MAX_TOKENS=$MAX_TOKENS"

RESET_STEP_FLAG=""
if [ -n "$RESET_STEP" ]; then
    RESET_STEP_FLAG="--reset_step"
fi

WANDB_FLAG=""
if [ -n "$WANDB" ] && [ "$WANDB" != "0" ]; then
    WANDB_FLAG="--wandb --wandb_project $WANDB_PROJECT --wandb_name $WANDB_RUN_NAME"
    if [ -n "$WANDB_GROUP" ]; then
        WANDB_FLAG="$WANDB_FLAG --wandb_group $WANDB_GROUP"
    fi
fi

FREEZE_DECODER_FLAG=""
if [ -n "$FREEZE_DECODER" ]; then
    FREEZE_DECODER_FLAG="--freeze_decoder"
fi

LORA_FLAG=""
if [ -n "$LORA_RANK" ]; then
    LORA_FLAG="--lora_rank $LORA_RANK --lora_alpha $LORA_ALPHA"
fi

NO_ENCODER_RESIDUAL_FLAG=""
if [ -n "$NO_ENCODER_RESIDUAL" ] && [ "$NO_ENCODER_RESIDUAL" != "0" ]; then
    NO_ENCODER_RESIDUAL_FLAG="--no_encoder_residual"
fi

DEBUG_FIRST_STEPS_FLAG=""
if [ "$DEBUG_FIRST_STEPS" -gt 0 ]; then
    DEBUG_FIRST_STEPS_FLAG="--debug_first_steps $DEBUG_FIRST_STEPS"
fi

AC_FLAG=""
if [ -n "$ACTIVATION_CHECKPOINT" ]; then
    AC_FLAG="--activation_checkpoint"
fi

srun --export=ALL --environment="$ENV_TOML" bash -c '
export LOCAL_RANK=$((SLURM_LOCALID))
export RANK=$((SLURM_NODEID * 4 + SLURM_LOCALID))
export CUDA_VISIBLE_DEVICES=$LOCAL_RANK
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
'"$PYTHON"' -m zip2zip_core.train \
    --data_dir '"$DATA_DIR"' \
    --output_dir '"$OUTPUT_DIR"' \
    --model_config '"$MODEL_CONFIG"' \
    --tokenizer '"$TOKENIZER"' \
    --resume_from '"$RESUME_FROM"' \
    --max_subtokens '"$MAX_SUBTOKENS"' \
    --encoder_dim '"$ENCODER_DIM"' \
    --encoder_intermediate_size '"$ENCODER_INTERMEDIATE_SIZE"' \
    --hyper_encoder_type '"$HYPER_ENCODER_TYPE"' \
    --max_active_codebook_size '"$MAX_ACTIVE_CODEBOOK_SIZE"' \
    --steps '"$STEPS"' \
    --max_tokens '"$MAX_TOKENS"' \
    --seq_len '"$SEQ_LEN"' \
    --local_batch_size '"$LOCAL_BATCH_SIZE"' \
    --gradient_accumulation_steps '"$GRAD_ACCUM"' \
    --num_workers '"$NUM_WORKERS"' \
    --lr '"$LR"' \
    --min_lr '"$MIN_LR"' \
    --warmup_steps '"$WARMUP_STEPS"' \
    --save_freq '"$SAVE_FREQ"' \
    --log_freq '"$LOG_FREQ"' \
    --no_remap_codebook --hyper_causal_mask \
    '"$AC_FLAG"' \
    '"$FREEZE_DECODER_FLAG"' \
    '"$LORA_FLAG"' \
    '"$NO_ENCODER_RESIDUAL_FLAG"' \
    '"$RESET_STEP_FLAG"' \
    '"$DEBUG_FIRST_STEPS_FLAG"' \
    '"$HYPER_LR_FLAG"' \
    '"$WANDB_FLAG"'
'
