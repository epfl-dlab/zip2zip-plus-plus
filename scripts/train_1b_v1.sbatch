#!/bin/bash
#SBATCH --job-name=z2z-1b-v1
#SBATCH --partition=normal
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=64
#SBATCH --time=8:00:00
#SBATCH --environment=/capstor/store/cscs/swissai/a0101/env.toml
#SBATCH --output=%x_%j.out
#SBATCH --account=a0101

set -euo pipefail

# ---------- paths ----------
# Nathan defaults:
# export SCRATCH=/iopsstor/scratch/cscs/nathanrchn
# PROJECT_DIR=/capstor/scratch/cscs/nathanrchn/zip2zip-core
# OUTPUT_DIR default: /users/nathanrchn/capstor_scratch/zip2zip-outputs/zip2zip-1b-v1
export SCRATCH=/iopsstor/scratch/cscs/mxx
export HF_HOME=$SCRATCH/.cache/huggingface
PROJECT_DIR=/capstor/scratch/cscs/mxx/zip2zip-core

# Use the uv venv (has torch nightly with varlen_attn + all deps)
export PYTHON=$PROJECT_DIR/.venv/bin/python

# export WANDB_API_KEY=<your-wandb-api-key>

# 3 variants to run:
# 1) no zip2zip, max_subtokens=1 (baseline)
# 2) zip2zip, max_subtokens=4
# 3) zip2zip nested merge, max_subtokens=4

# ---------- training config ----------
# 1M total batch size: local_bs=8 * seq_len=4096 * 16 GPUs * grad_accum=2 = 1,048,576 tokens/step
# 15B tokens / 1M = 15,000 steps
DATA_DIR=${DATA_DIR:-/capstor/store/cscs/swissai/a0101/mxx/zip2zip-data/llaza-20B-tokens-128shards}
OUTPUT_DIR=${OUTPUT_DIR:-$SCRATCH/zip2zip-outputs/zip2zip-1b-v1}
MAX_SUBTOKENS=${MAX_SUBTOKENS:-4}
ENCODER_DIM=${ENCODER_DIM:-512}
ENCODER_INTERMEDIATE_SIZE=${ENCODER_INTERMEDIATE_SIZE:-2048}
HYPER_ENCODER_TYPE=${HYPER_ENCODER_TYPE:-flat}
MAX_ACTIVE_CODEBOOK_SIZE=${MAX_ACTIVE_CODEBOOK_SIZE:-4096}
STEPS=${STEPS:-20000}
MAX_TOKENS=${MAX_TOKENS:-}
SEQ_LEN=${SEQ_LEN:-4096}
LOCAL_BATCH_SIZE=${LOCAL_BATCH_SIZE:-2}
GRAD_ACCUM=${GRAD_ACCUM:-4}
NUM_WORKERS=${NUM_WORKERS:-4}
LR=${LR:-3e-4}
MIN_LR=${MIN_LR:-3e-5}
WARMUP_STEPS=${WARMUP_STEPS:-500}
SAVE_FREQ=${SAVE_FREQ:-1000}
LOG_FREQ=${LOG_FREQ:-10}
RESUME_FROM=${RESUME_FROM:-}
RESUME_FROM_HF=${RESUME_FROM_HF:-}
RESUME_HF_REVISION=${RESUME_HF_REVISION:-}
PUSH_TO_HUB=${PUSH_TO_HUB:-}
WANDB=${WANDB:-}
WANDB_PROJECT=${WANDB_PROJECT:-zip2zip-core}
WANDB_RUN_NAME=${WANDB_RUN_NAME:-}
MODE=${MODE:-lm}
MODEL_CONFIG=${MODEL_CONFIG:-1B}
TOKEN_TYPE_LOSS_WEIGHT=${TOKEN_TYPE_LOSS_WEIGHT:-0.0}
NO_ENCODER_RESIDUAL=${NO_ENCODER_RESIDUAL:-}

mkdir -p "$OUTPUT_DIR"

# ---------- NCCL / network ----------
export NCCL_DEBUG=WARN
export NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3
export FI_CXI_DISABLE_HOST_REGISTER=1
export FI_MR_CACHE_MONITOR=userfaultfd
export FI_CXI_DEFAULT_CQ_SIZE=262144
export FI_CXI_DEFAULT_TX_SIZE=256

HEAD_NODE=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n1)
export MASTER_ADDR=$HEAD_NODE
export MASTER_PORT=29500
export WORLD_SIZE=$((SLURM_NNODES * 4))

echo "=== zip2zip-core 1B v1 training ==="
echo "NODES=$SLURM_NNODES  WORLD_SIZE=$WORLD_SIZE  MAX_SUBTOKENS=$MAX_SUBTOKENS  MODE=$MODE  MODEL=$MODEL_CONFIG"
echo "TOTAL_BATCH=1M tokens  LOCAL_BS=$LOCAL_BATCH_SIZE  GRAD_ACCUM=$GRAD_ACCUM  STEPS=$STEPS"
echo "HYPER_ENCODER_TYPE=$HYPER_ENCODER_TYPE"
echo "OUTPUT_DIR=$OUTPUT_DIR"

RESUME_FLAG=""
if [ -n "$RESUME_FROM" ]; then
    RESUME_FLAG="--resume_from $RESUME_FROM"
fi
if [ -n "$RESUME_FROM_HF" ]; then
    RESUME_FLAG="--resume_from_hf $RESUME_FROM_HF"
    if [ -n "$RESUME_HF_REVISION" ]; then
        RESUME_FLAG="$RESUME_FLAG --resume_hf_revision $RESUME_HF_REVISION"
    fi
fi

PUSH_FLAG=""
if [ -n "$PUSH_TO_HUB" ]; then
    PUSH_FLAG="--push_to_hub $PUSH_TO_HUB"
fi

WANDB_FLAG=""
if [ -n "$WANDB" ]; then
    WANDB_FLAG="--wandb --wandb_project $WANDB_PROJECT"
    if [ -n "$WANDB_RUN_NAME" ]; then
        WANDB_FLAG="$WANDB_FLAG --wandb_name $WANDB_RUN_NAME"
    fi
fi

MAX_TOKENS_FLAG=""
if [ -n "$MAX_TOKENS" ]; then
    MAX_TOKENS_FLAG="--max_tokens $MAX_TOKENS"
fi

NO_ENCODER_RESIDUAL_FLAG=""
if [ -n "$NO_ENCODER_RESIDUAL" ] && [ "$NO_ENCODER_RESIDUAL" != "0" ]; then
    NO_ENCODER_RESIDUAL_FLAG="--no_encoder_residual"
fi

srun --export=ALL bash -c '
export LOCAL_RANK=$((SLURM_LOCALID))
export RANK=$((SLURM_NODEID * 4 + SLURM_LOCALID))
export CUDA_VISIBLE_DEVICES=$LOCAL_RANK
export PYTHONUNBUFFERED=1
'"$PYTHON"' -m zip2zip_core.train \
    --data_dir '"$DATA_DIR"' \
    --output_dir '"$OUTPUT_DIR"' \
    --model_config '"$MODEL_CONFIG"' \
    --max_subtokens '"$MAX_SUBTOKENS"' \
    --encoder_dim '"$ENCODER_DIM"' \
    --encoder_intermediate_size '"$ENCODER_INTERMEDIATE_SIZE"' \
    --hyper_encoder_type '"$HYPER_ENCODER_TYPE"' \
    --max_active_codebook_size '"$MAX_ACTIVE_CODEBOOK_SIZE"' \
    --steps '"$STEPS"' \
    '"$MAX_TOKENS_FLAG"' \
    --seq_len '"$SEQ_LEN"' \
    --local_batch_size '"$LOCAL_BATCH_SIZE"' \
    --gradient_accumulation_steps '"$GRAD_ACCUM"' \
    --num_workers '"$NUM_WORKERS"' \
    --lr '"$LR"' \
    --min_lr '"$MIN_LR"' \
    --warmup_steps '"$WARMUP_STEPS"' \
    --save_freq '"$SAVE_FREQ"' \
    --log_freq '"$LOG_FREQ"' \
    --mode '"$MODE"' \
    --token_type_loss_weight '"$TOKEN_TYPE_LOSS_WEIGHT"' \
    --no_remap_codebook --hyper_causal_mask \
    '"$NO_ENCODER_RESIDUAL_FLAG"' \
    '"$RESUME_FLAG"' \
    '"$PUSH_FLAG"' \
    '"$WANDB_FLAG"'
'
