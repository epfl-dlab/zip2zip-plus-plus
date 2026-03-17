#!/bin/bash
#SBATCH --job-name=z2z-train
#SBATCH --partition=normal
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=64
#SBATCH --time=12:00:00
#SBATCH --environment=/capstor/scratch/cscs/nathanrchn/images/vs:251215/env.toml
#SBATCH --output=%x_%j.out
#SBATCH --account=a0101

set -euo pipefail

# ---------- paths ----------
export SCRATCH=/iopsstor/scratch/cscs/nathanrchn
export HF_HOME=$SCRATCH/.cache/huggingface
PROJECT_DIR=/capstor/scratch/cscs/nathanrchn/zip2zip-core

# Use the uv venv (has torch nightly with varlen_attn + all deps)
export PYTHON=$PROJECT_DIR/.venv/bin/python

export WANDB_API_KEY=<your-wandb-api-key>

# ---------- training config (override via environment) ----------
DATA_DIR=${DATA_DIR:-$SCRATCH/zip2zip-data/fineweb-llama3-tokens}
OUTPUT_DIR=${OUTPUT_DIR:-/users/nathanrchn/capstor_scratch/zip2zip-outputs}
MAX_SUBTOKENS=${MAX_SUBTOKENS:-4}
ENCODER_DIM=${ENCODER_DIM:-512}
ENCODER_INTERMEDIATE_SIZE=${ENCODER_INTERMEDIATE_SIZE:-2048}
MAX_ACTIVE_CODEBOOK_SIZE=${MAX_ACTIVE_CODEBOOK_SIZE:-2048}
STEPS=${STEPS:-19000}
SEQ_LEN=${SEQ_LEN:-4096}
LOCAL_BATCH_SIZE=${LOCAL_BATCH_SIZE:-4}
GRAD_ACCUM=${GRAD_ACCUM:-2}
NUM_WORKERS=${NUM_WORKERS:-4}
LR=${LR:-3e-4}
MIN_LR=${MIN_LR:-3e-5}
WARMUP_STEPS=${WARMUP_STEPS:-500}
SAVE_FREQ=${SAVE_FREQ:-1000}
LOG_FREQ=${LOG_FREQ:-10}
RESUME_FROM=${RESUME_FROM:-}
WANDB=${WANDB:-}
WANDB_PROJECT=${WANDB_PROJECT:-zip2zip-core}
WANDB_RUN_NAME=${WANDB_RUN_NAME:-}

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

echo "=== zip2zip-core training ==="
echo "NODES=$SLURM_NNODES  WORLD_SIZE=$WORLD_SIZE  MAX_SUBTOKENS=$MAX_SUBTOKENS"
echo "OUTPUT_DIR=$OUTPUT_DIR"

RESUME_FLAG=""
if [ -n "$RESUME_FROM" ]; then
    RESUME_FLAG="--resume_from $RESUME_FROM"
fi

WANDB_FLAG=""
if [ -n "$WANDB" ]; then
    WANDB_FLAG="--wandb --wandb_project $WANDB_PROJECT"
    if [ -n "$WANDB_RUN_NAME" ]; then
        WANDB_FLAG="$WANDB_FLAG --wandb_name $WANDB_RUN_NAME"
    fi
fi

srun --export=ALL bash -c '
export LOCAL_RANK=$((SLURM_LOCALID))
export RANK=$((SLURM_NODEID * 4 + SLURM_LOCALID))
export CUDA_VISIBLE_DEVICES=$LOCAL_RANK
export PYTHONUNBUFFERED=1
'"$PYTHON"' -m zip2zip_core.train \
    --data_dir '"$DATA_DIR"' \
    --output_dir '"$OUTPUT_DIR"' \
    --max_subtokens '"$MAX_SUBTOKENS"' \
    --encoder_dim '"$ENCODER_DIM"' \
    --encoder_intermediate_size '"$ENCODER_INTERMEDIATE_SIZE"' \
    --max_active_codebook_size '"$MAX_ACTIVE_CODEBOOK_SIZE"' \
    --steps '"$STEPS"' \
    --seq_len '"$SEQ_LEN"' \
    --local_batch_size '"$LOCAL_BATCH_SIZE"' \
    --gradient_accumulation_steps '"$GRAD_ACCUM"' \
    --num_workers '"$NUM_WORKERS"' \
    --lr '"$LR"' \
    --min_lr '"$MIN_LR"' \
    --warmup_steps '"$WARMUP_STEPS"' \
    --save_freq '"$SAVE_FREQ"' \
    --log_freq '"$LOG_FREQ"' \
    '"$RESUME_FLAG"' \
    '"$WANDB_FLAG"'
'
