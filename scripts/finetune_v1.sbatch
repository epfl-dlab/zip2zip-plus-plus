#!/bin/bash
#SBATCH --job-name=z2z-finetune-v1
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=64
#SBATCH --time=8:00:00
#SBATCH --output=%x_%j.out
#SBATCH --account=a0101

set -euo pipefail

# ---------- paths ----------
export SCRATCH=/iopsstor/scratch/cscs/mxx
export HF_HOME=$SCRATCH/.cache/huggingface
PROJECT_DIR=/capstor/scratch/cscs/mxx/zip2zip-core

# Use the uv venv (has torch nightly with varlen_attn + all deps)
export PYTHON=$PROJECT_DIR/.venv/bin/python

# export WANDB_API_KEY=<your-wandb-api-key>

# ---------- finetuning config (override via environment) ----------
DATA_DIR=${DATA_DIR:-/capstor/scratch/cscs/mxx/zip2zip-data/llaza-1B-tokens-8shards}
OUTPUT_BASE=${OUTPUT_BASE:-$SCRATCH/zip2zip-outputs}
RUN_NAME=${RUN_NAME:-llaza_1b_llama32_instruct_ms2_1bt_v1}
OUTPUT_DIR=${OUTPUT_DIR:-${OUTPUT_BASE}/${RUN_NAME}}

INIT_FROM_HF=${INIT_FROM_HF:-meta-llama/Llama-3.2-1B-Instruct}
MODEL_CONFIG=${MODEL_CONFIG:-1B}
MAX_SUBTOKENS=${MAX_SUBTOKENS:-2}
MAX_ACTIVE_CODEBOOK_SIZE=${MAX_ACTIVE_CODEBOOK_SIZE:-4096}

MAX_TOKENS=${MAX_TOKENS:-1000000000}
STEPS=${STEPS:-19000}
SEQ_LEN=${SEQ_LEN:-4096}
LOCAL_BATCH_SIZE=${LOCAL_BATCH_SIZE:-1}
GRAD_ACCUM=${GRAD_ACCUM:-8}
NUM_WORKERS=${NUM_WORKERS:-4}

LR=${LR:-5e-5}
MIN_LR=${MIN_LR:-5e-6}
WARMUP_STEPS=${WARMUP_STEPS:-500}
SAVE_FREQ=${SAVE_FREQ:-1000}
LOG_FREQ=${LOG_FREQ:-10}

ENCODER_DIM=${ENCODER_DIM:-512}
ENCODER_INTERMEDIATE_SIZE=${ENCODER_INTERMEDIATE_SIZE:-2048}
HYPER_ENCODER_TYPE=${HYPER_ENCODER_TYPE:-flat}

WANDB=${WANDB:-1}
WANDB_PROJECT=${WANDB_PROJECT:-zip2zip-core}
WANDB_RUN_NAME=${WANDB_RUN_NAME:-$RUN_NAME}
WANDB_GROUP=${WANDB_GROUP:-}

FREEZE_DECODER=${FREEZE_DECODER:-}
LORA_RANK=${LORA_RANK:-}
LORA_ALPHA=${LORA_ALPHA:-1.0}

RESUME_FROM=${RESUME_FROM:-}
RESET_STEP=${RESET_STEP:-}
NO_ENCODER_RESIDUAL=${NO_ENCODER_RESIDUAL:-}
DEBUG_FIRST_STEPS=${DEBUG_FIRST_STEPS:-0}

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

echo "=== zip2zip-core finetune v1 ==="
echo "NODES=$SLURM_NNODES  WORLD_SIZE=$WORLD_SIZE  MODEL=$MODEL_CONFIG  MAX_SUBTOKENS=$MAX_SUBTOKENS"
echo "INIT_FROM_HF=$INIT_FROM_HF"
echo "DATA_DIR=$DATA_DIR"
echo "OUTPUT_DIR=$OUTPUT_DIR"
echo "RUN_NAME=$RUN_NAME"
echo "MAX_TOKENS=$MAX_TOKENS"

RESUME_FLAG=""
if [ -n "$RESUME_FROM" ]; then
    RESUME_FLAG="--resume_from $RESUME_FROM"
fi

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
if [ -n "$NO_ENCODER_RESIDUAL" ]; then
    NO_ENCODER_RESIDUAL_FLAG="--no_encoder_residual"
fi

DEBUG_FIRST_STEPS_FLAG=""
if [ "$DEBUG_FIRST_STEPS" -gt 0 ]; then
    DEBUG_FIRST_STEPS_FLAG="--debug_first_steps $DEBUG_FIRST_STEPS"
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
    --init_from_hf '"$INIT_FROM_HF"' \
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
    '"$FREEZE_DECODER_FLAG"' \
    '"$LORA_FLAG"' \
    '"$NO_ENCODER_RESIDUAL_FLAG"' \
    '"$RESUME_FLAG"' \
    '"$RESET_STEP_FLAG"' \
    '"$DEBUG_FIRST_STEPS_FLAG"' \
    '"$WANDB_FLAG"'
'
