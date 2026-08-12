#!/bin/bash
#SBATCH --job-name=z2z-phi35-scratch
#SBATCH --partition=normal
#SBATCH --nodes=32
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=64
#SBATCH --time=8:00:00
#SBATCH --output=%x_%j.out
#SBATCH --account=a0101

# NOTE: --environment is intentionally NOT a #SBATCH directive. Putting the
# container at the batch level makes `srun` itself run inside the container,
# where it cannot load the host Slingshot switch plugin
# (switch_hpe_slingshot.so -> libjson-c.so.5 missing) and dies with
# "Can't find plugin for switch/hpe_slingshot". Instead we pass --environment
# on the srun line below, so the launcher runs on the host and only the task
# runs in the container (which still provides the OFI/libfabric NCCL stack
# needed for multi-node Slingshot networking).
ENV_TOML=/capstor/store/cscs/swissai/a0101/env.toml

set -euo pipefail

# ============================================================================
# From-scratch pretraining of the Phi-3.5-mini architecture (3.8B) with zip2zip.
#   - Base architecture aligned to Phi-3.5-mini (MODEL_CONFIG=Phi3.5-mini):
#     dim 3072, 32 layers, MHA 32/32, ffn 8192, vocab 32064, rope theta 1e4,
#     untied input/output embeddings.
#   - zip2zip components follow the previous Llama from-scratch run (core
#     defaults): flat hyper-encoder, encoder_dim 512, intermediate 2048,
#     max_codebook_size 4096, max_subtokens 3.
#   - Data must be pretokenized with the Phi tokenizer (see scripts/pretokenize.py
#     --model_name microsoft/Phi-3.5-mini-instruct).
#   - NO --init_from_hf  => trained from scratch.
# ============================================================================

# ---------- paths ----------
export SCRATCH=/iopsstor/scratch/cscs/mxx
export HF_HOME=$SCRATCH/.cache/huggingface
PROJECT_DIR=/capstor/scratch/cscs/mxx/zip2zip-core

# Use the uv venv (has torch nightly with varlen_attn + all deps)
export PYTHON=$PROJECT_DIR/.venv/bin/python

# export WANDB_API_KEY=<your-wandb-api-key>

# ---------- training config ----------
# 1M total batch size: local_bs=1 * seq_len=4096 * 128 GPUs * grad_accum=2 = 1,048,576 tokens/step
# 20B tokens / 1M ~= 19,073 steps  (STEPS=20000 covers the full 20B)
# Sizing: the 1B run did ~0.97s/step on 32 GPUs (-> 20B in ~5.1h). The 3.8B
# model is ~3.7x the FLOPs/step, so 128 GPUs (4x) lands ~0.9-1.0s/step -> 20B
# in ~5-6h, comfortably inside an 8h job.
DATA_DIR=${DATA_DIR:-/capstor/store/cscs/swissai/a0101/mxx/zip2zip-data/phi-20B-tokens-512shards}
OUTPUT_DIR=${OUTPUT_DIR:-/capstor/scratch/cscs/mxx/zip2zip-outputs/candidate-Phi35-MS3-flat-20BT}
TOKENIZER=${TOKENIZER:-microsoft/Phi-3.5-mini-instruct}
MODEL_CONFIG=${MODEL_CONFIG:-Phi3.5-mini}
MAX_SUBTOKENS=${MAX_SUBTOKENS:-3}
ENCODER_DIM=${ENCODER_DIM:-512}
ENCODER_INTERMEDIATE_SIZE=${ENCODER_INTERMEDIATE_SIZE:-2048}
HYPER_ENCODER_TYPE=${HYPER_ENCODER_TYPE:-flat}
MAX_ACTIVE_CODEBOOK_SIZE=${MAX_ACTIVE_CODEBOOK_SIZE:-4096}
STEPS=${STEPS:-20000}
MAX_TOKENS=${MAX_TOKENS:-}
SEQ_LEN=${SEQ_LEN:-4096}
LOCAL_BATCH_SIZE=${LOCAL_BATCH_SIZE:-1}
GRAD_ACCUM=${GRAD_ACCUM:-2}
# 20B pretraining data loads fine with 4 workers (matches the previous Llama
# from-scratch run). NOTE: the 1BT SFT data hangs with >1 worker — the SFT
# script (finetune_phi35_v1.sbatch) defaults NUM_WORKERS=1 for that reason.
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
WANDB_PROJECT=${WANDB_PROJECT:-candidate-Phi35-1B}
WANDB_RUN_NAME=${WANDB_RUN_NAME:-}
MODE=${MODE:-lm}
TOKEN_TYPE_LOSS_WEIGHT=${TOKEN_TYPE_LOSS_WEIGHT:-0.0}
NO_ENCODER_RESIDUAL=${NO_ENCODER_RESIDUAL:-}
# FSDP2 shards params/grads/optimizer across all ranks (see zip2zip_core.train),
# so a 3.8B model needs only ~footprint/world_size per GPU and fits comfortably.
# Activation checkpointing is optional with FSDP; enable for extra headroom by
# setting ACTIVATION_CHECKPOINT=1.
ACTIVATION_CHECKPOINT=${ACTIVATION_CHECKPOINT:-}

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

echo "=== zip2zip-core Phi-3.5-mini from-scratch pretraining ==="
echo "NODES=$SLURM_NNODES  WORLD_SIZE=$WORLD_SIZE  MAX_SUBTOKENS=$MAX_SUBTOKENS  MODE=$MODE  MODEL=$MODEL_CONFIG"
echo "TOTAL_BATCH=1M tokens  LOCAL_BS=$LOCAL_BATCH_SIZE  GRAD_ACCUM=$GRAD_ACCUM  STEPS=$STEPS"
echo "HYPER_ENCODER_TYPE=$HYPER_ENCODER_TYPE  TOKENIZER=$TOKENIZER"
echo "DATA_DIR=$DATA_DIR"
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
    '"$AC_FLAG"' \
    '"$RESUME_FLAG"' \
    '"$PUSH_FLAG"' \
    '"$WANDB_FLAG"'
'
