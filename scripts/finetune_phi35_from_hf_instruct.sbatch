#!/bin/bash
#SBATCH --job-name=z2z-phi35-fthf
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
# Reproduce the RELEASED zip2zip-Phi-3.5-mini-instruct recipe in zip2zip-core:
# zip2zip-FINETUNE from the *pretrained* microsoft/Phi-3.5-mini-instruct
# (NOT from-scratch). Base decoder FROZEN + LoRA; hyper-encoder trained fresh.
#
#   - --init_from_hf microsoft/Phi-3.5-mini-instruct  (Phi-3 fused qkv/gate_up
#     are auto-split to Llama layout in train.load_hf_pretrained)
#   - --freeze_decoder --lora_rank 32 --lora_alpha 32  (matches adapter_config)
#   - max_subtokens=3, active codebook=2048, encoder dim 3072 / 2 layers / 32 heads
#     (≈ released transformer encoder; zip2zip-core uses a single encoder, so it
#      is NOT bit-identical to released's separate in/out encoders — accepted)
#   - LR 3e-4 -> 1e-5 cosine; data = Phi-tokenized zip2zip-1B SFT shards
#
# Target reference (released zip2zip_config.json / adapter_config.json):
#   initial_vocab_size 32011 / vocab 32064 — zip2zip-core keeps 32064 (its hyper
#   boundary is config.vocab_size); disabled_ids auto-derive to the released
#   [0,1,2,32000..32010] set, so compression behavior matches.
# ============================================================================

# ---------- paths ----------
export SCRATCH=/iopsstor/scratch/cscs/mxx
export HF_HOME=$SCRATCH/.cache/huggingface
PROJECT_DIR=/capstor/scratch/cscs/mxx/zip2zip-core

# Use the uv venv (has torch nightly with varlen_attn + all deps)
export PYTHON=$PROJECT_DIR/.venv/bin/python

# export WANDB_API_KEY=<your-wandb-api-key>

# ---------- config (override via environment) ----------
DATA_DIR=${DATA_DIR:-/capstor/store/cscs/swissai/a0101/mxx/zip2zip-data/phi-1B-sft-8shards}
OUTPUT_BASE=${OUTPUT_BASE:-/capstor/scratch/cscs/mxx/zip2zip-outputs}
RUN_NAME=${RUN_NAME:-candidate-Phi35-MS3-ft-from-hf-instruct}
OUTPUT_DIR=${OUTPUT_DIR:-${OUTPUT_BASE}/${RUN_NAME}}

INIT_FROM_HF=${INIT_FROM_HF:-microsoft/Phi-3.5-mini-instruct}
TOKENIZER=${TOKENIZER:-microsoft/Phi-3.5-mini-instruct}
MODEL_CONFIG=${MODEL_CONFIG:-Phi3.5-mini}
MAX_SUBTOKENS=${MAX_SUBTOKENS:-3}
MAX_ACTIVE_CODEBOOK_SIZE=${MAX_ACTIVE_CODEBOOK_SIZE:-2048}

# token-budget training (released used ~260M tok; default to a full 1B pass and
# rely on intermediate checkpoints + eval to pick the best). Override as needed.
MAX_TOKENS=${MAX_TOKENS:-1000000000}
STEPS=${STEPS:-19000}
SEQ_LEN=${SEQ_LEN:-4096}
LOCAL_BATCH_SIZE=${LOCAL_BATCH_SIZE:-1}
GRAD_ACCUM=${GRAD_ACCUM:-8}
# 1BT datasets hang with the default 4 dataloader workers — keep at 1.
NUM_WORKERS=${NUM_WORKERS:-1}

# Released recipe: cosine 3e-4 -> 1e-5.
LR=${LR:-3e-4}
MIN_LR=${MIN_LR:-1e-5}
# Optional differential LR for the from-scratch hyper-encoder (leave empty = use LR).
HYPER_LR=${HYPER_LR:-}
MIN_HYPER_LR=${MIN_HYPER_LR:-}
WARMUP_STEPS=${WARMUP_STEPS:-500}
SAVE_FREQ=${SAVE_FREQ:-500}
LOG_FREQ=${LOG_FREQ:-10}

# Hyper-encoder ≈ released transformer: hidden 3072 (= model dim, so no in/out
# projection), 2 layers, 32 heads, intermediate 4*dim=12288.
ENCODER_DIM=${ENCODER_DIM:-3072}
ENCODER_N_LAYERS=${ENCODER_N_LAYERS:-2}
ENCODER_N_HEADS=${ENCODER_N_HEADS:-32}
ENCODER_INTERMEDIATE_SIZE=${ENCODER_INTERMEDIATE_SIZE:-12288}
HYPER_ENCODER_TYPE=${HYPER_ENCODER_TYPE:-flat}

# Frozen base + LoRA (matches released adapter_config: r=32, alpha=32).
FREEZE_DECODER=${FREEZE_DECODER:-1}
LORA_RANK=${LORA_RANK:-32}
LORA_ALPHA=${LORA_ALPHA:-32}

WANDB=${WANDB:-1}
WANDB_PROJECT=${WANDB_PROJECT:-candidate-Phi35-1B}
WANDB_RUN_NAME=${WANDB_RUN_NAME:-$RUN_NAME}
WANDB_GROUP=${WANDB_GROUP:-}

# Resume a previous run of THIS finetune (NOT for init_from_hf — that's the start).
RESUME_FROM=${RESUME_FROM:-}
RESET_STEP=${RESET_STEP:-}
NO_ENCODER_RESIDUAL=${NO_ENCODER_RESIDUAL:-}
DEBUG_FIRST_STEPS=${DEBUG_FIRST_STEPS:-0}
ACTIVATION_CHECKPOINT=${ACTIVATION_CHECKPOINT:-}

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

echo "=== zip2zip-core Phi-3.5-mini FINETUNE-from-pretrained ==="
echo "NODES=$SLURM_NNODES  WORLD_SIZE=$WORLD_SIZE  MODEL=$MODEL_CONFIG  MAX_SUBTOKENS=$MAX_SUBTOKENS"
echo "INIT_FROM_HF=$INIT_FROM_HF  TOKENIZER=$TOKENIZER"
echo "FREEZE_DECODER=$FREEZE_DECODER  LORA_RANK=$LORA_RANK  LORA_ALPHA=$LORA_ALPHA"
echo "ENCODER: type=$HYPER_ENCODER_TYPE dim=$ENCODER_DIM layers=$ENCODER_N_LAYERS heads=$ENCODER_N_HEADS inter=$ENCODER_INTERMEDIATE_SIZE"
echo "CODEBOOK(active)=$MAX_ACTIVE_CODEBOOK_SIZE  LR=$LR->$MIN_LR  MAX_TOKENS=$MAX_TOKENS"
echo "DATA_DIR=$DATA_DIR"
echo "OUTPUT_DIR=$OUTPUT_DIR  RUN_NAME=$RUN_NAME"

# ---------- assemble optional flags ----------
HYPER_LR_FLAG=""
if [ -n "$HYPER_LR" ]; then
    HYPER_LR_FLAG="--hyper_lr $HYPER_LR"
    if [ -n "$MIN_HYPER_LR" ]; then
        HYPER_LR_FLAG="$HYPER_LR_FLAG --min_hyper_lr $MIN_HYPER_LR"
    fi
fi

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
if [ -n "$FREEZE_DECODER" ] && [ "$FREEZE_DECODER" != "0" ]; then
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
    --init_from_hf '"$INIT_FROM_HF"' \
    --max_subtokens '"$MAX_SUBTOKENS"' \
    --encoder_dim '"$ENCODER_DIM"' \
    --encoder_n_layers '"$ENCODER_N_LAYERS"' \
    --encoder_n_heads '"$ENCODER_N_HEADS"' \
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
    '"$RESUME_FLAG"' \
    '"$RESET_STEP_FLAG"' \
    '"$DEBUG_FIRST_STEPS_FLAG"' \
    '"$HYPER_LR_FLAG"' \
    '"$WANDB_FLAG"'
'
