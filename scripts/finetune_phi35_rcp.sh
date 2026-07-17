#!/bin/bash
# Reproduce the RELEASED epfl-dlab/zip2zip-Phi-3.5-mini-instruct-v0.1 recipe on
# the RCP cluster (Run:AI, single node). Counterpart of
# finetune_phi35_from_hf_instruct.sbatch (CSCS) — same train.py flags, but the
# defaults here match the ACTUAL released run, verified against ozz-main
# reproducibility/phi_3.5_4b_train_cfg.yaml and the released HF repo's
# zip2zip_config.json / adapter_config.json:
#
#   max_subtokens=4 (NOT 3)          seq_len=2048 (NOT 4096)
#   32,768 tokens/optimizer-step     8,000 steps  (~262M tokens total)
#   cosine 3e-4 -> 1e-5, warmup 800  (= ozz-main min(1000, 0.1*max_steps))
#   AdamW betas (0.9, 0.999), weight_decay 0.01  (ozz-main used torch defaults)
#   base decoder FROZEN + LoRA r=32 alpha=32 on all attn+MLP projections
#   hyper-encoder: dim 3072 / 2 layers / 32 heads / intermediate 12288 (=4*dim),
#   learnable positions; active codebook 2048
#
# Known accepted deviations from the released run (not bit-identical):
#   - single hyper-encoder vs released untied input/output encoder pair
#   - hyper boundary at vocab_size 32064 vs 32011 (disabled_ids auto-derive to
#     the released [0,1,2,32000..32010] set, so compression behavior matches)
#   - packed token stream + on-the-fly LZW vs ozz per-doc padded samples with
#     offline compression
#   - loss masked to assistant turns (mask_*.npy) vs ozz plain CLM on all
#     tokens (move the mask_*.npy out of DATA_DIR to train CLM-style)
#   - fp32 params + bf16 autocast vs ozz float8 frozen base
#   - no early stopping (ozz stopped on train-loss plateau, patience 5 x 500 steps)
#
# ── Usage (Run:AI) ──────────────────────────────────────────────────────
#
#   # Smoke test (1 GPU, 20 steps, no W&B):
#   runai submit --name ft-phi35-smoke \
#     --image ghcr.io/jkminder/dlab-runai-images/pytorch:master \
#     --gpu 1 --cpu 16 --memory 128Gi \
#     --pvc dlab-scratch:/mnt --large-shm --node-pools h100 \
#     --environment STEPS=20 --environment SAVE_FREQ=10 --environment LOG_FREQ=1 \
#     --environment WANDB=0 \
#     -- "bash /dlabscratch1/<your-username>/code/zip2zip-core/scripts/finetune_phi35_rcp.sh"
#
#   # Full reproduction run (4 GPUs). WANDB_ENTITY must be your own personal
#   # W&B username/entity until epfl-dlab org access is granted — train.py
#   # falls back to a hardcoded "epfl-dlab" entity otherwise (src/zip2zip_core/
#   # project.py), which will fail if you're not a member yet.
#   runai submit --name ft-phi35-repro \
#     --image ghcr.io/jkminder/dlab-runai-images/pytorch:master \
#     --gpu 4 --cpu 32 --memory 256Gi \
#     --pvc dlab-scratch:/mnt --large-shm --node-pools h100 \
#     --environment WANDB_API_KEY=<your-key> \
#     --environment WANDB_ENTITY=<your-personal-wandb-username> \
#     -- "bash /dlabscratch1/<your-username>/code/zip2zip-core/scripts/finetune_phi35_rcp.sh"
#
# NOTE (DLAB image): pass the command string directly after `--` (the image
# entrypoint wraps it in bash -c itself; adding your own `bash -c` silently no-ops).
#
set -euo pipefail

SCRATCH=${SCRATCH:-/dlabscratch1/gentilin}
export HF_HOME=${HF_HOME:-$SCRATCH/.cache/huggingface}
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PROJECT_DIR=${PROJECT_DIR:-$SCRATCH/code/zip2zip-core}
LOG_DIR=$SCRATCH/logs/train
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# ---------- venv (uv-managed, mirrors the CSCS setup) ----------
cd "$PROJECT_DIR"
if [ ! -x .venv/bin/python ]; then
    if ! command -v uv >/dev/null 2>&1; then
        echo "[finetune_rcp] Installing uv..."
        pip install --quiet --user uv
        export PATH="$HOME/.local/bin:$PATH"
    fi
    echo "[finetune_rcp] Creating venv with uv sync (torch nightly cu130)..."
    uv sync --frozen
    # If the cu130 nightly wheels are incompatible with the node driver, install
    # a stable torch (cu128) into .venv instead and rerun with DISABLE_VARLEN=1
    # (hyper-encoder falls back to the padded attention path).
fi
PYTHON=$PROJECT_DIR/.venv/bin/python

# ---------- config (defaults = faithful released recipe; override via env) ----------
NUM_GPUS=${NUM_GPUS:-$(nvidia-smi -L | wc -l | tr -d ' ')}

DATA_DIR=${DATA_DIR:-$SCRATCH/datasets/phi-1B-sft-8shards-eosfix}
OUTPUT_BASE=${OUTPUT_BASE:-$SCRATCH/zip2zip-outputs}
RUN_NAME=${RUN_NAME:-repro-Phi35-v0.1-fthf}
OUTPUT_DIR=${OUTPUT_DIR:-${OUTPUT_BASE}/${RUN_NAME}}

INIT_FROM_HF=${INIT_FROM_HF:-microsoft/Phi-3.5-mini-instruct}
TOKENIZER=${TOKENIZER:-microsoft/Phi-3.5-mini-instruct}
MODEL_CONFIG=${MODEL_CONFIG:-Phi3.5-mini}
MAX_SUBTOKENS=${MAX_SUBTOKENS:-4}
MAX_ACTIVE_CODEBOOK_SIZE=${MAX_ACTIVE_CODEBOOK_SIZE:-2048}
# MAX_CODEBOOK_SIZE=0 disables compression entirely (the LZW encoder becomes an
# exact identity and the hyper path never executes): this is the
# continual-pretraining control. Checkpoints trained with 0 are base-mode-only
# at eval (they contain a random, never-trained hyper-encoder).
MAX_CODEBOOK_SIZE=${MAX_CODEBOOK_SIZE:-4096}
# DISABLE_DIGIT_IDS=1: digits never LZW-merge into hypertokens during training.
# Evaluate such checkpoints with the matching eval-side flag.
DISABLE_DIGIT_IDS=${DISABLE_DIGIT_IDS:-}
DIGIT_FLAG=""
if [ -n "$DISABLE_DIGIT_IDS" ] && [ "$DISABLE_DIGIT_IDS" != "0" ]; then
    DIGIT_FLAG="--disable_digit_ids"
fi

# Released run: 8000 steps x 32,768 tokens/step = ~262M tokens.
STEPS=${STEPS:-8000}
SEQ_LEN=${SEQ_LEN:-2048}
LOCAL_BATCH_SIZE=${LOCAL_BATCH_SIZE:-1}
# Keep 32,768 tokens/step regardless of GPU count: accum = 32768/(bs*seq*gpus).
GRAD_ACCUM=${GRAD_ACCUM:-$(( 32768 / (LOCAL_BATCH_SIZE * SEQ_LEN * NUM_GPUS) ))}
[ "$GRAD_ACCUM" -lt 1 ] && GRAD_ACCUM=1
# 1BT datasets hang with the default 4 dataloader workers — keep at 1.
NUM_WORKERS=${NUM_WORKERS:-1}

# train.py's main loop only checks --max_tokens when it's set at all (it
# silently ignores --steps/--stop_at in that case — see train.py's
# `while True:` loop). Deriving the MAX_TOKENS default from STEPS keeps the
# two in sync, so overriding STEPS alone (e.g. for a smoke test) actually
# shortens the run instead of being silently ignored.
TOKENS_PER_STEP=$(( LOCAL_BATCH_SIZE * SEQ_LEN * NUM_GPUS * GRAD_ACCUM ))
MAX_TOKENS=${MAX_TOKENS:-$(( STEPS * TOKENS_PER_STEP ))}

LR=${LR:-3e-4}
MIN_LR=${MIN_LR:-1e-5}
WARMUP_STEPS=${WARMUP_STEPS:-800}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.01}
ADAM_BETA2=${ADAM_BETA2:-0.999}
HYPER_LR=${HYPER_LR:-}
MIN_HYPER_LR=${MIN_HYPER_LR:-}
SAVE_FREQ=${SAVE_FREQ:-500}
LOG_FREQ=${LOG_FREQ:-10}

ENCODER_DIM=${ENCODER_DIM:-3072}
ENCODER_N_LAYERS=${ENCODER_N_LAYERS:-2}
ENCODER_N_HEADS=${ENCODER_N_HEADS:-32}
ENCODER_INTERMEDIATE_SIZE=${ENCODER_INTERMEDIATE_SIZE:-12288}
HYPER_ENCODER_TYPE=${HYPER_ENCODER_TYPE:-flat}

FREEZE_DECODER=${FREEZE_DECODER:-1}
LORA_RANK=${LORA_RANK:-32}
LORA_ALPHA=${LORA_ALPHA:-32}

WANDB=${WANDB:-1}
# Left unset, no --wandb_project flag is passed at all, so train.py falls back
# to its own canonical default (WANDB_PROJECT in src/zip2zip_core/project.py,
# currently "llaza") — the same project the eval scripts already log to.
# Set explicitly (e.g. WANDB_PROJECT=zip2zip-core) to log elsewhere.
WANDB_PROJECT=${WANDB_PROJECT:-}
WANDB_RUN_NAME=${WANDB_RUN_NAME:-$RUN_NAME}
WANDB_GROUP=${WANDB_GROUP:-}
# Pre-chosen W&B run id (8 lowercase alnum chars). Lets an orchestrator (e.g.
# scripts/pipeline_ft_eval_rcp.sh) create the training run under a known id and
# later log eval results into the SAME run via eval_harness --resume_wandb_id.
# Also disables train.py's random name suffixing, so the W&B run name is
# exactly WANDB_RUN_NAME.
WANDB_ID=${WANDB_ID:-}
# Until epfl-dlab org access is granted, set this to your own personal W&B
# username. Left unset, train.py falls back to a hardcoded "epfl-dlab" entity
# (src/zip2zip_core/project.py) and the run will fail if you're not a member.
WANDB_ENTITY=${WANDB_ENTITY:-}

# Checkpoint push to HF Hub: off by default; set HF_REPO=org/name to enable
# (eval_ckpt_rcp.sh can then evaluate straight from the Hub).
HF_REPO=${HF_REPO:-}

RESUME_FROM=${RESUME_FROM:-}
RESET_STEP=${RESET_STEP:-}
NO_ENCODER_RESIDUAL=${NO_ENCODER_RESIDUAL:-}
DEBUG_FIRST_STEPS=${DEBUG_FIRST_STEPS:-0}
ACTIVATION_CHECKPOINT=${ACTIVATION_CHECKPOINT:-}
DISABLE_VARLEN=${DISABLE_VARLEN:-}

# Do NOT pre-create OUTPUT_DIR: train.py creates it itself, and its collision
# logic appends "(1)" whenever the dir already exists — pre-creating it here is
# what silently moved every run's checkpoints into a "(N)"-suffixed folder.
LOGFILE="$LOG_DIR/finetune_${RUN_NAME}_${TIMESTAMP}.log"

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
    WANDB_FLAG="--wandb --wandb_name $WANDB_RUN_NAME"
    if [ -n "$WANDB_PROJECT" ]; then
        WANDB_FLAG="$WANDB_FLAG --wandb_project $WANDB_PROJECT"
    fi
    if [ -n "$WANDB_GROUP" ]; then
        WANDB_FLAG="$WANDB_FLAG --wandb_group $WANDB_GROUP"
    fi
    if [ -n "$WANDB_ID" ]; then
        WANDB_FLAG="$WANDB_FLAG --wandb_id $WANDB_ID --wandb_resume allow"
    fi
    if [ -n "$WANDB_ENTITY" ]; then
        WANDB_FLAG="$WANDB_FLAG --wandb_entity $WANDB_ENTITY"
    else
        echo "WARNING: WANDB_ENTITY is not set — train.py will default to the" \
             "hardcoded 'epfl-dlab' entity and the run will fail unless you're" \
             "already a member. Set WANDB_ENTITY=<your-personal-wandb-username>."
    fi
fi

HF_REPO_FLAG="--no_hf_repo"
if [ -n "$HF_REPO" ]; then
    HF_REPO_FLAG="--hf_repo $HF_REPO"
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

DISABLE_VARLEN_FLAG=""
if [ -n "$DISABLE_VARLEN" ] && [ "$DISABLE_VARLEN" != "0" ]; then
    DISABLE_VARLEN_FLAG="--disable_varlen"
fi

{
echo "=== zip2zip-core Phi-3.5-mini FINETUNE-from-pretrained (RCP) ==="
echo "NUM_GPUS=$NUM_GPUS  MODEL=$MODEL_CONFIG  MAX_SUBTOKENS=$MAX_SUBTOKENS"
echo "INIT_FROM_HF=$INIT_FROM_HF  TOKENIZER=$TOKENIZER"
echo "FREEZE_DECODER=$FREEZE_DECODER  LORA_RANK=$LORA_RANK  LORA_ALPHA=$LORA_ALPHA"
echo "ENCODER: type=$HYPER_ENCODER_TYPE dim=$ENCODER_DIM layers=$ENCODER_N_LAYERS heads=$ENCODER_N_HEADS inter=$ENCODER_INTERMEDIATE_SIZE"
echo "CODEBOOK(active)=$MAX_ACTIVE_CODEBOOK_SIZE  LR=$LR->$MIN_LR warmup=$WARMUP_STEPS wd=$WEIGHT_DECAY beta2=$ADAM_BETA2"
echo "SEQ_LEN=$SEQ_LEN  GRAD_ACCUM=$GRAD_ACCUM  TOKENS/STEP=$TOKENS_PER_STEP (released: 32768)"
echo "STEPS=$STEPS  MAX_TOKENS=$MAX_TOKENS"
echo "DATA_DIR=$DATA_DIR"
echo "OUTPUT_DIR=$OUTPUT_DIR  RUN_NAME=$RUN_NAME"
if [ "$TOKENS_PER_STEP" -ne 32768 ]; then
    echo "WARNING: tokens/step != 32768 — not the released batch size."
fi

"$PYTHON" -m torch.distributed.run --standalone --nproc_per_node="$NUM_GPUS" \
    -m zip2zip_core.train \
    --data_dir "$DATA_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --model_config "$MODEL_CONFIG" \
    --tokenizer "$TOKENIZER" \
    --init_from_hf "$INIT_FROM_HF" \
    --max_subtokens "$MAX_SUBTOKENS" \
    --encoder_dim "$ENCODER_DIM" \
    --encoder_n_layers "$ENCODER_N_LAYERS" \
    --encoder_n_heads "$ENCODER_N_HEADS" \
    --encoder_intermediate_size "$ENCODER_INTERMEDIATE_SIZE" \
    --hyper_encoder_type "$HYPER_ENCODER_TYPE" \
    --max_active_codebook_size "$MAX_ACTIVE_CODEBOOK_SIZE" \
    --max_codebook_size "$MAX_CODEBOOK_SIZE" \
    --steps "$STEPS" \
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
    --no_remap_codebook --hyper_causal_mask \
    $DIGIT_FLAG \
    $HF_REPO_FLAG \
    $AC_FLAG \
    $DISABLE_VARLEN_FLAG \
    $FREEZE_DECODER_FLAG \
    $LORA_FLAG \
    $NO_ENCODER_RESIDUAL_FLAG \
    $RESUME_FLAG \
    $RESET_STEP_FLAG \
    $DEBUG_FIRST_STEPS_FLAG \
    $HYPER_LR_FLAG \
    $WANDB_FLAG

} 2>&1 | tee "$LOGFILE"
