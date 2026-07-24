#!/bin/bash
# Gated pipeline launch: run the full invariant test suite and a 20-step
# smoke with the SAME recipe env vars, assert the recipe's features actually
# engaged in the smoke log, and only then chain into pipeline_ft_eval_rcp.sh.
# Fail-closed: any red step kills the job BEFORE the real run starts — no
# half-started run, no W&B pollution, nothing to clean up.
#
#   runai submit --name ft-<run> \
#     --image ghcr.io/jkminder/dlab-runai-images/pytorch:master \
#     --gpu 4 --cpu 16 --memory 128Gi \
#     --pvc dlab-scratch:/mnt --large-shm --node-pools h100 \
#     --environment WANDB_API_KEY=$WANDB_API_KEY \
#     --environment RUN_NAME=<real-run-name> \
#     --environment DISABLE_DIGIT_IDS=1 --environment UNTIED_HYPER_ENCODER=1 \
#     --environment BASE_TOKEN_POSITIONS=1 --environment TOKEN_TYPE_LOSS_WEIGHT=0.05 \
#     -- "bash /dlabscratch1/<user>/code/zip2zip-core/scripts/gated_pipeline_rcp.sh"
#
# All recipe envs are inherited by the smoke and the pipeline alike, so the
# smoke exercises exactly the configuration the real run will train.
set -euo pipefail

SCRATCH=${SCRATCH:-/dlabscratch1/gentilin}
PROJECT_DIR=$SCRATCH/code/zip2zip-core
BRANCH=${BRANCH:-finetuning-andrea}
RUN_NAME=${RUN_NAME:?Must set RUN_NAME to the real run name}

cd "$PROJECT_DIR"
git fetch origin
git checkout "$BRANCH"
git pull --ff-only origin "$BRANCH"
echo "=== gated launch at $(git rev-parse --short HEAD) for RUN_NAME=$RUN_NAME ==="

echo "=== gate 1/3: invariant test suites (CPU, deterministic) ==="
for t in test_untied_hyper_encoder test_deeper_hyper_encoder \
         test_base_positions test_token_type_loss; do
    echo "--- tests/$t.py"
    CUDA_VISIBLE_DEVICES= .venv/bin/python "tests/$t.py"
done

echo "=== gate 2/3: 20-step smoke with the run's exact recipe ==="
SMOKE_NAME="gate-smoke-$RUN_NAME"
RUN_NAME=$SMOKE_NAME STEPS=20 SAVE_FREQ=10 LOG_FREQ=1 WANDB=0 \
    bash scripts/finetune_phi35_rcp.sh
SMOKE_LOG=$(ls -t "$SCRATCH"/logs/train/finetune_"$SMOKE_NAME"_*.log | head -1)

echo "=== gate 3/3: feature engagement asserts on $SMOKE_LOG ==="
if [ -n "${BASE_TOKEN_POSITIONS:-}" ] && [ "${BASE_TOKEN_POSITIONS:-0}" != "0" ]; then
    grep -q "\[base_token_positions\] enabled" "$SMOKE_LOG"
    echo "    base_token_positions: engaged"
fi
if [ "${TOKEN_TYPE_LOSS_WEIGHT:-0}" != "0" ]; then
    grep -q "type_loss=" "$SMOKE_LOG"
    echo "    token_type_loss: engaged"
fi
if [ -n "${UNTIED_HYPER_ENCODER:-}" ] && [ "${UNTIED_HYPER_ENCODER:-0}" != "0" ]; then
    grep -q "UNTIED_FLAG='--untied_hyper_encoder'" "$SMOKE_LOG"
    echo "    untied_hyper_encoder: engaged"
fi
rm -rf "$SCRATCH"/zip2zip-outputs/"$SMOKE_NAME"*

echo "=== all gates green -> starting the real pipeline run ==="
exec bash scripts/pipeline_ft_eval_rcp.sh
