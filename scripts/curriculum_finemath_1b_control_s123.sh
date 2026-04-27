#!/bin/bash
set -e

# Control experiment: curriculum training with RANDOM weights at each phase transition.
# Same step/LR schedule as the real curriculum, but model weights are re-initialized randomly
# instead of loaded from the previous phase checkpoint.
# Purpose: verify whether curriculum pre-training actually matters, or if only the
# LR schedule position drives the results.

DATA_DIR=/mnt/scratch/data/finemath-10bt
OUTPUT_BASE=/mnt/scratch/checkpoints/zip2zip_1b_curriculum_16k_rand_control_s123
TOTAL_STEPS=16000
PHASE_STEPS=4000
WANDB_RUN_ID=${WANDB_RUN_ID:-curriculum_1b_rand_control_s123_$(date +%s)}

echo "========================================"
echo "Phase 1: max_subtokens=2 (steps 0→${PHASE_STEPS})"
echo "========================================"
torchrun --nproc_per_node=4 -m zip2zip_core.train \
    --data_dir "$DATA_DIR" \
    --output_dir "${OUTPUT_BASE}" \
    --max_subtokens 2 \
    --max_active_codebook_size 4096 \
    --steps "$TOTAL_STEPS" \
    --stop_at "$PHASE_STEPS" \
    --save_freq "$PHASE_STEPS" \
    --wandb \
    --wandb_project "zip2zip-core-GB200" \
    --wandb_name "zip2zip_1b_curriculum_rand_control_s123" \
    --wandb_id "$WANDB_RUN_ID" --wandb_resume allow \
    --no_remap_codebook --hyper_causal_mask \
    --seed 123

echo "========================================"
echo "Phase 2: max_subtokens=3 (steps ${PHASE_STEPS}→$((PHASE_STEPS*2))) [random weights]"
echo "========================================"
torchrun --nproc_per_node=4 -m zip2zip_core.train \
    --data_dir "$DATA_DIR" \
    --output_dir "${OUTPUT_BASE}" \
    --max_subtokens 3 \
    --max_active_codebook_size 4096 \
    --steps "$TOTAL_STEPS" \
    --stop_at "$((PHASE_STEPS*2))" \
    --save_freq "$PHASE_STEPS" \
    --resume_from "${OUTPUT_BASE}/step_${PHASE_STEPS}" \
    --random_weights \
    --wandb \
    --wandb_project "zip2zip-core-GB200" \
    --wandb_name "zip2zip_1b_curriculum_rand_control_s123" \
    --wandb_id "$WANDB_RUN_ID" --wandb_resume must \
    --no_remap_codebook --hyper_causal_mask \
    --seed 123

echo "========================================"
echo "Phase 3: max_subtokens=4 (steps $((PHASE_STEPS*2))→$((PHASE_STEPS*3))) [random weights]"
echo "========================================"
torchrun --nproc_per_node=4 -m zip2zip_core.train \
    --data_dir "$DATA_DIR" \
    --output_dir "${OUTPUT_BASE}" \
    --max_subtokens 4 \
    --max_active_codebook_size 4096 \
    --steps "$TOTAL_STEPS" \
    --stop_at "$((PHASE_STEPS*3))" \
    --save_freq "$PHASE_STEPS" \
    --resume_from "${OUTPUT_BASE}/step_$((PHASE_STEPS*2))" \
    --random_weights \
    --wandb \
    --wandb_project "zip2zip-core-GB200" \
    --wandb_name "zip2zip_1b_curriculum_rand_control_s123" \
    --wandb_id "$WANDB_RUN_ID" --wandb_resume must \
    --no_remap_codebook --hyper_causal_mask \
    --seed 123

echo "========================================"
echo "Phase 4: max_subtokens=8 (steps $((PHASE_STEPS*3))→${TOTAL_STEPS}) [random weights]"
echo "========================================"
torchrun --nproc_per_node=4 -m zip2zip_core.train \
    --data_dir "$DATA_DIR" \
    --output_dir "${OUTPUT_BASE}" \
    --max_subtokens 8 \
    --max_active_codebook_size 4096 \
    --steps "$TOTAL_STEPS" \
    --stop_at "$TOTAL_STEPS" \
    --save_freq "$PHASE_STEPS" \
    --resume_from "${OUTPUT_BASE}/step_$((PHASE_STEPS*3))" \
    --random_weights \
    --wandb \
    --wandb_project "zip2zip-core-GB200" \
    --wandb_name "zip2zip_1b_curriculum_rand_control_s123" \
    --wandb_id "$WANDB_RUN_ID" --wandb_resume must \
    --no_remap_codebook --hyper_causal_mask \
    --seed 123

echo "Curriculum random-weights control experiment complete!"
