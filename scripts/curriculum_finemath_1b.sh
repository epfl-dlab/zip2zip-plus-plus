#!/bin/bash
set -e

# Curriculum training: ms=1→2→3→4, global 16000-step cosine schedule
# Each phase runs 4000 steps, resuming from the previous phase's checkpoint

DATA_DIR=/mnt/scratch/data/finemath-10bt
OUTPUT_BASE=/mnt/scratch/checkpoints/zip2zip_1b_curriculum
TOTAL_STEPS=16000
PHASE_STEPS=4000

echo "========================================"
echo "Phase 1: max_subtokens=1 (steps 0→${PHASE_STEPS})"
echo "========================================"
torchrun --nproc_per_node=4 -m zip2zip_core.train \
    --data_dir "$DATA_DIR" \
    --output_dir "${OUTPUT_BASE}" \
    --max_subtokens 1 \
    --max_active_codebook_size 4096 \
    --steps "$TOTAL_STEPS" \
    --save_freq "$PHASE_STEPS" \
    --wandb \
    --wandb_project "zip2zip-core-GB200" \
    --wandb_name "zip2zip_1b_curriculum" \
    --no_remap_codebook --hyper_causal_mask

echo "========================================"
echo "Phase 2: max_subtokens=2 (steps ${PHASE_STEPS}→$((PHASE_STEPS*2)))"
echo "========================================"
torchrun --nproc_per_node=4 -m zip2zip_core.train \
    --data_dir "$DATA_DIR" \
    --output_dir "${OUTPUT_BASE}" \
    --max_subtokens 2 \
    --max_active_codebook_size 4096 \
    --steps "$TOTAL_STEPS" \
    --save_freq "$PHASE_STEPS" \
    --resume_from "${OUTPUT_BASE}/step_${PHASE_STEPS}" \
    --wandb \
    --wandb_project "zip2zip-core-GB200" \
    --wandb_name "zip2zip_1b_curriculum" \
    --no_remap_codebook --hyper_causal_mask

echo "========================================"
echo "Phase 3: max_subtokens=3 (steps $((PHASE_STEPS*2))→$((PHASE_STEPS*3)))"
echo "========================================"
torchrun --nproc_per_node=4 -m zip2zip_core.train \
    --data_dir "$DATA_DIR" \
    --output_dir "${OUTPUT_BASE}" \
    --max_subtokens 3 \
    --max_active_codebook_size 4096 \
    --steps "$TOTAL_STEPS" \
    --save_freq "$PHASE_STEPS" \
    --resume_from "${OUTPUT_BASE}/step_$((PHASE_STEPS*2))" \
    --wandb \
    --wandb_project "zip2zip-core-GB200" \
    --wandb_name "zip2zip_1b_curriculum" \
    --no_remap_codebook --hyper_causal_mask

echo "========================================"
echo "Phase 4: max_subtokens=4 (steps $((PHASE_STEPS*3))→${TOTAL_STEPS})"
echo "========================================"
torchrun --nproc_per_node=4 -m zip2zip_core.train \
    --data_dir "$DATA_DIR" \
    --output_dir "${OUTPUT_BASE}" \
    --max_subtokens 4 \
    --max_active_codebook_size 4096 \
    --steps "$TOTAL_STEPS" \
    --save_freq "$PHASE_STEPS" \
    --resume_from "${OUTPUT_BASE}/step_$((PHASE_STEPS*3))" \
    --wandb \
    --wandb_project "zip2zip-core-GB200" \
    --wandb_name "zip2zip_1b_curriculum" \
    --no_remap_codebook --hyper_causal_mask

echo "Curriculum training complete!"

# ---------- Baseline: direct ms=4 for 16000 steps (no curriculum) ----------
echo "========================================"
echo "Baseline: max_subtokens=4, 16000 steps (no curriculum)"
echo "========================================"
torchrun --nproc_per_node=4 -m zip2zip_core.train \
    --data_dir "$DATA_DIR" \
    --output_dir "/mnt/scratch/checkpoints/zip2zip_1b_ms4_curriculum_no" \
    --max_subtokens 4 \
    --max_active_codebook_size 4096 \
    --steps "$TOTAL_STEPS" \
    --save_freq "$PHASE_STEPS" \
    --wandb \
    --wandb_project "zip2zip-core-GB200" \
    --wandb_name "zip2zip_1b_ms4_baseline" \
    --no_remap_codebook --hyper_causal_mask

echo "All training complete!"
