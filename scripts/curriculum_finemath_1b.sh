#!/bin/bash
set -e

# Curriculum training: ms=1→2→3→4, global 16000-step cosine schedule
# Each phase runs 4000 steps, resuming from the previous phase's checkpoint

DATA_DIR=/mnt/scratch/data/finemath-10bt
OUTPUT_BASE=/mnt/scratch/checkpoints/zip2zip_1b_curriculum_16k
TOTAL_STEPS=16000
PHASE_STEPS=4000
WANDB_RUN_ID=${WANDB_RUN_ID:-curriculum_1b_$(date +%s)}

# echo "========================================"
# echo "Phase 1: max_subtokens=2 (steps 0→${PHASE_STEPS})"
# echo "========================================"
# torchrun --nproc_per_node=4 -m zip2zip_core.train \
#     --data_dir "$DATA_DIR" \
#     --output_dir "${OUTPUT_BASE}" \
#     --max_subtokens 2 \
#     --max_active_codebook_size 4096 \
#     --steps "$TOTAL_STEPS" \
#     --stop_at "$PHASE_STEPS" \
#     --save_freq "$PHASE_STEPS" \
#     --wandb \
#     --wandb_project "zip2zip-core-GB200" \
#     --wandb_name "zip2zip_1b_curriculum" \
#     --wandb_id "$WANDB_RUN_ID" --wandb_resume allow \
#     --no_remap_codebook --hyper_causal_mask

# echo "========================================"
# echo "Phase 2: max_subtokens=3 (steps ${PHASE_STEPS}→$((PHASE_STEPS*2)))"
# echo "========================================"
# torchrun --nproc_per_node=4 -m zip2zip_core.train \
#     --data_dir "$DATA_DIR" \
#     --output_dir "${OUTPUT_BASE}" \
#     --max_subtokens 3 \
#     --max_active_codebook_size 4096 \
#     --steps "$TOTAL_STEPS" \
#     --stop_at "$((PHASE_STEPS*2))" \
#     --save_freq "$PHASE_STEPS" \
#     --resume_from "${OUTPUT_BASE}/step_${PHASE_STEPS}" \
#     --wandb \
#     --wandb_project "zip2zip-core-GB200" \
#     --wandb_name "zip2zip_1b_curriculum" \
#     --wandb_id "$WANDB_RUN_ID" --wandb_resume must \
#     --no_remap_codebook --hyper_causal_mask

# echo "========================================"
# echo "Phase 3: max_subtokens=4 (steps $((PHASE_STEPS*2))→$((PHASE_STEPS*3)))"
# echo "========================================"
# torchrun --nproc_per_node=4 -m zip2zip_core.train \
#     --data_dir "$DATA_DIR" \
#     --output_dir "${OUTPUT_BASE}" \
#     --max_subtokens 4 \
#     --max_active_codebook_size 4096 \
#     --steps "$TOTAL_STEPS" \
#     --stop_at "$((PHASE_STEPS*3))" \
#     --save_freq "$PHASE_STEPS" \
#     --resume_from "${OUTPUT_BASE}/step_$((PHASE_STEPS*2))" \
#     --wandb \
#     --wandb_project "zip2zip-core-GB200" \
#     --wandb_name "zip2zip_1b_curriculum" \
#     --wandb_id "$WANDB_RUN_ID" --wandb_resume must \
#     --no_remap_codebook --hyper_causal_mask

# echo "========================================"
# echo "Phase 4: max_subtokens=8 (steps $((PHASE_STEPS*3))→${TOTAL_STEPS})"
# echo "========================================"
# torchrun --nproc_per_node=4 -m zip2zip_core.train \
#     --data_dir "$DATA_DIR" \
#     --output_dir "${OUTPUT_BASE}" \
#     --max_subtokens 8 \
#     --max_active_codebook_size 4096 \
#     --steps "$TOTAL_STEPS" \
#     --stop_at "$TOTAL_STEPS" \
#     --save_freq "$PHASE_STEPS" \
#     --resume_from "${OUTPUT_BASE}/step_$((PHASE_STEPS*3))" \
#     --wandb \
#     --wandb_project "zip2zip-core-GB200" \
#     --wandb_name "zip2zip_1b_curriculum" \
#     --wandb_id "$WANDB_RUN_ID" --wandb_resume must \
#     --no_remap_codebook --hyper_causal_mask

# echo "Curriculum training complete!"

# ---------- ms=1→8 curriculum: 12K baseline then 4K ms=8 ----------
MS8_OUTPUT=/mnt/scratch/checkpoints/zip2zip_1b_curriculum_ms8
MS8_TOTAL=16000
MS8_WANDB_ID=${MS8_WANDB_ID:-curriculum_1b_ms8_$(date +%s)}

echo "========================================"
echo "ms8 Phase 1: max_subtokens=1 (steps 0→12000)"
echo "========================================"
torchrun --nproc_per_node=4 -m zip2zip_core.train \
    --data_dir "$DATA_DIR" \
    --output_dir "${MS8_OUTPUT}" \
    --max_subtokens 1 \
    --max_active_codebook_size 4096 \
    --steps "$MS8_TOTAL" \
    --stop_at 12000 \
    --save_freq 4000 \
    --wandb \
    --wandb_project "zip2zip-core-GB200" \
    --wandb_name "zip2zip_1b_curriculum_ms8" \
    --wandb_id "$MS8_WANDB_ID" --wandb_resume allow \
    --no_remap_codebook --hyper_causal_mask

echo "========================================"
echo "ms8 Phase 2: max_subtokens=8 (steps 12000→16000)"
echo "========================================"
torchrun --nproc_per_node=4 -m zip2zip_core.train \
    --data_dir "$DATA_DIR" \
    --output_dir "${MS8_OUTPUT}" \
    --max_subtokens 8 \
    --max_active_codebook_size 4096 \
    --steps "$MS8_TOTAL" \
    --stop_at "$MS8_TOTAL" \
    --save_freq 4000 \
    --resume_from "${MS8_OUTPUT}/step_12000" \
    --wandb \
    --wandb_project "zip2zip-core-GB200" \
    --wandb_name "zip2zip_1b_curriculum_ms8" \
    --wandb_id "$MS8_WANDB_ID" --wandb_resume must \
    --no_remap_codebook --hyper_causal_mask

# ---------- Baseline: direct ms=8 for 16K steps (no curriculum) ----------
echo "========================================"
echo "Baseline: max_subtokens=8, 16000 steps (no curriculum)"
echo "========================================"
torchrun --nproc_per_node=4 -m zip2zip_core.train \
    --data_dir "$DATA_DIR" \
    --output_dir "/mnt/scratch/checkpoints/zip2zip_1b_ms8_no_curriculum" \
    --max_subtokens 8 \
    --max_active_codebook_size 4096 \
    --steps 16000 \
    --save_freq 4000 \
    --wandb \
    --wandb_project "zip2zip-core-GB200" \
    --wandb_name "zip2zip_1b_ms8_no_curriculum" \
    --no_remap_codebook --hyper_causal_mask

echo "All training complete!"
