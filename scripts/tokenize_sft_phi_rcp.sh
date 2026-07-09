#!/bin/bash
# Tokenize epfl-dlab/zip2zip-1B with the Phi-3.5-mini tokenizer + chat template
# into packed SFT shards (shard_*.npy + mask_*.npy + manifest.json) on the RCP
# PVC. CPU-only job — no GPU needed. Expected output: ~1B tokens, 8 full shards
# of 125M tokens each (~5GB tokens + masks).
#
# ── Usage (Run:AI) ──────────────────────────────────────────────────────
#
#   runai submit --name tokenize-phi-sft \
#     --image ghcr.io/jkminder/dlab-runai-images/pytorch:master \
#     --cpu 16 --memory 96Gi \
#     --pvc dlab-scratch:/mnt \
#     -- "bash /dlabscratch1/<your-username>/code/zip2zip-core/scripts/tokenize_sft_phi_rcp.sh"
#
# The script refuses to overwrite: if OUTPUT_DIR already contains .npy files,
# it exits with FileExistsError — remove them first.
#
set -euo pipefail

SCRATCH=${SCRATCH:-/dlabscratch1/gentilin}
export HF_HOME=${HF_HOME:-$SCRATCH/.cache/huggingface}
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

export OUTPUT_DIR=${OUTPUT_DIR:-$SCRATCH/datasets/phi-1B-sft-8shards}

PROJECT_DIR=${PROJECT_DIR:-$SCRATCH/code/zip2zip-core}
LOG_DIR=$SCRATCH/logs/tokenize
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOGFILE="$LOG_DIR/tokenize_phi_sft_${TIMESTAMP}.log"

# ---------- venv (lightweight: only needs datasets/transformers/numpy) ----------
VENV_DIR=$SCRATCH/.venvs/tokenize
if [ ! -f "$VENV_DIR/bin/activate" ]; then
    echo "[tokenize] Creating venv at $VENV_DIR..."
    python -m venv --system-site-packages "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"
pip install --quiet "datasets" "transformers" "numpy"

cd "$PROJECT_DIR"

{
echo "=== Phi-3.5 SFT tokenization (epfl-dlab/zip2zip-1B) ==="
echo "  OUTPUT_DIR: $OUTPUT_DIR"
echo "  HF_HOME:    $HF_HOME"
echo "======================================================="

python scripts/create_sft_dataset_zip2zip1B_phi.py

echo "--- manifest ---"
cat "$OUTPUT_DIR/manifest.json"

echo "--- sanity: max token id must be < 32064 (Phi vocab) ---"
python - <<'EOF'
import os, numpy as np
d = os.environ["OUTPUT_DIR"]
shard = sorted(f for f in os.listdir(d) if f.startswith("shard_"))[0]
a = np.load(os.path.join(d, shard), mmap_mode="r")
m = int(a[:10_000_000].max())
print(f"{shard}: max token id in first 10M = {m}")
assert m < 32064, "Token id >= 32064 — this is NOT Phi-tokenized data!"
EOF

} 2>&1 | tee "$LOGFILE"
