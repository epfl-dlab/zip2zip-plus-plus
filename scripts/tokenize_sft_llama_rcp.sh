#!/bin/bash
# Tokenize epfl-dlab/zip2zip-1B with the Llama-3.2-1B-Instruct tokenizer + chat
# template into packed SFT shards (shard_*.npy + mask_*.npy + manifest.json) on
# the RCP PVC. CPU-only job — no GPU needed. Expected output: ~1B tokens, 8
# full shards of 125M tokens each (~5GB tokens + masks). Mirrors
# tokenize_sft_phi_rcp.sh; the dataset is the Llama twin of
# phi-1B-sft-8shards-eosfix (eosfix, no mathchat).
#
# meta-llama/Llama-3.2-1B-Instruct is a gated repo: the job needs either a
# warm HF cache or HF_TOKEN in the environment.
#
# ── Usage (Run:AI) ──────────────────────────────────────────────────────
#
#   runai submit --name tokenize-llama-sft \
#     --image ghcr.io/jkminder/dlab-runai-images/pytorch:master \
#     --cpu 16 --memory 96Gi \
#     --pvc dlab-scratch:/mnt \
#     -- "bash /dlabscratch1/<your-username>/code/zip2zip-core/scripts/tokenize_sft_llama_rcp.sh"
#
# The script refuses to overwrite: if OUTPUT_DIR already contains .npy files,
# it exits with FileExistsError — remove them first.
#
set -euo pipefail

Z2Z_SCRATCH=${Z2Z_SCRATCH:-/dlabscratch1/gentilin}
export HF_HOME=${HF_HOME:-$Z2Z_SCRATCH/.cache/huggingface}
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

export OUTPUT_DIR=${OUTPUT_DIR:-$Z2Z_SCRATCH/datasets/llama32-1B-sft-8shards-eosfix}

PROJECT_DIR=${PROJECT_DIR:-$Z2Z_SCRATCH/code/zip2zip-core}
LOG_DIR=$Z2Z_SCRATCH/logs/tokenize
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOGFILE="$LOG_DIR/tokenize_llama_sft_${TIMESTAMP}.log"

# ---------- venv (lightweight: only needs datasets/transformers/numpy) ----------
VENV_DIR=$Z2Z_SCRATCH/.venvs/tokenize
if [ ! -f "$VENV_DIR/bin/activate" ]; then
    echo "[tokenize] Creating venv at $VENV_DIR..."
    python -m venv --system-site-packages "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"
pip install --quiet "datasets" "transformers" "numpy"

cd "$PROJECT_DIR"

{
echo "=== Llama-3.2 SFT tokenization (epfl-dlab/zip2zip-1B) ==="
echo "  OUTPUT_DIR: $OUTPUT_DIR"
echo "  HF_HOME:    $HF_HOME"
echo "========================================================="

python scripts/create_sft_dataset_zip2zip1B.py

echo "--- manifest ---"
cat "$OUTPUT_DIR/manifest.json"

echo "--- sanity: token ids in Llama vocab, EOS in loss ---"
python - <<'EOF'
import os, numpy as np
d = os.environ["OUTPUT_DIR"]
shard = sorted(f for f in os.listdir(d) if f.startswith("shard_"))[0]
mask = shard.replace("shard_", "mask_")
a = np.load(os.path.join(d, shard), mmap_mode="r")[:10_000_000]
m = np.load(os.path.join(d, mask), mmap_mode="r")[:10_000_000]
top = int(a.max())
print(f"{shard}: max token id in first 10M = {top}")
assert top < 128256, "Token id >= 128256 — this is NOT Llama-tokenized data!"
eos_pos = a == 128001
n_eos = int(eos_pos.sum())
n_eos_in_loss = int((m[eos_pos] == 1).sum())
print(f"<|end_of_text|> occurrences: {n_eos}, with mask=1: {n_eos_in_loss}")
assert n_eos > 0, "No <|end_of_text|> found — doc separator missing!"
# literal "<|end_of_text|>" strings inside raw web/code docs also tokenize to
# 128001 and may carry mask=0 inside chat turns, so require a ratio not equality
assert n_eos_in_loss / n_eos > 0.99, "Doc EOS mostly mask=0 — eosfix not applied!"
bos_pos = a == 128000
print(f"<|begin_of_text|> occurrences: {int(bos_pos.sum())}, with mask=1: {int((m[bos_pos] == 1).sum())} (expected 0)")
EOF

} 2>&1 | tee "$LOGFILE"
