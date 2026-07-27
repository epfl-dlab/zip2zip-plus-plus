#!/bin/bash
# Pre-tokenize epfl-dlab/llaza-20B into raw pretraining shards (shard_*.npy +
# manifest.json, NO mask_*.npy — raw text has no assistant turns) on the RCP PVC.
# CPU-only job — no GPU needed. Expected output: ~20.0B tokens, 513 shards of
# 39M tokens each (~75 GiB).
#
# Defaults reproduce the one recorded llaza-20B invocation in this repo
# (scripts/README-phi35.md:26-30, which produced the CSCS phi-20B-tokens-512shards
# corpus): Phi-3.5-mini tokenizer, --tokens_per_shard 39000000. Keeping both means
# this corpus is byte-identical to that one, since the document order is fixed by
# shuffle(seed=42) inside pretokenize.py.
#
# ── Usage (Run:AI) ──────────────────────────────────────────────────────
#
#   runai submit --name pretok-llaza20b-phi \
#     --image ghcr.io/jkminder/dlab-runai-images/pytorch:master \
#     --cpu 32 --memory 128Gi \
#     --pvc dlab-scratch:/mnt \
#     -- "bash /dlabscratch1/<your-username>/code/zip2zip-core/scripts/pretokenize_llaza20b_rcp.sh"
#
# For a Llama-tokenized corpus instead:
#   TOKENIZER=$PWD/assets/hf_tokenizer/Llama-3.1-8B \
#   OUTPUT_DIR=$SCRATCH/datasets/llaza-20B-tokens-513shards  bash <this script>
#
# Safe to re-run: pretokenize.py resumes from manifest.json, and refuses to touch
# a shard directory whose resume point is unknown rather than duplicating data.
#
set -euo pipefail

Z2Z_SCRATCH=${Z2Z_SCRATCH:-/dlabscratch1/gentilin}
export HF_HOME=${HF_HOME:-$Z2Z_SCRATCH/.cache/huggingface}
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

export OUTPUT_DIR=${OUTPUT_DIR:-$Z2Z_SCRATCH/datasets/phi-20B-tokens-513shards}
export TOKENIZER=${TOKENIZER:-microsoft/Phi-3.5-mini-instruct}
DATASET=${DATASET:-epfl-dlab/llaza-20B}
TOKENS_PER_SHARD=${TOKENS_PER_SHARD:-39000000}
TARGET_TOKENS=${TARGET_TOKENS:-20e9}
NUM_WORKERS=${NUM_WORKERS:-16}

PROJECT_DIR=${PROJECT_DIR:-$Z2Z_SCRATCH/code/zip2zip-core}
LOG_DIR=$Z2Z_SCRATCH/logs/tokenize
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOGFILE="$LOG_DIR/pretokenize_llaza20b_${TIMESTAMP}.log"

# ---------- disk preflight ----------
# load_dataset() here is NOT streaming and .map() is eager, so the whole 20B-token
# split is tokenized into the HF Arrow cache *before* the first shard is written.
# Peak footprint is ~3-4x the output: ~46GB parquet download + ~160GB tokenized
# Arrow (int64) + ~75GiB of shards. Set SKIP_DISK_CHECK=1 to override.
need_gb() {  # need_gb <path> <required GiB>
    local path=$1 required=$2 avail
    mkdir -p "$path"
    avail=$(df -PB1073741824 "$path" | awk 'NR==2 {print $4}')
    echo "[disk] $path: ${avail} GiB available, ${required} GiB required"
    if [ "${SKIP_DISK_CHECK:-0}" != "1" ] && [ "$avail" -lt "$required" ]; then
        echo "FATAL: not enough free space on $path (${avail} GiB < ${required} GiB)."
        echo "  The HF cache alone needs ~200 GiB for a 20B-token corpus."
        echo "  Free space, point HF_HOME at another filesystem, or set SKIP_DISK_CHECK=1."
        exit 1
    fi
}
need_gb "$HF_HOME" "${REQUIRE_CACHE_GB:-220}"
need_gb "$OUTPUT_DIR" "${REQUIRE_OUTPUT_GB:-85}"

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
echo "=== llaza-20B pretokenization (raw pretraining shards) ==="
echo "  commit:           $(git -C "$PROJECT_DIR" rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo "  DATASET:          $DATASET"
echo "  TOKENIZER:        $TOKENIZER"
echo "  OUTPUT_DIR:       $OUTPUT_DIR"
echo "  TOKENS_PER_SHARD: $TOKENS_PER_SHARD"
echo "  TARGET_TOKENS:    $TARGET_TOKENS"
echo "  NUM_WORKERS:      $NUM_WORKERS"
echo "  HF_HOME:          $HF_HOME"
echo "=========================================================="

python scripts/pretokenize.py \
    --output_dir "$OUTPUT_DIR" \
    --model_name "$TOKENIZER" \
    --dataset "$DATASET" \
    --dataset_split train \
    --column text \
    --target_tokens "$TARGET_TOKENS" \
    --tokens_per_shard "$TOKENS_PER_SHARD" \
    --min_doc_length 50 \
    --num_workers "$NUM_WORKERS"

echo "--- manifest ---"
cat "$OUTPUT_DIR/manifest.json"

echo "--- sanity: shard count, dtype, shape, token ids within tokenizer vocab ---"
python - <<'EOF'
import os, glob
import numpy as np
from transformers import AutoTokenizer

d = os.environ["OUTPUT_DIR"]
vocab = len(AutoTokenizer.from_pretrained(os.environ["TOKENIZER"]))
paths = sorted(glob.glob(os.path.join(d, "shard_*.npy")))
print(f"shards: {len(paths)}  tokenizer vocab: {vocab}")

lengths = set()
for p in paths:
    a = np.load(p, mmap_mode="r")
    assert a.dtype == np.uint32, f"{p}: dtype {a.dtype}, expected uint32"
    assert a.ndim == 1, f"{p}: shape {a.shape}, expected flat 1-D"
    lengths.add(int(a.shape[0]))
print(f"distinct shard lengths: {sorted(lengths)}")
print(f"total tokens: {sum(int(np.load(p, mmap_mode='r').shape[0]) for p in paths)/1e9:.3f}B")

# A tokenizer mismatch is the failure this catches: Llama ids (up to 128255) in a
# corpus labelled Phi silently mis-offset every hyper-token id at training time.
m = int(np.load(paths[0], mmap_mode="r")[:10_000_000].max())
print(f"max token id in first 10M of {os.path.basename(paths[0])} = {m}")
assert m < vocab, f"token id {m} >= vocab {vocab} — wrong tokenizer for this data!"
assert not glob.glob(os.path.join(d, "mask_*.npy")), \
    "unexpected mask_*.npy: this is a raw pretraining corpus, it must have no masks"
EOF

} 2>&1 | tee "$LOGFILE"
