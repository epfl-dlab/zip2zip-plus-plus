#!/bin/bash
# Pre-tokenize epfl-dlab/llaza-200B into raw pretraining shards (shard_*.npy +
# manifest.json, NO mask_*.npy — raw text has no assistant turns) on the RCP PVC.
# CPU-only job — no GPU needed. Expected output: exactly 5129 full shards of
# 39M tokens (200.031B tokens, ~745 GiB); the sanity block at the end asserts
# this, so a dataset that runs dry short of the target fails loudly.
#
# Same recipe as scripts/pretokenize_llaza20b_rcp.sh, one axis changed: the
# dataset (and hence the target). Tokenizer and --tokens_per_shard stay at the
# 20B values, so shard granularity is identical across both corpora. Note the
# corpora are NOT prefixes of each other: shuffle(seed=42) runs over a different
# document list, so phi-20B shards are not the first 513 shards of this output.
#
# llaza-200B is the full mixture llaza-20B was drawn from (50% FineWeb-edu,
# 20% The Stack, 10% FineMath, 20% FineWeb2-HQ): 190.2M docs, 492 GB parquet,
# ~218B tokens under Llama-3.1, so ~240B+ under Phi's smaller 32k vocab —
# comfortably more than the 200e9 target.
#
# ── Scale warning ───────────────────────────────────────────────────────
# pretokenize.py is EAGER: load_dataset() is not streaming and .map() tokenizes
# the WHOLE split into the HF Arrow cache before shard 0 is written (the 200e9
# target only stops shard writing, not tokenization). Peak disk, all on the
# same PVC with the defaults below:
#   ~460 GiB  parquet download (hub cache)
#   ~860 GiB  raw-text Arrow, single copy (filter/shuffle add only ~3 GiB of
#             indices caches — they do not copy the text)
#   ~2.0 TiB  tokenized int64 Arrow for the full ~240-270B-token corpus
#   ~750 GiB  output shards (uint32) — the only permanent part
# ≈ 4.0 TiB peak. Once the run completes, reclaim ~3.3 TiB by deleting the
# llaza-200B entries under $HF_HOME/datasets and $HF_HOME/hub.
#
# Expect roughly 3 days (measured 20B baseline, 2026-07-27 log: 9.5 h total =
# ~3 h download+map at 16 workers + ~6.4 h shard writing at ~0.8M tok/s; the
# map scales with NUM_WORKERS, the shard-writing loop is single-threaded and
# does not). Interruptions are cheap:
# pretokenize.py resumes exactly from manifest.json, completed Arrow cache
# pieces are reused, and the disk preflight below credits whatever a previous
# attempt already wrote — so Run:AI auto-retries resume useful work instead of
# tripping a guard. One exception: a kill mid-np.save leaves a truncated
# highest-numbered shard and the resume refuses to start (as a numpy mmap
# ValueError, or the guided "Inconsistent state" message, depending on where
# the kill landed); delete that one shard file and resubmit.
#
# NUM_WORKERS here is datasets' num_proc, NOT the dataloader knob: the training
# rule "long runs need NUM_WORKERS=0" does not apply, and 0 here means half the
# node's cores (cgroup-unaware) — leave it matched to the --cpu request.
#
# ── Usage (Run:AI) ──────────────────────────────────────────────────────
#
#   runai-rcp-prod submit --name pretok-llaza200b-phi \
#     --image ghcr.io/jkminder/dlab-runai-images/pytorch:master \
#     --cpu 32 --memory 192Gi \
#     --pvc dlab-scratch:/mnt \
#     -- bash /dlabscratch1/<your-username>/code/zip2zip-core/scripts/pretokenize_llaza200b_rcp.sh
#
# (192Gi, not the 20B script's 128Gi: twice the map workers, and The Stack's
# multi-MB outlier documents make map-phase RSS the one thing that scales.)
#
# Pass the command as a bare argument list, NOT as a quoted string and NOT as
# `bash -c '...'`: the DLAB image entrypoint wraps whatever it receives in its own
# `/bin/bash -c "..."`, so any quoting of your own is lost and an inline shell
# snippet is truncated at the first `;`.
#
set -euo pipefail

Z2Z_SCRATCH=${Z2Z_SCRATCH:-/dlabscratch1/gentilin}
export HF_HOME=${HF_HOME:-$Z2Z_SCRATCH/.cache/huggingface}
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

export OUTPUT_DIR=${OUTPUT_DIR:-$Z2Z_SCRATCH/datasets/phi-200B-tokens-5129shards}
export TOKENIZER=${TOKENIZER:-microsoft/Phi-3.5-mini-instruct}
DATASET=${DATASET:-epfl-dlab/llaza-200B}
export TOKENS_PER_SHARD=${TOKENS_PER_SHARD:-39000000}
export TARGET_TOKENS=${TARGET_TOKENS:-200e9}
NUM_WORKERS=${NUM_WORKERS:-32}
# Parallel processes for load_dataset's download+prepare phase. The 2026-08-13
# attempt measured ~3 MB/s on the single-process python downloader — parallel
# files times hf_transfer (below) is what turns ~2 weeks of download into hours.
# Does not change the prepared row order, so shuffle(seed=42) is unaffected.
DOWNLOAD_NUM_PROC=${DOWNLOAD_NUM_PROC:-16}

PROJECT_DIR=${PROJECT_DIR:-$Z2Z_SCRATCH/code/zip2zip-core}
LOG_DIR=$Z2Z_SCRATCH/logs/tokenize
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOGFILE="$LOG_DIR/pretokenize_llaza200b_${TIMESTAMP}.log"

# ---------- disk preflight ----------
# Requirements describe a from-scratch run (see scale warning above). Space
# this run's own artifacts already occupy — the HF caches and existing shards —
# is credited back, so a Run:AI auto-retry of a partially-done run passes
# instead of failing on the space its first attempt consumed. $HF_HOME/datasets
# is credited wholesale — any unrelated cache living there weakens the guard by
# its own size (today that is ~77 GiB, noise; do not park other multi-hundred-GiB
# caches there mid-run). When cache and output share a filesystem (the default),
# the two requirements are checked as one sum. SKIP_DISK_CHECK=1 disables the
# free-space enforcement (df/du still run); it is also the escape hatch for
# re-running an already-complete output dir after the caches were cleaned up,
# which would otherwise demand the full from-scratch requirement again.
REQUIRE_CACHE_GB=${REQUIRE_CACHE_GB:-3600}
REQUIRE_OUTPUT_GB=${REQUIRE_OUTPUT_GB:-800}

used_gb() {  # total GiB already on disk under the given paths (absent = 0)
    local sum=0 d g
    for d in "$@"; do
        [ -e "$d" ] || continue
        g=$(du -s -B1073741824 "$d" 2>/dev/null | awk '{print $1}')
        sum=$((sum + ${g:-0}))
    done
    echo "$sum"
}
free_gb() { df -PB1073741824 "$1" | awk 'NR==2 {print $4}'; }
fs_of()   { df -P "$1" | awk 'NR==2 {print $1}'; }

check_free() {  # check_free <path> <required GiB> <label>
    local avail
    avail=$(free_gb "$1")
    echo "[disk] $3: ${avail} GiB free on $1, ${2} GiB still required"
    if [ "${SKIP_DISK_CHECK:-0}" != "1" ] && [ "$avail" -lt "$2" ]; then
        echo "FATAL: not enough free space on $1 (${avail} GiB < ${2} GiB)."
        echo "  Free space, point HF_HOME at another filesystem, or set SKIP_DISK_CHECK=1."
        exit 1
    fi
}

mkdir -p "$HF_HOME" "$OUTPUT_DIR"
CACHE_USED=$(used_gb "$HF_HOME/datasets" "$HF_HOME/hub/datasets--${DATASET//\//--}")
OUTPUT_USED=$(used_gb "$OUTPUT_DIR")
CACHE_NEED=$(( REQUIRE_CACHE_GB > CACHE_USED ? REQUIRE_CACHE_GB - CACHE_USED : 0 ))
OUTPUT_NEED=$(( REQUIRE_OUTPUT_GB > OUTPUT_USED ? REQUIRE_OUTPUT_GB - OUTPUT_USED : 0 ))
echo "[disk] already written: cache ${CACHE_USED} GiB, output ${OUTPUT_USED} GiB"
if [ "$(fs_of "$HF_HOME")" = "$(fs_of "$OUTPUT_DIR")" ]; then
    check_free "$HF_HOME" $((CACHE_NEED + OUTPUT_NEED)) "cache+output (same filesystem)"
else
    check_free "$HF_HOME" "$CACHE_NEED" "cache"
    check_free "$OUTPUT_DIR" "$OUTPUT_NEED" "output"
fi

# ---------- venv (lightweight: only needs datasets/transformers/numpy) ----------
# datasets/transformers are pinned to the image versions that built the 20B
# corpus (checked 2026-08-12). The venv uses --system-site-packages over the
# MUTABLE pytorch:master tag, so without pins a re-pulled image can change the
# datasets version mid-campaign, which changes the .map fingerprint and
# silently re-tokenizes ~2 TiB of Arrow next to the old cache. With the pins,
# pip is a no-op while the image still matches and installs the exact known
# version into the venv the moment it drifts.
VENV_DIR=$Z2Z_SCRATCH/.venvs/tokenize
if [ ! -f "$VENV_DIR/bin/activate" ]; then
    echo "[tokenize] Creating venv at $VENV_DIR..."
    python -m venv --system-site-packages "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"
pip install --quiet "datasets==2.18.0" "transformers==4.56.2" "numpy" "hf_transfer==0.1.8"

# hf_transfer: Rust multi-chunk downloader. The stock python path measured
# ~3 MB/s single-stream on 2026-08-13 and died 7 straight Run:AI attempts on
# mid-file ChunkedEncodingError; hf_transfer saturates the link and retries
# per chunk. Guarded on importability so a pip hiccup degrades to the slow
# downloader instead of a hard error at the first file. NOTE: hf_transfer does
# not resume *.incomplete files left by killed python-path attempts — delete
# those from the hub cache before relaunching (completed blobs are reused).
if python -c "import hf_transfer" 2>/dev/null; then
    export HF_HUB_ENABLE_HF_TRANSFER=1
    echo "[tokenize] hf_transfer enabled for downloads"
else
    echo "[tokenize] hf_transfer unavailable — using the slow python downloader"
fi
export HF_HUB_DOWNLOAD_TIMEOUT=${HF_HUB_DOWNLOAD_TIMEOUT:-60}

cd "$PROJECT_DIR"

{
echo "=== llaza-200B pretokenization (raw pretraining shards) ==="
echo "  commit:           $(git -C "$PROJECT_DIR" rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo "  DATASET:          $DATASET"
echo "  TOKENIZER:        $TOKENIZER"
echo "  OUTPUT_DIR:       $OUTPUT_DIR"
echo "  TOKENS_PER_SHARD: $TOKENS_PER_SHARD"
echo "  TARGET_TOKENS:    $TARGET_TOKENS"
echo "  NUM_WORKERS:      $NUM_WORKERS"
echo "  DOWNLOAD_NUM_PROC:$DOWNLOAD_NUM_PROC"
echo "  HF_HOME:          $HF_HOME"
echo "==========================================================="

# Retry HERE rather than through Run:AI's 6 silent pod retries (which the
# 2026-08-13 attempt exhausted into a Failed job on transient HF network
# errors). Resume is cheap by design — manifest.json plus the hub/Arrow
# caches — so each retry continues from wherever the last attempt died.
PRETOK_RETRIES=${PRETOK_RETRIES:-20}
attempt=1
until python scripts/pretokenize.py \
    --output_dir "$OUTPUT_DIR" \
    --model_name "$TOKENIZER" \
    --dataset "$DATASET" \
    --dataset_split train \
    --column text \
    --target_tokens "$TARGET_TOKENS" \
    --tokens_per_shard "$TOKENS_PER_SHARD" \
    --min_doc_length 50 \
    --num_workers "$NUM_WORKERS" \
    --download_num_proc "$DOWNLOAD_NUM_PROC"
do
    rc=$?
    if [ "$attempt" -ge "$PRETOK_RETRIES" ]; then
        echo "FATAL: pretokenize.py failed $attempt times (last rc=$rc) — giving up."
        exit "$rc"
    fi
    attempt=$((attempt + 1))
    echo "[tokenize] pretokenize.py failed (rc=$rc) — retry $attempt/$PRETOK_RETRIES in 60s"
    sleep 60
done

echo "--- manifest ---"
cat "$OUTPUT_DIR/manifest.json"

echo "--- sanity: shard count, dtype, shape, token ids within tokenizer vocab ---"
python - <<'EOF'
import math
import os, glob
import numpy as np
from transformers import AutoTokenizer

d = os.environ["OUTPUT_DIR"]
tps = int(os.environ["TOKENS_PER_SHARD"])
target = int(float(os.environ["TARGET_TOKENS"]))
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

# The corpus is supposed to exceed the target, in which case every shard is
# full and the count is exact. Fewer/short shards mean the dataset ran dry:
# pretokenize.py marks that complete=True without complaint, so catch it here.
expected = math.ceil(target / tps)
assert len(paths) == expected, \
    f"{len(paths)} shards, expected {expected} — dataset exhausted before the target?"
assert lengths == {tps}, \
    f"shard lengths {sorted(lengths)} != {{{tps}}} — a short shard means the dataset ran dry"

# A tokenizer mismatch is the failure this catches: Llama ids (up to 128255) in a
# corpus labelled Phi silently mis-offset every hyper-token id at training time.
m = int(np.load(paths[0], mmap_mode="r")[:10_000_000].max())
print(f"max token id in first 10M of {os.path.basename(paths[0])} = {m}")
assert m < vocab, f"token id {m} >= vocab {vocab} — wrong tokenizer for this data!"
assert not glob.glob(os.path.join(d, "mask_*.npy")), \
    "unexpected mask_*.npy: this is a raw pretraining corpus, it must have no masks"
EOF

} 2>&1 | tee "$LOGFILE"
