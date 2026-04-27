"""
Measure how much LZW segmentation changes across merge sizes.

For the same base token sequences, compress with ms=2,3,4,8 and compute
pairwise token-level edit distance between the resulting segmentations,
treating each compressed token (expanded to its base-token tuple) as an atom.

High edit distance across ms transitions → curriculum transfer hard.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))

import numpy as np
from zip2zip_compression import LZWCompressor

# ── Config ────────────────────────────────────────────────────────────────────
SHARD       = "/mnt/scratch/data/finemath-10bt/shard_00000.npy"
SEQ_LEN     = 512          # base tokens per sample
N_SAMPLES   = 200
VOCAB_SIZE  = 128256       # Llama-3
PAD_ID      = 128001
MAX_CB      = 4096
MS_LIST     = [2, 3, 4, 8]
SEED        = 42
# ─────────────────────────────────────────────────────────────────────────────


def levenshtein(a, b):
    """Token-level edit distance between two lists (each element is a tuple)."""
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, n + 1):
            temp = dp[j]
            dp[j] = prev if a[i-1] == b[j-1] else 1 + min(prev, dp[j], dp[j-1])
            prev = temp
    return dp[n]


def compress_to_chunks(sequences, ms):
    """Returns list of segmentations: each segmentation is a list of tuples."""
    compressor = LZWCompressor(
        initial_vocab_size=VOCAB_SIZE,
        max_codebook_size=MAX_CB,
        max_subtokens=ms,
        pad_token_id=PAD_ID,
        disabled_ids=[],
    )
    encodings, _, codebooks = compressor.batch_encode(sequences)
    results = []
    for enc, cb in zip(encodings, codebooks):
        cb_dict = cb.to_dict()
        chunks = []
        for tok_id in enc:
            if tok_id == PAD_ID:
                break
            chunks.append(tuple(cb_dict[tok_id]) if tok_id in cb_dict else (tok_id,))
        results.append(chunks)
    return results


def norm_edit(a, b):
    d = levenshtein(a, b)
    return d / max(len(a), len(b), 1)


def main():
    rng = np.random.default_rng(SEED)
    data = np.load(SHARD)

    # Sample N_SAMPLES non-overlapping sequences
    total = len(data) // SEQ_LEN
    idxs = rng.choice(total, size=N_SAMPLES, replace=False)
    sequences = [data[i*SEQ_LEN:(i+1)*SEQ_LEN].tolist() for i in idxs]

    print(f"Loaded {N_SAMPLES} sequences of length {SEQ_LEN}")
    print(f"Compressing with ms = {MS_LIST} ...\n")

    # Compress each ms
    segmentations = {}
    for ms in MS_LIST:
        print(f"  ms={ms} ...", end=" ", flush=True)
        segmentations[ms] = compress_to_chunks(sequences, ms)
        avg_len = np.mean([len(s) for s in segmentations[ms]])
        print(f"avg compressed len = {avg_len:.1f}  "
              f"(ratio = {SEQ_LEN/avg_len:.2f}x)")

    # Pairwise edit distances
    print("\nNormalized edit distance between segmentations (mean ± std):")
    print(f"{'Pair':<12}  {'mean':>8}  {'std':>8}  {'median':>8}")
    print("-" * 45)
    pairs = [(MS_LIST[i], MS_LIST[i+1]) for i in range(len(MS_LIST)-1)]
    pairs += [(MS_LIST[0], MS_LIST[-1])]  # also ms=2 vs ms=8

    for ms_a, ms_b in pairs:
        dists = [norm_edit(segmentations[ms_a][i], segmentations[ms_b][i])
                 for i in range(N_SAMPLES)]
        dists = np.array(dists)
        print(f"ms={ms_a}→ms={ms_b}  {dists.mean():>8.3f}  "
              f"{dists.std():>8.3f}  {np.median(dists):>8.3f}")

    # Baseline: same ms, two halves of the data (different sequences, same ms)
    print("\nBaseline: same ms=2, different sequences (should be ~0 by construction)")
    # Instead: compare ms=2 sequence to itself shifted by 1 → should be 0
    # Better baseline: ms=2 vs ms=2 on DIFFERENT sequences of similar content
    # Approximate by comparing odd vs even indexed samples
    base_dists = [norm_edit(segmentations[2][i], segmentations[2][i])
                  for i in range(N_SAMPLES)]
    print(f"  self-distance ms=2: {np.mean(base_dists):.3f} (sanity check, should be 0.000)")


if __name__ == "__main__":
    main()
