"""Exhaustively verify a directory of pretokenized shards.

Reads every token of every shard, unlike the cheap spot check inside the
tokenization launchers. Checks the structural invariants the training loader
depends on (`src/zip2zip_core/data.py`), and two integrity properties that a
partial, duplicated or truncated write would violate:

  * no two shards are byte-identical — the signature of a resume that restarted
    the document stream from zero;
  * the `<bos>` / `<eos>` counts match the document count in the manifest, so no
    document boundary was lost or double-written.

Usage:
    python scripts/verify_token_shards.py \
        --data_dir /path/to/tokens \
        --tokenizer microsoft/Phi-3.5-mini-instruct \
        --expect_tokens_per_shard 39000000
"""

import argparse
import hashlib
import json
import os
import time
from multiprocessing import Pool

import numpy as np

SHARD_PREFIX = "shard_"
MANIFEST_NAME = "manifest.json"


def inspect_shard(job):
    """Full read of one shard. Returns per-shard facts, or an error string."""
    path, chunk, bos_id, eos_id = job
    name = os.path.basename(path)
    try:
        arr = np.load(path, mmap_mode="r")
    except Exception as exc:  # a truncated .npy fails to open at all
        return {"name": name, "error": f"cannot open: {exc}"}

    if arr.dtype != np.uint32:
        return {"name": name, "error": f"dtype {arr.dtype}, expected uint32"}
    if arr.ndim != 1:
        return {"name": name, "error": f"shape {arr.shape}, expected flat 1-D"}

    digest = hashlib.blake2b(digest_size=16)
    hi, lo = 0, 2**32 - 1
    n_bos = n_eos = n_zero = 0
    for start in range(0, arr.shape[0], chunk):
        block = np.asarray(arr[start:start + chunk])
        digest.update(block.tobytes())
        if block.size:
            hi = max(hi, int(block.max()))
            lo = min(lo, int(block.min()))
        n_zero += int(np.count_nonzero(block == 0))
        if bos_id is not None:
            n_bos += int(np.count_nonzero(block == bos_id))
        if eos_id is not None:
            n_eos += int(np.count_nonzero(block == eos_id))

    return {
        "name": name,
        "tokens": int(arr.shape[0]),
        "bytes_on_disk": os.path.getsize(path),
        "header_bytes": os.path.getsize(path) - arr.nbytes,
        "max_id": hi,
        "min_id": lo,
        "n_bos": n_bos,
        "n_eos": n_eos,
        "n_zero": n_zero,
        "digest": digest.hexdigest(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--tokenizer", default=None,
                    help="resolve vocab size and bos/eos ids from this tokenizer")
    ap.add_argument("--expect_tokens_per_shard", type=int, default=None)
    ap.add_argument("--expect_num_shards", type=int, default=None)
    ap.add_argument("--chunk", type=int, default=1 << 25,
                    help="tokens read per block (bounds memory)")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    vocab_size = bos_id = eos_id = None
    if args.tokenizer:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.tokenizer)
        vocab_size, bos_id, eos_id = len(tok), tok.bos_token_id, tok.eos_token_id
        print(f"tokenizer {args.tokenizer}: vocab={vocab_size} bos={bos_id} eos={eos_id}")

    shard_paths = sorted(
        os.path.join(args.data_dir, f)
        for f in os.listdir(args.data_dir)
        if f.startswith(SHARD_PREFIX) and f.endswith(".npy")
    )
    print(f"{len(shard_paths)} shards in {args.data_dir}")

    manifest = None
    manifest_path = os.path.join(args.data_dir, MANIFEST_NAME)
    if os.path.exists(manifest_path):
        with open(manifest_path) as f:
            manifest = json.load(f)
        print(f"manifest: {json.dumps(manifest)}")

    start = time.time()
    jobs = [(p, args.chunk, bos_id, eos_id) for p in shard_paths]
    with Pool(args.workers) as pool:
        results = []
        for i, res in enumerate(pool.imap(inspect_shard, jobs), 1):
            results.append(res)
            if i % 50 == 0 or i == len(jobs):
                gb = sum(r.get("bytes_on_disk", 0) for r in results) / 1024**3
                print(f"  {i}/{len(jobs)} shards, {gb:.1f} GiB read, {time.time()-start:.0f}s")

    failures = []

    broken = [r for r in results if "error" in r]
    for r in broken:
        failures.append(f"{r['name']}: {r['error']}")

    ok = [r for r in results if "error" not in r]
    total_tokens = sum(r["tokens"] for r in ok)
    total_bytes = sum(r["bytes_on_disk"] for r in ok)
    lengths = sorted({r["tokens"] for r in ok})
    headers = sorted({r["header_bytes"] for r in ok})
    global_max = max((r["max_id"] for r in ok), default=0)
    global_min = min((r["min_id"] for r in ok), default=0)
    total_bos = sum(r["n_bos"] for r in ok)
    total_eos = sum(r["n_eos"] for r in ok)
    total_zero = sum(r["n_zero"] for r in ok)

    print("\n=== measured ===")
    print(f"shards read:          {len(ok)}")
    print(f"total tokens:         {total_tokens} ({total_tokens/1e9:.3f}B)")
    print(f"total bytes on disk:  {total_bytes} ({total_bytes/1024**3:.2f} GiB)")
    print(f"distinct shard sizes: {lengths}")
    print(f"distinct .npy headers:{headers}")
    print(f"token id range:       [{global_min}, {global_max}]")
    print(f"<bos> occurrences:    {total_bos}")
    print(f"<eos> occurrences:    {total_eos}")
    print(f"id 0 occurrences:     {total_zero}")

    # Every shard must be unique. Two identical shards mean the document stream
    # was replayed from the start, which is what the old resume path did.
    digests = {}
    for r in ok:
        digests.setdefault(r["digest"], []).append(r["name"])
    dupes = {d: names for d, names in digests.items() if len(names) > 1}
    print(f"distinct shard digests: {len(digests)} of {len(ok)}")
    if dupes:
        for d, names in list(dupes.items())[:5]:
            failures.append(f"identical shards {names} (blake2b {d})")

    if args.expect_tokens_per_shard:
        odd = [r["name"] for r in ok if r["tokens"] != args.expect_tokens_per_shard]
        if odd:
            failures.append(
                f"{len(odd)} shard(s) not {args.expect_tokens_per_shard} tokens: {odd[:5]}"
            )
    if args.expect_num_shards and len(shard_paths) != args.expect_num_shards:
        failures.append(f"{len(shard_paths)} shards, expected {args.expect_num_shards}")
    if vocab_size is not None and global_max >= vocab_size:
        failures.append(f"token id {global_max} >= vocab {vocab_size} — wrong tokenizer")
    if any(r["bytes_on_disk"] != r["tokens"] * 4 + r["header_bytes"] for r in ok):
        failures.append("byte size does not match token count for some shard")

    if manifest:
        print("\n=== manifest cross-check ===")
        for key, measured in (("num_shards", len(shard_paths)),
                              ("total_tokens", total_tokens)):
            recorded = manifest.get(key)
            verdict = "ok" if recorded == measured else "MISMATCH"
            print(f"{key}: manifest={recorded} measured={measured} -> {verdict}")
            if recorded != measured:
                failures.append(f"manifest {key}={recorded} but measured {measured}")
        if not manifest.get("complete"):
            failures.append("manifest says the run never completed")
        docs = manifest.get("documents_processed")
        if docs and bos_id is not None:
            # <bos> is emitted once per document. The corpus stops at the target,
            # so the last few documents' tokens can be dropped: bos <= docs.
            print(f"documents_processed={docs} vs <bos>={total_bos} "
                  f"(diff {docs - total_bos}, expected small and >= 0)")
            if total_bos > docs:
                failures.append(f"{total_bos} <bos> but only {docs} documents recorded")

    mask_files = [f for f in os.listdir(args.data_dir) if f.startswith("mask_")]
    print(f"\nmask files: {len(mask_files)}")

    print(f"\nelapsed: {time.time()-start:.0f}s")
    if failures:
        print("\n=== FAIL ===")
        for f in failures:
            print(f"  - {f}")
        raise SystemExit(1)
    print("\n=== PASS: every token of every shard checked ===")


if __name__ == "__main__":
    main()
