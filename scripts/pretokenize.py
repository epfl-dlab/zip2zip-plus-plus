"""Pre-tokenize a text dataset into flat token shards.

Saves tokenized data as numpy files for efficient loading during training: one
`shard_%05d.npy` of `--tokens_per_shard` uint32 token ids per shard, plus a
`manifest.json` recording progress and provenance. Uses half the available CPUs
for parallel tokenization by default.

Interrupted runs resume exactly. The manifest records how many documents were
consumed from the shuffled stream — deterministic, since the seed is fixed — and
a `pending_after_*.bin` file holds the tokens left over after the last full
shard. Both are rewritten atomically after every shard. A shard directory with
no manifest cannot be resumed, because there is no way to know where the stream
stopped; the script refuses to touch it rather than write duplicates. See
--assume_docs_processed.

Usage:
    python scripts/pretokenize.py \
        --output_dir /path/to/output \
        --dataset epfl-dlab/llaza-20B \
        --model_name microsoft/Phi-3.5-mini-instruct \
        --target_tokens 20e9 \
        --tokens_per_shard 39000000
"""

import argparse
import json
import os
import time
from multiprocessing import cpu_count

import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer

SHARD_PREFIX = "shard_"
PENDING_PREFIX = "pending_after_"
MANIFEST_NAME = "manifest.json"


def _pending_name(num_shards):
    """Leftover-token file for a given shard count.

    Deliberately not a `.npy` file: the training loader treats every non-`mask_`
    `.npy` in the directory as a token shard (src/zip2zip_core/data.py).
    """
    return f"{PENDING_PREFIX}{num_shards:05d}.bin"


def _write_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)


def _write_bin(path, arr):
    tmp = path + ".tmp"
    arr.tofile(tmp)
    os.replace(tmp, path)


def find_shards(output_dir):
    """Token shards on disk, in load order."""
    if not os.path.isdir(output_dir):
        return []
    return sorted(
        os.path.join(output_dir, f)
        for f in os.listdir(output_dir)
        if f.startswith(SHARD_PREFIX) and f.endswith(".npy")
    )


def measure_shards(shard_paths):
    """Token count of each shard, read from the .npy headers only."""
    lengths = []
    for path in shard_paths:
        arr = np.load(path, mmap_mode="r")
        if arr.ndim != 1 or arr.dtype != np.uint32:
            raise SystemExit(
                f"{path}: expected a flat 1-D uint32 array, got shape {arr.shape} "
                f"dtype {arr.dtype} — not a shard written by this script."
            )
        lengths.append(int(arr.shape[0]))
    return lengths


def save_progress(output_dir, num_shards, total_tokens, docs_processed,
                  tokens_per_shard, token_buffer, complete):
    """Record enough state to resume exactly. Called after every shard.

    Pending file first, manifest second: the manifest is what makes a pending
    file live, so a crash between the two leaves the older consistent state in
    place. Stale pending files are dropped only once the new manifest is
    committed.
    """
    pending_path = os.path.join(output_dir, _pending_name(num_shards))
    _write_bin(pending_path, np.array(token_buffer, dtype=np.uint32))

    _write_json(os.path.join(output_dir, MANIFEST_NAME), {
        "total_tokens": total_tokens,
        "documents_processed": docs_processed,
        "num_shards": num_shards,
        "tokens_per_shard": tokens_per_shard,
        "pending_tokens": len(token_buffer),
        "has_loss_masks": False,
        "token_dtype": "uint32",
        "complete": complete,
    })

    keep = os.path.basename(pending_path)
    for name in os.listdir(output_dir):
        if name.startswith(PENDING_PREFIX) and name != keep:
            os.remove(os.path.join(output_dir, name))


def load_progress(output_dir, shard_paths, tokens_per_shard, assume_docs_processed):
    """Reconstruct the resume point from an existing shard directory.

    Refuses to guess. Resuming needs the number of documents already consumed,
    and that cannot be inferred from the shards: the shuffle seed is fixed, so
    re-reading the stream from document 0 writes byte-identical duplicates of
    the existing shards into higher-numbered files.
    """
    lengths = measure_shards(shard_paths)
    num_shards = len(shard_paths)
    total_tokens = sum(lengths)

    manifest = None
    manifest_path = os.path.join(output_dir, MANIFEST_NAME)
    if os.path.exists(manifest_path):
        with open(manifest_path) as f:
            manifest = json.load(f)

    complete = bool(manifest and manifest.get("complete"))

    if manifest and "documents_processed" in manifest:
        if manifest.get("num_shards") != num_shards:
            raise SystemExit(
                f"Inconsistent state in {output_dir}: {MANIFEST_NAME} records "
                f"{manifest.get('num_shards')} shards but {num_shards} are on disk. "
                "The job died mid-flush. Delete the highest-numbered "
                f"{SHARD_PREFIX}*.npy and re-run to resume exactly."
            )
        docs_processed = int(manifest["documents_processed"])
    elif assume_docs_processed >= 0:
        docs_processed = assume_docs_processed
        print(f"No usable {MANIFEST_NAME}; trusting --assume_docs_processed {docs_processed}.")
    else:
        raise SystemExit(
            f"{output_dir} holds {num_shards} shard(s) but no usable {MANIFEST_NAME}, "
            "so the resume point is unknown. Continuing would re-read the dataset "
            "from document 0 and duplicate the existing shards. Either empty the "
            "directory and start over, or pass --assume_docs_processed N if you know "
            "how many documents produced those shards (every shard line in the run "
            "log prints the count)."
        )

    if not complete:
        short = [
            (os.path.basename(p), n)
            for p, n in zip(shard_paths, lengths)
            if n != tokens_per_shard
        ]
        if short:
            raise SystemExit(
                f"Shard(s) in {output_dir} are not {tokens_per_shard} tokens long: "
                f"{short[:3]}. Either a shard was truncated by a kill, or these were "
                "written with a different --tokens_per_shard. Delete them and re-run."
            )

    token_buffer = []
    pending_path = os.path.join(output_dir, _pending_name(num_shards))
    if os.path.exists(pending_path):
        token_buffer = np.fromfile(pending_path, dtype=np.uint32).tolist()
    elif not complete:
        print(
            f"WARNING: {os.path.basename(pending_path)} is missing, so the partial "
            "shard buffer was lost. Resuming leaves a gap of at most one document."
        )

    return {
        "num_shards": num_shards,
        "total_tokens": total_tokens,
        "docs_processed": docs_processed,
        "token_buffer": token_buffer,
        "complete": complete,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.1-8B")
    parser.add_argument("--dataset", type=str, default="epfl-dlab/llaza-20B")
    parser.add_argument("--dataset_name", type=str, default=None)
    parser.add_argument("--dataset_split", type=str, default="train")
    parser.add_argument("--column", type=str, default="text")
    parser.add_argument("--tokens_per_shard", type=int, default=1000_000_000)
    parser.add_argument("--target_tokens", type=float, default=20e9)
    parser.add_argument("--min_doc_length", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=10_000)
    parser.add_argument("--num_workers", type=int, default=0, help="Number of workers (0 = all CPUs)")
    parser.add_argument("--download_num_proc", type=int, default=0,
                        help="Parallel processes for load_dataset's download+prepare "
                             "phase (0 = datasets' default single process — the "
                             "historical behavior). Parallelizes across the repo's "
                             "source files without changing the prepared row order, "
                             "so shuffle(seed=42) output is unchanged.")
    parser.add_argument("--assume_docs_processed", type=int, default=-1,
                        help="Resume a shard directory that has no manifest.json by "
                             "supplying the number of documents already consumed. A "
                             "wrong value silently duplicates or drops data.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    target_tokens = int(args.target_tokens)

    # Inspect a previous attempt before touching the dataset, so an unresumable
    # directory fails in seconds rather than after the download and tokenization.
    resume = None
    shard_paths = find_shards(args.output_dir)
    if shard_paths:
        resume = load_progress(args.output_dir, shard_paths, args.tokens_per_shard,
                               args.assume_docs_processed)
        if resume["complete"]:
            print(f"{args.output_dir} is already complete: "
                  f"{resume['total_tokens']/1e9:.2f}B tokens in {resume['num_shards']} shards.")
            return
        if resume["total_tokens"] >= target_tokens:
            print(f"Already have enough tokens: {resume['total_tokens']/1e9:.2f}B "
                  f"in {resume['num_shards']} shards.")
            return
        print(f"Resuming from shard {resume['num_shards']}: "
              f"{resume['total_tokens']/1e9:.2f}B tokens done, "
              f"{resume['docs_processed']} documents consumed, "
              f"{len(resume['token_buffer'])} tokens pending")

    num_workers = args.num_workers if args.num_workers > 0 else cpu_count()//2

    print(f"Loading tokenizer from {args.model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    bos_id = tokenizer.bos_token_id
    eos_id = tokenizer.eos_token_id
    vocab_size = len(tokenizer)

    print(f"vocab_size={vocab_size}, bos={bos_id}, eos={eos_id}")
    print(f"Output dir: {args.output_dir}")
    print(f"Target: {args.target_tokens/1e9:.1f}B tokens")
    print(f"Using {num_workers} workers, batch_size={args.batch_size}")

    print("Loading dataset...")
    load_kwargs = {
        "path": args.dataset,
        "split": args.dataset_split,
    }
    if args.dataset_name:
        load_kwargs["name"] = args.dataset_name
    if args.download_num_proc > 0:
        load_kwargs["num_proc"] = args.download_num_proc
    ds = load_dataset(**load_kwargs)
    ds = ds.shuffle(seed=42)

    # Filter short documents
    ds = ds.filter(
        lambda x: x[args.column] is not None and len(x[args.column]) >= args.min_doc_length,
        num_proc=num_workers,
    )

    # Tokenize in parallel using datasets .map()
    def tokenize(batch):
        all_tokens = []
        for text in batch[args.column]:
            tokens = tokenizer.encode(text, add_special_tokens=False)
            all_tokens.append([bos_id] + tokens + [eos_id])
        return {"tokens": all_tokens}

    ds = ds.map(
        tokenize,
        batched=True,
        batch_size=args.batch_size,
        num_proc=num_workers,
        remove_columns=ds.column_names,
    )

    # Skip the documents the previous attempt already consumed. Sound because
    # the shuffle seed is fixed, so this stream is identical run to run.
    if resume and resume["docs_processed"]:
        skip = resume["docs_processed"]
        if hasattr(ds, "select"):
            if skip >= len(ds):
                print(f"All {len(ds)} documents were already consumed; nothing left to do.")
                save_progress(args.output_dir, resume["num_shards"], resume["total_tokens"],
                              skip, args.tokens_per_shard, [], complete=True)
                return
            ds = ds.select(range(skip, len(ds)))
        else:
            ds = ds.skip(skip)

    shard_idx = resume["num_shards"] if resume else 0
    token_buffer = list(resume["token_buffer"]) if resume else []
    total_tokens = resume["total_tokens"] if resume else 0
    docs_processed = resume["docs_processed"] if resume else 0
    resume_tokens = total_tokens

    start_time = time.time()

    for sample in ds:
        token_buffer.extend(sample["tokens"])
        docs_processed += 1

        while len(token_buffer) >= args.tokens_per_shard:
            shard_tokens = np.array(token_buffer[:args.tokens_per_shard], dtype=np.uint32)
            shard_path = os.path.join(args.output_dir, f"{SHARD_PREFIX}{shard_idx:05d}.npy")
            np.save(shard_path, shard_tokens)

            total_tokens += args.tokens_per_shard
            token_buffer = token_buffer[args.tokens_per_shard:]
            shard_idx += 1
            save_progress(args.output_dir, shard_idx, total_tokens, docs_processed,
                          args.tokens_per_shard, token_buffer, complete=False)
            elapsed = time.time() - start_time
            tok_per_sec = (total_tokens - resume_tokens) / elapsed if elapsed > 0 else 0
            print(f"Saved shard {shard_idx - 1}: {shard_path} ({total_tokens/1e9:.2f}B tokens, {docs_processed} docs, {tok_per_sec/1e6:.2f}M tok/s)")

            if total_tokens >= target_tokens:
                break

        if total_tokens >= target_tokens:
            break

    # Save remaining tokens if any
    if token_buffer and total_tokens < target_tokens:
        shard_tokens = np.array(token_buffer, dtype=np.uint32)
        shard_path = os.path.join(args.output_dir, f"{SHARD_PREFIX}{shard_idx:05d}.npy")
        np.save(shard_path, shard_tokens)
        total_tokens += len(token_buffer)
        shard_idx += 1
        print(f"Saved final shard {shard_idx - 1}: {shard_path} ({total_tokens/1e9:.2f}B tokens)")

    save_progress(args.output_dir, shard_idx, total_tokens, docs_processed,
                  args.tokens_per_shard, [], complete=True)

    elapsed = time.time() - start_time
    tok_per_sec = (total_tokens - resume_tokens) / elapsed if elapsed > 0 else 0
    print(f"\nDone! Total: {total_tokens/1e9:.2f}B tokens in {shard_idx} shards")
    print(f"Documents processed: {docs_processed}")
    print(f"Elapsed: {elapsed:.1f}s, Average: {tok_per_sec/1e6:.2f}M tok/s (this session)")


if __name__ == "__main__":
    main()
