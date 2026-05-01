"""Pre-tokenize a text dataset with the Llama 3 tokenizer.

Saves tokenized data as numpy files for efficient loading during training.
Each shard contains ~100M tokens. Uses all available CPUs for parallel tokenization.

Usage:
    python scripts/pretokenize.py \
        --output_dir /path/to/output \
        --dataset HuggingFaceFW/fineweb \
        --dataset_name sample-10BT \
        --target_tokens 10.5e9
"""

import argparse
import os
import time
from multiprocessing import cpu_count

import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.1-8B")
    parser.add_argument("--dataset", type=str, default="epfl-dlab/zip2zip-plus-mixture-20b")
    parser.add_argument("--dataset_name", type=str, default=None)
    parser.add_argument("--dataset_split", type=str, default="train")
    parser.add_argument("--column", type=str, default="text")
    parser.add_argument("--tokens_per_shard", type=int, default=1000_000_000)
    parser.add_argument("--target_tokens", type=float, default=20e9)
    parser.add_argument("--min_doc_length", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=10_000)
    parser.add_argument("--num_workers", type=int, default=0, help="Number of workers (0 = all CPUs)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

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

    shard_idx = 0
    token_buffer = []
    total_tokens = 0
    docs_processed = 0
    target_tokens = int(args.target_tokens)

    # Check for existing shards to resume
    existing_shards = sorted([f for f in os.listdir(args.output_dir) if f.endswith(".npy")])
    if existing_shards:
        shard_idx = len(existing_shards)
        total_tokens = shard_idx * args.tokens_per_shard
        print(f"Resuming from shard {shard_idx}, ~{total_tokens/1e9:.2f}B tokens already done")
        if total_tokens >= target_tokens:
            print("Already have enough tokens!")
            return

    start_time = time.time()

    for sample in ds:
        token_buffer.extend(sample["tokens"])
        docs_processed += 1

        while len(token_buffer) >= args.tokens_per_shard:
            shard_tokens = np.array(token_buffer[:args.tokens_per_shard], dtype=np.uint32)
            shard_path = os.path.join(args.output_dir, f"shard_{shard_idx:05d}.npy")
            np.save(shard_path, shard_tokens)

            total_tokens += args.tokens_per_shard
            token_buffer = token_buffer[args.tokens_per_shard:]
            elapsed = time.time() - start_time
            tok_per_sec = total_tokens / elapsed
            print(f"Saved shard {shard_idx}: {shard_path} ({total_tokens/1e9:.2f}B tokens, {docs_processed} docs, {tok_per_sec/1e6:.2f}M tok/s)")
            shard_idx += 1

            if total_tokens >= target_tokens:
                break

        if total_tokens >= target_tokens:
            break

    # Save remaining tokens if any
    if token_buffer and total_tokens < target_tokens:
        shard_tokens = np.array(token_buffer, dtype=np.uint32)
        shard_path = os.path.join(args.output_dir, f"shard_{shard_idx:05d}.npy")
        np.save(shard_path, shard_tokens)
        total_tokens += len(token_buffer)
        print(f"Saved final shard {shard_idx}: {shard_path} ({total_tokens/1e9:.2f}B tokens)")

    elapsed = time.time() - start_time
    tok_per_sec = total_tokens / elapsed if elapsed > 0 else 0
    print(f"\nDone! Total: {total_tokens/1e9:.2f}B tokens in {shard_idx+1} shards")
    print(f"Documents processed: {docs_processed}")
    print(f"Elapsed: {elapsed:.1f}s, Average: {tok_per_sec/1e6:.2f}M tok/s")


if __name__ == "__main__":
    main()
