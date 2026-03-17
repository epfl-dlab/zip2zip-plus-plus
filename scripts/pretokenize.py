"""Pre-tokenize a text dataset with the Llama 3 tokenizer.

Saves tokenized data as numpy files for efficient loading during training.
Each shard contains ~100M tokens.

Usage:
    python scripts/pretokenize.py \
        --output_dir /path/to/output \
        --dataset HuggingFaceFW/fineweb \
        --dataset_name sample-10BT \
        --target_tokens 10.5e9
"""

import argparse
import os

import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.1-8B")
    parser.add_argument("--dataset", type=str, default="HuggingFaceFW/fineweb")
    parser.add_argument("--dataset_name", type=str, default="sample-10BT")
    parser.add_argument("--dataset_split", type=str, default="train")
    parser.add_argument("--column", type=str, default="text")
    parser.add_argument("--tokens_per_shard", type=int, default=100_000_000)
    parser.add_argument("--target_tokens", type=float, default=10.5e9)
    parser.add_argument("--min_doc_length", type=int, default=50)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading tokenizer from {args.model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    bos_id = tokenizer.bos_token_id
    eos_id = tokenizer.eos_token_id
    vocab_size = len(tokenizer)

    print(f"vocab_size={vocab_size}, bos={bos_id}, eos={eos_id}")
    print(f"Output dir: {args.output_dir}")
    print(f"Target: {args.target_tokens/1e9:.1f}B tokens")

    print("Loading dataset (streaming)...")
    ds = load_dataset(
        args.dataset,
        name=args.dataset_name,
        split=args.dataset_split,
        streaming=True,
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

    for sample in ds:
        if args.column not in sample:
            raise KeyError(f"Column '{args.column}' not found in sample. Available columns: {list(sample.keys())}")

        text = sample[args.column]
        if not text or len(text) < args.min_doc_length:
            continue

        tokens = tokenizer.encode(text, add_special_tokens=False)
        tokens = [bos_id] + tokens + [eos_id]
        token_buffer.extend(tokens)
        docs_processed += 1

        while len(token_buffer) >= args.tokens_per_shard:
            shard_tokens = np.array(token_buffer[:args.tokens_per_shard], dtype=np.uint32)
            shard_path = os.path.join(args.output_dir, f"shard_{shard_idx:05d}.npy")
            np.save(shard_path, shard_tokens)

            total_tokens += args.tokens_per_shard
            token_buffer = token_buffer[args.tokens_per_shard:]
            print(f"Saved shard {shard_idx}: {shard_path} ({total_tokens/1e9:.2f}B tokens, {docs_processed} docs)")
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

    print(f"\nDone! Total: {total_tokens/1e9:.2f}B tokens in {shard_idx+1} shards")
    print(f"Documents processed: {docs_processed}")


if __name__ == "__main__":
    main()
