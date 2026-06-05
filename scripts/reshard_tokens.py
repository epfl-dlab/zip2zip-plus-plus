#!/usr/bin/env python3
"""Reshard pre-tokenized token streams without re-tokenizing.

This script reads existing token shards from an input directory and rewrites
them into a new directory with a different shard count. It streams data shard
by shard, so peak memory stays bounded by the target shard size rather than the
full dataset size.

If matching `mask_*.npy` files are present for every token shard, they are
resharded in lock-step.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--num_shards", type=int, required=True)
    parser.add_argument(
        "--prefix",
        type=str,
        default="shard_",
        help="Filename prefix for token shards in both input and output dirs.",
    )
    return parser.parse_args()


def find_token_shards(input_dir: Path, prefix: str) -> list[Path]:
    shards = sorted(
        p
        for p in input_dir.iterdir()
        if p.is_file() and p.name.startswith(prefix) and p.suffix == ".npy"
    )
    if not shards:
        raise ValueError(f"No token shards matching {prefix}*.npy found in {input_dir}")
    return shards


def find_mask_shards(token_shards: list[Path]) -> list[Path] | None:
    mask_shards = [p.with_name(p.name.replace("shard_", "mask_", 1)) for p in token_shards]
    has_masks = [p.exists() for p in mask_shards]
    if any(has_masks) and not all(has_masks):
        missing = [str(p) for p, ok in zip(mask_shards, has_masks) if not ok]
        raise ValueError(f"Found partial mask shards; missing examples: {missing[:3]}")
    return mask_shards if all(has_masks) else None


def load_shard(path: Path, dtype: np.dtype) -> np.ndarray:
    arr = np.load(path, mmap_mode="r")
    if arr.dtype != dtype:
        raise ValueError(f"{path} has dtype {arr.dtype}, expected {dtype}")
    if arr.ndim != 1:
        raise ValueError(f"{path} has shape {arr.shape}, expected flat 1D array")
    return arr


def compute_target_sizes(total_items: int, num_shards: int) -> list[int]:
    base = total_items // num_shards
    return [base] * num_shards


def save_array(path: Path, array: np.ndarray) -> None:
    np.save(path, array)


def reshard_stream(
    input_shards: list[Path],
    output_dir: Path,
    target_sizes: list[int],
    output_prefix: str,
    dtype: np.dtype,
) -> None:
    out_idx = 0
    out_pos = 0
    out_buf = np.empty(target_sizes[out_idx], dtype=dtype)

    for in_idx, shard_path in enumerate(input_shards):
        shard = load_shard(shard_path, dtype)
        in_pos = 0
        shard_len = shard.shape[0]
        print(f"[{in_idx + 1}/{len(input_shards)}] Reading {shard_path.name} ({shard_len:,} items)")

        while in_pos < shard_len:
            remaining_in = shard_len - in_pos
            remaining_out = target_sizes[out_idx] - out_pos
            take = min(remaining_in, remaining_out)
            out_buf[out_pos : out_pos + take] = shard[in_pos : in_pos + take]
            in_pos += take
            out_pos += take

            if out_pos == target_sizes[out_idx]:
                out_path = output_dir / f"{output_prefix}{out_idx:05d}.npy"
                save_array(out_path, out_buf)
                print(f"  wrote {out_path.name} ({target_sizes[out_idx]:,} items)")
                out_idx += 1
                if out_idx == len(target_sizes):
                    break
                out_buf = np.empty(target_sizes[out_idx], dtype=dtype)
                out_pos = 0

    if out_idx != len(target_sizes):
        raise RuntimeError(
            f"Expected to write {len(target_sizes)} shards, only wrote {out_idx}"
        )


def main() -> None:
    args = parse_args()
    if args.num_shards <= 0:
        raise ValueError("--num_shards must be positive")

    args.output_dir.mkdir(parents=True, exist_ok=False)

    token_shards = find_token_shards(args.input_dir, args.prefix)
    mask_shards = find_mask_shards(token_shards)

    token_lengths = [load_shard(path, np.uint32).shape[0] for path in token_shards]
    total_tokens = sum(token_lengths)
    target_sizes = compute_target_sizes(total_tokens, args.num_shards)

    print(f"Input dir: {args.input_dir}")
    print(f"Output dir: {args.output_dir}")
    print(f"Input token shards: {len(token_shards)}")
    print(f"Output token shards: {args.num_shards}")
    print(f"Total tokens: {total_tokens:,}")
    print(
        "Target shard sizes: "
        f"min={min(target_sizes):,}, max={max(target_sizes):,}, "
        f"remainder={total_tokens % args.num_shards}"
    )

    reshard_stream(
        input_shards=token_shards,
        output_dir=args.output_dir,
        target_sizes=target_sizes,
        output_prefix="shard_",
        dtype=np.uint32,
    )

    if mask_shards is not None:
        total_masks = sum(load_shard(path, np.uint8).shape[0] for path in mask_shards)
        if total_masks != total_tokens:
            raise ValueError(
                f"Total mask items {total_masks:,} do not match total tokens {total_tokens:,}"
            )
        print("Resharding matching mask shards...")
        reshard_stream(
            input_shards=mask_shards,
            output_dir=args.output_dir,
            target_sizes=target_sizes,
            output_prefix="mask_",
            dtype=np.uint8,
        )

    print("Done.")


if __name__ == "__main__":
    main()
