# Data Pipeline

## Overview

Training data flows through two stages:

1. **Pre-tokenization**: Raw text → `.npy` shards of base token IDs
2. **On-the-fly LZW compression**: `Zip2ZipDataset` reads shards and compresses them into mixed base+hypertoken sequences at training time

## Pre-tokenization

```bash
uv run python scripts/pretokenize.py \
    --output_dir /path/to/tokens \
    --dataset epfl-dlab/llaza-20B \
    --target_tokens 20e9
```

### What it does

1. Loads a HuggingFace dataset (default: `epfl-dlab/llaza-20B`)
2. Tokenizes each document with the Llama 3 tokenizer, wrapping with `<bos>` / `<eos>`
3. Concatenates all documents into a continuous token stream
4. Writes fixed-size shards as `shard_00000.npy`, `shard_00001.npy`, etc.

Each shard contains ~1B tokens as `uint32` values. The script supports resuming — if shards already exist, it picks up from where it left off.

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--output_dir` | (required) | Output directory for `.npy` shards |
| `--dataset` | `epfl-dlab/llaza-20B` | HuggingFace dataset path |
| `--dataset_name` | None | Dataset config name (e.g. `sample-10BT`) |
| `--model_name` | `meta-llama/Llama-3.1-8B` | Tokenizer to use |
| `--tokens_per_shard` | 1B | Tokens per shard file |
| `--target_tokens` | 20B | Total tokens to produce |
| `--min_doc_length` | 50 | Filter out short documents (characters) |

### Loss masks (optional)

For SFT data where loss should only be computed on certain spans (e.g. assistant turns), the dataset supports **loss mask files**. For each `shard_00000.npy`, place a corresponding `mask_00000.npy` with the same length, where each element is 0 (ignore) or 1 (compute loss).

If any mask files are present, all shards must have matching masks.

## Zip2ZipDataset

`src/zip2zip_core/data.py` — an `IterableDataset` that reads pre-tokenized shards and applies LZW compression on-the-fly.

### LZW compression

Each training sample is produced by:

1. Reading a chunk of `seq_len * 2` base tokens from the current shard
2. Running `LZWCompressor.encode()` to compress them into a mixed sequence of base tokens and hypertokens
3. Truncating to `seq_len + 1` tokens (for input/label split)

The LZW compressor is configured with:
- `initial_vocab_size=128256` (Llama 3 vocab)
- `max_codebook_size=4096` (max hypertoken entries)
- `max_subtokens` (compression window, e.g. 2, 3, or 4)
- `disabled_ids` — all Llama 3.1 special tokens (128000–128255) are excluded from compression

### Codebook remapping

By default (`remap_codebook=True`), the dataset compacts each sample's codebook to only contain entries that actually appear in the sequence. This means hypertoken IDs are remapped to a dense range `[vocab_size, vocab_size + K)` where `K` is the number of unique hypertokens used in that sample.

This is important for efficiency — instead of the model operating over the full `max_codebook_size=4096` entries, it only processes the entries actually used (typically a few hundred per sample).

Disable with `--no_remap_codebook` for ablation experiments.

### Training modes

#### `lm` mode (default)

Standard next-token prediction on compressed sequences. The input is the compressed token stream, and labels are shifted by one position.

When loss masks are present, they are propagated through compression: a compressed token's mask is True only if all its constituent base tokens have mask=1.

#### `compress` mode

Explicit compression/decompression task. Each sample is formatted as:

```
<DECOMPRESS> base_tokens <COMPRESS> compressed_tokens    (compression direction)
<COMPRESS> compressed_tokens <DECOMPRESS> base_tokens    (decompression direction)
```

Direction is chosen randomly (50/50) per sample. Loss is only computed on the second part (the part the model must predict). Special token IDs used: `COMPRESS_TOKEN_ID = 128002`, `DECOMPRESS_TOKEN_ID = 128003`.

### Cross-document boundaries

Documents are concatenated end-to-end with `<bos>` and `<eos>` boundaries. The dataset reads fixed-size chunks from this stream, so a single chunk may span multiple documents. There is no special handling to prevent cross-document attention — the model sees the full sequence including document boundaries.

## Collation

`remap_collate_fn` pads variable-size codebooks to a fixed `max_active_codebook_size` so that `torch.compile` sees a single static tensor shape (avoiding recompilations).

Each batch contains:
- `input_ids`: `(B, seq_len)` — mixed base + hypertoken IDs
- `codebook`: `(B, max_active_codebook_size, max_subtokens)` — padded codebook entries
- `n_base_tokens`: `(B,)` — number of base tokens represented by the compressed sequence
- `labels`: `(B, seq_len)` — target token IDs (`-100` for masked positions)

## Shard distribution

Shards are distributed across ranks round-robin: rank `r` gets shards `r, r + world_size, r + 2*world_size, ...`. When using DataLoader workers, each worker further subdivides its rank's shards.

The dataset loops infinitely and supports checkpointing via `state_dict()` / `load_state_dict()` (tracks current shard index and offset).
