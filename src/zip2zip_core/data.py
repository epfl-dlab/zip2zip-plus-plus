"""Zip2Zip dataset and data loading utilities."""

import os
import random
from functools import partial

import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset, get_worker_info
from torch.distributed.checkpoint.stateful import Stateful

from zip2zip_compression import LZWCompressor

# Special tokens for compression/decompression task (unused Llama3 special token slots)
COMPRESS_TOKEN_ID = 128002
DECOMPRESS_TOKEN_ID = 128003
LLAMA31_8B_SPECIAL_TOKEN_IDS = tuple(range(128000, 128256))


class Zip2ZipDataset(IterableDataset, Stateful):
    """Pre-tokenized dataset with on-the-fly LZW compression.

    Supports two modes:
    - "lm": Language modeling on compressed sequences (original behavior).
    - "compress": Compression/decompression task. Sequences have format
      <DECOMPRESS> base_tokens <COMPRESS> compressed_tokens (or reverse).
      Loss is only computed on the second part.

    Yields (input_dict, labels) where codebook is compacted to only contain
    entries that appear in the token sequence (variable size per sample).
    """

    def __init__(
        self,
        data_dir: str,
        seq_len: int,
        max_subtokens: int,
        max_codebook_size: int = 4096,
        initial_vocab_size: int = 128256,
        bos_token_id: int = 128000,
        eos_token_id: int = 128001,
        pad_token_id: int = 128001,
        rank: int = 0,
        world_size: int = 1,
        mode: str = "lm",
        remap_codebook: bool = True,
    ):
        self.data_dir = data_dir
        self.seq_len = seq_len
        self.max_subtokens = max_subtokens
        self.max_codebook_size = max_codebook_size
        self.initial_vocab_size = initial_vocab_size
        self.pad_token_id = pad_token_id
        self.mode = mode
        self.remap_codebook = remap_codebook
        self.base_chunk_len = seq_len * 2

        shard_files = sorted(
            [
                os.path.join(data_dir, f)
                for f in os.listdir(data_dir)
                if f.endswith(".npy") and not f.startswith("mask_")
            ]
        )
        if not shard_files:
            raise ValueError(f"No .npy files in {data_dir}")

        mask_files = [self._mask_path_for_shard(path) for path in shard_files]
        has_masks = [os.path.exists(path) for path in mask_files]
        if any(has_masks) and not all(has_masks):
            missing = [path for path, exists in zip(mask_files, has_masks) if not exists]
            raise ValueError(f"Found partial loss masks in {data_dir}; missing {missing[:3]}")

        # Distribute shards
        self.shard_files = shard_files[rank::world_size]
        self.mask_files = mask_files[rank::world_size] if all(has_masks) else None
        self._shard_idx = 0
        self._offset = 0

        self.compressor_args = dict(
            initial_vocab_size=initial_vocab_size,
            max_codebook_size=max_codebook_size,
            max_subtokens=max_subtokens,
            pad_token_id=pad_token_id,
            disabled_ids=list(LLAMA31_8B_SPECIAL_TOKEN_IDS),
        )

    @staticmethod
    def _mask_path_for_shard(shard_path: str) -> str:
        dirname = os.path.dirname(shard_path)
        basename = os.path.basename(shard_path)
        if basename.startswith("shard_"):
            return os.path.join(dirname, basename.replace("shard_", "mask_", 1))
        return os.path.join(dirname, f"mask_{basename}")

    @staticmethod
    def _load_token_shard(shard_path: str):
        try:
            return np.load(shard_path, mmap_mode="r")
        except ValueError as exc:
            # Dolma writes flat binary arrays with a .npy suffix, not np.save files.
            if "pickle" not in str(exc).lower():
                raise
            dtype = np.dtype(np.uint32)
            size = os.path.getsize(shard_path)
            if size % dtype.itemsize != 0:
                raise ValueError(
                    f"{shard_path} is neither a NumPy .npy file nor a raw uint32 shard"
                ) from exc
            return np.memmap(
                shard_path,
                dtype=dtype,
                mode="r",
                shape=(size // dtype.itemsize,),
            )

    def _codebook_to_tensor(self, codebook) -> torch.LongTensor:
        cb_dict = codebook.to_dict()
        cb_list = []
        for i in range(self.max_codebook_size):
            hyper_id = self.initial_vocab_size + i
            if hyper_id in cb_dict:
                entry = list(cb_dict[hyper_id])
                while len(entry) < self.max_subtokens:
                    entry.append(self.pad_token_id)
                cb_list.append(entry[: self.max_subtokens])
            else:
                cb_list.append([self.pad_token_id] * self.max_subtokens)
        return torch.LongTensor(cb_list)

    def _remap_hyper_ids(self, x, y, codebook):
        """Remap hyper IDs to a compact 0..K-1 range, return compacted codebook.

        Args:
            x: (T,) input token IDs
            y: (T,) label token IDs
            codebook: (max_codebook_size, max_subtokens) full codebook

        Returns:
            x_new, y_new: remapped token IDs with hyper IDs in [vocab, vocab+K)
            compact_cb: (K, max_subtokens) only the used codebook entries
        """
        vocab = self.initial_vocab_size

        # Find all unique hyper codebook indices used in input or labels
        all_tokens = torch.cat([x, y])
        hyper_mask = all_tokens >= vocab
        if not hyper_mask.any():
            # No hyper tokens — return 0-entry codebook
            empty = torch.full(
                (0, self.max_subtokens), self.pad_token_id, dtype=codebook.dtype
            )
            return x, y, empty

        used_cb_indices = (all_tokens[hyper_mask] - vocab).unique(sorted=True)
        K = used_cb_indices.shape[0]

        # Compact codebook: only the used rows
        compact_cb = codebook[used_cb_indices]  # (K, max_subtokens)

        # Build old→new index remap
        remap = torch.empty(self.max_codebook_size, dtype=torch.long)
        remap[used_cb_indices] = torch.arange(K)

        # Remap tokens in-place
        x_new = x.clone()
        y_new = y.clone()
        for tokens in (x_new, y_new):
            mask = tokens >= vocab
            if mask.any():
                tokens[mask] = vocab + remap[tokens[mask] - vocab]

        return x_new, y_new, compact_cb

    def _compressed_loss_mask(self, compressed, base_mask, codebook):
        cb_dict = codebook.to_dict()
        offset = 0
        compressed_mask = []
        for tok in compressed:
            span_len = len(cb_dict[tok]) if tok >= self.initial_vocab_size else 1
            span = base_mask[offset : offset + span_len]
            compressed_mask.append(bool(np.all(span)))
            offset += span_len
        return torch.BoolTensor(compressed_mask)

    def _iter_lm(self, shards, mask_shards=None):
        """Original language modeling mode: next-token prediction on compressed sequences."""
        while True:
            for shard_idx in range(self._shard_idx, len(shards)):
                self._shard_idx = shard_idx
                shard = self._load_token_shard(shards[shard_idx])
                mask_shard = np.load(mask_shards[shard_idx]) if mask_shards is not None else None
                if mask_shard is not None and len(mask_shard) != len(shard):
                    raise ValueError(f"Mask length mismatch for {shards[shard_idx]}")
                offset = self._offset
                self._offset = 0

                while offset + self.base_chunk_len <= len(shard):
                    chunk = shard[offset : offset + self.base_chunk_len].tolist()
                    chunk_mask = (
                        mask_shard[offset : offset + self.base_chunk_len]
                        if mask_shard is not None
                        else None
                    )
                    offset += self.base_chunk_len

                    compressor = LZWCompressor(**self.compressor_args)
                    try:
                        compressed, _, codebook = compressor.encode(
                            chunk, padding="do_not_pad", truncation=False
                        )
                    except Exception:
                        continue

                    if len(compressed) < self.seq_len + 1:
                        continue

                    compressed = compressed[: self.seq_len + 1]

                    # Count base tokens represented by compressed sequence
                    cb_dict = codebook.to_dict()
                    n_base_tokens = sum(
                        len(cb_dict[tok]) if tok >= self.initial_vocab_size else 1
                        for tok in compressed
                    )

                    x = torch.LongTensor(compressed[:-1])
                    y = torch.LongTensor(compressed[1:])
                    if chunk_mask is not None:
                        loss_mask = self._compressed_loss_mask(compressed, chunk_mask, codebook)[1:]
                        if not loss_mask.any():
                            continue
                        y[~loss_mask] = -100
                    cb = self._codebook_to_tensor(codebook)

                    # Remap to compact codebook
                    if self.remap_codebook:
                        x, y, cb = self._remap_hyper_ids(x, y, cb)

                    yield {"input": x, "codebook": cb, "n_base_tokens": n_base_tokens}, y

            # Loop
            self._shard_idx = 0
            self._offset = 0

    def _iter_compress(self, shards):
        """Compression/decompression task mode.

        Produces sequences of the form:
          <DECOMPRESS> base_tokens <COMPRESS> compressed_tokens  (compression direction)
          <COMPRESS> compressed_tokens <DECOMPRESS> base_tokens  (decompression direction)

        Loss is only computed on the second part (labels set to -100 for the first part).
        Direction is chosen randomly (50/50) per sample.
        """
        vocab = self.initial_vocab_size
        target_total = self.seq_len - 1  # content budget (excluding 2 special tokens)

        while True:
            for shard_idx in range(self._shard_idx, len(shards)):
                self._shard_idx = shard_idx
                shard = self._load_token_shard(shards[shard_idx])
                offset = self._offset
                self._offset = 0

                while offset + self.base_chunk_len <= len(shard):
                    chunk = shard[offset : offset + self.base_chunk_len].tolist()

                    compressor = LZWCompressor(**self.compressor_args)
                    try:
                        compressed, _, codebook = compressor.encode(
                            chunk, padding="do_not_pad", truncation=False
                        )
                    except Exception:
                        offset += self.base_chunk_len
                        continue

                    cb_dict = codebook.to_dict()

                    # Find how many compressed tokens (+ corresponding base tokens) fit
                    base_count = 0
                    comp_len = 0
                    for i, tok in enumerate(compressed):
                        tok_base = len(cb_dict[tok]) if tok >= vocab else 1
                        if base_count + tok_base + (i + 1) > target_total:
                            break
                        base_count += tok_base
                        comp_len = i + 1

                    if comp_len < 10:
                        offset += self.base_chunk_len
                        continue

                    # Advance by used base tokens only (data-efficient)
                    offset += base_count

                    base_tokens = list(chunk[:base_count])
                    comp_tokens = list(compressed[:comp_len])

                    # Choose direction randomly
                    if random.random() < 0.5:
                        # Compression: given base, produce compressed
                        full = [DECOMPRESS_TOKEN_ID] + base_tokens + [COMPRESS_TOKEN_ID] + comp_tokens
                        loss_start = base_count + 1
                    else:
                        # Decompression: given compressed, produce base
                        full = [COMPRESS_TOKEN_ID] + comp_tokens + [DECOMPRESS_TOKEN_ID] + base_tokens
                        loss_start = comp_len + 1

                    loss_end = loss_start + (comp_len if full[0] == DECOMPRESS_TOKEN_ID else base_count)

                    # Pad to exactly seq_len + 1 tokens
                    pad_len = (self.seq_len + 1) - len(full)
                    if pad_len > 0:
                        full += [self.pad_token_id] * pad_len

                    x = torch.LongTensor(full[:-1])
                    y = torch.LongTensor(full[1:])

                    # Mask: only compute loss on the second part
                    loss_mask = torch.zeros(self.seq_len, dtype=torch.bool)
                    loss_mask[loss_start:loss_end] = True
                    y[~loss_mask] = -100

                    cb = self._codebook_to_tensor(codebook)
                    if self.remap_codebook:
                        x, y, cb = self._remap_hyper_ids(x, y, cb)

                    yield {"input": x, "codebook": cb, "n_base_tokens": base_count}, y

            # Loop
            self._shard_idx = 0
            self._offset = 0

    def __iter__(self):
        worker_info = get_worker_info()
        if worker_info is not None:
            shards = self.shard_files[worker_info.id :: worker_info.num_workers]
        else:
            shards = self.shard_files

        if self.mode == "compress":
            yield from self._iter_compress(shards)
        else:
            yield from self._iter_lm(shards, self.mask_files)

    def state_dict(self):
        return {"shard_idx": self._shard_idx, "offset": self._offset}

    def load_state_dict(self, state_dict):
        self._shard_idx = state_dict["shard_idx"]
        self._offset = state_dict["offset"]


def remap_collate_fn(batch, pad_token_id, max_subtokens, max_active_codebook_size):
    """Collate variable-size codebooks by padding to a fixed size.

    All codebooks are padded to max_active_codebook_size so torch.compile
    sees a single static shape (no recompilations).
    """
    inputs_list, labels_list = zip(*batch)

    input_ids = torch.stack([inp["input"] for inp in inputs_list])
    labels = torch.stack(labels_list)

    K = max_active_codebook_size

    padded_cbs = []
    for inp in inputs_list:
        cb = inp["codebook"]  # (K_i, max_subtokens)
        k = cb.shape[0]
        if k > K:
            cb = cb[:K]
        elif k < K:
            pad = torch.full(
                (K - k, max_subtokens), pad_token_id, dtype=cb.dtype
            )
            cb = torch.cat([cb, pad], dim=0)
        padded_cbs.append(cb)

    codebooks = torch.stack(padded_cbs)

    n_base_tokens = torch.tensor([inp["n_base_tokens"] for inp in inputs_list], dtype=torch.long)

    return {"input": input_ids, "codebook": codebooks, "n_base_tokens": n_base_tokens}, labels


def build_dataloader(
    data_dir: str,
    seq_len: int,
    max_subtokens: int,
    max_codebook_size: int,
    max_active_codebook_size: int,
    local_batch_size: int,
    rank: int = 0,
    world_size: int = 1,
    num_workers: int = 0,
    pad_token_id: int = 128001,
    mode: str = "lm",
    remap_codebook: bool = True,
) -> DataLoader:
    """Build a Zip2Zip DataLoader with remapping collation."""
    dataset = Zip2ZipDataset(
        data_dir=data_dir,
        seq_len=seq_len,
        max_subtokens=max_subtokens,
        max_codebook_size=max_codebook_size,
        rank=rank,
        world_size=world_size,
        mode=mode,
        remap_codebook=remap_codebook,
    )

    collate_fn = partial(
        remap_collate_fn,
        pad_token_id=pad_token_id,
        max_subtokens=max_subtokens,
        max_active_codebook_size=max_active_codebook_size,
    )

    return DataLoader(
        dataset,
        batch_size=local_batch_size,
        collate_fn=collate_fn,
        num_workers=num_workers,
        prefetch_factor=2 if num_workers > 0 else None,
        persistent_workers=num_workers > 0,
    )
