"""Zip2Zip dataset and data loading utilities."""

import os
from functools import partial

import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset, get_worker_info
from torch.distributed.checkpoint.stateful import Stateful

from zip2zip_compression import LZWCompressor


class Zip2ZipDataset(IterableDataset, Stateful):
    """Pre-tokenized dataset with on-the-fly LZW compression.

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
    ):
        self.data_dir = data_dir
        self.seq_len = seq_len
        self.max_subtokens = max_subtokens
        self.max_codebook_size = max_codebook_size
        self.initial_vocab_size = initial_vocab_size
        self.pad_token_id = pad_token_id
        self.base_chunk_len = seq_len * 2

        shard_files = sorted(
            [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".npy")]
        )
        if not shard_files:
            raise ValueError(f"No .npy files in {data_dir}")

        # Distribute shards
        self.shard_files = shard_files[rank::world_size]
        self._shard_idx = 0
        self._offset = 0

        self.compressor_args = dict(
            initial_vocab_size=initial_vocab_size,
            max_codebook_size=max_codebook_size,
            max_subtokens=max_subtokens,
            pad_token_id=pad_token_id,
            disabled_ids=[bos_token_id, eos_token_id],
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

    def __iter__(self):
        worker_info = get_worker_info()
        if worker_info is not None:
            shards = self.shard_files[worker_info.id :: worker_info.num_workers]
        else:
            shards = self.shard_files

        while True:
            for shard_idx in range(self._shard_idx, len(shards)):
                self._shard_idx = shard_idx
                shard = np.load(shards[shard_idx])
                offset = self._offset
                self._offset = 0

                while offset + self.base_chunk_len <= len(shard):
                    chunk = shard[offset : offset + self.base_chunk_len].tolist()
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
                    x = torch.LongTensor(compressed[:-1])
                    y = torch.LongTensor(compressed[1:])
                    cb = self._codebook_to_tensor(codebook)

                    # Remap to compact codebook
                    x, y, cb = self._remap_hyper_ids(x, y, cb)

                    yield {"input": x, "codebook": cb}, y

            # Loop
            self._shard_idx = 0
            self._offset = 0

    def state_dict(self):
        return {"shard_idx": self._shard_idx, "offset": self._offset}

    def load_state_dict(self, state_dict):
        self._shard_idx = state_dict["shard_idx"]
        self._offset = state_dict["offset"]


def remap_collate_fn(
    batch,
    pad_token_id,
    max_subtokens,
    max_active_codebook_size,
    initial_vocab_size,
):
    """Collate variable-size codebooks by padding to a fixed size.

    All codebooks are padded to max_active_codebook_size so torch.compile
    sees a single static shape (no recompilations).
    """
    inputs_list, labels_list = zip(*batch)

    input_ids = torch.stack([inp["input"] for inp in inputs_list])
    labels = torch.stack(labels_list)
    n_base_tokens = (labels < initial_vocab_size).sum(dim=1)

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

    return {
        "input": input_ids,
        "codebook": codebooks,
        "n_base_tokens": n_base_tokens,
    }, labels


def build_dataloader(
    data_dir: str,
    seq_len: int,
    max_subtokens: int,
    max_codebook_size: int,
    max_active_codebook_size: int,
    local_batch_size: int,
    mode: str = "lm",
    rank: int = 0,
    world_size: int = 1,
    num_workers: int = 0,
    pad_token_id: int = 128001,
    initial_vocab_size: int = 128256,
) -> DataLoader:
    """Build a Zip2Zip DataLoader with remapping collation."""
    if mode != "lm":
        raise NotImplementedError(
            f"Unsupported dataloader mode: {mode!r}. Only 'lm' is implemented."
        )

    dataset = Zip2ZipDataset(
        data_dir=data_dir,
        seq_len=seq_len,
        max_subtokens=max_subtokens,
        max_codebook_size=max_codebook_size,
        initial_vocab_size=initial_vocab_size,
        rank=rank,
        world_size=world_size,
    )

    collate_fn = partial(
        remap_collate_fn,
        pad_token_id=pad_token_id,
        max_subtokens=max_subtokens,
        max_active_codebook_size=max_active_codebook_size,
        initial_vocab_size=initial_vocab_size,
    )

    return DataLoader(
        dataset,
        batch_size=local_batch_size,
        collate_fn=collate_fn,
        num_workers=num_workers,
        prefetch_factor=2 if num_workers > 0 else None,
        persistent_workers=num_workers > 0,
    )
