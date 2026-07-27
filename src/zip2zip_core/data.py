"""Zip2Zip dataset and data loading utilities."""

import os
import random
import time
from functools import partial

import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset, get_worker_info
from torch.distributed.checkpoint.stateful import Stateful

from zip2zip_compression import CodebookManager, CompressionConfig, LZWCompressor

# Special tokens for compression/decompression task (unused Llama3 special token slots)
COMPRESS_TOKEN_ID = 128002
DECOMPRESS_TOKEN_ID = 128003
LLAMA31_8B_SPECIAL_TOKEN_IDS = tuple(range(128000, 128256))


def _debug_data_log(enabled: bool, rank: int, worker_id: int, message: str):
    if not enabled:
        return
    ts = time.strftime("%H:%M:%S")
    print(f"[data-debug {ts}] [rank{rank}] [worker{worker_id}] {message}", flush=True)


def online_codebook_counts(compressed_ids, codebook, compressor_args):
    """Rows installed by the online decoder after each compressed token.

    Training has the encoder's final codebook, but generation builds the same
    dictionary incrementally from each emitted token's base-token expansion.
    Replay that exact CodebookManager path and return a scalar count per
    position. LZW row indices are sequential, so row ``k`` is available iff
    ``k < count[t]``.
    """
    cb_dict = codebook.to_dict()
    vocab_size = compressor_args["initial_vocab_size"]
    manager = CodebookManager(CompressionConfig(**compressor_args))
    installed = 0
    counts = []

    for tok in compressed_ids:
        tok = int(tok)
        if tok < vocab_size:
            expansion = [tok]
        else:
            expansion = cb_dict.get(tok)
            if expansion is None:
                raise RuntimeError(
                    f"compressed token {tok} is absent from the final codebook"
                )
            expansion = list(expansion)

        _, update_indices = manager.update_codebooks([expansion])
        new_indices = [int(i) for i in update_indices[0]]
        expected = list(range(installed, installed + len(new_indices)))
        if new_indices != expected:
            raise RuntimeError(
                "online codebook rows were not installed sequentially: "
                f"expected {expected}, got {new_indices}"
            )
        installed += len(new_indices)
        counts.append(installed)

    return torch.tensor(counts, dtype=torch.long)


def online_unavailable_targets(
    labels, counts, vocab_size, max_codebook_entries
):
    """Targets outside the current decoder or model output vocabulary.

    The codec has rare legal unknown-next-ID cases (force-merge/KwKwK): the
    next target is exactly the first uninstalled row. The current generation
    runtime has no embedding for that row, so v0.6.5 excludes it from the
    teacher-forced objective instead of leaking its final-codebook embedding.
    Rows beyond the model's active codebook are likewise unscorable. A decoder
    gap larger than one is not a supported special case and fails loudly.
    """
    if labels.shape != counts.shape:
        raise ValueError(
            f"labels/counts shape mismatch: {tuple(labels.shape)} vs "
            f"{tuple(counts.shape)}"
        )
    slots = labels - vocab_size
    is_hyper = labels >= vocab_size
    decoder_unavailable = is_hyper & (slots >= counts)
    if decoder_unavailable.any() and not torch.equal(
        slots[decoder_unavailable], counts[decoder_unavailable]
    ):
        raise RuntimeError(
            "online codebook target is more than the single legal next row "
            "ahead of decoder state"
        )
    return decoder_unavailable | (
        is_hyper & (slots >= max_codebook_entries)
    )


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
        disabled_ids: tuple[int, ...] | list[int] | None = None,
        rank: int = 0,
        world_size: int = 1,
        mode: str = "lm",
        remap_codebook: bool = True,
        online_codebook_mask: bool = False,
        max_active_codebook_size: int | None = None,
        debug_samples: int = 0,
    ):
        self.data_dir = data_dir
        self.seq_len = seq_len
        self.max_subtokens = max_subtokens
        self.max_codebook_size = max_codebook_size
        self.initial_vocab_size = initial_vocab_size
        self.pad_token_id = pad_token_id
        self.mode = mode
        self.remap_codebook = remap_codebook
        self.online_codebook_mask = online_codebook_mask
        self.max_active_codebook_size = (
            max_codebook_size
            if max_active_codebook_size is None
            else max_active_codebook_size
        )
        self.base_chunk_len = seq_len * 2
        self.rank = rank
        self.debug_samples = debug_samples
        self._debug_samples_emitted = 0
        self._debug_event_budget = max(16, debug_samples * 8) if debug_samples > 0 else 0
        self._debug_worker_id = 0

        shard_files = sorted(
            [
                os.path.join(data_dir, f)
                for f in os.listdir(data_dir)
                if f.endswith(".npy") and not f.startswith("mask_")
            ]
        )
        if not shard_files:
            raise ValueError(f"No .npy files in {data_dir}")
        if world_size > len(shard_files):
            raise ValueError(
                f"world_size={world_size} exceeds available token shards={len(shard_files)} "
                f"in {data_dir}. Each rank needs at least one shard with the current "
                "rank::world_size partitioning."
            )
        if self.online_codebook_mask and self.mode != "lm":
            raise ValueError("online_codebook_mask is supported only in mode='lm'")
        if self.online_codebook_mask and self.remap_codebook:
            raise ValueError(
                "online_codebook_mask requires remap_codebook=False because "
                "decoder-installed counts refer to original LZW row indices"
            )
        if (
            self.online_codebook_mask
            and self.max_active_codebook_size <= 0
        ):
            raise ValueError(
                "online_codebook_mask requires max_active_codebook_size > 0"
            )

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

        # Special-token ids that must never be merged into the codebook. Defaults
        # to the Llama-3.1 reserved range for back-compat; callers training on a
        # different tokenizer (e.g. Phi-3.5-mini) must pass the matching ids.
        if disabled_ids is None:
            disabled_ids = LLAMA31_8B_SPECIAL_TOKEN_IDS
        self.compressor_args = dict(
            initial_vocab_size=initial_vocab_size,
            max_codebook_size=max_codebook_size,
            max_subtokens=max_subtokens,
            pad_token_id=pad_token_id,
            disabled_ids=list(disabled_ids),
        )

    def _debug_enabled(self) -> bool:
        return self.debug_samples > 0 and self._debug_event_budget > 0

    def _debug_log(self, message: str):
        enabled = self._debug_enabled()
        _debug_data_log(enabled, self.rank, self._debug_worker_id, message)
        if enabled:
            self._debug_event_budget -= 1

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
        if self.max_codebook_size == 0:
            # Keep the (rows, max_subtokens) shape even with zero rows — a bare
            # LongTensor([]) is 1-D and only collates by torch's legacy
            # empty-1D cat special case.
            return torch.full(
                (0, self.max_subtokens), self.pad_token_id, dtype=torch.long
            )
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
                self._debug_log(
                    f"loading lm shard idx={shard_idx} path={os.path.basename(shards[shard_idx])} "
                    f"resume_offset={self._offset}"
                )
                shard = self._load_token_shard(shards[shard_idx])
                mask_shard = np.load(mask_shards[shard_idx]) if mask_shards is not None else None
                if mask_shard is not None and len(mask_shard) != len(shard):
                    raise ValueError(f"Mask length mismatch for {shards[shard_idx]}")
                # self._offset is the RESUME position and must keep tracking the
                # live one: it is synced to `offset` at every yield below and
                # zeroed only once this shard is exhausted. Zeroing it here (the
                # historical behaviour) made state_dict() always report offset=0,
                # so a resumed run silently replayed the shard from its start.
                offset = self._offset
                self._debug_log(
                    f"loaded lm shard idx={shard_idx} tokens={len(shard)} "
                    f"has_mask={mask_shard is not None} start_offset={offset}"
                )

                while offset + self.base_chunk_len <= len(shard):
                    if self._debug_samples_emitted < self.debug_samples:
                        self._debug_log(
                            f"lm chunk shard_idx={shard_idx} offset={offset} "
                            f"chunk_len={self.base_chunk_len} before encode"
                        )
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
                        self._debug_log(
                            f"lm encode failed shard_idx={shard_idx} offset={offset - self.base_chunk_len}"
                        )
                        continue

                    if self._debug_samples_emitted < self.debug_samples:
                        self._debug_log(
                            f"lm encoded shard_idx={shard_idx} offset={offset - self.base_chunk_len} "
                            f"compressed_len={len(compressed)}"
                        )

                    if len(compressed) < self.seq_len + 1:
                        continue

                    compressed = compressed[: self.seq_len + 1]

                    # A window with ZERO hypertokens makes this rank skip the
                    # hyper-encoder forward (model gate: (codebook != pad).any())
                    # while other ranks run it — the FSDP collectives then
                    # deadlock until the NCCL watchdog aborts the job (observed
                    # in the first digitsafe run: rare all-number windows have
                    # no LZW merges once digits are disabled). Skip such windows
                    # when compression is on; the uncompressed control
                    # (max_codebook_size=0) skips the hyper path on ALL ranks
                    # uniformly, which is safe, so it must keep every window.
                    if self.max_codebook_size > 0 and not any(
                        t >= self.initial_vocab_size for t in compressed
                    ):
                        self._debug_log(
                            f"lm skip zero-hypertoken window shard_idx={shard_idx} "
                            f"offset={offset - self.base_chunk_len}"
                        )
                        continue

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

                    available_counts = None
                    online_skipped_targets = 0
                    if self.online_codebook_mask:
                        out_of_range_input = x >= (
                            self.initial_vocab_size
                            + self.max_active_codebook_size
                        )
                        if out_of_range_input.any():
                            bad_id = int(x[out_of_range_input][0].item())
                            raise RuntimeError(
                                f"input hyper-token {bad_id} exceeds the active "
                                f"codebook size {self.max_active_codebook_size}"
                            )
                        available_counts = online_codebook_counts(
                            compressed[:-1], codebook, self.compressor_args
                        )
                        unavailable = online_unavailable_targets(
                            y,
                            available_counts,
                            self.initial_vocab_size,
                            self.max_active_codebook_size,
                        )
                        online_skipped_targets = int(unavailable.sum().item())
                        y[unavailable] = -100
                        if not (y != -100).any():
                            continue

                    cb = self._codebook_to_tensor(codebook)

                    # Remap to compact codebook
                    if self.remap_codebook:
                        x, y, cb = self._remap_hyper_ids(x, y, cb)

                    if self._debug_samples_emitted < self.debug_samples:
                        self._debug_log(
                            f"lm yield sample_idx={self._debug_samples_emitted} shard_idx={shard_idx} "
                            f"offset={offset - self.base_chunk_len} input={tuple(x.shape)} "
                            f"codebook={tuple(cb.shape)} n_base_tokens={n_base_tokens}"
                        )
                    self._debug_samples_emitted += 1
                    sample = {
                        "input": x,
                        "codebook": cb,
                        "n_base_tokens": n_base_tokens,
                    }
                    if self.online_codebook_mask:
                        sample["codebook_counts"] = available_counts
                        sample["online_skipped_targets"] = online_skipped_targets
                    # Publish the resume position. A checkpoint can only be taken
                    # between yields, so syncing here is exactly sufficient: it
                    # names the first chunk this shard has NOT yet emitted.
                    self._offset = offset
                    yield sample, y

                # Shard exhausted: the next one starts at its own beginning.
                self._offset = 0

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
                self._debug_log(
                    f"loading compress shard idx={shard_idx} path={os.path.basename(shards[shard_idx])} "
                    f"resume_offset={self._offset}"
                )
                shard = self._load_token_shard(shards[shard_idx])
                # Same contract as _iter_lm: synced at every yield, zeroed only
                # once the shard is exhausted.
                offset = self._offset
                self._debug_log(
                    f"loaded compress shard idx={shard_idx} tokens={len(shard)} start_offset={offset}"
                )

                while offset + self.base_chunk_len <= len(shard):
                    if self._debug_samples_emitted < self.debug_samples:
                        self._debug_log(
                            f"compress chunk shard_idx={shard_idx} offset={offset} "
                            f"chunk_len={self.base_chunk_len} before encode"
                        )
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

                    if self._debug_samples_emitted < self.debug_samples:
                        self._debug_log(
                            f"compress yield sample_idx={self._debug_samples_emitted} shard_idx={shard_idx} "
                            f"offset={offset} input={tuple(x.shape)} codebook={tuple(cb.shape)} "
                            f"base_count={base_count} comp_len={comp_len}"
                        )
                    self._debug_samples_emitted += 1
                    self._offset = offset
                    yield {"input": x, "codebook": cb, "n_base_tokens": base_count}, y

                # Shard exhausted: the next one starts at its own beginning.
                self._offset = 0

            # Loop
            self._shard_idx = 0
            self._offset = 0

    def __iter__(self):
        worker_info = get_worker_info()
        masks = self.mask_files
        if worker_info is not None:
            self._debug_worker_id = worker_info.id
            shards = self.shard_files[worker_info.id :: worker_info.num_workers]
            # The masks MUST be sliced the same way. Passing the rank-level list
            # while the shards are worker-sliced pairs worker w's j-th shard with
            # mask j: right only while num_workers <= 1, and silently wrong after
            # that — both files are full-size, so the length check never fires and
            # the run trains on another shard's loss mask.
            if masks is not None:
                masks = masks[worker_info.id :: worker_info.num_workers]
            if not shards:
                # An empty slice makes `for shard_idx in range(0, 0)` inside
                # `while True` spin without ever yielding, which stalls the
                # DataLoader's in-order fetch and hangs the rank instead of
                # failing it.
                raise ValueError(
                    f"worker {worker_info.id} of {worker_info.num_workers} got no "
                    f"shards: this rank owns only {len(self.shard_files)}. Use at "
                    f"most --num_workers {len(self.shard_files)}"
                )
        else:
            self._debug_worker_id = 0
            shards = self.shard_files

        self._debug_log(
            f"iter start mode={self.mode} assigned_shards={len(shards)} "
            f"shard_preview={[os.path.basename(s) for s in shards[:3]]} "
            f"resume_state=(shard_idx={self._shard_idx}, offset={self._offset})"
        )

        if self.mode == "compress":
            yield from self._iter_compress(shards)
        else:
            yield from self._iter_lm(shards, masks)

    def _shard_identity(self):
        """What this rank's stream IS, not merely how long it is.

        A shard COUNT cannot tell two corpora apart: every dataset in this
        project has 8 shards, so at world_size=4 they all report 2 and a resume
        onto a different DATA_DIR would seek into unrelated files without
        complaint. Basenames identify the corpus; sizes catch a regenerated
        dataset that kept the filenames.
        """
        return [
            (os.path.basename(path), os.path.getsize(path))
            for path in self.shard_files
        ]

    def _mask_identity(self):
        """Identity of this rank's loss masks, or None when the corpus has none.

        The masks decide WHICH labels enter the loss, so swapping them mid-run
        mixes two objectives in one run. They live in separate files from the
        token shards, so identical tokens with regenerated masks would otherwise
        pass every other check.
        """
        if self.mask_files is None:
            return None
        return [
            (os.path.basename(path), os.path.getsize(path))
            for path in self.mask_files
        ]

    def state_dict(self):
        """Position in this rank's data stream, for an exact resume."""
        return {
            "shard_idx": self._shard_idx,
            "offset": self._offset,
            "mode": self.mode,
            "shards": self._shard_identity(),
            "masks": self._mask_identity(),
        }

    def check_state_dict(self, state_dict):
        """Raise if `state_dict` describes a different stream. Never mutates.

        Split out from load_state_dict so the caller can decide what to do about
        a mismatch collectively, instead of some ranks raising while others
        proceed into the next collective and hang.
        """
        saved_mode = state_dict.get("mode", self.mode)
        if saved_mode != self.mode:
            raise ValueError(
                f"position was saved in mode={saved_mode!r} but this run is "
                f"mode={self.mode!r}; the two iterate shards differently, so the "
                "saved offset does not transfer"
            )
        current = self._shard_identity()
        saved_shards = state_dict.get("shards")
        if saved_shards is None:
            raise ValueError(
                "position records no shard identity (written by an older build); "
                "it cannot be proven to belong to this dataset"
            )
        saved_shards = [tuple(entry) for entry in saved_shards]
        if saved_shards != current:
            saved_names = [name for name, _ in saved_shards]
            current_names = [name for name, _ in current]
            detail = (
                f"names {saved_names} vs {current_names}"
                if saved_names != current_names
                else "same names but different file sizes (dataset regenerated)"
            )
            raise ValueError(
                f"position belongs to a different dataset for this rank: {detail}. "
                f"Check DATA_DIR (currently {self.data_dir})"
            )
        # Masks were added to the recorded identity after `shards`, so a state
        # written by an older build simply has no "masks" key: skip rather than
        # refuse, or resuming any checkpoint from before the change would break.
        if "masks" in state_dict:
            saved_masks = state_dict["masks"]
            current_masks = self._mask_identity()
            if saved_masks is not None:
                saved_masks = [tuple(entry) for entry in saved_masks]
            if saved_masks != current_masks:
                if (saved_masks is None) != (current_masks is None):
                    detail = (
                        "the checkpoint trained WITH loss masks and this corpus has "
                        "none" if current_masks is None else
                        "the checkpoint trained WITHOUT loss masks and this corpus "
                        "has them"
                    )
                else:
                    detail = (
                        f"masks differ: {[n for n, _ in saved_masks]} vs "
                        f"{[n for n, _ in current_masks]}"
                        if [n for n, _ in saved_masks] != [n for n, _ in current_masks]
                        else "same mask names but different sizes (regenerated)"
                    )
                raise ValueError(
                    f"position belongs to a different loss objective: {detail}. "
                    "Which labels enter the loss would change mid-run"
                )

        if not 0 <= state_dict["shard_idx"] < len(self.shard_files):
            raise ValueError(
                f"saved shard_idx={state_dict['shard_idx']} is out of range for "
                f"{len(self.shard_files)} shards"
            )
        if state_dict["offset"] < 0:
            raise ValueError(f"saved offset={state_dict['offset']} is negative")

    def load_state_dict(self, state_dict):
        self.check_state_dict(state_dict)
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

    inputs = {
        "input": input_ids,
        "codebook": codebooks,
        "n_base_tokens": n_base_tokens,
    }
    if "codebook_counts" in inputs_list[0]:
        inputs["codebook_counts"] = torch.stack(
            [inp["codebook_counts"] for inp in inputs_list]
        )
        inputs["online_skipped_targets"] = torch.tensor(
            [inp["online_skipped_targets"] for inp in inputs_list],
            dtype=torch.long,
        )

    return inputs, labels


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
    initial_vocab_size: int = 128256,
    disabled_ids: tuple[int, ...] | list[int] | None = None,
    mode: str = "lm",
    remap_codebook: bool = True,
    online_codebook_mask: bool = False,
    debug_samples: int = 0,
) -> DataLoader:
    """Build a Zip2Zip DataLoader with remapping collation."""
    dataset = Zip2ZipDataset(
        data_dir=data_dir,
        seq_len=seq_len,
        max_subtokens=max_subtokens,
        max_codebook_size=max_codebook_size,
        initial_vocab_size=initial_vocab_size,
        pad_token_id=pad_token_id,
        disabled_ids=disabled_ids,
        rank=rank,
        world_size=world_size,
        mode=mode,
        remap_codebook=remap_codebook,
        online_codebook_mask=online_codebook_mask,
        max_active_codebook_size=max_active_codebook_size,
        debug_samples=debug_samples,
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
