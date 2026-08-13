"""Zip2Zip dataset and data loading utilities."""

import os
import random
import re
import time
from functools import partial

import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset, get_worker_info
from torch.distributed.checkpoint.stateful import Stateful

from zip2zip_compression import CodebookManager, CompressionConfig, LZWCompressor

# Default markers for the compression/decompression task -- unused Llama 3
# special-token slots (<|reserved_special_token_0/1|>). They are ONLY valid for a
# Llama-3 vocabulary: under Phi-3.5-mini (vocab_size=32064) both ids are out of
# range, so a run on another tokenizer must pass its own pair. Derive them with
# ``default_transducer_token_ids`` rather than hardcoding a second constant here.
COMPRESS_TOKEN_ID = 128002
DECOMPRESS_TOKEN_ID = 128003
LLAMA31_8B_SPECIAL_TOKEN_IDS = tuple(range(128000, 128256))

# Which half of a compress-mode sample carries the loss. Emitted per sample so
# training can report the two directions separately WITHOUT changing what it
# optimizes: the objective stays the joint mixed one. A single mixed curve cannot
# say which direction a model failed to learn, and the mix is not even 50/50 by
# token count -- the decompression direction supervises `compression_ratio` times
# more target tokens per sample than the compression direction.
DIRECTION_COMPRESS = 1    # given base tokens, produce the compressed sequence
DIRECTION_DECOMPRESS = 0  # given the compressed sequence, produce base tokens

# Which direction(s) a compress-mode run is trained on. "both" is the historical
# behaviour (one model, 50/50 per sample) and reproduces the published Llama
# sweep. "compress"/"decompress" train a SINGLE-direction model, which is what
# makes a run's loss unambiguous: with "both", one curve blends two tasks whose
# target-token counts differ by the compression ratio, so it cannot say which
# direction the model actually learned.
TRANSDUCER_DIRECTIONS = ("both", "compress", "decompress")

# Reserved/unused special-token names across the tokenizers this repo trains on:
# Llama 3 spells them <|reserved_special_token_N|>, Phi-3.5 <|placeholderN|>.
_UNUSED_SPECIAL_TOKEN_RE = re.compile(r"^<\|(?:reserved_special_token_|placeholder)\d+\|>$")


def default_transducer_token_ids(tokenizer, vocab_size: int) -> tuple[int, int]:
    """(compress_token_id, decompress_token_id) for ``tokenizer``.

    The transducer task in ``mode=compress`` needs two markers that (a) exist in
    the model's embedding table and (b) the LZW compressor will never merge into
    a hyper-token -- otherwise the separator itself gets absorbed into a codebook
    entry and the task becomes ill-defined. Both tokenizers used here reserve
    unused special-token slots for exactly this; take the two lowest.

    Returns (128002, 128003) for Llama 3 and (32002, 32003) for Phi-3.5-mini,
    i.e. the historical Llama defaults are reproduced, not merely approximated.
    """
    candidates = sorted(
        i
        for name, i in (tokenizer.get_added_vocab() or {}).items()
        if _UNUSED_SPECIAL_TOKEN_RE.match(name) and 0 <= i < vocab_size
    )
    if len(candidates) < 2:
        raise ValueError(
            f"tokenizer exposes {len(candidates)} unused reserved special-token slots "
            f"below vocab_size={vocab_size}, need 2 for the compress/decompress "
            "markers. Pass --compress_token_id / --decompress_token_id explicitly."
        )
    return candidates[0], candidates[1]


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

    # lossless_windows: how many base_chunk_len chunks a single window may read
    # while extending (see _iter_lm). 3 chunks = 3*2*seq_len raw tokens covers
    # any compression ratio below 3*2*seq_len/(seq_len+1) ~= 6x; natural text
    # never reaches that, and a bounded cap guarantees forward progress on
    # pathological streams (the region is then skipped whole).
    LOSSLESS_MAX_WINDOW_CHUNKS = 3

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
        include_base_view: bool = False,
        lossless_windows: bool = False,
        compress_token_id: int = COMPRESS_TOKEN_ID,
        decompress_token_id: int = DECOMPRESS_TOKEN_ID,
        direction: str = "both",
    ):
        self.data_dir = data_dir
        self.seq_len = seq_len
        self.max_subtokens = max_subtokens
        self.max_codebook_size = max_codebook_size
        self.initial_vocab_size = initial_vocab_size
        self.pad_token_id = pad_token_id
        self.mode = mode
        self.compress_token_id = compress_token_id
        self.decompress_token_id = decompress_token_id
        self.direction = direction
        self.remap_codebook = remap_codebook
        self.online_codebook_mask = online_codebook_mask
        self.include_base_view = include_base_view
        # Off (default): historical behavior, bit-identical — every window
        # reads a fixed 2*seq_len chunk, advances by that amount, drops the
        # compressed tail beyond seq_len+1 tokens, and skips windows that
        # compress to fewer than seq_len+1 tokens. On: lm-mode windows tile
        # the stream exactly — the offset advances by the base-token span of
        # the emitted (truncated) window, and under-filled windows extend
        # their raw input instead of being dropped. Only _iter_lm changes;
        # "compress" mode ignores the flag.
        # Measured negative on the paired 8k v0.6.4 run (2026-08-13): GSM8K
        # -8.2pt real (p=1e-08), ppl and gen compression worse — the legacy
        # ratio>=2.0 skip is an accidental quality filter. Keep off for real
        # runs; diagnostic lever only.
        self.lossless_windows = bool(lossless_windows)
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
        if self.include_base_view and self.mode != "lm":
            raise ValueError("include_base_view is supported only in mode='lm'")
        if self.lossless_windows and self.mode != "lm":
            raise ValueError("lossless_windows is supported only in mode='lm'")
        if self.lossless_windows and self.include_base_view:
            raise ValueError(
                "lossless_windows is incompatible with include_base_view: the "
                "base view is drawn from the full raw chunk, which under the "
                "lossless advance overlaps the next compressed window"
            )
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

        if self.mode == "compress":
            if direction not in TRANSDUCER_DIRECTIONS:
                raise ValueError(
                    f"direction={direction!r} must be one of {TRANSDUCER_DIRECTIONS}"
                )
            # The markers are ordinary embedding rows, so an id past the base
            # vocabulary would be read as a hyper-token and gather out of bounds
            # (a CUDA device-side assert, far from its cause). They must also be
            # in disabled_ids: an LZW-mergeable separator gets absorbed into a
            # codebook entry and silently redefines the task.
            for name, tok_id in (
                ("compress_token_id", self.compress_token_id),
                ("decompress_token_id", self.decompress_token_id),
            ):
                if not 0 <= tok_id < initial_vocab_size:
                    raise ValueError(
                        f"{name}={tok_id} is outside the base vocabulary "
                        f"[0, {initial_vocab_size}). The Llama-3 defaults "
                        f"({COMPRESS_TOKEN_ID}, {DECOMPRESS_TOKEN_ID}) do not carry over "
                        "to another tokenizer -- derive the pair with "
                        "data.default_transducer_token_ids()."
                    )
                if tok_id not in set(disabled_ids):
                    raise ValueError(
                        f"{name}={tok_id} is not in disabled_ids, so LZW may merge the "
                        "task separator into a hyper-token. Derive disabled_ids from "
                        "zip2zip_core.disabled_ids.compute_disabled_ids()."
                    )
            if self.compress_token_id == self.decompress_token_id:
                raise ValueError(
                    f"compress_token_id and decompress_token_id are both "
                    f"{self.compress_token_id}; the two task directions would be "
                    "indistinguishable to the model."
                )

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

    def _count_expanded_base_tokens(self, tokens, codebook) -> int:
        """Number of base tokens represented by a compressed token sequence."""
        cb_dict = codebook.to_dict()
        return sum(
            len(cb_dict[int(tok)])
            if int(tok) >= self.initial_vocab_size
            else 1
            for tok in tokens
            if int(tok) != -100
        )

    def _count_unmasked_target_base_tokens(self, labels, codebook) -> int:
        """Base tokens represented by labels that actually enter the loss.

        This is the corrected ``target_n_base_tokens`` denominator. The legacy
        ``n_base_tokens`` field intentionally retains the historical count over
        the complete compressed ``T+1`` window for curve comparability.
        """
        return self._count_expanded_base_tokens(labels.tolist(), codebook)

    def _make_base_view(self, chunk, chunk_mask):
        """An uncompressed CLM window from the same raw ``2 * seq_len`` chunk.

        For masked corpora, choose the earliest length-``seq_len`` target window
        with the most valid labels. This is deterministic across ranks/workers,
        preserves the original assistant mask, and avoids selecting an all-masked
        base replay view when the compressed sample itself has valid targets.
        """
        n = self.seq_len
        if len(chunk) < n + 1:
            raise ValueError(
                f"base replay needs at least seq_len+1={n + 1} raw tokens, "
                f"got {len(chunk)}"
            )

        start = 0
        mask = None
        if chunk_mask is not None:
            mask = np.asarray(chunk_mask, dtype=np.bool_)
            if len(mask) != len(chunk):
                raise ValueError(
                    f"base replay chunk/mask length mismatch: "
                    f"{len(chunk)} vs {len(mask)}"
                )
            # For start s, labels correspond to raw mask[s+1 : s+n+1].
            prefix = np.concatenate(
                [np.zeros(1, dtype=np.int64), np.cumsum(mask, dtype=np.int64)]
            )
            n_starts = len(chunk) - n
            valid_counts = (
                prefix[n + 1 : n + 1 + n_starts]
                - prefix[1 : 1 + n_starts]
            )
            start = int(valid_counts.argmax())

        base_input = torch.LongTensor(chunk[start : start + n])
        base_labels = torch.LongTensor(chunk[start + 1 : start + n + 1])
        if mask is not None:
            target_mask = torch.from_numpy(
                mask[start + 1 : start + n + 1].copy()
            )
            base_labels[~target_mask] = -100

        base_n_base_tokens = int((base_labels != -100).sum().item())
        if base_n_base_tokens == 0:
            raise RuntimeError(
                "base replay selected an all-masked window from a compressed "
                "sample that was expected to contain valid targets"
            )
        return base_input, base_labels, base_n_base_tokens

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
                    # `start` is this window's fixed origin; `offset` keeps the
                    # historical eager advance so every legacy `continue` below
                    # behaves exactly as before. Lossless mode re-derives the
                    # advance from the emitted window instead, always relative
                    # to `start`.
                    start = offset
                    chunk_len = self.base_chunk_len
                    if self._debug_samples_emitted < self.debug_samples:
                        self._debug_log(
                            f"lm chunk shard_idx={shard_idx} offset={offset} "
                            f"chunk_len={chunk_len} before encode"
                        )
                    chunk = shard[start : start + chunk_len].tolist()
                    chunk_mask = (
                        mask_shard[start : start + chunk_len]
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
                        if not self.lossless_windows:
                            continue
                        # The window under-filled: this text compresses at
                        # >= base_chunk_len/seq_len (= 2.0x). The historical
                        # skip silently removed exactly the most compressible
                        # windows from training; instead, extend the raw input
                        # and re-encode until seq_len+1 compressed tokens
                        # exist (bounded by LOSSLESS_MAX_WINDOW_CHUNKS and the
                        # shard end).
                        while (
                            len(compressed) < self.seq_len + 1
                            and chunk_len
                            < self.base_chunk_len * self.LOSSLESS_MAX_WINDOW_CHUNKS
                            and start + chunk_len + self.base_chunk_len <= len(shard)
                        ):
                            chunk_len += self.base_chunk_len
                            chunk = shard[start : start + chunk_len].tolist()
                            if mask_shard is not None:
                                chunk_mask = mask_shard[start : start + chunk_len]
                            compressor = LZWCompressor(**self.compressor_args)
                            try:
                                compressed, _, codebook = compressor.encode(
                                    chunk, padding="do_not_pad", truncation=False
                                )
                            except Exception:
                                self._debug_log(
                                    f"lm encode failed during lossless extension "
                                    f"shard_idx={shard_idx} start={start} "
                                    f"chunk_len={chunk_len}"
                                )
                                break
                        if len(compressed) < self.seq_len + 1:
                            # Still under-filled at the cap, the shard end, or
                            # after a failed re-encode. An UNTRUNCATED encode
                            # covers its entire raw input, so advancing by its
                            # expansion skips exactly the region that cannot
                            # fill a window — nothing before or after it.
                            self._debug_log(
                                f"lm skip unfillable lossless window "
                                f"shard_idx={shard_idx} start={start} "
                                f"chunk_len={chunk_len} "
                                f"compressed_len={len(compressed)}"
                            )
                            offset = start + self._count_expanded_base_tokens(
                                compressed, codebook
                            )
                            continue

                    compressed = compressed[: self.seq_len + 1]

                    # A window with ZERO hypertokens in the INPUT slice makes
                    # this rank's hyper path diverge from the other ranks and
                    # the FSDP collectives deadlock until the NCCL watchdog
                    # aborts the job. The model branches on x = compressed[:-1],
                    # so scan exactly that slice: scanning the full window let
                    # through a window whose only hypertoken was the final,
                    # label-only token, which orphaned hyper_encoder from the
                    # loss on one rank (pt20B step 95231; survivable since the
                    # unconditional mix in _embed_tokens_train, but the guard
                    # must match what the model sees). First observed as rare
                    # all-number windows in the digitsafe run. The uncompressed
                    # control (max_codebook_size=0) skips the hyper path on ALL
                    # ranks uniformly, which is safe, so it keeps every window.
                    if self.max_codebook_size > 0 and not any(
                        t >= self.initial_vocab_size for t in compressed[:-1]
                    ):
                        self._debug_log(
                            f"lm skip zero-hypertoken window shard_idx={shard_idx} "
                            f"offset={offset - self.base_chunk_len}"
                        )
                        if self.lossless_windows:
                            # Skip exactly the window's base span so the text
                            # behind the truncation point is not lost with it.
                            offset = start + self._count_expanded_base_tokens(
                                compressed, codebook
                            )
                        continue

                    # Without remap the hyper ids index raw LZW rows and the
                    # collate silently truncates the codebook to
                    # max_active_codebook_size rows. Inside a seq_len+1 window
                    # LZW cannot yet have created row max_active — but that is
                    # a property of the current seq_len, not of this code, so
                    # fail loudly here instead of surfacing as a CUDA
                    # device-side assert in the embedding gather.
                    if self.max_codebook_size > 0 and not self.remap_codebook:
                        highest = max(compressed)
                        if highest >= self.initial_vocab_size + self.max_active_codebook_size:
                            raise ValueError(
                                f"compressed window references LZW row "
                                f"{highest - self.initial_vocab_size} but the "
                                f"collate truncates the codebook to "
                                f"max_active_codebook_size="
                                f"{self.max_active_codebook_size} rows"
                            )

                    # Historical reporting denominator: expand the complete
                    # compressed T+1 window, including the first input-only
                    # token and targets that may subsequently be masked. Keep
                    # this exact definition under its old field name so every
                    # v0.6.x curve remains comparable.
                    n_base_tokens = self._count_expanded_base_tokens(
                        compressed, codebook
                    )

                    x = torch.LongTensor(compressed[:-1])
                    y = torch.LongTensor(compressed[1:])
                    if chunk_mask is not None:
                        loss_mask = self._compressed_loss_mask(compressed, chunk_mask, codebook)[1:]
                        if not loss_mask.any():
                            if self.lossless_windows:
                                offset = start + n_base_tokens
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
                            if self.lossless_windows:
                                offset = start + n_base_tokens
                            continue

                    # Corrected reporting denominator: expand only target
                    # labels that survived every mask above. This deliberately
                    # does not affect ``backward_loss``.
                    target_n_base_tokens = self._count_unmasked_target_base_tokens(
                        y, codebook
                    )

                    cb = self._codebook_to_tensor(codebook)

                    # Remap to compact codebook
                    if self.remap_codebook:
                        x, y, cb = self._remap_hyper_ids(x, y, cb)

                    if self._debug_samples_emitted < self.debug_samples:
                        self._debug_log(
                            f"lm yield sample_idx={self._debug_samples_emitted} shard_idx={shard_idx} "
                            f"offset={offset - self.base_chunk_len} input={tuple(x.shape)} "
                            f"codebook={tuple(cb.shape)} n_base_tokens={n_base_tokens} "
                            f"target_n_base_tokens={target_n_base_tokens}"
                        )
                    self._debug_samples_emitted += 1
                    sample = {
                        "input": x,
                        "codebook": cb,
                        "n_base_tokens": n_base_tokens,
                        "target_n_base_tokens": target_n_base_tokens,
                    }
                    if self.include_base_view:
                        (
                            sample["base_input"],
                            sample["base_labels"],
                            sample["base_n_base_tokens"],
                        ) = self._make_base_view(chunk, chunk_mask)
                    if self.online_codebook_mask:
                        sample["codebook_counts"] = available_counts
                        sample["online_skipped_targets"] = online_skipped_targets
                    if self.lossless_windows:
                        # Advance by exactly the base span of the emitted
                        # window (n_base_tokens expands the full truncated
                        # T+1 window): the next window starts on the first
                        # base token the truncation dropped, so windows tile
                        # the stream with no gap and no overlap.
                        offset = start + n_base_tokens
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

        The marker ids are per-tokenizer (self.compress_token_id /
        self.decompress_token_id), not the Llama-3 module constants: under
        Phi-3.5-mini those constants are past the end of the embedding table.
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

                    # A single-direction run must NOT consume the RNG: drawing
                    # here and discarding the result would leave two runs on the
                    # same seed reading different data.
                    if self.direction == "both":
                        compressing = random.random() < 0.5
                    else:
                        compressing = self.direction == "compress"
                    if compressing:
                        # Compression: given base, produce compressed
                        full = (
                            [self.decompress_token_id]
                            + base_tokens
                            + [self.compress_token_id]
                            + comp_tokens
                        )
                        loss_start = base_count + 1
                    else:
                        # Decompression: given compressed, produce base
                        full = (
                            [self.compress_token_id]
                            + comp_tokens
                            + [self.decompress_token_id]
                            + base_tokens
                        )
                        loss_start = comp_len + 1

                    loss_end = loss_start + (comp_len if compressing else base_count)

                    # Pad to exactly seq_len + 1 tokens
                    pad_len = (self.seq_len + 1) - len(full)
                    if pad_len > 0:
                        full += [self.pad_token_id] * pad_len

                    # Same hazard as _iter_lm: a window with ZERO hypertokens in
                    # the INPUT slice leaves this rank's hyper path unused while
                    # the other ranks take it, and the FSDP collectives deadlock
                    # until the NCCL watchdog aborts the job. Here the window can
                    # be hypertoken-free in either direction (LZW found no merge
                    # inside the budget), so scan exactly what the model branches
                    # on -- x = full[:-1], after padding.
                    if self.max_codebook_size > 0 and not any(
                        t >= vocab for t in full[:-1]
                    ):
                        self._debug_log(
                            f"compress skip zero-hypertoken window shard_idx={shard_idx} "
                            f"offset={offset - base_count} comp_len={comp_len}"
                        )
                        continue

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
                    # In either task direction, the supervised second half
                    # represents exactly ``base_count`` base tokens. Publish
                    # the explicit corrected denominator here too so collation
                    # never has to silently substitute the legacy field.
                    yield {
                        "input": x,
                        "codebook": cb,
                        "n_base_tokens": base_count,
                        "target_n_base_tokens": base_count,
                        "direction": (
                            DIRECTION_COMPRESS if compressing else DIRECTION_DECOMPRESS
                        ),
                    }, y

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
    missing_target_denominator = [
        index
        for index, inp in enumerate(inputs_list)
        if "target_n_base_tokens" not in inp
    ]
    if missing_target_denominator:
        raise ValueError(
            "target_n_base_tokens is required for every sample; missing at "
            f"batch indices {missing_target_denominator}. The LM path must "
            "report its masked-target expansion explicitly."
        )
    target_n_base_tokens = torch.tensor(
        [inp["target_n_base_tokens"] for inp in inputs_list],
        dtype=torch.long,
    )

    inputs = {
        "input": input_ids,
        "codebook": codebooks,
        "n_base_tokens": n_base_tokens,
        "target_n_base_tokens": target_n_base_tokens,
    }
    base_fields = {"base_input", "base_labels", "base_n_base_tokens"}
    has_base_fields = [base_fields <= set(inp) for inp in inputs_list]
    if any(has_base_fields):
        if not all(has_base_fields):
            raise ValueError("base replay fields must be present on every sample")
        inputs["base_input"] = torch.stack(
            [inp["base_input"] for inp in inputs_list]
        )
        inputs["base_labels"] = torch.stack(
            [inp["base_labels"] for inp in inputs_list]
        )
        inputs["base_n_base_tokens"] = torch.tensor(
            [inp["base_n_base_tokens"] for inp in inputs_list],
            dtype=torch.long,
        )
    # Which transducer direction each sample carries (compress mode only; see
    # DIRECTION_COMPRESS/DIRECTION_DECOMPRESS). The lm path has no direction, so
    # the key is absent there rather than defaulted to one of the two -- training
    # reads it to split its reporting and must not be handed a fabricated value.
    if "direction" in inputs_list[0]:
        missing_direction = [
            index for index, inp in enumerate(inputs_list) if "direction" not in inp
        ]
        if missing_direction:
            raise ValueError(
                "direction must be present on every sample of a batch once any "
                f"sample has it; missing at batch indices {missing_direction}"
            )
        inputs["direction"] = torch.tensor(
            [inp["direction"] for inp in inputs_list], dtype=torch.long
        )
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
    include_base_view: bool = False,
    lossless_windows: bool = False,
    compress_token_id: int = COMPRESS_TOKEN_ID,
    decompress_token_id: int = DECOMPRESS_TOKEN_ID,
    direction: str = "both",
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
        include_base_view=include_base_view,
        debug_samples=debug_samples,
        lossless_windows=lossless_windows,
        compress_token_id=compress_token_id,
        decompress_token_id=decompress_token_id,
        direction=direction,
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
