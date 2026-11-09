#!/usr/bin/env python3
"""Build a shared-window SFT dataset (shared_S) and >2x drop audit JSON.

Given a pre-tokenized SFT shard directory (e.g. phi-1B-sft-8shards-eosfix),
this script inspects every fixed ``2 * seq_len`` chunk in shard order using the
exact eligibility rules of ``Zip2ZipDataset._iter_lm`` (with historical
``lossless_windows=False``, ``online_codebook_mask=False``, ``remap_codebook=False``)
across all Phi-3.5 SFT training regimes:

  1. Uncompressed control (``max_codebook_size=0``)
  2. zip2zip baseline (``v0.2``, ``disable_digit_ids=False``)
  3. Digit-protected zip2zip / zip2zip++ (``v0.4`` / ``v0.6.4``, ``disable_digit_ids=True``)

Only chunks that pass ALL regimes are written to ``--dst_dir`` (preserving the
8-shard ``shard_XXXXX.npy`` + ``mask_XXXXX.npy`` layout). Training any of the
four variants on ``--dst_dir`` therefore experiences ZERO runtime window skips
and consumes the exact same raw chunk at every optimizer step on every rank.

Simultaneously, the script writes a comprehensive JSON audit (``--audit_json``)
covering:
  - Per-shard and total window retention / drop reason counts (``ratio_gt_2x``,
    ``zero_hypertoken``, ``all_masked``) and ``v0.2 -> v0.4`` transitions.
  - Token-type and structural composition of retained vs ``>2x`` dropped windows
    (digit, whitespace/indentation, LaTeX/math, code punctuation, 4-gram
    repetition, lexical diversity, chat-turn markers, document boundaries),
    without storing raw training text.
  - Exact step-matched training token budgets over ``--steps_budget_chunks``
    emitted windows (default ``128,000 = 8,000 * 16``) for ``world_size`` in
    ``(1, 2, 4)`` on both the original dataset and ``shared_S``.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import re
import sys
import time
from typing import Any

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.path.join(_ROOT, "src") not in sys.path:
    sys.path.insert(0, os.path.join(_ROOT, "src"))
if os.path.join(_ROOT, "ext", "torchtitan") not in sys.path:
    sys.path.insert(0, os.path.join(_ROOT, "ext", "torchtitan"))

from zip2zip_compression import LZWCompressor  # noqa: E402
from zip2zip_core.data import Zip2ZipDataset  # noqa: E402
from zip2zip_core.disabled_ids import compute_disabled_ids, digit_ids  # noqa: E402

STATUS_OK = 0
STATUS_RATIO_GT_2X = 1
STATUS_ZERO_HYPERTOKEN = 2
STATUS_ALL_MASKED = 3
STATUS_CODEBOOK_OVERFLOW = 4

STATUS_NAMES = {
    STATUS_OK: "ok",
    STATUS_RATIO_GT_2X: "ratio_gt_2x",
    STATUS_ZERO_HYPERTOKEN: "zero_hypertoken",
    STATUS_ALL_MASKED: "all_masked",
    STATUS_CODEBOOK_OVERFLOW: "codebook_overflow",
}

_BYTE_TOKEN_RE = re.compile(r"^<0x([0-9A-Fa-f]{2})>$")
_LATEX_CMD_RE = re.compile(
    r"(?:\\|frac|sqrt|cdot|begin|end|left|right|mathrm|mathbf|mathbb|"
    r"alpha|beta|gamma|delta|theta|lambda|sigma|omega|sum|int|prod|infty)"
)
_MATH_CHARS = set("^_${}=+-*/<>≤≥≠±×÷∑∏∫√∞∂∇∈∉⊂⊆∪∩")
_CODE_KEYWORDS = {
    "def", "class", "return", "import", "from", "self", "elif", "lambda",
    "yield", "async", "await", "const", "function", "var", "let", "public",
    "private", "static", "void", "include", "struct", "namespace",
}
_CODE_PUNCT_SUBSTRS = (
    "()", "[]", "{}", "==", "!=", "<=", ">=", "->", "=>", "::", "//", "/*",
    "*/", "&&", "||", "+=", "-=", "__",
)


def _decode_piece_for_audit(piece: str) -> str:
    m = _BYTE_TOKEN_RE.match(piece)
    if m:
        b = int(m.group(1), 16)
        return chr(b) if b < 128 else ""
    return piece.replace("▁", " ").replace("Ġ", " ")


def build_vocab_audit_tables(tokenizer, vocab_size: int) -> dict[str, Any]:
    """Precompute uint8 token-category lookup arrays of length ``vocab_size``."""
    is_digit = np.zeros(vocab_size, dtype=np.uint8)
    is_ws_indent = np.zeros(vocab_size, dtype=np.uint8)
    is_latex_math = np.zeros(vocab_size, dtype=np.uint8)
    is_code_punct = np.zeros(vocab_size, dtype=np.uint8)
    is_chat_marker = np.zeros(vocab_size, dtype=np.uint8)

    base_special = set(compute_disabled_ids(tokenizer, vocab_size, disable_digit_ids=False))
    for i in digit_ids(tokenizer, vocab_size):
        if 0 <= i < vocab_size:
            is_digit[i] = 1

    added_vocab = tokenizer.get_added_vocab() or {}
    for tok_str, tok_id in added_vocab.items():
        if 0 <= tok_id < vocab_size and (
            "user" in tok_str.lower()
            or "assistant" in tok_str.lower()
            or tok_str in ("<|end|>", "<|eot_id|>", "<|im_start|>", "<|im_end|>")
        ):
            is_chat_marker[tok_id] = 1

    vocab = tokenizer.get_vocab()
    for piece, tok_id in vocab.items():
        if not (0 <= tok_id < vocab_size) or tok_id in base_special:
            continue
        norm = _decode_piece_for_audit(piece)
        if not norm:
            continue
        if norm.strip() == "":
            is_ws_indent[tok_id] = 1
            continue
        stripped = norm.strip()
        if any(ch in _MATH_CHARS for ch in norm) or _LATEX_CMD_RE.search(norm):
            is_latex_math[tok_id] = 1
        if (
            stripped in _CODE_KEYWORDS
            or any(sub in norm for sub in _CODE_PUNCT_SUBSTRS)
            or any(ch in "();[]{}" for ch in norm)
        ):
            is_code_punct[tok_id] = 1

    eos_id = tokenizer.eos_token_id
    return {
        "is_digit": is_digit,
        "is_ws_indent": is_ws_indent,
        "is_latex_math": is_latex_math,
        "is_code_punct": is_code_punct,
        "is_chat_marker": is_chat_marker,
        "eos_token_id": int(eos_id) if eos_id is not None else -1,
        "disabled_v02": compute_disabled_ids(tokenizer, vocab_size, disable_digit_ids=False),
        "disabled_v04": compute_disabled_ids(tokenizer, vocab_size, disable_digit_ids=True),
    }


def _eval_compressed_chunk(
    chunk_list: list[int],
    chunk_mask: np.ndarray | None,
    compressor_args: dict[str, Any],
    seq_len: int,
    vocab_size: int,
    max_active_codebook_size: int,
) -> tuple[int, int, int, int]:
    """Mirror ``Zip2ZipDataset._iter_lm`` on one ``2 * seq_len`` chunk.

    Returns ``(status, full_compressed_len, n_base_tokens, target_n_base_tokens)``.
    """
    compressor = LZWCompressor(**compressor_args)
    compressed, _, codebook = compressor.encode(
        chunk_list, padding="do_not_pad", truncation=False
    )
    comp_len = len(compressed)
    if comp_len < seq_len + 1:
        return STATUS_RATIO_GT_2X, comp_len, 0, 0

    win = compressed[: seq_len + 1]
    if not any(t >= vocab_size for t in win[:-1]):
        return STATUS_ZERO_HYPERTOKEN, comp_len, 0, 0

    highest = max(win)
    if highest >= vocab_size + max_active_codebook_size:
        return STATUS_CODEBOOK_OVERFLOW, comp_len, 0, 0

    cb_dict = codebook.to_dict()
    spans = [len(cb_dict[t]) if t >= vocab_size else 1 for t in win]
    n_base_tokens = int(sum(spans))

    if chunk_mask is None:
        target_n_base = int(sum(spans[1:]))
        return STATUS_OK, comp_len, n_base_tokens, target_n_base

    cursor = spans[0]
    target_n_base = 0
    any_valid = False
    for span_len in spans[1:]:
        if bool(np.all(chunk_mask[cursor : cursor + span_len])):
            any_valid = True
            target_n_base += span_len
        cursor += span_len

    if not any_valid:
        return STATUS_ALL_MASKED, comp_len, 0, 0
    return STATUS_OK, comp_len, n_base_tokens, target_n_base


def _fourgram_repeat_rate(chunk_arr: np.ndarray, vocab_size: int) -> float:
    n = len(chunk_arr)
    if n < 4:
        return 0.0
    u = chunk_arr.astype(np.uint64, copy=False)
    if vocab_size <= 65536:
        packed = (u[:-3] << 48) | (u[1:-2] << 32) | (u[2:-1] << 16) | u[3:]
    else:
        packed = (
            u[:-3] * np.uint64(1099511628211)
            ^ u[1:-2] * np.uint64(14695981039346656037)
            ^ u[2:-1] * np.uint64(6364136223846793005)
            ^ u[3:]
        )
    return float(1.0 - (len(np.unique(packed)) / float(n - 3)))


def _classify_bucket(
    repeat_4g: float,
    digit_frac: float,
    latex_frac: float,
    ws_frac: float,
    code_frac: float,
    has_chat: bool,
) -> int:
    if repeat_4g >= 0.35:
        return 0  # high_repetition_boilerplate
    if (digit_frac + latex_frac) >= 0.18:
        return 1  # math_or_numeric_heavy
    if (ws_frac + code_frac) >= 0.22:
        return 2  # code_or_indentation_heavy
    if has_chat:
        return 3  # chat_dialogue
    return 4  # general_prose


BUCKET_NAMES = (
    "high_repetition_boilerplate",
    "math_or_numeric_heavy",
    "code_or_indentation_heavy",
    "chat_dialogue",
    "general_prose",
)

FEATURE_NAMES = (
    "digit_frac",
    "ws_indent_frac",
    "latex_math_frac",
    "code_punct_frac",
    "fourgram_repeat_rate",
    "unique_token_ratio",
    "unmasked_frac",
    "full_chunk_ratio_v02",
    "full_chunk_ratio_v04",
)


def _process_single_shard(task: dict[str, Any]) -> dict[str, Any]:
    shard_idx = task["shard_idx"]
    shard_path = task["shard_path"]
    mask_path = task["mask_path"]
    dst_shard_path = task["dst_shard_path"]
    dst_mask_path = task["dst_mask_path"]
    seq_len = task["seq_len"]
    vocab_size = task["vocab_size"]
    pad_token_id = task["pad_token_id"]
    max_subtokens = task["max_subtokens"]
    max_codebook_size = task["max_codebook_size"]
    max_active_codebook_size = task["max_active_codebook_size"]
    tables = task["tables"]

    t0 = time.perf_counter()
    shard = Zip2ZipDataset._load_token_shard(shard_path)
    mask_shard = np.load(mask_path, mmap_mode="r") if mask_path is not None else None
    if mask_shard is not None and len(mask_shard) != len(shard):
        raise ValueError(f"Mask length mismatch for {shard_path}")

    chunk_len = 2 * seq_len
    n_chunks = len(shard) // chunk_len
    tail_tokens_ignored = int(len(shard) - n_chunks * chunk_len)

    comp_v02 = dict(
        initial_vocab_size=vocab_size,
        max_codebook_size=max_codebook_size,
        max_subtokens=max_subtokens,
        pad_token_id=pad_token_id,
        disabled_ids=list(tables["disabled_v02"]),
    )
    comp_v04 = dict(
        initial_vocab_size=vocab_size,
        max_codebook_size=max_codebook_size,
        max_subtokens=max_subtokens,
        pad_token_id=pad_token_id,
        disabled_ids=list(tables["disabled_v04"]),
    )

    is_digit = tables["is_digit"]
    is_ws = tables["is_ws_indent"]
    is_latex = tables["is_latex_math"]
    is_code = tables["is_code_punct"]
    is_chat = tables["is_chat_marker"]

    status_uncomp = np.empty(n_chunks, dtype=np.uint8)
    status_v02 = np.empty(n_chunks, dtype=np.uint8)
    status_v04 = np.empty(n_chunks, dtype=np.uint8)
    n_base_v02 = np.zeros(n_chunks, dtype=np.uint32)
    tgt_base_v02 = np.zeros(n_chunks, dtype=np.uint32)
    n_base_v04 = np.zeros(n_chunks, dtype=np.uint32)
    tgt_base_v04 = np.zeros(n_chunks, dtype=np.uint32)
    tgt_base_uncomp = np.zeros(n_chunks, dtype=np.uint32)
    buckets = np.empty(n_chunks, dtype=np.uint8)
    features = np.empty((n_chunks, len(FEATURE_NAMES)), dtype=np.float32)

    kept_indices: list[int] = []
    inv_len = 1.0 / float(chunk_len)

    for idx in range(n_chunks):
        start = idx * chunk_len
        chunk_arr = np.asarray(shard[start : start + chunk_len])
        chunk_list = chunk_arr.tolist()
        c_mask = (
            np.asarray(mask_shard[start : start + chunk_len])
            if mask_shard is not None
            else None
        )

        if c_mask is None:
            status_uncomp[idx] = STATUS_OK
            tgt_base_uncomp[idx] = seq_len
            unmasked_frac = 1.0
        else:
            unmasked_targets = int(np.count_nonzero(c_mask[1 : seq_len + 1]))
            if unmasked_targets > 0:
                status_uncomp[idx] = STATUS_OK
                tgt_base_uncomp[idx] = unmasked_targets
            else:
                status_uncomp[idx] = STATUS_ALL_MASKED
            unmasked_frac = float(np.count_nonzero(c_mask)) * inv_len

        st02, clen02, nb02, tb02 = _eval_compressed_chunk(
            chunk_list, c_mask, comp_v02, seq_len, vocab_size, max_active_codebook_size
        )
        st04, clen04, nb04, tb04 = _eval_compressed_chunk(
            chunk_list, c_mask, comp_v04, seq_len, vocab_size, max_active_codebook_size
        )
        if st02 == STATUS_CODEBOOK_OVERFLOW or st04 == STATUS_CODEBOOK_OVERFLOW:
            raise ValueError(
                f"Chunk {idx} in {shard_path} exceeded max_active_codebook_size={max_active_codebook_size}"
            )

        status_v02[idx] = st02
        n_base_v02[idx] = nb02
        tgt_base_v02[idx] = tb02
        status_v04[idx] = st04
        n_base_v04[idx] = nb04
        tgt_base_v04[idx] = tb04

        if status_uncomp[idx] == STATUS_OK and st02 == STATUS_OK and st04 == STATUS_OK:
            kept_indices.append(idx)

        clamped = np.clip(chunk_arr, 0, vocab_size - 1)
        d_frac = float(is_digit[clamped].sum()) * inv_len
        w_frac = float(is_ws[clamped].sum()) * inv_len
        l_frac = float(is_latex[clamped].sum()) * inv_len
        c_frac = float(is_code[clamped].sum()) * inv_len
        has_chat = bool(is_chat[clamped].any())
        rep_4g = _fourgram_repeat_rate(chunk_arr, vocab_size)
        uniq_ratio = float(len(np.unique(chunk_arr))) * inv_len
        ratio02 = float(chunk_len) / float(max(1, clen02))
        ratio04 = float(chunk_len) / float(max(1, clen04))

        features[idx] = (
            d_frac,
            w_frac,
            l_frac,
            c_frac,
            rep_4g,
            uniq_ratio,
            unmasked_frac,
            ratio02,
            ratio04,
        )
        buckets[idx] = _classify_bucket(
            rep_4g, d_frac, l_frac, w_frac, c_frac, has_chat
        )

    # Materialize filtered shard & mask arrays cleanly in one contiguous write.
    n_kept = len(kept_indices)
    out_tokens = np.empty(n_kept * chunk_len, dtype=shard.dtype)
    out_mask = (
        np.empty(n_kept * chunk_len, dtype=mask_shard.dtype)
        if mask_shard is not None
        else None
    )
    for out_pos, src_idx in enumerate(kept_indices):
        s_start = src_idx * chunk_len
        d_start = out_pos * chunk_len
        out_tokens[d_start : d_start + chunk_len] = shard[s_start : s_start + chunk_len]
        if out_mask is not None:
            out_mask[d_start : d_start + chunk_len] = mask_shard[
                s_start : s_start + chunk_len
            ]

    np.save(dst_shard_path, out_tokens)
    if out_mask is not None and dst_mask_path is not None:
        np.save(dst_mask_path, out_mask)

    elapsed = time.perf_counter() - t0
    print(
        f"[sharedS] shard_{shard_idx:05d}: kept {n_kept:,} / {n_chunks:,} chunks "
        f"({100.0 * n_kept / max(1, n_chunks):.2f}%) in {elapsed:.1f}s",
        flush=True,
    )
    return {
        "shard_idx": shard_idx,
        "src_tokens": int(len(shard)),
        "dst_tokens": int(len(out_tokens)),
        "tail_tokens_ignored": tail_tokens_ignored,
        "n_chunks": n_chunks,
        "n_kept": n_kept,
        "token_dtype": str(shard.dtype),
        "mask_dtype": str(mask_shard.dtype) if mask_shard is not None else None,
        "elapsed_sec": round(elapsed, 2),
        "status_uncomp": status_uncomp,
        "status_v02": status_v02,
        "status_v04": status_v04,
        "n_base_v02": n_base_v02,
        "tgt_base_v02": tgt_base_v02,
        "n_base_v04": n_base_v04,
        "tgt_base_v04": tgt_base_v04,
        "tgt_base_uncomp": tgt_base_uncomp,
        "buckets": buckets,
        "features": features,
    }


def _summarize_subset(
    mask: np.ndarray,
    features: np.ndarray,
    buckets: np.ndarray,
    total_chunks: int,
) -> dict[str, Any]:
    count = int(mask.sum())
    if count == 0:
        return {
            "chunks": 0,
            "share_of_all_chunks": 0.0,
            "feature_means": {},
            "feature_medians": {},
            "bucket_counts": {k: 0 for k in BUCKET_NAMES},
            "bucket_shares": {k: 0.0 for k in BUCKET_NAMES},
        }
    sub_feat = features[mask]
    sub_bkt = buckets[mask]
    means = {
        name: round(float(sub_feat[:, j].mean()), 6)
        for j, name in enumerate(FEATURE_NAMES)
    }
    medians = {
        name: round(float(np.median(sub_feat[:, j])), 6)
        for j, name in enumerate(FEATURE_NAMES)
    }
    b_counts = {
        name: int((sub_bkt == b_id).sum())
        for b_id, name in enumerate(BUCKET_NAMES)
    }
    b_shares = {
        name: round(b_counts[name] / float(count), 6) for name in BUCKET_NAMES
    }
    return {
        "chunks": count,
        "share_of_all_chunks": round(count / float(max(1, total_chunks)), 6),
        "feature_means": means,
        "feature_medians": medians,
        "bucket_counts": b_counts,
        "bucket_shares": b_shares,
    }


def _simulate_training_budget(
    shard_results: list[dict[str, Any]],
    *,
    world_size: int,
    total_emitted_chunks: int,
    seq_len: int,
    use_shared_s: bool,
) -> dict[str, Any]:
    """Exact token & skip accounting over ``total_emitted_chunks`` across ranks."""
    if world_size > len(shard_results):
        return {"supported": False, "reason": "world_size > num_shards"}
    if total_emitted_chunks % world_size != 0:
        raise ValueError(
            f"total_emitted_chunks={total_emitted_chunks} not divisible by world_size={world_size}"
        )
    per_rank_quota = total_emitted_chunks // world_size
    chunk_len = 2 * seq_len

    out: dict[str, Any] = {"supported": True, "per_rank_emitted_chunks": per_rank_quota}
    variants = (
        ("uncompressed", "status_uncomp", None, "tgt_base_uncomp"),
        ("v0.2_zip2zip", "status_v02", "n_base_v02", "tgt_base_v02"),
        ("v0.4_v0.6.4_digitsafe", "status_v04", "n_base_v04", "tgt_base_v04"),
    )

    for var_name, st_key, nb_key, tb_key in variants:
        inspected = 0
        skipped = 0
        wrapped_epochs = 0
        sum_n_base = 0
        sum_tgt_base = 0

        for rank in range(world_size):
            rank_shards = shard_results[rank::world_size]
            # Build per-rank stream of chunks (either filtered shared_S or original).
            emitted_r = 0
            pass_idx = 0
            while emitted_r < per_rank_quota:
                if pass_idx > 0:
                    wrapped_epochs += 1
                progressed = False
                for sr in rank_shards:
                    shared_keep = (
                        (sr["status_uncomp"] == STATUS_OK)
                        & (sr["status_v02"] == STATUS_OK)
                        & (sr["status_v04"] == STATUS_OK)
                    )
                    indices = (
                        np.nonzero(shared_keep)[0]
                        if use_shared_s
                        else np.arange(sr["n_chunks"])
                    )
                    st_arr = sr[st_key]
                    nb_arr = sr[nb_key] if nb_key is not None else None
                    tb_arr = sr[tb_key]
                    for c_idx in indices:
                        inspected += 1
                        if st_arr[c_idx] != STATUS_OK:
                            skipped += 1
                            continue
                        emitted_r += 1
                        progressed = True
                        sum_n_base += (
                            int(nb_arr[c_idx]) if nb_arr is not None else (seq_len + 1)
                        )
                        sum_tgt_base += int(tb_arr[c_idx])
                        if emitted_r >= per_rank_quota:
                            break
                    if emitted_r >= per_rank_quota:
                        break
                if not progressed:
                    return {
                        "supported": False,
                        "reason": f"rank {rank} has zero eligible chunks for {var_name}",
                    }
                pass_idx += 1

        out[var_name] = {
            "chunks_emitted": total_emitted_chunks,
            "chunks_inspected": inspected,
            "chunks_skipped_at_runtime": skipped,
            "runtime_skip_rate": round(skipped / float(max(1, inspected)), 6),
            "rank_epoch_wraps": wrapped_epochs,
            "model_input_tokens": total_emitted_chunks * seq_len,
            "raw_chunks_inspected_tokens": inspected * chunk_len,
            "window_expanded_base_tokens": sum_n_base,
            "supervised_target_base_tokens": sum_tgt_base,
            "mean_window_compression_ratio": round(
                sum_n_base / float(total_emitted_chunks * (seq_len + 1)), 6
            ),
        }
    return out


def build_shared_windows_dataset(
    *,
    src_dir: str,
    dst_dir: str,
    audit_json: str,
    tokenizer_name: str = "microsoft/Phi-3.5-mini-instruct",
    vocab_size: int = 32064,
    pad_token_id: int = 32000,
    seq_len: int = 2048,
    max_subtokens: int = 4,
    max_codebook_size: int = 4096,
    max_active_codebook_size: int = 2048,
    workers: int = 8,
    steps_budget_chunks: int = 128_000,
    overwrite: bool = False,
) -> dict[str, Any]:
    from transformers import AutoTokenizer

    shard_files = sorted(
        os.path.join(src_dir, f)
        for f in os.listdir(src_dir)
        if f.endswith(".npy") and not f.startswith("mask_")
    )
    if not shard_files:
        raise ValueError(f"No token shards (.npy) found in {src_dir}")

    mask_files = [Zip2ZipDataset._mask_path_for_shard(p) for p in shard_files]
    has_masks = [os.path.exists(p) for p in mask_files]
    if any(has_masks) and not all(has_masks):
        raise ValueError(f"Partial loss masks in {src_dir}")
    use_masks = all(has_masks)

    os.makedirs(dst_dir, exist_ok=True)
    existing_dst = [f for f in os.listdir(dst_dir) if f.endswith(".npy")]
    if existing_dst and not overwrite:
        raise FileExistsError(
            f"{dst_dir} already contains .npy shards ({existing_dst[:3]}); "
            "pass --overwrite to replace them."
        )

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    tables = build_vocab_audit_tables(tokenizer, vocab_size)

    tasks = []
    for i, sp in enumerate(shard_files):
        bname = os.path.basename(sp)
        dst_sp = os.path.join(dst_dir, bname)
        dst_mp = (
            Zip2ZipDataset._mask_path_for_shard(dst_sp) if use_masks else None
        )
        tasks.append(
            {
                "shard_idx": i,
                "shard_path": sp,
                "mask_path": mask_files[i] if use_masks else None,
                "dst_shard_path": dst_sp,
                "dst_mask_path": dst_mp,
                "seq_len": seq_len,
                "vocab_size": vocab_size,
                "pad_token_id": pad_token_id,
                "max_subtokens": max_subtokens,
                "max_codebook_size": max_codebook_size,
                "max_active_codebook_size": max_active_codebook_size,
                "tables": tables,
            }
        )

    t_start = time.perf_counter()
    n_proc = max(1, min(int(workers), len(tasks)))
    if n_proc == 1:
        shard_results = [_process_single_shard(t) for t in tasks]
    else:
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=n_proc) as pool:
            shard_results = pool.map(_process_single_shard, tasks)
    shard_results.sort(key=lambda r: r["shard_idx"])
    wall_sec = time.perf_counter() - t_start

    # Write manifest.json in dst_dir matching create_sft_dataset_zip2zip1B_phi.py.
    total_src_tokens = sum(r["src_tokens"] for r in shard_results)
    total_dst_tokens = sum(r["dst_tokens"] for r in shard_results)
    manifest = {
        "source_dataset_dir": os.path.abspath(src_dir),
        "filter_policy": "shared_S_intersection_uncompressed_v0.2_v0.4_v0.6.4",
        "total_tokens": total_dst_tokens,
        "source_total_tokens": total_src_tokens,
        "num_shards": len(shard_results),
        "seq_len": seq_len,
        "base_chunk_len": 2 * seq_len,
        "max_subtokens": max_subtokens,
        "max_codebook_size": max_codebook_size,
        "max_active_codebook_size": max_active_codebook_size,
        "has_loss_masks": use_masks,
        "token_dtype": shard_results[0]["token_dtype"],
        "mask_dtype": shard_results[0]["mask_dtype"],
    }
    with open(os.path.join(dst_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    # Concatenate per-chunk arrays for global dataset audit summaries.
    st_uncomp = np.concatenate([r["status_uncomp"] for r in shard_results])
    st_v02 = np.concatenate([r["status_v02"] for r in shard_results])
    st_v04 = np.concatenate([r["status_v04"] for r in shard_results])
    all_buckets = np.concatenate([r["buckets"] for r in shard_results])
    all_features = np.concatenate([r["features"] for r in shard_results], axis=0)
    total_chunks = int(len(st_uncomp))

    shared_mask = (
        (st_uncomp == STATUS_OK)
        & (st_v02 == STATUS_OK)
        & (st_v04 == STATUS_OK)
    )
    rescued_by_digit = (st_v02 == STATUS_RATIO_GT_2X) & (st_v04 == STATUS_OK)
    newly_zero_hyper_in_v04 = (st_v02 == STATUS_OK) & (
        st_v04 == STATUS_ZERO_HYPERTOKEN
    )

    def _status_counts(arr: np.ndarray) -> dict[str, int]:
        return {
            name: int((arr == code).sum())
            for code, name in STATUS_NAMES.items()
            if code != STATUS_CODEBOOK_OVERFLOW
        }

    per_shard_summary = []
    for r in shard_results:
        per_shard_summary.append(
            {
                "shard_idx": r["shard_idx"],
                "src_tokens": r["src_tokens"],
                "dst_tokens": r["dst_tokens"],
                "tail_tokens_ignored": r["tail_tokens_ignored"],
                "n_chunks": r["n_chunks"],
                "n_kept_shared_S": r["n_kept"],
                "retention_rate": round(
                    r["n_kept"] / float(max(1, r["n_chunks"])), 6
                ),
                "uncompressed": _status_counts(r["status_uncomp"]),
                "v0.2_zip2zip": _status_counts(r["status_v02"]),
                "v0.4_v0.6.4_digitsafe": _status_counts(r["status_v04"]),
                "elapsed_sec": r["elapsed_sec"],
            }
        )

    budgets = {}
    for ws in (1, 2, 4):
        if steps_budget_chunks % ws == 0 and ws <= len(shard_results):
            budgets[f"world_size_{ws}"] = {
                "original_dataset": _simulate_training_budget(
                    shard_results,
                    world_size=ws,
                    total_emitted_chunks=steps_budget_chunks,
                    seq_len=seq_len,
                    use_shared_s=False,
                ),
                "shared_S_dataset": _simulate_training_budget(
                    shard_results,
                    world_size=ws,
                    total_emitted_chunks=steps_budget_chunks,
                    seq_len=seq_len,
                    use_shared_s=True,
                ),
            }

    audit = {
        "config": {
            "src_dir": os.path.abspath(src_dir),
            "dst_dir": os.path.abspath(dst_dir),
            "tokenizer": tokenizer_name,
            "vocab_size": vocab_size,
            "pad_token_id": pad_token_id,
            "seq_len": seq_len,
            "base_chunk_len": 2 * seq_len,
            "max_subtokens": max_subtokens,
            "max_codebook_size": max_codebook_size,
            "max_active_codebook_size": max_active_codebook_size,
            "disabled_ids_v02_count": len(tables["disabled_v02"]),
            "disabled_ids_v04_count": len(tables["disabled_v04"]),
            "steps_budget_chunks": steps_budget_chunks,
            "wall_seconds": round(wall_sec, 2),
        },
        "totals": {
            "total_chunks": total_chunks,
            "shared_S_kept_chunks": int(shared_mask.sum()),
            "shared_S_dropped_chunks": int((~shared_mask).sum()),
            "shared_S_retention_rate": round(
                float(shared_mask.sum()) / float(max(1, total_chunks)), 6
            ),
            "source_total_tokens": total_src_tokens,
            "shared_S_total_tokens": total_dst_tokens,
            "uncompressed_counts": _status_counts(st_uncomp),
            "v0.2_zip2zip_counts": _status_counts(st_v02),
            "v0.4_v0.6.4_digitsafe_counts": _status_counts(st_v04),
            "transitions_v02_to_v04": {
                "rescued_from_ratio_gt_2x_by_digit_protection": int(
                    rescued_by_digit.sum()
                ),
                "newly_dropped_zero_hypertoken_in_v04": int(
                    newly_zero_hyper_in_v04.sum()
                ),
                "both_ratio_gt_2x": int(
                    (
                        (st_v02 == STATUS_RATIO_GT_2X)
                        & (st_v04 == STATUS_RATIO_GT_2X)
                    ).sum()
                ),
            },
        },
        "content_composition": {
            "all_chunks": _summarize_subset(
                np.ones(total_chunks, dtype=bool),
                all_features,
                all_buckets,
                total_chunks,
            ),
            "shared_S_kept": _summarize_subset(
                shared_mask, all_features, all_buckets, total_chunks
            ),
            "dropped_v02_ratio_gt_2x": _summarize_subset(
                st_v02 == STATUS_RATIO_GT_2X,
                all_features,
                all_buckets,
                total_chunks,
            ),
            "dropped_v04_ratio_gt_2x": _summarize_subset(
                st_v04 == STATUS_RATIO_GT_2X,
                all_features,
                all_buckets,
                total_chunks,
            ),
            "rescued_by_digit_protection": _summarize_subset(
                rescued_by_digit, all_features, all_buckets, total_chunks
            ),
            "dropped_v04_zero_hypertoken": _summarize_subset(
                st_v04 == STATUS_ZERO_HYPERTOKEN,
                all_features,
                all_buckets,
                total_chunks,
            ),
            "dropped_all_masked": _summarize_subset(
                (st_uncomp == STATUS_ALL_MASKED)
                | (st_v02 == STATUS_ALL_MASKED)
                | (st_v04 == STATUS_ALL_MASKED),
                all_features,
                all_buckets,
                total_chunks,
            ),
        },
        "step_matched_training_budgets": budgets,
        "per_shard": per_shard_summary,
    }

    if audit_json:
        audit_dir = os.path.dirname(os.path.abspath(audit_json))
        if audit_dir:
            os.makedirs(audit_dir, exist_ok=True)
        with open(audit_json, "w") as f:
            json.dump(audit, f, indent=2)
        print(f"[sharedS] Wrote audit JSON -> {audit_json}", flush=True)

    return audit


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--src_dir", required=True, help="Input SFT shard directory.")
    p.add_argument("--dst_dir", required=True, help="Output shared_S shard directory.")
    p.add_argument(
        "--audit_json",
        required=True,
        help="Output JSON path for window-drop and token-budget audit.",
    )
    p.add_argument("--tokenizer", default="microsoft/Phi-3.5-mini-instruct")
    p.add_argument("--vocab_size", type=int, default=32064)
    p.add_argument("--pad_token_id", type=int, default=32000)
    p.add_argument("--seq_len", type=int, default=2048)
    p.add_argument("--max_subtokens", type=int, default=4)
    p.add_argument("--max_codebook_size", type=int, default=4096)
    p.add_argument("--max_active_codebook_size", type=int, default=2048)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument(
        "--steps_budget_chunks",
        type=int,
        default=128_000,
        help="Total emitted windows across all ranks for step-budget simulation "
        "(default 128000 = 8000 steps * 16 windows/step).",
    )
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    audit = build_shared_windows_dataset(
        src_dir=args.src_dir,
        dst_dir=args.dst_dir,
        audit_json=args.audit_json,
        tokenizer_name=args.tokenizer,
        vocab_size=args.vocab_size,
        pad_token_id=args.pad_token_id,
        seq_len=args.seq_len,
        max_subtokens=args.max_subtokens,
        max_codebook_size=args.max_codebook_size,
        max_active_codebook_size=args.max_active_codebook_size,
        workers=args.workers,
        steps_budget_chunks=args.steps_budget_chunks,
        overwrite=args.overwrite,
    )
    totals = audit["totals"]
    print(
        f"[sharedS] DONE: kept {totals['shared_S_kept_chunks']:,} / "
        f"{totals['total_chunks']:,} chunks "
        f"({100.0 * totals['shared_S_retention_rate']:.2f}%), "
        f"tokens={totals['shared_S_total_tokens']:,}"
    )


if __name__ == "__main__":
    main()
