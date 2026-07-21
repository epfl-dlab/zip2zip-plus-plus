"""Single source of truth for which token ids the LZW compressor must never
merge into a hyper-token codebook entry.

Training (train.py), eval (lm_eval_adapter.py), and HF export (export_phi.py /
export.py) all have to agree on this set for a checkpoint's whole lifecycle --
computing it independently in each place is exactly how the train/eval
mismatch bugs documented in docs/evaluation.md happened (chat-turn markers,
then digits). Always derive it here instead of re-deriving it locally.
"""

from __future__ import annotations


def base_disabled_ids(tokenizer, vocab_size: int) -> set[int]:
    """Special + added-vocab token ids (chat markers, etc.) -- never merged,
    regardless of any protection flag."""
    special = set(tokenizer.all_special_ids or [])
    added = set(tokenizer.get_added_vocab().values())
    return {i for i in (special | added) if 0 <= i < vocab_size}


def digit_ids(tokenizer, vocab_size: int) -> set[int]:
    """Token ids for single digits 0-9, with and without the SentencePiece
    word-boundary marker -- gated behind --disable_digit_ids."""
    pieces = {str(d) for d in range(10)} | {f"▁{d}" for d in range(10)}
    return {
        i
        for piece, i in tokenizer.get_vocab().items()
        if piece in pieces and 0 <= i < vocab_size
    }


def compute_disabled_ids(
    tokenizer, vocab_size: int, *, disable_digit_ids: bool = False
) -> list[int]:
    """The full disabled_ids set for a given training/eval/export configuration."""
    ids = base_disabled_ids(tokenizer, vocab_size)
    if disable_digit_ids:
        ids |= digit_ids(tokenizer, vocab_size)
    return sorted(ids)
