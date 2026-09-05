"""Single source of truth for which token ids the LZW compressor must never
merge into a hyper-token codebook entry.

Training (train.py), eval (lm_eval_adapter.py), and HF export (export_phi.py /
export.py) all have to agree on this set for a checkpoint's whole lifecycle --
computing it independently in each place is exactly how the train/eval
mismatch bugs documented in docs/evaluation.md happened (chat-turn markers,
then digits). Always derive it here instead of re-deriving it locally.
"""

from __future__ import annotations

import re


def base_disabled_ids(tokenizer, vocab_size: int) -> set[int]:
    """Special + added-vocab token ids (chat markers, etc.) -- never merged,
    regardless of any protection flag."""
    special = set(tokenizer.all_special_ids or [])
    added = set(tokenizer.get_added_vocab().values())
    return {i for i in (special | added) if 0 <= i < vocab_size}


# ASCII-digit runs only. Phi-3.5's SentencePiece vocab exposes exactly "0".."9"
# (digits are always split), so this yields the same 10 ids it always did.
# Llama-3's byte-level BPE pre-tokenizes digits in groups of up to three, so
# every 1-, 2- and 3-digit string is its own piece (10 + 100 + 1000 = 1110) and
# all of them must be protected for numbers to keep their base segmentation.
# Deliberately not \p{N} / str.isdigit(): both vocabs carry superscript and
# fraction pieces (², ½, ₀, ...) that were never protected historically.
_DIGIT_PIECE = re.compile(r"[▁Ġ]?[0-9]+")


def digit_ids(tokenizer, vocab_size: int) -> set[int]:
    """Token ids of every vocab piece that is a run of ASCII digits, with or
    without a leading word-boundary marker -- gated behind --disable_digit_ids."""
    return {
        i
        for piece, i in tokenizer.get_vocab().items()
        if _DIGIT_PIECE.fullmatch(piece) and 0 <= i < vocab_size
    }


def compute_disabled_ids(
    tokenizer, vocab_size: int, *, disable_digit_ids: bool = False
) -> list[int]:
    """The full disabled_ids set for a given training/eval/export configuration."""
    ids = base_disabled_ids(tokenizer, vocab_size)
    if disable_digit_ids:
        ids |= digit_ids(tokenizer, vocab_size)
    return sorted(ids)
