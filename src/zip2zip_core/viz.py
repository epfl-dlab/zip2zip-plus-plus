"""Visualization utilities for zip2zip token sequences.

Architecture: colorizer (strategy) + renderer (single function).

Colorizers:  (token_ids, codebook, ...) → List[ColoredToken]   # decide colors
Renderer:    (List[ColoredToken], tokenizer) → str              # decode + apply ANSI
"""
from __future__ import annotations

import dataclasses
import random
from typing import Dict, List, Optional, Set

from transformers import PreTrainedTokenizerBase


# ── data ─────────────────────────────────────────────────────────────────────

RESET = "\033[0m"

BLUE = "\033[34m"
YELLOW = "\033[33m"
ORANGE = "\033[38;5;208m"
RED = "\033[31m"
DARK_RED = "\033[38;5;88m"
BROWN = "\033[38;5;130m"
BLACK = "\033[30m"

NGRAM_COLORS = {
    1: BLUE,
    2: YELLOW,
    3: ORANGE,
    4: RED,
    5: DARK_RED,
}


@dataclasses.dataclass
class ColoredToken:
    token_ids: List[int]
    color: str = RESET


# ── colorizers (strategy) ────────────────────────────────────────────────────

def colorize_by_ngram(
    token_ids: List[int],
    codebook: Dict[int, List[int]],
    special_token_ids: Optional[Set[int]] = None,
) -> List[ColoredToken]:
    """Color tokens by n-gram size: blue=1, yellow=2, orange=3, red=4, dark_red=5."""
    if special_token_ids is None:
        special_token_ids = set()
    out = []
    for tid in token_ids:
        base_ids = codebook.get(tid, [tid])
        if tid in special_token_ids:
            color = BLACK
        else:
            color = NGRAM_COLORS.get(len(base_ids), BROWN)
        out.append(ColoredToken(token_ids=base_ids, color=color))
    return out


def colorize_by_ppl(
    token_ids: List[int],
    ppl_values: List[float],
    codebook: Dict[int, List[int]],
) -> List[ColoredToken]:
    """Color tokens by perplexity — blue gradient (low ppl = bright, high = dim)."""
    max_ppl = max(ppl_values)
    out = []
    for tid, ppl in zip(token_ids, ppl_values):
        base_ids = codebook.get(tid, [tid])
        intensity = int(255 * (1 - ppl / max_ppl))
        color = f"\033[38;2;{intensity};{intensity};255m"
        out.append(ColoredToken(token_ids=base_ids, color=color))
    return out


def colorize_random(
    token_ids: List[int],
    codebook: Optional[Dict[int, List[int]]] = None,
) -> List[ColoredToken]:
    """Color each token a random color."""
    if codebook is None:
        codebook = {}
    out = []
    for tid in token_ids:
        base_ids = codebook.get(tid, [tid])
        color = f"\033[38;5;{random.randint(0, 255)}m"
        out.append(ColoredToken(token_ids=base_ids, color=color))
    return out


# ── renderer ─────────────────────────────────────────────────────────────────

def _decode_preserving_leading_space(
    tokenizer: PreTrainedTokenizerBase,
    token_ids: List[int],
) -> str:
    """Decode token IDs, preserving the leading space that SentencePiece drops."""
    raw_tokens = tokenizer.convert_ids_to_tokens(token_ids)
    if any(t is None for t in raw_tokens):
        raise ValueError(f"Non-decodable token IDs: {token_ids}")
    if raw_tokens[0].startswith(chr(9601)):
        return " " + tokenizer.decode(token_ids)
    return tokenizer.decode(token_ids)


def render_colored_tokens(
    colored_tokens: List[ColoredToken],
    tokenizer: PreTrainedTokenizerBase,
) -> str:
    """Render colored tokens to an ANSI-colored string."""
    out = ""
    for ct in colored_tokens:
        text = _decode_preserving_leading_space(tokenizer, ct.token_ids)
        out += f"{ct.color}{text}{RESET}"
    return out
