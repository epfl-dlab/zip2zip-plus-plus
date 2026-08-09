"""Multi-view (segmentation-marginalized) scoring for evaluation.

An LZW-compressed target admits many token sequences that decode to the same
base-token string: the hypertoken for (a, b, c) can equally be produced as
a·b·c, ab·c, or abc. Strict scoring credits only the canonical compressed id,
so probability mass the model puts on the other, equally correct, views is
counted as error. The multi-view probability sums that mass back.

The exact sum over all segmentations of an n-token expansion has O(2^(n-1))
terms; this module implements the tractable first-token upper bound from the
zip2zip++ paper (eq. 6-7): only the *first* token of each valid segmentation
is scored, and the distinct first tokens of the segmentations of (y1..yn) are
exactly the tokens whose own expansion is a non-empty prefix of (y1..yn):

    p*(x | ctx) = p(y1 | ctx) + sum over L in 2..n of p(h_L | ctx)

where h_L is the codebook entry expanding to (y1..yL), when it exists. Since
the canonical target is in the set, p* >= p_strict always, so the multi-view
perplexity is a lower (more favorable) bound and strict/multi-view bracket
the model's true quality. For a base-token target the set degenerates to the
singleton {x} and multi-view equals strict.

Availability note: LZW dictionaries are prefix-closed — an entry W·z is only
ever inserted when W is already an entry — so every prefix entry of a target
was created strictly before the target itself. Whenever the canonical target
is a legal emission, so is every candidate; no decode-time replay is needed.

Aggregation is denominator-free by design: lm-eval reports every rolling
perplexity metric as a monotone function of sum(loglik)/D for some per-task
denominator D (bytes, words — whose counting basis is task-specific, e.g.
wikitext counts the RAW page, not the detokenized request string). Instead of
re-deriving D and risking a basis mismatch, `derive_multi_view_metrics`
rescales the reported strict metric with the ratio of the two loglikelihood
sums, so D cancels exactly.

Deliberately torch-free and lm-eval-free so the unit tests exercise the exact
production logic anywhere. Training-side "relaxed" metrics in train.py are a
separate implementation and are intentionally left untouched.
"""

from __future__ import annotations

import math
from typing import Dict, List, Sequence, Tuple


def build_expansion_index(
    cb_dict: Dict[int, Sequence[int]],
) -> Dict[Tuple[int, ...], List[int]]:
    """Map expansion tuple -> hyper token ids with exactly that expansion.

    `cb_dict` is `codebook.to_dict()`: absolute hyper id (>= vocab_size) ->
    base-id sequence. A well-formed LZW codebook has unique expansions; the
    list form is defensive so a duplicated entry still contributes each id
    once rather than shadowing the other.
    """
    index: Dict[Tuple[int, ...], List[int]] = {}
    for hid, exp in cb_dict.items():
        index.setdefault(tuple(exp), []).append(hid)
    return index


def multi_view_candidates(
    target_id: int,
    cb_dict: Dict[int, Sequence[int]],
    expansion_index: Dict[Tuple[int, ...], List[int]],
    vocab_size: int,
) -> List[int]:
    """Distinct first tokens of all valid segmentations of the target.

    Base-token target: the singleton [target_id]. Hypertoken target with
    expansion (y1..yn): y1 as a base token, plus every codebook entry whose
    expansion is (y1..yL) for L in 2..n — which includes the target itself.
    """
    if target_id < vocab_size:
        return [target_id]
    exp = tuple(cb_dict[target_id])
    if not exp:
        # Real LZW entries are never empty (length >= 2); degrade to strict
        # scoring rather than crash on a malformed codebook.
        return [target_id]
    cands: List[int] = [exp[0]]
    for length in range(2, len(exp) + 1):
        cands.extend(expansion_index.get(exp[:length], ()))
    if target_id not in cands:
        # Unreachable when expansion_index was built from the same cb_dict;
        # kept so a stale index can never drop the canonical target and
        # silently break the p* >= p_strict invariant.
        cands.append(target_id)
    seen = set()
    out: List[int] = []
    for c in cands:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


class MultiViewAccumulator:
    """Per-task strict and multi-view loglikelihood sums over rolling docs.

    Deliberately stores no byte/word counts: the strict sums are accumulated
    from the very values handed back to lm-eval, and the multi-view metrics
    are later derived from the harness-reported strict metrics via
    `derive_multi_view_metrics`, where the denominator cancels.
    """

    def __init__(self) -> None:
        self._tasks: Dict[str, Dict[str, float]] = {}

    def add_document(
        self,
        task: str | None,
        strict_logprob: float,
        multi_view_logprob: float,
        n_hyper_targets: int,
        n_targets: int,
    ) -> None:
        s = self._tasks.setdefault(
            task or "unknown",
            dict(
                strict_loglik_sum=0.0,
                multi_view_loglik_sum=0.0,
                docs=0,
                targets_scored=0,
                hyper_targets_scored=0,
            ),
        )
        s["strict_loglik_sum"] += strict_logprob
        s["multi_view_loglik_sum"] += multi_view_logprob
        s["docs"] += 1
        s["targets_scored"] += n_targets
        s["hyper_targets_scored"] += n_hyper_targets

    def summary(self) -> Dict[str, Dict[str, float]]:
        return {task: dict(s) for task, s in self._tasks.items()}


def derive_multi_view_metrics(
    sums: Dict[str, float], reported: Dict[str, float]
) -> Dict[str, float]:
    """Derive multi-view metrics from one task's sums and its reported strict
    metrics ({'word_perplexity': ..., 'byte_perplexity': ..., 'bits_per_byte':
    ...}, whichever the task declares).

    lm-eval reports ppl = exp(-S/D) and bpb = -S/(D*ln2) with S the strict
    loglikelihood sum and D the task's denominator. Substituting the
    multi-view sum M rescales the exponent by r = M/S with D cancelled:

        ppl_mv = ppl_strict ** r        bpb_mv = bpb_strict * r

    Both sums are accumulated from the same requests lm-eval aggregates, so
    the identity is exact up to float summation order. r is in [0, 1] because
    the candidate set contains the canonical target (M >= S, both <= 0), so
    every multi-view metric is at most its strict counterpart.

    `implied_byte_denominator` back-solves D from the reported byte
    perplexity; it must land on (near-)integer byte counts, which callers use
    as the alignment self-check between our sums and lm-eval's aggregation.

    Empty when nothing was scored (S == 0) — degenerate, not an error.
    """
    strict_sum = sums["strict_loglik_sum"]
    mv_sum = sums["multi_view_loglik_sum"]
    out: Dict[str, float] = {}
    if strict_sum == 0.0:
        return out
    ratio = mv_sum / strict_sum
    for name in ("word_perplexity", "byte_perplexity"):
        value = reported.get(name)
        if isinstance(value, (int, float)) and value > 0:
            out[f"multi_view_{name}"] = value ** ratio
    bpb = reported.get("bits_per_byte")
    if isinstance(bpb, (int, float)):
        out["multi_view_bits_per_byte"] = bpb * ratio
        # strict_bpb - multi_view_bpb >= 0: the share of the strict loss that
        # is pure segmentation ambiguity, not model error.
        out["segmentation_gap_bits_per_byte"] = bpb * (1.0 - ratio)
    byte_ppl = reported.get("byte_perplexity")
    if isinstance(byte_ppl, (int, float)) and byte_ppl > 0 and byte_ppl != 1.0:
        out["implied_byte_denominator"] = -strict_sum / math.log(byte_ppl)
    return out
