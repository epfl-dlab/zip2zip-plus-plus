"""Unit tests for the multi-view (segmentation-marginalized) eval metrics.

Covers the pure logic in zip2zip_core.multi_view — candidate-set construction,
the accumulator, and the denominator-free metric derivation — plus, when the
rust compressor is installed, the LZW prefix-closure invariant the candidate
availability argument relies on.
"""

import math

import pytest

from zip2zip_core.multi_view import (
    MultiViewAccumulator,
    build_expansion_index,
    derive_multi_view_metrics,
    multi_view_candidates,
)

V = 100  # toy base vocab size; hyper ids start here


def _index(cb):
    return build_expansion_index(cb)


# ─────────────────────────── candidate sets ────────────────────────────────


def test_base_target_is_singleton():
    cb = {V + 0: [1, 2]}
    assert multi_view_candidates(7, cb, _index(cb), V) == [7]


def test_paper_example_target_abc():
    """Target expanding to (a,b,c) -> p(a) + p(ab) + p(abc): candidates are the
    first base token, the (a,b) entry, and the target itself."""
    a, b, c = 1, 2, 3
    H_ab, H_abc = V + 0, V + 1
    cb = {H_ab: [a, b], H_abc: [a, b, c]}
    cands = multi_view_candidates(H_abc, cb, _index(cb), V)
    assert cands == [a, H_ab, H_abc]


def test_missing_intermediate_prefix_is_skipped():
    cb = {V + 1: [1, 2, 3]}  # no (1,2) entry
    cands = multi_view_candidates(V + 1, cb, _index(cb), V)
    assert cands == [1, V + 1]


def test_two_token_target():
    cb = {V + 0: [4, 5]}
    assert multi_view_candidates(V + 0, cb, _index(cb), V) == [4, V + 0]


def test_duplicate_expansion_counts_each_id_once():
    """A (defensively handled) duplicated expansion contributes both ids, each
    exactly once."""
    cb = {V + 0: [1, 2], V + 1: [1, 2], V + 2: [1, 2, 3]}
    cands = multi_view_candidates(V + 2, cb, _index(cb), V)
    assert cands == [1, V + 0, V + 1, V + 2]
    assert len(cands) == len(set(cands))


def test_target_survives_a_stale_index():
    """The canonical target must be in the set even if the index is stale,
    or p* >= p_strict silently breaks."""
    cb = {V + 1: [1, 2, 3]}
    cands = multi_view_candidates(V + 1, cb, {}, V)
    assert cands == [1, V + 1]


def test_multi_view_bound_is_at_least_strict():
    """log(sum(exp)) over a candidate set containing the target can never be
    below the target's own logprob."""
    lp = {1: -2.3, V + 0: -1.7, V + 1: -4.1}
    cb = {V + 0: [1, 2], V + 1: [1, 2, 3]}
    cands = multi_view_candidates(V + 1, cb, _index(cb), V)
    mv = math.log(sum(math.exp(lp[c]) for c in cands))
    assert mv >= lp[V + 1]


# ─────────────────────────── accumulator ───────────────────────────────────


def test_accumulator_sums_documents_per_task():
    acc = MultiViewAccumulator()
    acc.add_document("pile", -2.0, -1.5, 1, 2)
    acc.add_document("pile", -2.0, -1.5, 0, 2)
    acc.add_document("mc4", -1.0, -1.0, 0, 1)  # separate task
    s = acc.summary()
    assert s["pile"] == dict(
        strict_loglik_sum=-4.0,
        multi_view_loglik_sum=-3.0,
        docs=2,
        targets_scored=4,
        hyper_targets_scored=1,
    )
    assert s["mc4"]["docs"] == 1
    assert s["mc4"]["strict_loglik_sum"] == -1.0


def test_accumulator_counts_unscored_documents():
    acc = MultiViewAccumulator()
    acc.add_document("wikitext", -3.0, -3.0, 0, 1)
    acc.add_document("wikitext", 0.0, 0.0, 0, 0)  # doc too short to score
    s = acc.summary()["wikitext"]
    assert s["docs"] == 2
    assert s["strict_loglik_sum"] == -3.0


def test_accumulator_none_task_buckets_as_unknown():
    acc = MultiViewAccumulator()
    acc.add_document(None, -1.0, -1.0, 0, 1)
    assert "unknown" in acc.summary()


# ─────────────────────── denominator-free derivation ───────────────────────


def test_derivation_matches_hand_computed_metrics():
    """S=-4, M=-3 with byte denominator 5 and word denominator 2: the derived
    values must equal what direct aggregation with those denominators gives."""
    sums = dict(strict_loglik_sum=-4.0, multi_view_loglik_sum=-3.0)
    ln2 = math.log(2)
    reported = {
        "byte_perplexity": math.exp(4.0 / 5),
        "word_perplexity": math.exp(4.0 / 2),
        "bits_per_byte": 4.0 / (5 * ln2),
    }
    out = derive_multi_view_metrics(sums, reported)
    assert abs(out["multi_view_byte_perplexity"] - math.exp(3.0 / 5)) < 1e-12
    assert abs(out["multi_view_word_perplexity"] - math.exp(3.0 / 2)) < 1e-12
    assert abs(out["multi_view_bits_per_byte"] - 3.0 / (5 * ln2)) < 1e-12
    # gap = strict_bpb - multi_view_bpb, non-negative
    assert abs(out["segmentation_gap_bits_per_byte"] - 1.0 / (5 * ln2)) < 1e-12
    assert out["segmentation_gap_bits_per_byte"] >= 0
    # back-solved denominator lands exactly on the byte count
    assert abs(out["implied_byte_denominator"] - 5.0) < 1e-9


def test_derivation_never_beats_strict():
    sums = dict(strict_loglik_sum=-10.0, multi_view_loglik_sum=-7.5)
    reported = {"byte_perplexity": 3.2, "bits_per_byte": 1.7}
    out = derive_multi_view_metrics(sums, reported)
    assert out["multi_view_byte_perplexity"] <= reported["byte_perplexity"]
    assert out["multi_view_bits_per_byte"] <= reported["bits_per_byte"]


def test_derivation_equal_sums_reproduce_strict():
    """No hypertokens scored (base mode, or zero merges): multi-view must
    reproduce the strict metrics exactly."""
    sums = dict(strict_loglik_sum=-6.0, multi_view_loglik_sum=-6.0)
    reported = {"byte_perplexity": 2.5, "bits_per_byte": 1.3}
    out = derive_multi_view_metrics(sums, reported)
    assert out["multi_view_byte_perplexity"] == pytest.approx(2.5, abs=1e-12)
    assert out["multi_view_bits_per_byte"] == pytest.approx(1.3, abs=1e-12)
    assert out["segmentation_gap_bits_per_byte"] == pytest.approx(0.0, abs=1e-12)


def test_derivation_degenerate_and_partial_inputs():
    # nothing scored -> nothing derived
    assert derive_multi_view_metrics(
        dict(strict_loglik_sum=0.0, multi_view_loglik_sum=0.0), {"byte_perplexity": 2.0}
    ) == {}
    # only bits_per_byte reported -> only bpb-family derived
    out = derive_multi_view_metrics(
        dict(strict_loglik_sum=-4.0, multi_view_loglik_sum=-3.0),
        {"bits_per_byte": 1.0},
    )
    assert set(out) == {"multi_view_bits_per_byte", "segmentation_gap_bits_per_byte"}
    # byte_perplexity == 1.0 must not divide by log(1) == 0
    out = derive_multi_view_metrics(
        dict(strict_loglik_sum=-4.0, multi_view_loglik_sum=-3.0),
        {"byte_perplexity": 1.0},
    )
    assert "implied_byte_denominator" not in out


# ───────────────────────── LZW prefix closure ───────────────────────────────


def test_lzw_codebooks_are_prefix_closed():
    """The candidate availability argument: every proper prefix (length >= 2)
    of a codebook entry is itself an entry, so prefix candidates of any legal
    target were created before the target. Exercised on the real compressor
    with the same kwargs shape the adapter uses."""
    zc = pytest.importorskip("zip2zip_compression")
    compressor = zc.LZWCompressor(
        initial_vocab_size=V,
        max_codebook_size=64,
        max_subtokens=4,
        pad_token_id=0,
        disabled_ids=[],
    )
    ids = [1, 2, 3, 4] * 32 + [5, 6] * 16 + [1, 2, 3, 4] * 8
    compressed, _, codebook = compressor.encode(
        ids, padding="do_not_pad", truncation=False
    )
    cb = codebook.to_dict()
    assert any(t >= V for t in compressed), "test input produced no hypertokens"
    index = build_expansion_index(cb)
    for hid, exp in cb.items():
        exp = tuple(exp)
        assert len(exp) >= 2
        for length in range(2, len(exp)):
            assert exp[:length] in index, (
                f"entry {hid} expansion {exp} lacks prefix entry {exp[:length]}"
            )
    # And the candidate set of every emitted hypertoken is fully resolvable.
    for tok in compressed:
        if tok >= V:
            cands = multi_view_candidates(tok, cb, index, V)
            assert cands[0] == tuple(cb[tok])[0]
            assert tok in cands
