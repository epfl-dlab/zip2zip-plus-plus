"""Regression tests for compressed/base-view metric stratification."""

import math

import torch

import zip2zip_core.train as train_module
from zip2zip_core.train import (
    finalize_mixed_objective,
    finalize_view_metric_sums,
    new_view_metric_sums,
)


def test_compat_metrics_match_the_historical_per_micro_formula():
    """Top-level keys retain the historical rank-microbatch weighting."""
    bucket = new_view_metric_sums()
    microbatches = (
        # nll, legacy base tokens, valid tokens, correct, hyper correct/count
        (6.0, 3, 2, 1, 0, 0),
        (20.0, 5, 4, 3, 1, 2),
        (42.0, 7, 6, 3, 3, 3),
        (72.0, 9, 8, 6, 1, 4),
    )
    old_loss_terms = []
    old_acc_terms = []
    old_hyper_terms = []
    for (
        nll,
        legacy_n,
        valid_n,
        correct_n,
        hyper_correct_n,
        hyper_n,
    ) in microbatches:
        bucket["nll_sum"] += nll
        bucket["legacy_base_count"] += legacy_n
        bucket["target_base_count"] += legacy_n
        bucket["valid_count"] += valid_n
        bucket["correct"] += correct_n
        bucket["hyper_correct"] += hyper_correct_n
        bucket["hyper_count"] += hyper_n
        bucket["micro_count"] += 1

        loss_term = nll / legacy_n
        acc_term = correct_n / valid_n
        hyper_term = hyper_correct_n / hyper_n if hyper_n else 0.0
        old_loss_terms.append(loss_term)
        old_acc_terms.append(acc_term)
        old_hyper_terms.append(hyper_term)
        bucket["compat_loss_legacy_sum"] += loss_term
        bucket["compat_acc_sum"] += acc_term
        # The old accumulator did not add anything for an absent class, but it
        # still divided by every microbatch at log time.
        if hyper_n:
            bucket["compat_hyper_acc_sum"] += hyper_term

    metrics = finalize_view_metric_sums(bucket)
    n_micro = len(microbatches)
    assert math.isclose(
        metrics["compat_loss_legacy"],
        sum(old_loss_terms) / n_micro,
    )
    assert math.isclose(metrics["compat_acc"], sum(old_acc_terms) / n_micro)
    assert math.isclose(
        metrics["compat_hyper_acc"],
        sum(old_hyper_terms) / n_micro,
    )

    # Canonical slash-prefixed metrics deliberately use raw-count weighting,
    # so they need not equal the historical mean of per-microbatch ratios.
    assert math.isclose(metrics["loss_legacy_base_token"], 140 / 24)
    assert not math.isclose(
        metrics["loss_legacy_base_token"],
        metrics["compat_loss_legacy"],
    )


def test_compressed_metrics_are_not_diluted_by_base_replay():
    compressed = new_view_metric_sums()
    compressed.update(
        {
            "nll_sum": 60.0,
            "valid_count": 30.0,
            "legacy_base_count": 90.0,
            "target_base_count": 45.0,
            "correct": 21.0,
            "base_correct": 13.0,
            "base_count": 20.0,
            "hyper_correct": 8.0,
            "hyper_count": 10.0,
            "relaxed_hyper_correct": 9.0,
            "micro_count": 3.0,
            "compat_hyper_acc_sum": 2.4,
            "compat_loss_legacy_sum": 2.0,
            "compat_loss_target_sum": 4.0,
            "compat_compression_legacy_sum": 9.0,
            "compat_compression_target_sum": 4.5,
        }
    )
    before = finalize_view_metric_sums(compressed)

    # A deliberately terrible base view belongs to a different bucket and
    # therefore cannot mechanically turn 0.8 compressed hyper accuracy into
    # 0.6 when replay is 25%.
    base_view = new_view_metric_sums()
    base_view.update(
        {
            "nll_sum": 999.0,
            "valid_count": 10.0,
            "legacy_base_count": 10.0,
            "target_base_count": 10.0,
            "micro_count": 1.0,
        }
    )
    finalize_view_metric_sums(base_view)
    after = finalize_view_metric_sums(compressed)

    assert after == before
    assert math.isclose(after["hyper_token_acc"], 0.8)
    assert math.isclose(after["compat_hyper_acc"], 0.8)
    assert math.isclose(after["loss_legacy_base_token"], 2 / 3)
    assert math.isclose(after["loss_target_base_token"], 4 / 3)
    assert math.isclose(after["compression_legacy"], 3.0)
    assert math.isclose(after["compression_target"], 1.5)


def test_token_type_raw_counts_and_absent_class_compat_semantics():
    compressed = new_view_metric_sums()
    compressed.update(
        {
            "type_nll_sum": 8.0,
            "type_count": 10.0,
            "type_correct": 7.0,
            "base_type_correct": 3.0,
            "base_type_count": 4.0,
            "hyper_type_correct": 3.0,
            "hyper_type_count": 6.0,
            "micro_count": 2.0,
            "compat_type_loss_sum": 1.6,
            "compat_type_acc_sum": 1.4,
            "compat_base_type_acc_sum": 1.5,
            # Only one of the two rank-microbatches contains the hyper class.
            # The historical accumulator adds 0.5 once then divides by two.
            "compat_hyper_type_acc_sum": 0.5,
            "compat_hyper_ratio_sum": 0.6,
        }
    )
    metrics = finalize_view_metric_sums(compressed)
    assert math.isclose(metrics["type_loss"], 0.8)
    assert math.isclose(metrics["type_acc"], 0.7)
    assert math.isclose(metrics["base_type_acc"], 0.75)
    assert math.isclose(metrics["hyper_type_acc"], 0.5)
    assert math.isclose(metrics["hyper_ratio"], 0.6)
    assert math.isclose(metrics["compat_type_loss"], 0.8)
    assert math.isclose(metrics["compat_type_acc"], 0.7)
    assert math.isclose(metrics["compat_base_type_acc"], 0.75)
    assert math.isclose(metrics["compat_hyper_type_acc"], 0.25)
    assert math.isclose(metrics["compat_hyper_ratio"], 0.3)

    absent = new_view_metric_sums()
    absent["micro_count"] = 1.0
    absent_metrics = finalize_view_metric_sums(absent)
    assert absent_metrics["hyper_token_acc"] == 0.0
    assert absent_metrics["hyper_type_acc"] == 0.0
    assert math.isfinite(absent_metrics["hyper_token_acc"])


def test_mixed_objective_weights_every_rank_microbatch_once():
    compressed = new_view_metric_sums()
    compressed["compat_backward_loss_sum"] = 9.0
    compressed["micro_count"] = 3.0
    base_view = new_view_metric_sums()
    base_view["compat_backward_loss_sum"] = 5.0
    base_view["micro_count"] = 1.0

    assert math.isclose(
        finalize_mixed_objective(compressed, base_view),
        3.5,
    )
    assert finalize_mixed_objective(
        new_view_metric_sums(),
        new_view_metric_sums(),
    ) == 0.0


def test_all_reduce_sums_every_metric_field(monkeypatch):
    local = new_view_metric_sums()
    local.update(
        {
            "nll_sum": 3.0,
            "valid_count": 2.0,
            "micro_count": 1.0,
            "compat_backward_loss_sum": 1.5,
        }
    )
    calls = []

    def fake_all_reduce(values, op):
        calls.append((values.dtype, op))
        values.mul_(4)

    monkeypatch.setattr(train_module.dist, "all_reduce", fake_all_reduce)
    reduced = train_module._all_reduce_view_metric_sums(
        local, torch.device("cpu")
    )

    assert calls == [(torch.float64, train_module.dist.ReduceOp.SUM)]
    assert set(reduced) == set(local)
    assert reduced["nll_sum"] == 12.0
    assert reduced["valid_count"] == 8.0
    assert reduced["micro_count"] == 4.0
    assert reduced["compat_backward_loss_sum"] == 6.0
