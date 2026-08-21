"""Correctness invariants for the v0.6.5 online codebook mask.

v0.6.5 = v0.6.4 + one behavior change: teacher-forced hyper-token logits use
the rows actually installed by the incremental decoder, rather than assuming
row k exists at compressed position k. The low-level flag is default-off.

Runs on CPU, both as a script and under pytest.
"""
import contextlib
import dataclasses
import importlib
import io
import os
import random
import sys
import tempfile
import types

import numpy as np
import torch
from zip2zip_compression import CodebookManager, CompressionConfig, LZWCompressor

from _hyper_common import build_model, make_harness, summarize
from zip2zip_core.configs import zip2zip_llama_configs
from zip2zip_core.data import (
    Zip2ZipDataset,
    online_codebook_counts,
    online_unavailable_targets,
    remap_collate_fn,
)

V = 64
PAD = 0
MAX_CB = 32


def codec_args(max_subtokens=4, disabled_ids=()):
    return {
        "initial_vocab_size": V,
        "max_codebook_size": MAX_CB,
        "max_subtokens": max_subtokens,
        "pad_token_id": PAD,
        "disabled_ids": list(disabled_ids),
    }


def encode(seq, max_subtokens=4, disabled_ids=()):
    args = codec_args(max_subtokens, disabled_ids)
    compressed, _, codebook = LZWCompressor(**args).encode(
        seq, padding="do_not_pad", truncation=False
    )
    return list(compressed), codebook, args


def strip_pad(row):
    return [int(x) for x in row if int(x) != PAD]


def reference_counts(compressed, codebook, args):
    """Independent runtime trace, including row-content and replay checks."""
    cb = codebook.to_dict()
    expanded_manager = CodebookManager(CompressionConfig(**args))
    direct_manager = CodebookManager(CompressionConfig(**args))
    installed = 0
    counts = []

    for tok in compressed:
        expansion = [tok] if tok < V else list(cb[tok])
        flat, indices = expanded_manager.update_codebooks([expansion])
        new = [int(i) for i in indices[0]]
        assert new == list(range(installed, installed + len(new)))
        for row_i, slot in enumerate(new):
            got = strip_pad(
                flat[0][
                    row_i * args["max_subtokens"]:
                    (row_i + 1) * args["max_subtokens"]
                ]
            )
            assert got == list(cb[V + slot])
        installed += len(new)
        counts.append(installed)

        direct_manager.update_codebooks([[tok]])
        assert (
            expanded_manager.get_codebooks()[0].to_dict()
            == direct_manager.get_codebooks()[0].to_dict()
        )

    return counts


def small_model():
    base = zip2zip_llama_configs["Phi3.5-mini"]
    dim = 128
    cfg = dataclasses.replace(
        base,
        dim=dim,
        n_layers=1,
        vocab_size=V,
        pad_token_id=PAD,
        max_codebook_size=6,
        max_subtokens=4,
        tie_hyper_encoder=False,
        encoder_dim=dim,
        encoder_n_layers=1,
        encoder_n_heads=4,
        encoder_intermediate_size=256,
        token_type_loss_weight=0.0,
        tok_embeddings=dataclasses.replace(
            base.tok_embeddings, init_std=dim ** -0.5
        ),
        layer=dataclasses.replace(
            base.layer,
            feed_forward=dataclasses.replace(
                base.layer.feed_forward, hidden_dim=256
            ),
            attention=dataclasses.replace(
                base.layer.attention, n_heads=8, n_kv_heads=8
            ),
        ),
        rope=dataclasses.replace(base.rope, dim=dim // 8, max_seq_len=64),
    )
    return build_model(cfg, seed=65), cfg


def model_fixture(cfg):
    tokens = torch.tensor([[1, 2, V, V + 1]])
    codebook = torch.full(
        (1, cfg.max_codebook_size, cfg.max_subtokens),
        cfg.pad_token_id,
        dtype=torch.long,
    )
    for k in range(4):
        codebook[0, k, :2] = torch.tensor([10 + k, 20 + k])
    return tokens, codebook


def cli_rejects(extra):
    import zip2zip_core.train as train_mod

    saved = sys.argv
    err = io.StringIO()
    try:
        sys.argv = [
            "train.py",
            "--data_dir", "/tmp/data",
            "--output_dir", "/tmp/out",
            *extra,
        ]
        with contextlib.redirect_stderr(err):
            try:
                train_mod.main()
            except SystemExit as exc:
                return exc.code == 2
    finally:
        sys.argv = saved
    return False


def import_eval_adapter():
    """Import the real scorer while stubbing lm-eval if it is not installed."""
    created = []
    try:
        return importlib.import_module("zip2zip_core.lm_eval_adapter"), created
    except ModuleNotFoundError as exc:
        if exc.name != "lm_eval":
            raise

    lm_eval = types.ModuleType("lm_eval")
    lm_eval.__path__ = []
    api = types.ModuleType("lm_eval.api")
    api.__path__ = []
    api_model = types.ModuleType("lm_eval.api.model")
    api_registry = types.ModuleType("lm_eval.api.registry")
    utils = types.ModuleType("lm_eval.utils")
    api_model.LM = object
    api_registry.register_model = lambda _name: lambda cls: cls
    utils.get_rolling_token_windows = lambda *args, **kwargs: ()
    utils.make_disjoint_window = lambda window: window
    stubs = {
        "lm_eval": lm_eval,
        "lm_eval.api": api,
        "lm_eval.api.model": api_model,
        "lm_eval.api.registry": api_registry,
        "lm_eval.utils": utils,
    }
    for name, module in stubs.items():
        sys.modules[name] = module
        created.append(name)
    return importlib.import_module("zip2zip_core.lm_eval_adapter"), created


def main():
    results, check = make_harness()

    # O1: ideal LZW is already one decoder row behind the encoder-time mask.
    comp, cb, args = encode([1, 2, 1, 2])
    counts = online_codebook_counts(comp, cb, args)
    check("O1_simple_compression", comp == [1, 2, V], f"compressed={comp}")
    check("O1_counts_are_after_each_input",
          counts.tolist() == [0, 1, 2],
          f"counts={counts.tolist()}")
    check("O1_training_prefix_counts",
          counts[:-1].tolist() == [0, 1],
          "A predicts B with 0 rows; B predicts AB with 1 row")

    # O2: disabled tokens and max_subtokens force-emits widen the gap.
    comp_d, cb_d, args_d = encode(
        [1, 9, 2, 3, 4, 5, 6, 7], disabled_ids=[9]
    )
    counts_d = online_codebook_counts(comp_d, cb_d, args_d).tolist()
    check("O2_disabled_trace",
          comp_d == [1, 9, 2, 3, 4, 5, 6, 7]
          and counts_d == [0, 0, 0, 1, 2, 3, 4, 5],
          f"compressed={comp_d} counts={counts_d}")

    comp_f, cb_f, args_f = encode(
        [1, 2, 1, 2, 3, 4, 5, 6, 7], max_subtokens=2
    )
    counts_f = online_codebook_counts(comp_f, cb_f, args_f).tolist()
    check("O2_force_emit_trace",
          comp_f == [1, 2, V, 3, 4, 5, 6, 7]
          and counts_f == [0, 1, 2, 2, 3, 4, 5, 6],
          f"compressed={comp_f} counts={counts_f}")

    # O3: random/adversarial traces match the real Rust manager, including
    # direct compressed-token replay and installed row contents.
    rng = random.Random(65)
    random_ok = True
    checked_positions = 0
    for max_subtokens in (2, 3, 4):
        for use_disabled in (False, True):
            disabled = [9] if use_disabled else []
            for _ in range(80):
                seq = [rng.randint(1, 12) for _ in range(rng.randint(4, 80))]
                comp_r, cb_r, args_r = encode(
                    seq, max_subtokens=max_subtokens, disabled_ids=disabled
                )
                got = online_codebook_counts(comp_r, cb_r, args_r).tolist()
                want = reference_counts(comp_r, cb_r, args_r)
                checked_positions += len(comp_r)
                if got != want:
                    random_ok = False
                    break
    check("O3_rust_reference_matches",
          random_ok and checked_positions > 10_000,
          f"{checked_positions} compressed positions")

    # O4: KwKwK is a legal encoded next ID but not in the current generator's
    # output vocabulary. Exclude exactly that target; the following row exists.
    comp_k, cb_k, args_k = encode([1, 2, 1, 2, 1, 2, 1])
    x_counts = online_codebook_counts(comp_k[:-1], cb_k, args_k)
    labels = torch.tensor(comp_k[1:])
    unavailable = online_unavailable_targets(labels, x_counts, V, MAX_CB)
    check("O4_kwkwk_shape",
          comp_k == [1, 2, V, V + 2]
          and x_counts.tolist() == [0, 1, 2],
          f"compressed={comp_k} counts={x_counts.tolist()}")
    check("O4_only_unknown_next_target_skipped",
          unavailable.tolist() == [False, False, True],
          f"unavailable={unavailable.tolist()}")
    sft_labels = torch.tensor([V, -100, 7])
    sft_unavailable = online_unavailable_targets(
        sft_labels, torch.tensor([0, 0, 0]), V, MAX_CB
    )
    check("O4_existing_sft_mask_is_preserved",
          sft_unavailable.tolist() == [True, False, False],
          f"unavailable={sft_unavailable.tolist()}")
    gap_raises = False
    try:
        online_unavailable_targets(
            torch.tensor([V + 3]), torch.tensor([1]), V, MAX_CB
        )
    except RuntimeError:
        gap_raises = True
    check("O4_gap_larger_than_one_fails", gap_raises,
          "unsupported codec drift cannot be silently ignored")
    outside_model = online_unavailable_targets(
        torch.tensor([V + 2]), torch.tensor([3]), V, 2
    )
    check("O4_target_outside_active_codebook_is_unavailable",
          outside_model.tolist() == [True],
          "an installed decoder row can still be outside the model's K logits")

    # O5/O6: model-level contrast. None is exactly the historical path; exact
    # counts change only unavailable hyper logits. Pad rows remain masked.
    model, cfg = small_model()
    tokens, codebook = model_fixture(cfg)
    with torch.no_grad():
        legacy = model(tokens, codebook=codebook, hyper_causal_mask=True)
        explicit_none = model(
            tokens,
            codebook=codebook,
            hyper_causal_mask=True,
            codebook_counts=None,
        )
        exact = model(
            tokens,
            codebook=codebook,
            hyper_causal_mask=True,
            codebook_counts=torch.tensor([[0, 1, 1, 2]]),
        )
    check("O5_flag_off_bit_identical",
          torch.equal(legacy, explicit_none),
          "passing no exact counts reproduces the old forward bit-for-bit")
    legacy_finite = torch.isfinite(legacy[0, :, V:])
    exact_finite = torch.isfinite(exact[0, :, V:])
    expected_legacy = torch.tensor([
        [True, False, False, False, False, False],
        [True, True, False, False, False, False],
        [True, True, True, False, False, False],
        [True, True, True, True, False, False],
    ])
    expected_exact = torch.tensor([
        [False, False, False, False, False, False],
        [True, False, False, False, False, False],
        [True, False, False, False, False, False],
        [True, True, False, False, False, False],
    ])
    check("O6_legacy_mask_unchanged",
          torch.equal(legacy_finite, expected_legacy),
          f"finite={legacy_finite.tolist()}")
    check("O6_exact_mask_effect",
          torch.equal(exact_finite, expected_exact),
          f"finite={exact_finite.tolist()}")
    check("O6_masked_logits_are_negative_infinity",
          torch.isneginf(legacy[0, :, V:][~expected_legacy]).all()
          and torch.isneginf(exact[0, :, V:][~expected_exact]).all(),
          "unavailable and padding rows cannot become NaN")
    common = exact_finite
    exact_hyper = exact[0, :, V:]
    legacy_hyper = legacy[0, :, V:]
    check("O6_allowed_and_base_logits_unchanged",
          torch.equal(exact[0, :, :V], legacy[0, :, :V])
          and torch.equal(exact_hyper[common], legacy_hyper[common]),
          "only newly unavailable hyper logits change")

    shape_raises = False
    try:
        model(
            tokens,
            codebook=codebook,
            codebook_counts=torch.zeros((1, 3), dtype=torch.long),
        )
    except ValueError:
        shape_raises = True
    check("O6_bad_count_shape_fails", shape_raises,
          "misaligned per-position metadata cannot broadcast silently")

    # O7: collation is opt-in, but both reporting denominators are explicit.
    base_sample = {
        "input": torch.tensor([1, 2, 3]),
        "codebook": torch.tensor([[1, 2, PAD, PAD]]),
        "n_base_tokens": 3,
        "target_n_base_tokens": 3,
    }
    old_inputs, _ = remap_collate_fn(
        [(base_sample, torch.tensor([2, 3, 4]))],
        PAD, 4, 4,
    )
    online_sample = {
        **base_sample,
        "codebook_counts": torch.tensor([0, 1, 1]),
        "online_skipped_targets": 1,
    }
    online_inputs, _ = remap_collate_fn(
        [(online_sample, torch.tensor([2, 3, -100]))],
        PAD, 4, 4,
    )
    check("O7_flag_off_collate_carries_explicit_target_denominator",
          set(old_inputs)
          == {"input", "codebook", "n_base_tokens", "target_n_base_tokens"}
          and old_inputs["target_n_base_tokens"].tolist() == [3],
          f"keys={sorted(old_inputs)}")
    missing_target_raises = False
    try:
        missing_target = {
            key: value
            for key, value in base_sample.items()
            if key != "target_n_base_tokens"
        }
        remap_collate_fn(
            [(missing_target, torch.tensor([2, 3, 4]))],
            PAD, 4, 4,
        )
    except ValueError as exc:
        missing_target_raises = "target_n_base_tokens is required" in str(exc)
    check("O7_missing_target_denominator_fails_loudly",
          missing_target_raises,
          "LM collation cannot silently substitute the legacy denominator")
    check("O7_online_metadata_collates",
          online_inputs["codebook_counts"].shape == (1, 3)
          and online_inputs["codebook_counts"].tolist() == [[0, 1, 1]]
          and online_inputs["online_skipped_targets"].tolist() == [1],
          f"keys={sorted(online_inputs)}")

    # O8: exercise the real dataset path, including target masking and metadata.
    with tempfile.TemporaryDirectory() as td:
        np.save(
            os.path.join(td, "shard_000.npy"),
            np.array([1, 2, 1, 2, 1, 2, 1, 3], dtype=np.uint32),
        )
        dataset = Zip2ZipDataset(
            td,
            seq_len=4,
            max_subtokens=4,
            max_codebook_size=MAX_CB,
            initial_vocab_size=V,
            pad_token_id=PAD,
            disabled_ids=[],
            online_codebook_mask=True,
            remap_codebook=False,
        )
        yielded_inputs, yielded_labels = next(iter(dataset))
        check("O8_dataset_exact_trace",
              yielded_inputs["input"].tolist() == [1, 2, V, V + 2]
              and yielded_inputs["codebook_counts"].tolist() == [0, 1, 2, 3]
              and yielded_labels.tolist() == [2, V, -100, 3]
              and yielded_inputs["online_skipped_targets"] == 1,
              f"x={yielded_inputs['input'].tolist()} "
              f"counts={yielded_inputs['codebook_counts'].tolist()} "
              f"y={yielded_labels.tolist()}")

    # O9: unsupported dataset/CLI combinations fail before distributed/GPU work.
    with tempfile.TemporaryDirectory() as td:
        np.save(os.path.join(td, "shard_000.npy"), np.arange(32, dtype=np.uint32))
        remap_raises = mode_raises = active_input_raises = False
        try:
            Zip2ZipDataset(
                td, 4, 2, initial_vocab_size=V,
                online_codebook_mask=True, remap_codebook=True,
            )
        except ValueError:
            remap_raises = True
        try:
            Zip2ZipDataset(
                td, 4, 2, initial_vocab_size=V, mode="compress",
                online_codebook_mask=True, remap_codebook=False,
            )
        except ValueError:
            mode_raises = True
        np.save(
            os.path.join(td, "shard_000.npy"),
            np.array([1, 2, 1, 2, 1, 2, 1, 3], dtype=np.uint32),
        )
        try:
            limited_dataset = Zip2ZipDataset(
                td,
                seq_len=4,
                max_subtokens=4,
                max_codebook_size=MAX_CB,
                initial_vocab_size=V,
                pad_token_id=PAD,
                disabled_ids=[],
                online_codebook_mask=True,
                max_active_codebook_size=2,
                remap_codebook=False,
            )
            next(iter(limited_dataset))
        except RuntimeError:
            active_input_raises = True
    check("O9_dataset_guards",
          remap_raises and mode_raises and active_input_raises,
          "remap, non-LM mode, and input IDs outside active K fail loudly")
    check("O9_cli_guards",
          cli_rejects(["--online_codebook_mask"])
          and cli_rejects([
              "--online_codebook_mask", "--no_remap_codebook", "--mode", "compress"
          ])
          and cli_rejects([
              "--online_codebook_mask", "--no_remap_codebook",
              "--max_codebook_size", "0"
          ])
          and cli_rejects([
              "--online_codebook_mask", "--no_remap_codebook",
              "--max_active_codebook_size", "0"
          ]),
          "all invalid combinations stop before distributed initialization")

    # O10: the behavior flag is resume-hard in both directions.
    import zip2zip_core.train as train_mod
    saved_dist = train_mod.dist
    train_mod.dist = types.SimpleNamespace(get_rank=lambda: 0)
    try:
        with tempfile.TemporaryDirectory() as td:
            common = {
                "disable_digit_ids": True,
                "max_codebook_size": 4096,
                "max_active_codebook_size": 2048,
                "tokenizer": "microsoft/Phi-3.5-mini-instruct",
                "untied_hyper_encoder": True,
                "base_token_positions": True,
                "zero_init_encoder_output": True,
                "no_encoder_residual": False,
                "token_type_loss_weight": 0.05,
                "encoder_dim": 3072,
                "encoder_n_layers": 2,
                "encoder_n_heads": 32,
                "encoder_intermediate_size": 12288,
                "max_subtokens": 4,
                "seq_len": 2048,
                "data_dir": "/data",
                "warmstart_steps": 0,
            }
            args_on = types.SimpleNamespace(
                **common,
                online_codebook_mask=True,
                allow_resume_mismatch=False,
            )

            def rejected(recorded, requested=args_on):
                torch.save({"args": recorded}, os.path.join(td, "meta.pt"))
                try:
                    train_mod.validate_resume_args(td, requested)
                    return False
                except ValueError:
                    return True

            recorded_off = dict(common, online_codebook_mask=False)
            recorded_legacy = dict(common)
            recorded_on = dict(common, online_codebook_mask=True)
            check("O10_resume_on_vs_off_rejected", rejected(recorded_off),
                  "v0.6.4 cannot be resumed and relabeled v0.6.5")
            check("O10_legacy_absence_is_off", rejected(recorded_legacy),
                  "missing old meta key is interpreted as false")
            args_off = types.SimpleNamespace(
                **common,
                online_codebook_mask=False,
                allow_resume_mismatch=False,
            )
            check("O10_resume_off_vs_on_rejected",
                  rejected(recorded_on, args_off),
                  "v0.6.5 cannot be resumed and relabeled v0.6.4")
            recorded_other_k = dict(
                recorded_on, max_active_codebook_size=1024
            )
            check("O10_active_codebook_change_rejected",
                  rejected(recorded_other_k),
                  "changing K would change which targets are trainable")
            torch.save({"args": recorded_on}, os.path.join(td, "meta.pt"))
            matching = True
            try:
                train_mod.validate_resume_args(td, args_on)
            except ValueError:
                matching = False
            check("O10_matching_resume_allowed", matching,
                  "an unchanged v0.6.5 lineage resumes")
    finally:
        train_mod.dist = saved_dist

    # O11: the real compressed lm-eval scorer uses exact counts and excludes
    # the same one unknown-next target. The historical path scores all targets.
    adapter, lm_eval_stubs = import_eval_adapter()
    try:
        scorer_model, scorer_cfg = small_model()
        scorer_model.eval()
        lm = object.__new__(adapter.Zip2ZipLM)
        lm.cfg = scorer_cfg
        lm.model = scorer_model
        lm._device = torch.device("cpu")
        lm._dtype = torch.bfloat16
        lm._compressor_kwargs = {
            **codec_args(),
            "max_codebook_size": scorer_cfg.max_codebook_size,
        }
        lm.hyper_causal_mask = True
        lm.online_codebook_mask = True
        lm.online_codebook_mask_active = True
        lm.multi_view = False
        lm.compression_stats = {
            "in_comp": 0,
            "in_base": 0,
            "gen_comp": 0,
            "gen_base": 0,
            "online_skipped_targets": 0,
            "online_replay_requests": 0,
            "online_replay_tokens": 0,
            "online_replay_seconds": 0.0,
        }
        exact_score = lm._score_compressed(
            [1, 2, 1, 2, 1, 2, 1], cont_start_base=0
        )
        exact_skipped = lm.compression_stats["online_skipped_targets"]
        exact_timing = lm.compression_summary()
        check("O11_eval_replay_cost_is_measured",
              exact_timing["online_replay_requests"] == 1
              and exact_timing["online_replay_tokens"] == 3
              and exact_timing["online_replay_seconds"] >= 0
              and exact_timing["online_replay_us_per_token"] >= 0,
              f"timing={exact_timing}")

        lm.eval_mode = "compressed"
        lm._max_length = 64
        lm.tok_encode = lambda text: (
            [1, 2] if text == "ctx" else [1, 2, 1, 2, 1, 2, 1]
        )
        mc_raises = False
        try:
            lm.loglikelihood([types.SimpleNamespace(args=("ctx", "cont"))])
        except RuntimeError as exc:
            mc_raises = (
                "bias the request score and is_greedy result" in str(exc)
            )
        check("O11_mc_unavailable_target_fails_loudly",
              mc_raises,
              "MC scoring cannot silently omit a negative logprob term")

        lm.online_codebook_mask = False
        lm.online_codebook_mask_active = False
        lm.compression_stats = {
            "in_comp": 0,
            "in_base": 0,
            "gen_comp": 0,
            "gen_base": 0,
        }
        legacy_score = lm._score_compressed(
            [1, 2, 1, 2, 1, 2, 1], cont_start_base=0
        )
        check("O11_eval_scorer_exact_mask_and_skip",
              exact_score[2:4] == (2, 3)
              and exact_skipped == 1
              and legacy_score[2:4] == (3, 6),
              f"exact counts={exact_score[2:4]} skipped={exact_skipped} "
              f"legacy counts={legacy_score[2:4]}")
    finally:
        if lm_eval_stubs:
            sys.modules.pop("zip2zip_core.lm_eval_adapter", None)
            for name in reversed(lm_eval_stubs):
                sys.modules.pop(name, None)

    return summarize(results)


def test_online_codebook_mask_invariants():
    failed = main()
    assert not failed, f"failed invariants: {failed}"


def test_exact_multi_view_scorer_brackets_strict_and_first_token():
    adapter, lm_eval_stubs = import_eval_adapter()
    try:
        scorer_model, scorer_cfg = small_model()
        scorer_model.eval()
        lm = object.__new__(adapter.Zip2ZipLM)
        lm.cfg = scorer_cfg
        lm.model = scorer_model
        lm._device = torch.device("cpu")
        lm._dtype = torch.bfloat16
        lm._max_length = 64
        lm._compressor_kwargs = {
            **codec_args(),
            "max_codebook_size": scorer_cfg.max_codebook_size,
        }
        lm.hyper_causal_mask = True
        lm.online_codebook_mask = True
        lm.online_codebook_mask_active = True
        lm.multi_view = True
        lm.compression_stats = {
            "in_comp": 0,
            "in_base": 0,
            "gen_comp": 0,
            "gen_base": 0,
            "online_skipped_targets": 0,
            "online_replay_requests": 0,
            "online_replay_tokens": 0,
            "online_replay_seconds": 0.0,
        }

        strict, _, _, _, first_token, n_hyper, exact = lm._score_compressed(
            [1, 2, 1, 2],
            cont_start_base=0,
            compute_multi_view=True,
            compute_exact_multi_view=True,
        )
        assert n_hyper == 1
        assert strict <= exact <= first_token
        assert strict < exact
    finally:
        if lm_eval_stubs:
            sys.modules.pop("zip2zip_core.lm_eval_adapter", None)
            for name in reversed(lm_eval_stubs):
                sys.modules.pop(name, None)


if __name__ == "__main__":
    sys.exit(1 if main() else 0)
