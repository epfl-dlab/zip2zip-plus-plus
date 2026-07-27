"""Focused invariants for opt-in uncompressed base-view replay.

Runnable directly as well as under pytest:

    python tests/test_base_view_replay.py

The compressor is replaced with a deterministic in-memory fixture so this suite
tests data plumbing and loss-mask accounting rather than the Rust codec.
"""

import os
import inspect
import sys
import tempfile

import numpy as np
import torch

import zip2zip_core.data as data_mod
from zip2zip_core.data import (
    Zip2ZipDataset,
    build_dataloader,
    remap_collate_fn,
)
from zip2zip_core.train import base_view_replay_microsteps


SEQ_LEN = 4
VOCAB_SIZE = 100
PAD_ID = 0
RAW_CHUNK = np.arange(10, 18, dtype=np.uint32)
# Targets for a base window starting at zero are raw indices 1..4:
# valid, valid, masked, valid.
LOSS_MASK = np.array(
    [False, True, True, False, True, True, False, True], dtype=np.bool_
)


class _FakeCodebook:
    def to_dict(self):
        # The compressed stream below partitions RAW_CHUNK into spans 1,2,1,3,1.
        return {
            VOCAB_SIZE: [11, 12],
            VOCAB_SIZE + 1: [14, 15, 16],
        }


class _FakeCompressor:
    def __init__(self, **_kwargs):
        pass

    def encode(self, chunk, **_kwargs):
        assert list(chunk[: len(RAW_CHUNK)]) == RAW_CHUNK.tolist()
        compressed = [10, VOCAB_SIZE, 13, VOCAB_SIZE + 1, 17]
        return compressed, None, _FakeCodebook()


def _write_fixture(root: str, *, with_mask: bool = True) -> None:
    np.save(os.path.join(root, "shard_000.npy"), RAW_CHUNK)
    if with_mask:
        np.save(os.path.join(root, "mask_000.npy"), LOSS_MASK)


def _dataset(root: str, *, include_base_view: bool) -> Zip2ZipDataset:
    return Zip2ZipDataset(
        data_dir=root,
        seq_len=SEQ_LEN,
        max_subtokens=3,
        max_codebook_size=2,
        max_active_codebook_size=2,
        initial_vocab_size=VOCAB_SIZE,
        pad_token_id=PAD_ID,
        disabled_ids=[],
        remap_codebook=False,
        include_base_view=include_base_view,
    )


def main():
    results = {}

    def check(name, condition, detail=""):
        results[name] = bool(condition)
        print(f"[{'PASS' if condition else 'FAIL'}] {name} {detail}")

    original_compressor = data_mod.LZWCompressor
    data_mod.LZWCompressor = _FakeCompressor
    try:
        with tempfile.TemporaryDirectory() as root:
            _write_fixture(root)

            legacy_sample, legacy_labels = next(iter(_dataset(
                root, include_base_view=False
            )))
            replay_sample, replay_labels = next(iter(_dataset(
                root, include_base_view=True
            )))

            # BV1: opt-in only. Apart from the intentional n_base_tokens reporting
            # fix, disabling replay preserves the historical sample contract.
            check(
                "BV1_legacy_keys_unchanged",
                set(legacy_sample) == {"input", "codebook", "n_base_tokens"},
                f"keys={sorted(legacy_sample)}",
            )
            check(
                "BV1_replay_does_not_change_compressed_view",
                torch.equal(legacy_sample["input"], replay_sample["input"])
                and torch.equal(legacy_sample["codebook"], replay_sample["codebook"])
                and torch.equal(legacy_labels, replay_labels),
                f"labels={replay_labels.tolist()}",
            )

            # Compressed targets are [span2, masked span1, masked span3, span1].
            # Only 2+1 base tokens enter the loss; the former implementation
            # incorrectly counted all 8 input+target base tokens.
            check(
                "BV2_compressed_n_base_counts_only_unmasked_targets",
                replay_labels.tolist() == [VOCAB_SIZE, -100, -100, 17]
                and replay_sample["n_base_tokens"] == 3,
                f"labels={replay_labels.tolist()} "
                f"n_base={replay_sample['n_base_tokens']}",
            )

            # The deterministic earliest best window is raw[0:5]. Its raw target
            # mask is [T,T,F,T], so the base replay view must preserve -100.
            check(
                "BV3_base_view_same_chunk_and_mask_preserved",
                replay_sample["base_input"].tolist() == [10, 11, 12, 13]
                and replay_sample["base_labels"].tolist() == [11, 12, -100, 14]
                and replay_sample["base_n_base_tokens"] == 3,
                f"x={replay_sample['base_input'].tolist()} "
                f"y={replay_sample['base_labels'].tolist()} "
                f"n={replay_sample['base_n_base_tokens']}",
            )

            inputs, labels = remap_collate_fn(
                [(replay_sample, replay_labels), (replay_sample, replay_labels)],
                pad_token_id=PAD_ID,
                max_subtokens=3,
                max_active_codebook_size=2,
            )
            check(
                "BV4_collate_carries_optional_base_fields",
                inputs["base_input"].shape == (2, SEQ_LEN)
                and inputs["base_labels"].shape == (2, SEQ_LEN)
                and inputs["base_n_base_tokens"].tolist() == [3, 3]
                and torch.equal(labels[0], replay_labels),
                f"keys={sorted(inputs)}",
            )

            loader = build_dataloader(
                data_dir=root,
                seq_len=SEQ_LEN,
                max_subtokens=3,
                max_codebook_size=2,
                max_active_codebook_size=2,
                local_batch_size=1,
                initial_vocab_size=VOCAB_SIZE,
                pad_token_id=PAD_ID,
                disabled_ids=[],
                remap_codebook=False,
                include_base_view=True,
            )
            check(
                "BV5_build_dataloader_plumbs_opt_in",
                loader.dataset.include_base_view is True,
                f"include_base_view={loader.dataset.include_base_view}",
            )

            rejected = False
            try:
                Zip2ZipDataset(
                    data_dir=root,
                    seq_len=SEQ_LEN,
                    max_subtokens=3,
                    max_codebook_size=2,
                    initial_vocab_size=VOCAB_SIZE,
                    pad_token_id=PAD_ID,
                    disabled_ids=[],
                    mode="compress",
                    include_base_view=True,
                )
            except ValueError as exc:
                rejected = "only in mode='lm'" in str(exc)
            check(
                "BV6_replay_rejects_non_lm_mode",
                rejected,
                "compress/decompress mode has different sequence semantics",
            )

        with tempfile.TemporaryDirectory() as root:
            _write_fixture(root, with_mask=False)
            sample, labels = next(iter(_dataset(root, include_base_view=True)))
            check(
                "BV7_unmasked_counts_exclude_first_input_span",
                sample["n_base_tokens"] == 7
                and sample["base_n_base_tokens"] == SEQ_LEN
                and bool((labels != -100).all())
                and bool((sample["base_labels"] != -100).all()),
                f"compressed_n={sample['n_base_tokens']} "
                f"base_n={sample['base_n_base_tokens']}",
            )

        schedules = [
            base_view_replay_microsteps(step, 4, 0.25)
            for step in range(1, 9)
        ]
        check(
            "BV8_exact_quarter_rank_and_resume_stateless",
            schedules == [frozenset({0})] * 8
            and base_view_replay_microsteps(137, 4, 0.25)
            == frozenset({0}),
            "GA=4 yields one whole base microbatch at every optimizer step",
        )
        check(
            "BV8_final_microstep_always_compressed",
            all(3 not in selected for selected in schedules),
            "the FSDP sync microstep keeps the compressed-only graph active",
        )
        rejected_schedule = False
        try:
            base_view_replay_microsteps(1, 4, 0.80)
        except ValueError as exc:
            rejected_schedule = "final microstep" in str(exc)
        check(
            "BV8_rejects_schedule_without_compressed_sync_step",
            rejected_schedule,
            "unsafe replay probabilities fail before distributed initialization",
        )
        dataset_params = list(inspect.signature(Zip2ZipDataset).parameters)
        loader_params = list(inspect.signature(build_dataloader).parameters)
        check(
            "BV9_new_option_preserves_debug_samples_positional_slot",
            dataset_params.index("debug_samples")
            < dataset_params.index("include_base_view")
            and loader_params.index("debug_samples")
            < loader_params.index("include_base_view"),
            "include_base_view was appended after the existing public arguments",
        )
    finally:
        data_mod.LZWCompressor = original_compressor

    failed = [name for name, passed in results.items() if not passed]
    print(f"\n{len(results) - len(failed)}/{len(results)} passed")
    return failed


def test_base_view_replay_invariants():
    failed = main()
    assert not failed, f"failed invariants: {failed}"


if __name__ == "__main__":
    sys.exit(1 if main() else 0)
