"""Correctness invariants for resuming the data stream.

Before this was fixed, `--resume_from` restored step/model/optimizer but NOT the
dataloader position: `Zip2ZipDataset` advanced a LOCAL `offset` while
`self._offset` was only ever zeroed, so `state_dict()` always reported offset=0
and a resumed run silently replayed the shards from the beginning. That was
measured in the v0.7 run — post-resume step 6000 fed the batch the original run
had seen at step 500.

Invariants proven:
  D1  state_dict() tracks the LIVE position: the offset advances as samples are
      emitted (this is the assertion the old code failed).
  D2  the resume is exact: save after N samples, restore into a fresh dataset,
      and the following samples equal the uninterrupted run's.
  D3  crossing a shard boundary resets the offset, so a restore taken there does
      not seek into the middle of the next shard.
  D4  load_state_dict REFUSES a state describing a different stream (mode or
      shard count) instead of seeking to a meaningless spot.
  D5  train.py hard-fails a resume whose position cannot be restored, and only
      proceeds with --allow_data_replay.
  D6  the position is withheld (None) when --num_workers > 0, where it lives in
      worker processes and cannot be read back.

Runnable both as a script and under pytest. Uses tiny synthetic shards, no GPU.
"""

import os
import sys
import tempfile
import types

import numpy as np
import torch

from _hyper_common import make_harness, summarize
from zip2zip_core.data import Zip2ZipDataset


VOCAB = 256
PAD = 0
SEQ_LEN = 8          # base_chunk_len = 16
MAX_SUBTOKENS = 3
CHUNK = SEQ_LEN * 2


def write_shards(data_dir, n_shards=2, tokens_per_shard=1600):
    """Small-alphabet random streams.

    A 6-symbol alphabet makes bigrams repeat inside a 16-token window, so LZW
    finds merges and windows survive the zero-hypertoken skip. Deliberately NOT
    periodic: a tiled stream would emit byte-identical windows at different
    offsets, which would let a resume test pass without resuming anything.
    """
    rng = np.random.default_rng(0)
    for i in range(n_shards):
        stream = rng.integers(
            10, 16, size=tokens_per_shard, dtype=np.uint32
        ).astype(np.uint32)
        np.save(os.path.join(data_dir, f"shard_{i:03d}.npy"), stream)
        # _mask_path_for_shard rewrites the "shard_" prefix to "mask_", and the
        # shard scan skips anything starting with "mask_".
        np.save(
            os.path.join(data_dir, f"mask_{i:03d}.npy"),
            np.ones(tokens_per_shard, dtype=bool),
        )


def make_dataset(data_dir, mode="lm"):
    return Zip2ZipDataset(
        data_dir=data_dir,
        seq_len=SEQ_LEN,
        max_subtokens=MAX_SUBTOKENS,
        max_codebook_size=64,
        initial_vocab_size=VOCAB,
        pad_token_id=PAD,
        disabled_ids=[PAD],
        rank=0,
        world_size=1,
        mode=mode,
        remap_codebook=True,
        max_active_codebook_size=64,
    )


def take(dataset, n):
    """First n samples plus the position recorded after each."""
    out = []
    iterator = iter(dataset)
    for _ in range(n):
        sample, labels = next(iterator)
        out.append((sample["input"].clone(), labels.clone(), dict(dataset.state_dict())))
    return out


def main():
    results, check = make_harness()

    with tempfile.TemporaryDirectory() as data_dir:
        write_shards(data_dir)

        # ---- D1: the recorded position actually moves ----
        # Not every chunk yields a sample: a window whose compressed form is too
        # short, or that has no hypertoken at all, is skipped. So the offset
        # advances by a POSITIVE MULTIPLE of one chunk per emitted sample, and it
        # restarts at 0 on each new shard. The old code reported a constant 0.
        reference = take(make_dataset(data_dir), 8)
        positions = [(s["shard_idx"], s["offset"]) for _, _, s in reference]
        within_shard = [
            (a[1], b[1]) for a, b in zip(positions, positions[1:]) if a[0] == b[0]
        ]
        check(
            "D1_state_dict_tracks_the_live_position",
            len(reference) == 8
            and reference[0][2]["offset"] > 0
            and bool(within_shard)
            and all(later > earlier for earlier, later in within_shard),
            f"positions (shard, offset) after each sample: {positions}",
        )
        check(
            "D1_offset_advances_in_whole_chunks",
            all(offset % CHUNK == 0 for _, offset in positions)
            and all(
                (later - earlier) % CHUNK == 0 and later - earlier >= CHUNK
                for earlier, later in within_shard
            ),
            f"in-shard deltas: {[b - a for a, b in within_shard]} (chunk={CHUNK})",
        )

        # ---- D2: restoring reproduces the uninterrupted stream exactly ----
        cut = 3
        resumed = make_dataset(data_dir)
        resumed.load_state_dict(reference[cut - 1][2])
        tail = take(resumed, 4)
        expected = reference[cut : cut + 4]
        check(
            "D2_resume_reproduces_the_uninterrupted_stream",
            len(tail) == len(expected)
            and all(
                torch.equal(got[0], want[0]) and torch.equal(got[1], want[1])
                for got, want in zip(tail, expected)
            ),
            "samples after the restore match the samples the same stream "
            "produced without interruption",
        )
        check(
            "D2_fresh_dataset_would_have_replayed_from_zero",
            torch.equal(take(make_dataset(data_dir), 1)[0][0], reference[0][0])
            and not torch.equal(reference[0][0], expected[0][0]),
            "the bug this pins: an unrestored dataset re-emits sample 0",
        )

        # ---- D3: the offset resets when the stream crosses into a new shard ----
        long_run = take(make_dataset(data_dir), 150)
        crossings = [
            index
            for index, (_, _, state) in enumerate(long_run)
            if state["shard_idx"] == 1
        ]
        check(
            "D3_shard_transition_is_reached",
            bool(crossings),
            f"reached shard_idx=1 after {crossings[0] if crossings else None} samples",
        )
        if crossings:
            first = crossings[0]
            check(
                "D3_offset_restarts_inside_the_new_shard",
                long_run[first - 1][2]["shard_idx"] == 0
                and 0 < long_run[first][2]["offset"] < long_run[first - 1][2]["offset"],
                f"offset dropped from {long_run[first - 1][2]['offset']} (end of "
                f"shard 0) to {long_run[first][2]['offset']} (start of shard 1)",
            )
            boundary = make_dataset(data_dir)
            boundary.load_state_dict(long_run[first][2])
            check(
                "D3_restore_at_the_boundary_is_exact",
                torch.equal(take(boundary, 1)[0][0], long_run[first + 1][0]),
                "a checkpoint taken on the first sample of a new shard resumes "
                "at the second, not deep inside the shard",
            )

        # ---- D4: a state from a different stream is refused ----
        good = dict(reference[0][2])

        def refuses(state, text):
            try:
                make_dataset(data_dir).load_state_dict(state)
            except ValueError as exc:
                return text in str(exc)
            return False

        check(
            "D4_mode_mismatch_refused",
            refuses({**good, "mode": "compress"}, "iterate shards differently"),
            "an lm position cannot be applied to a compress run",
        )
        check(
            "D4_shard_count_mismatch_refused",
            refuses({**good, "n_shards": 7}, "shard_idx would point"),
            "a changed DATA_DIR or world_size is caught",
        )
        check(
            "D4_out_of_range_shard_refused",
            refuses({**good, "shard_idx": 99}, "out of range"),
            "a corrupt index fails loudly",
        )

        # ---- D5/D6: the train.py guards ----
        import zip2zip_core.train as train_mod

        saved_dist = train_mod.dist
        def fake_all_gather_object(out_list, obj):
            out_list[0] = obj

        train_mod.dist = types.SimpleNamespace(
            get_rank=lambda: 0,
            get_world_size=lambda: 1,
            all_gather_object=fake_all_gather_object,
        )
        try:
            loader = types.SimpleNamespace(dataset=make_dataset(data_dir))

            def args_for(**overrides):
                values = dict(num_workers=0, allow_data_replay=False)
                values.update(overrides)
                return types.SimpleNamespace(**values)

            with tempfile.TemporaryDirectory() as ckpt_dir:
                torch.save({"step": 10}, os.path.join(ckpt_dir, "meta.pt"))
                refused = False
                try:
                    train_mod._restore_loader_state(ckpt_dir, loader, args_for())
                except ValueError as exc:
                    refused = "REPLAY the shards" in str(exc)
                check(
                    "D5_legacy_checkpoint_without_position_is_a_hard_error",
                    refused,
                    "a resume that would silently replay data is refused",
                )

                allowed = True
                try:
                    train_mod._restore_loader_state(
                        ckpt_dir, loader, args_for(allow_data_replay=True)
                    )
                except ValueError:
                    allowed = False
                check(
                    "D5_allow_data_replay_is_the_explicit_escape_hatch",
                    allowed,
                    "the operator can still accept the replay deliberately",
                )

                torch.save(
                    {
                        "step": 10,
                        "loader_state": {
                            "world_size": 4,
                            "num_workers": 0,
                            "ranks": [good] * 4,
                        },
                    },
                    os.path.join(ckpt_dir, "meta.pt"),
                )
                world_refused = False
                try:
                    train_mod._restore_loader_state(ckpt_dir, loader, args_for())
                except ValueError as exc:
                    world_refused = "world_size" in str(exc)
                check(
                    "D5_world_size_change_is_a_hard_error",
                    world_refused,
                    "shards are split rank::world_size, so the position does not "
                    "transfer across a different rank count",
                )

            check(
                "D6_position_withheld_when_workers_own_it",
                train_mod._gather_loader_state(loader, args_for(num_workers=2))
                is None,
                "with --num_workers > 0 the main process would save a stale zero",
            )
            check(
                "D6_position_gathered_with_no_workers",
                (
                    train_mod._gather_loader_state(loader, args_for()) or {}
                ).get("world_size") == 1,
                "the default single-process loader is recoverable",
            )
        finally:
            train_mod.dist = saved_dist

    return summarize(results)


def test_resume_dataloader_invariants():
    failed = main()
    assert not failed, f"failed invariants: {failed}"


if __name__ == "__main__":
    sys.exit(1 if main() else 0)
