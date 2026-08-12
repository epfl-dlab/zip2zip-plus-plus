"""Correctness invariants for the LOSSLESS_WINDOWS data-stream mode.

Historical (default) behavior: every lm-mode window reads a fixed
``2 * seq_len`` chunk, advances by that amount unconditionally, truncates the
compressed window to ``seq_len + 1`` tokens (silently dropping the tail's
text), and drops any window that compresses below ``seq_len + 1`` tokens
(ratio >= 2.0 — exactly the most compressible text). ``lossless_windows=True``
makes windows tile the stream instead: the offset advances by the base-token
span of each emitted window, and under-filled windows extend their raw input
(bounded by ``LOSSLESS_MAX_WINDOW_CHUNKS``) instead of being dropped.

Invariants proven:
  L1  the flag defaults to OFF and the OFF stream is unchanged: same samples,
      same state offsets as an implicit-default dataset.
  L2  flag ON: windows tile the stream — every window starts exactly where
      the previous one's span ended, or after it by at least one full window
      span (a skipped window: zero-hypertoken / all-masked / unfillable).
      Overlaps and sub-window gaps are impossible.
  L3  a >= 2x-compressible region is dropped by the legacy stream but trained
      on by the lossless stream, via input extension.
  L4  a window that cannot fill before the shard end is skipped WHOLE and the
      stream wraps; the shard replays identically. (Mid-shard windows always
      fill within the cap: LZW warmup bounds the first seq_len+1 emissions to
      at most seq_len*(seq_len+3)/2 + ... < 3 chunks, so the unfillable branch
      is reachable only against the shard end or after an encode failure.)
  L5  the resume round-trip is exact under the flag: save after N samples,
      restore into a fresh dataset, and the following samples match.
  L6  no emitted window is ever fully masked, under either setting.

Runnable both as a script and under pytest. Tiny synthetic shards, no GPU.
"""

import os
import tempfile

import numpy as np
import torch

from zip2zip_core.data import Zip2ZipDataset

VOCAB = 256
PAD = 0
SEQ_LEN = 8          # base_chunk_len = 16
CHUNK = SEQ_LEN * 2


def _write_shard(data_dir, stream, mask=None, idx=0):
    stream = np.asarray(stream, dtype=np.uint32)
    np.save(os.path.join(data_dir, f"shard_{idx:03d}.npy"), stream)
    if mask is None:
        mask = np.ones(len(stream), dtype=bool)
    np.save(os.path.join(data_dir, f"mask_{idx:03d}.npy"), np.asarray(mask, dtype=bool))


def _mixed_stream(n=1600, seed=0):
    """Small-alphabet random stream: compressible enough to form hypertokens,
    not compressible enough (ratio < 2) to trigger the short-window path."""
    rng = np.random.default_rng(seed)
    return rng.integers(10, 16, size=n, dtype=np.uint32)


def _make_dataset(data_dir, max_subtokens=3, **kwargs):
    return Zip2ZipDataset(
        data_dir=data_dir,
        seq_len=SEQ_LEN,
        max_subtokens=max_subtokens,
        max_codebook_size=64,
        initial_vocab_size=VOCAB,
        pad_token_id=PAD,
        disabled_ids=[PAD],
        rank=0,
        world_size=1,
        mode="lm",
        remap_codebook=True,
        **kwargs,
    )


def _take(dataset, n):
    out = []
    it = iter(dataset)
    for _ in range(n):
        sample, y = next(it)
        out.append((sample, y, dict(dataset.state_dict())))
    return out


def test_L1_flag_off_stream_is_unchanged():
    with tempfile.TemporaryDirectory() as d:
        _write_shard(d, _mixed_stream())
        implicit = _take(_make_dataset(d), 12)
        explicit = _take(_make_dataset(d, lossless_windows=False), 12)
        assert _make_dataset(d).lossless_windows is False
        for (s_a, y_a, st_a), (s_b, y_b, st_b) in zip(implicit, explicit):
            assert torch.equal(s_a["input"], s_b["input"])
            assert torch.equal(y_a, y_b)
            assert st_a == st_b
        # Historical semantics: the offset advances by exactly one fixed
        # chunk per consumed window, so every position is a CHUNK multiple.
        for _, _, st in implicit:
            assert st["offset"] % CHUNK == 0


def test_L2_lossless_windows_tile_the_stream():
    with tempfile.TemporaryDirectory() as d:
        _write_shard(d, _mixed_stream())
        ds = _make_dataset(d, lossless_windows=True)
        prev_end, prev_shard = 0, 0
        adjacent = 0
        for sample, _, st in _take(ds, 12):
            if st["shard_idx"] != prev_shard:
                prev_end, prev_shard = st["offset"], st["shard_idx"]
                continue
            window_start = st["offset"] - sample["n_base_tokens"]
            gap = window_start - prev_end
            assert gap >= 0, "windows overlap: some text was trained twice"
            if gap == 0:
                adjacent += 1
            else:
                # A gap can only be a run of SKIPPED windows (zero-hypertoken,
                # all-masked, unfillable, encode failure), each of which
                # advances by at least one window span (or one legacy chunk).
                assert gap >= SEQ_LEN + 1, (
                    f"gap of {gap} tokens is smaller than any skippable "
                    f"window: text was silently lost"
                )
            prev_end = st["offset"]
            # a window never trains on less than seq_len+1 base tokens and
            # never advances past what it read
            assert SEQ_LEN + 1 <= sample["n_base_tokens"] <= CHUNK * ds.LOSSLESS_MAX_WINDOW_CHUNKS
        assert adjacent >= 1, (
            "no two consecutive yields were adjacent: exact tiling never exercised"
        )


def test_L3_compressible_region_dropped_legacy_trained_lossless():
    # First chunk: one repeated symbol -> LZW ratio >= 2 inside 16 tokens with
    # max_subtokens=3 -> the legacy stream must drop it. Then normal text.
    stream = np.concatenate([np.full(CHUNK, 7, dtype=np.uint32), _mixed_stream(1600, seed=1)])
    with tempfile.TemporaryDirectory() as d:
        _write_shard(d, stream)
        legacy_first = _take(_make_dataset(d), 1)[0]
        # legacy: the compressible first chunk was consumed and dropped, so
        # the first emitted window starts at or after the second chunk
        assert legacy_first[2]["offset"] >= 2 * CHUNK

        ds = _make_dataset(d, lossless_windows=True)
        sample, _, st = _take(ds, 1)[0]
        # lossless: the first window is emitted FROM the compressible region,
        # extended beyond one chunk to fill seq_len+1 compressed tokens
        assert st["offset"] == sample["n_base_tokens"]
        assert sample["n_base_tokens"] > CHUNK, (
            "extension did not run: the compressible region was not trained on"
        )


def test_L4_shard_end_unfillable_window_skipped_and_stream_wraps():
    # A 60-token single-symbol shard with max_subtokens=8, derived exactly
    # from the LZW warmup: [7]*16 -> 6 codes, [7]*32 -> 8, [7]*48 -> 10.
    # Window 1 (start 0) extends twice, fills at 48 raw tokens, and its
    # truncated 9 codes expand to 1+2+...+8+8 = 44 base tokens. Window 2
    # (start 44) cannot extend (44+16+16 > 60), stays at 6 < 9 codes, and
    # takes the unfillable-skip branch: advance by the untruncated encode's
    # full 16-token span -> offset 60 -> shard exhausted -> wrap. Pass 2 must
    # replay pass 1 identically.
    with tempfile.TemporaryDirectory() as d:
        _write_shard(d, np.full(60, 7, dtype=np.uint32))
        ds = _make_dataset(d, max_subtokens=8, lossless_windows=True)
        first, second = _take(ds, 2)
        s1, _, st1 = first
        s2, _, st2 = second
        assert s1["n_base_tokens"] == 44
        assert st1["shard_idx"] == 0 and st1["offset"] == 44
        # The 16-token tail was skipped via the unfillable branch and the
        # stream wrapped: the second yield is the second PASS's first window.
        assert st2["shard_idx"] == 0 and st2["offset"] == 44
        assert torch.equal(s1["input"], s2["input"])


def test_L5_lossless_resume_roundtrip_is_exact():
    with tempfile.TemporaryDirectory() as d:
        _write_shard(d, _mixed_stream())
        ds = _make_dataset(d, lossless_windows=True)
        _take(ds, 5)
        snapshot = dict(ds.state_dict())
        tail_direct = _take(ds, 4)

        resumed = _make_dataset(d, lossless_windows=True)
        resumed.load_state_dict(snapshot)
        tail_resumed = _take(resumed, 4)
        for (s_a, y_a, st_a), (s_b, y_b, st_b) in zip(tail_direct, tail_resumed):
            assert torch.equal(s_a["input"], s_b["input"])
            assert torch.equal(y_a, y_b)
            assert st_a == st_b


def test_L6_no_window_is_fully_masked_under_either_setting():
    stream = _mixed_stream(1600, seed=3)
    mask = np.ones(len(stream), dtype=bool)
    mask[: 3 * CHUNK] = False  # a fully masked leading region
    with tempfile.TemporaryDirectory() as d:
        _write_shard(d, stream, mask=mask)
        for flag in (False, True):
            for sample, y, _ in _take(_make_dataset(d, lossless_windows=flag), 6):
                assert (y != -100).any(), f"fully masked window emitted (lossless={flag})"


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"{name}: OK")
