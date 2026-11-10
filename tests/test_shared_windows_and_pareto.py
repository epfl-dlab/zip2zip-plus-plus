"""Correctness invariants for shared-window SFT data and eval compression narrowing:

1. ``scripts/build_shared_windows_sft.py`` (shared_S dataset + >2x window-drop audit):
   - Zero runtime skips across Uncompressed, v0.2, and v0.4/v0.6.4 on shared_S.
   - Exact step-by-step raw-chunk alignment across all variants.
   - Exact drop accounting (ratio_gt_2x, zero_hypertoken, all_masked, rescued).
2. ``Zip2ZipLM`` inference codec sweep on flat hyper-encoders (Pareto narrowing):
   - Narrowing ``eval_max_subtokens`` (1..checkpoint_max_subtokens) and
     ``eval_max_codebook_size`` (0..checkpoint_max_codebook_size) on flat
     hyper-encoders works across strict scoring, multi-view scoring, and
     cached/uncached compressed generation, while widening fails loudly.
3. ``scripts/eval_ckpt_rcp.sh`` launcher wiring for ``EVAL_MAX_SUBTOKENS`` and
   ``EVAL_MAX_CODEBOOK_SIZE`` (collision-free ``VARIANT_TAG`` + JSON audit).
"""

from __future__ import annotations

import dataclasses
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))
if str(REPO / "ext" / "torchtitan") not in sys.path:
    sys.path.insert(0, str(REPO / "ext" / "torchtitan"))
if str(REPO / "tests") not in sys.path:
    sys.path.insert(0, str(REPO / "tests"))

from _hyper_common import DEV, build_model  # noqa: E402
from zip2zip_core.configs import zip2zip_llama_configs  # noqa: E402
from zip2zip_core.data import Zip2ZipDataset  # noqa: E402
from zip2zip_core.disabled_ids import compute_disabled_ids  # noqa: E402


def _load_shared_builder_module():
    spec = importlib.util.spec_from_file_location(
        "_build_shared_windows_sft",
        REPO / "scripts" / "build_shared_windows_sft.py",
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


class _MockPhiTokenizer:
    """Offline deterministic tokenizer covering special, chat, digit, ws, math, code, prose."""

    def __init__(self, vocab_size: int = 128):
        self.vocab_size = vocab_size
        self.pad_token_id = 0
        self.bos_token_id = 1
        self.eos_token_id = 2
        self.eos_token = "<|endoftext|>"
        self.all_special_ids = [0, 1, 2]
        self.chat_template = None
        self.name_or_path = "mock/phi-test"

        self._added = {
            "<|assistant|>": 3,
            "<|user|>": 4,
            "<|end|>": 5,
        }
        self._id_to_piece: dict[int, str] = {
            0: "<unk>",
            1: "<s>",
            2: "<|endoftext|>",
            3: "<|assistant|>",
            4: "<|user|>",
            5: "<|end|>",
            6: "<0x0A>",
            7: "▁▁▁▁",
            8: "\\frac",
            9: "==",
        }
        for d in range(10):
            self._id_to_piece[10 + d] = str(d)
        self._id_to_piece[20] = "def"
        self._id_to_piece[21] = "return"
        for i in range(22, vocab_size):
            self._id_to_piece[i] = f"▁w{i}"
        self._vocab = {p: i for i, p in self._id_to_piece.items()}

    def __len__(self) -> int:
        return self.vocab_size

    def get_vocab(self) -> dict[str, int]:
        return dict(self._vocab)

    def get_added_vocab(self) -> dict[str, int]:
        return dict(self._added)

    def convert_ids_to_tokens(self, idx: int) -> str:
        return self._id_to_piece[idx]

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        if text == "\n":
            return [6]
        return [22 + (ord(ch) % 40) for ch in text]

    def decode(self, ids, skip_special_tokens: bool = True) -> str:
        out = []
        for idx in ids:
            idx = int(idx)
            if skip_special_tokens and idx in (0, 1, 2, 3, 4, 5):
                continue
            if idx == 6:
                out.append("\n")
            elif 10 <= idx <= 19:
                out.append(str(idx - 10))
            else:
                out.append(self._id_to_piece.get(idx, "x").replace("▁", " "))
        return "".join(out)


def _make_diverse_chunk(length: int, start_tok: int = 30) -> list[int]:
    """Low-compression chunk with guaranteed >=1 LZW repeat pair and len(comp) >= seq_len+1."""
    # Distinct tokens plus a few repeated pairs at the front so at least one
    # hypertoken appears in win[:-1] while overall compression stays well below 2x.
    head = [start_tok, start_tok + 1, start_tok, start_tok + 1, start_tok + 2, start_tok + 3, start_tok + 2, start_tok + 3]
    body = [25 + ((k * 7 + start_tok) % 95) for k in range(length - len(head))]
    return head + body


def test_shared_windows_zero_runtime_skips_and_exact_alignment(tmp_path, monkeypatch):
    import transformers

    vocab_size = 128
    seq_len = 32
    chunk_len = 2 * seq_len  # 64
    mock_tok = _MockPhiTokenizer(vocab_size=vocab_size)
    monkeypatch.setattr(
        transformers.AutoTokenizer,
        "from_pretrained",
        classmethod(lambda cls, *a, **kw: mock_tok),
    )

    builder = _load_shared_builder_module()

    # Construct 6 archetypes per shard across 2 shards:
    #   c0: valid diverse prose (kept)
    #   c1: >2x prose repetition [30, 31] * 32 -> ratio_gt_2x in BOTH v0.2 and v0.4
    #   c2: pure digit repetition [10, 11] * 32 -> ratio_gt_2x in v0.2, zero_hypertoken in v0.4
    #   c3: mixed digit repetition (36 tokens) + prose with 1 repeat pair + distinct tail (28 tokens)
    #       -> under v0.2 digits compress heavily so total compressed len < 33 (ratio_gt_2x);
    #       -> under v0.4 digits stay unmerged (36 tokens) + prose (<28 tokens with >=1 hyper) >= 33 -> RESCUED!
    #   c4: valid prose tokens, but mask=0 on first 40 positions -> all_masked in all regimes!
    #   c5: valid chat + prose chunk (kept)
    c0 = _make_diverse_chunk(chunk_len, start_tok=30)
    c1 = [30, 31] * (chunk_len // 2)
    c2 = [10, 11] * (chunk_len // 2)
    c3 = [40, 41, 40, 41] + ([12, 13] * 30)
    c4 = _make_diverse_chunk(chunk_len, start_tok=50)
    c5 = [4, 32, 33, 5, 3] + _make_diverse_chunk(chunk_len - 5, start_tok=60)
    assert all(len(c) == chunk_len for c in (c0, c1, c2, c3, c4, c5))

    m_ones = np.ones(chunk_len, dtype=np.uint8)
    m_zero_head = np.zeros(chunk_len, dtype=np.uint8)
    m_zero_head[55:] = 1  # chunk[1:33] is all 0, and compressed window is also all 0 in head

    src_dir = tmp_path / "src_sft"
    src_dir.mkdir()
    for s_idx in range(2):
        toks = np.array(c0 + c1 + c2 + c3 + c4 + c5, dtype=np.uint32)
        msks = np.concatenate(
            [m_ones, m_ones, m_ones, m_ones, m_zero_head, m_ones]
        ).astype(np.uint8)
        np.save(src_dir / f"shard_{s_idx:05d}.npy", toks)
        np.save(src_dir / f"mask_{s_idx:05d}.npy", msks)

    dst_dir = tmp_path / "dst_shared"
    audit_path = tmp_path / "audit.json"
    audit = builder.build_shared_windows_dataset(
        src_dir=str(src_dir),
        dst_dir=str(dst_dir),
        audit_json=str(audit_path),
        tokenizer_name="mock/phi-test",
        vocab_size=vocab_size,
        pad_token_id=0,
        seq_len=seq_len,
        max_subtokens=4,
        max_codebook_size=64,
        max_active_codebook_size=64,
        workers=1,
        steps_budget_chunks=4,
        overwrite=True,
    )

    totals = audit["totals"]
    assert totals["total_chunks"] == 12
    # Only c0 and c5 pass all three regimes (2 per shard * 2 shards = 4).
    assert totals["shared_S_kept_chunks"] == 4
    assert totals["transitions_v02_to_v04"]["rescued_from_ratio_gt_2x_by_digit_protection"] == 2
    assert totals["transitions_v02_to_v04"]["newly_dropped_zero_hypertoken_in_v04"] == 0
    assert totals["v0.4_v0.6.4_digitsafe_counts"]["zero_hypertoken"] == 2
    assert totals["uncompressed_counts"]["all_masked"] == 2

    # Budget check: 0 runtime skips on shared_S, >0 runtime skips on original!
    ws2 = audit["step_matched_training_budgets"]["world_size_2"]
    for var_key in ("uncompressed", "v0.2_zip2zip", "v0.4_v0.6.4_digitsafe"):
        assert ws2["shared_S_dataset"][var_key]["chunks_skipped_at_runtime"] == 0
        assert ws2["shared_S_dataset"][var_key]["runtime_skip_rate"] == 0.0
    assert ws2["original_dataset"]["v0.2_zip2zip"]["chunks_skipped_at_runtime"] > 0
    assert "shared_S_kept" in audit["content_composition"]

    # Ground-truth runtime check with the REAL Zip2ZipDataset across all 3 data regimes:
    dis_v02 = compute_disabled_ids(mock_tok, vocab_size, disable_digit_ids=False)
    dis_v04 = compute_disabled_ids(mock_tok, vocab_size, disable_digit_ids=True)

    ds_uncomp = Zip2ZipDataset(
        data_dir=str(dst_dir),
        seq_len=seq_len,
        max_subtokens=4,
        max_codebook_size=0,
        max_active_codebook_size=0,
        initial_vocab_size=vocab_size,
        pad_token_id=0,
        disabled_ids=dis_v04,
        remap_codebook=False,
    )
    ds_v02 = Zip2ZipDataset(
        data_dir=str(dst_dir),
        seq_len=seq_len,
        max_subtokens=4,
        max_codebook_size=64,
        max_active_codebook_size=64,
        initial_vocab_size=vocab_size,
        pad_token_id=0,
        disabled_ids=dis_v02,
        remap_codebook=False,
    )
    ds_v04 = Zip2ZipDataset(
        data_dir=str(dst_dir),
        seq_len=seq_len,
        max_subtokens=4,
        max_codebook_size=64,
        max_active_codebook_size=64,
        initial_vocab_size=vocab_size,
        pad_token_id=0,
        disabled_ids=dis_v04,
        remap_codebook=False,
    )

    it_u, it_02, it_04 = iter(ds_uncomp), iter(ds_v02), iter(ds_v04)
    expected_positions = [(0, chunk_len), (0, 2 * chunk_len), (1, chunk_len), (1, 2 * chunk_len)]
    for exp_shard, exp_offset in expected_positions:
        su, _ = next(it_u)
        s02, _ = next(it_02)
        s04, _ = next(it_04)
        # Zero skips means every dataset sits at the exact same (_shard_idx, _offset)!
        assert (ds_uncomp._shard_idx, ds_uncomp._offset) == (exp_shard, exp_offset)
        assert (ds_v02._shard_idx, ds_v02._offset) == (exp_shard, exp_offset)
        assert (ds_v04._shard_idx, ds_v04._offset) == (exp_shard, exp_offset)
        # First token of uncompressed and compressed windows is identical (base token 0 of the same raw chunk).
        assert int(su["input"][0]) == int(s02["input"][0]) == int(s04["input"][0])


def _save_tiny_flat_checkpoint(ckpt_dir: Path, *, max_subtokens: int = 4, max_codebook_size: int = 16):
    cfg_key = "_test_flat_pareto_narrowing"
    base = zip2zip_llama_configs["Phi3.5-mini"]
    cfg = dataclasses.replace(
        base,
        dim=64,
        n_layers=2,
        vocab_size=128,
        pad_token_id=0,
        max_codebook_size=max_codebook_size,
        max_subtokens=max_subtokens,
        hyper_encoder_type="flat",
        tie_hyper_encoder=False,
        base_token_positions=True,
        zero_init_encoder_output=True,
        token_type_loss_weight=0.05,
        encoder_dim=64,
        encoder_n_layers=1,
        encoder_n_heads=4,
        encoder_intermediate_size=128,
        tok_embeddings=dataclasses.replace(base.tok_embeddings, init_std=64**-0.5),
        layer=dataclasses.replace(
            base.layer,
            feed_forward=dataclasses.replace(base.layer.feed_forward, hidden_dim=128),
            attention=dataclasses.replace(base.layer.attention, n_heads=4, n_kv_heads=4),
        ),
        rope=dataclasses.replace(base.rope, dim=16, max_seq_len=256 * max_subtokens),
    )
    model = build_model(cfg, seed=7)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), ckpt_dir / "model.pt")
    torch.save(
        {
            "args": {
                "model_config": cfg_key,
                "tokenizer": "mock/phi-test",
                "max_subtokens": max_subtokens,
                "max_codebook_size": max_codebook_size,
                "hyper_encoder_type": "flat",
                "untied_hyper_encoder": True,
                "base_token_positions": True,
                "zero_init_encoder_output": True,
                "token_type_loss_weight": 0.05,
                "disable_digit_ids": True,
                "encoder_dim": 64,
                "encoder_n_layers": 1,
                "encoder_n_heads": 4,
                "encoder_intermediate_size": 128,
            }
        },
        ckpt_dir / "meta.pt",
    )
    return cfg_key, cfg


def test_flat_checkpoint_pareto_narrowing_score_and_generate(tmp_path, monkeypatch):
    pytest.importorskip("lm_eval")
    import transformers
    import zip2zip_core.lm_eval_adapter as adapter

    mock_tok = _MockPhiTokenizer(vocab_size=128)
    monkeypatch.setattr(
        transformers.AutoTokenizer,
        "from_pretrained",
        classmethod(lambda cls, *a, **kw: mock_tok),
    )

    ckpt_dir = tmp_path / "flat_v064_ckpt"
    cfg_key, cfg = _save_tiny_flat_checkpoint(ckpt_dir, max_subtokens=4, max_codebook_size=16)
    adapter.zip2zip_llama_configs[cfg_key] = cfg
    try:
        # 1. Widening beyond checkpoint limits must fail loudly.
        with pytest.raises(ValueError, match="Flat hyper-encoders"):
            adapter.Zip2ZipLM(
                pretrained=str(ckpt_dir),
                tokenizer="mock/phi-test",
                device=DEV,
                dtype="float32",
                eval_max_subtokens=5,
            )
        with pytest.raises(ValueError, match="eval_max_codebook_size"):
            adapter.Zip2ZipLM(
                pretrained=str(ckpt_dir),
                tokenizer="mock/phi-test",
                device=DEV,
                dtype="float32",
                eval_max_codebook_size=32,
            )
        with pytest.raises(ValueError, match="non-negative"):
            adapter.Zip2ZipLM(
                pretrained=str(ckpt_dir),
                tokenizer="mock/phi-test",
                device=DEV,
                dtype="float32",
                eval_max_codebook_size=-1,
            )

        def _make_lm(*, ms=None, cb=None, kv=True):
            lm = adapter.Zip2ZipLM(
                pretrained=str(ckpt_dir),
                tokenizer="mock/phi-test",
                device=DEV,
                dtype="float32",
                max_length=128,
                eval_max_subtokens=ms,
                eval_max_codebook_size=cb,
                use_kv_cache=kv,
            )
            lm.model.hyper_encoder.disable_varlen = True
            if getattr(lm.model, "hyper_output", None) is not None:
                lm.model.hyper_output.disable_varlen = True
            return lm

        seq = [30, 31, 32, 33] * 12  # forms 2-, 3-, and 4-token LZW merges

        # 2. Merge-length sweep (ms = 1, 2, 3, 4) on flat encoder:
        ratios_ms = {}
        for ms in (1, 2, 3, 4):
            lm = _make_lm(ms=ms)
            assert lm.checkpoint_max_subtokens == 4
            assert lm.eval_max_subtokens == ms
            assert lm.cfg.max_subtokens == 4  # model pos_embed width stays 4!
            if ms == 1:
                assert lm.eval_mode == "base"
            else:
                assert lm.eval_mode == "compressed"
            lp, _, n_comp, n_base, mv_lp, _, exact_lp = lm._score(
                seq, cont_start_base=4, compute_multi_view=True
            )
            assert np.isfinite(lp) and np.isfinite(mv_lp) and np.isfinite(exact_lp)
            assert n_comp > 0 and n_base > 0
            ratios_ms[ms] = lm.compression_summary()["input_compression_ratio"]

        assert ratios_ms[1] == pytest.approx(1.0)
        assert ratios_ms[1] < ratios_ms[2] < ratios_ms[3] <= ratios_ms[4]

        # 3. Codebook-capacity sweep (cb = 0, 2, 16) on flat encoder:
        ratios_cb = {}
        for cb_cap in (0, 2, 16):
            lm = _make_lm(cb=cb_cap)
            assert lm.checkpoint_max_codebook_size == 16
            assert lm.eval_max_codebook_size == cb_cap
            if cb_cap == 0:
                assert lm.eval_mode == "base"
            lp, _, _, _, _, _, exact_lp = lm._score(
                seq, cont_start_base=4, compute_multi_view=True
            )
            assert np.isfinite(lp) and np.isfinite(exact_lp)
            ratios_cb[cb_cap] = lm.compression_summary()["input_compression_ratio"]

        assert ratios_cb[0] == pytest.approx(1.0)
        assert ratios_cb[0] < ratios_cb[2] < ratios_cb[16]

        # 4. Compressed generation with narrowed flat encoder (ms=2, cb=4):
        #    Cached and uncached paths must produce bit-identical text, including
        #    when the sampler picks hypertokens and advances RustCodebookManager!
        lm_cached = _make_lm(ms=2, cb=4, kv=True)
        lm_uncached = _make_lm(ms=2, cb=4, kv=False)

        # Bias output head toward first codebook slots so hypertokens are sampled
        # and expanded during generation, exercising both seed & step-by-step updates.
        prompt_ids = [30, 31, 30, 31, 32, 33, 32, 33, 34, 35]
        out_cached = lm_cached._generate_compressed(
            list(prompt_ids),
            until=[],
            max_gen_toks=10,
            do_sample=False,
            temperature=0.0,
            top_p=1.0,
            top_k=0,
        )
        out_uncached = lm_uncached._generate_compressed(
            list(prompt_ids),
            until=[],
            max_gen_toks=10,
            do_sample=False,
            temperature=0.0,
            top_p=1.0,
            top_k=0,
        )
        assert out_cached == out_uncached

        # Force first generated token to be hyper slot V+0 (seeded merge [30, 31])
        # to guarantee step-by-step new_flat slicing at stride S_comp=2 runs cleanly.
        call_idx = {"n": 0}
        orig_sample = lm_cached._sample_next

        def _force_hyper_then_greedy(last_logits, *a, **kw):
            call_idx["n"] += 1
            if call_idx["n"] == 1:
                assert torch.isfinite(last_logits[128 + 0])
                # Slots >= eval_max_codebook_size (4) must remain masked (-inf)!
                assert not torch.isfinite(last_logits[128 + 4 :]).any()
                return 128 + 0
            return orig_sample(last_logits, *a, **kw)

        lm_cached._sample_next = _force_hyper_then_greedy
        gen_hyper = lm_cached._generate_compressed(
            list(prompt_ids),
            until=[],
            max_gen_toks=6,
            do_sample=False,
            temperature=0.0,
            top_p=1.0,
            top_k=0,
        )
        assert isinstance(gen_hyper, str) and len(gen_hyper) > 0
    finally:
        adapter.zip2zip_llama_configs.pop(cfg_key, None)


def test_eval_ckpt_launcher_pareto_flags_and_audit(tmp_path):
    launcher = REPO / "scripts" / "eval_ckpt_rcp.sh"
    pipeline = REPO / "scripts" / "pipeline_ft_eval_rcp.sh"
    subprocess.run(["bash", "-n", str(launcher)], check=True)
    subprocess.run(["bash", "-n", str(pipeline)], check=True)

    base_env = {
        **os.environ,
        "CKPT_DIR": "/tmp/fake_run/step_8000",
        "PRESET": "default",
        "Z2Z_SCRATCH": str(tmp_path / "scratch"),
        "EVAL_PATHS_ONLY": "1",
        "TIMESTAMP": "20261109_120000",
        "RUNAI_JOB_NAME": "pareto-test",
        "HOSTNAME": "node0",
    }
    base_env.pop("EVAL_MAX_SUBTOKENS", None)
    base_env.pop("EVAL_MAX_CODEBOOK_SIZE", None)

    default_out = subprocess.run(
        ["bash", str(launcher)], env=base_env, text=True, capture_output=True, check=True
    ).stdout
    assert "_mode-auto_20261109_120000_" in default_out

    swept_out = subprocess.run(
        ["bash", str(launcher)],
        env={**base_env, "EVAL_MAX_SUBTOKENS": "2", "EVAL_MAX_CODEBOOK_SIZE": "256"},
        text=True,
        capture_output=True,
        check=True,
    ).stdout
    assert "_mode-auto-ms2-cb256_20261109_120000_" in swept_out

    cb0_out = subprocess.run(
        ["bash", str(launcher)],
        env={**base_env, "EVAL_MAX_CODEBOOK_SIZE": "0"},
        text=True,
        capture_output=True,
        check=True,
    ).stdout
    assert "_mode-auto-cb0_20261109_120000_" in cb0_out

    # Extract and test the inline post-eval JSON audit script.
    source = launcher.read_text()
    audit_prog = source.split("<<'PY'\n")[-1].split("\nPY\n")[0]
    res_json = tmp_path / "results.json"
    res_json.write_text(
        json.dumps(
            {
                "args": {
                    "eval_mode": "base",
                    "eval_max_subtokens": 2,
                    "eval_max_codebook_size": 0,
                    "disable_digit_ids": True,
                    "disable_mathsym_ids": False,
                }
            }
        )
    )
    subprocess.run(
        [sys.executable, "-", str(res_json), "", "2", "0", "1", ""],
        input=audit_prog,
        text=True,
        check=True,
    )
    # Even if EVAL_MODE=compressed is passed alongside cb=0 or ms=1, the audit
    # recognizes the automatic switch to eval_mode="base".
    subprocess.run(
        [sys.executable, "-", str(res_json), "compressed", "2", "0", "1", ""],
        input=audit_prog,
        text=True,
        check=True,
    )
    bad = subprocess.run(
        [sys.executable, "-", str(res_json), "", "2", "128", "1", ""],
        input=audit_prog,
        text=True,
        capture_output=True,
    )
    assert bad.returncode != 0
    assert "eval-max-codebook-size audit FAILED" in bad.stderr
