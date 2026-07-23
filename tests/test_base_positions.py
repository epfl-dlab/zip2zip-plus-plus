"""Correctness invariants for base-token RoPE positions (v0.6.2).

v0.6.2 = the v0.5 canonical recipe (untied + digit-protected) with RoPE
positions following the UNCOMPRESSED stream: every token sits at the
base-space index of its last constituent (positions = cumsum(spans) - 1),
so relative distances keep their pretrained meaning regardless of local
compression ratio. Default off = one position per compressed token.

Runs on CPU with the fully shrunken config (decoder dim 256 / vocab 512 /
2 layers, encoder dim 128). fp32, varlen disabled.

Runnable both as a script (python tests/test_base_positions.py) and under
pytest. Shared mechanics live in tests/_hyper_common.py.

Invariants proven:
  B1  the config default is off (v0.1-v0.6.1 checkpoints unaffected).
  B2  position helper arithmetic: hand-built spans {3,2} at known indices
      give exactly the expected end-anchored cumsum positions; all-base
      rows reduce exactly to arange(T).
  B3  flag-on is bit-identical to flag-off on uncompressed input (all-base
      tokens + all-pad codebook take the plain path in both).
  B4  passing explicit positions=arange equals the default no-positions
      forward (pins the torchtitan freqs_cis gather contract).
  B5  with a hypertoken present, flag-on logits are IDENTICAL before the
      first hypertoken (causal prefix untouched) and DIFFERENT from it on
      (the feature actually reaches attention).
  B6  incremental-inference span buffer agrees with the training-path
      codebook derivation (train/generation cannot drift), and the flag-on
      inference forward is well-formed.
  B7  a fully merged window reaches base position seq_len*ms - 1 without
      overrunning a rope cache sized seq_len*ms (the train.py sizing rule).
  B8  resume guard: base_token_positions is a hard key — recorded-vs-
      requested mismatch raises; matching args pass.
"""
import dataclasses
import os
import sys
import tempfile
import types

from _hyper_common import build_model, fwd, make_harness, summarize

import torch
from zip2zip_core.configs import zip2zip_llama_configs

MS = 4
T = 16


def small_cfg(base_positions: bool):
    base = zip2zip_llama_configs["Phi3.5-mini"]
    return dataclasses.replace(
        base,
        dim=256,
        n_layers=2,
        vocab_size=512,
        pad_token_id=0,
        max_codebook_size=8,
        max_subtokens=MS,
        tie_hyper_encoder=False,
        base_token_positions=base_positions,
        encoder_dim=128,
        encoder_n_layers=2,
        encoder_n_heads=4,
        encoder_intermediate_size=256,
        tok_embeddings=dataclasses.replace(base.tok_embeddings, init_std=256 ** -0.5),
        layer=dataclasses.replace(
            base.layer,
            feed_forward=dataclasses.replace(base.layer.feed_forward, hidden_dim=512),
            attention=dataclasses.replace(base.layer.attention, n_heads=8, n_kv_heads=8),
        ),
        # sized like train.py does with the flag on: seq_len * max_subtokens
        rope=dataclasses.replace(base.rope, dim=256 // 8, max_seq_len=T * MS),
    )


def build(base_positions: bool, seed: int = 0):
    cfg = small_cfg(base_positions)
    return build_model(cfg, seed=seed), cfg


def spanned_input(cfg, dev):
    """tokens [b b H0 b H1 b b ...] with entry0 span 3, entry1 span 2."""
    V = cfg.vocab_size
    pad = cfg.pad_token_id
    K = cfg.max_codebook_size
    torch.manual_seed(7)
    toks = torch.randint(1, V, (1, T), device=dev)
    toks[0, 2] = V + 0
    toks[0, 4] = V + 1
    cb = torch.full((1, K, cfg.max_subtokens), pad, device=dev)
    cb[0, 0, :3] = torch.tensor([11, 12, 13], device=dev)  # span 3
    cb[0, 1, :2] = torch.tensor([21, 22], device=dev)      # span 2
    return toks, cb


def main():
    results, check = make_harness()

    # ---- B1: default off ----
    base = zip2zip_llama_configs["Phi3.5-mini"]
    check("B1_config_default_off", base.base_token_positions is False,
          "code default keeps v0.5 semantics")

    mon, cfg = build(base_positions=True, seed=1)
    dev = next(mon.parameters()).device
    toks, cb = spanned_input(cfg, dev)

    # ---- B2: helper arithmetic ----
    pos = mon._base_token_positions(toks, cb)
    # spans: [1,1,3,1,2,1,1,...] -> cumsum-1 = [0,1,4,5,7,8,9,...]
    expected = torch.tensor([0, 1, 4, 5, 7, 8], device=dev)
    check("B2_end_anchored_cumsum", bool(torch.equal(pos[0, :6], expected)),
          f"got {pos[0, :6].tolist()} expected {expected.tolist()}")
    all_base = torch.randint(1, cfg.vocab_size, (1, T), device=dev)
    pos_base = mon._base_token_positions(all_base, cb)
    check("B2_all_base_reduces_to_arange",
          bool(torch.equal(pos_base[0], torch.arange(T, device=dev))),
          "uncompressed rows get arange positions")

    # ---- B3: flag-on == flag-off on uncompressed input ----
    moff, _ = build(base_positions=False, seed=1)  # same seed -> same weights
    cb_allpad = torch.full_like(cb, cfg.pad_token_id)
    lo_on = fwd(mon, all_base, cb_allpad)
    lo_off = fwd(moff, all_base, cb_allpad)
    check("B3_uncompressed_bit_identical",
          bool(torch.equal(lo_on, lo_off)),
          "plain path unaffected by the flag")

    # ---- B4: explicit arange positions == default ----
    with torch.no_grad():
        out_default = moff(all_base, codebook=cb_allpad, hyper_causal_mask=True)
        out_explicit = moff(all_base, codebook=cb_allpad, hyper_causal_mask=True,
                            positions=torch.arange(T, device=dev).unsqueeze(0))
    out_default = (out_default[0] if isinstance(out_default, tuple) else out_default).float()
    out_explicit = (out_explicit[0] if isinstance(out_explicit, tuple) else out_explicit).float()
    check("B4_explicit_arange_matches_default",
          torch.allclose(out_default, out_explicit, atol=1e-5, rtol=1e-5),
          f"max|d|={float((out_default - out_explicit).abs().max()):.2e}")

    # ---- B5: prefix identical, suffix differs, with a hypertoken present ----
    l_on = fwd(mon, toks, cb)
    l_off = fwd(moff, toks, cb)
    first_hyper = 2
    prefix_same = torch.allclose(l_on[:, :first_hyper], l_off[:, :first_hyper],
                                 atol=1e-5, rtol=1e-5)
    Vv = cfg.vocab_size
    suffix_diff = float((l_on[:, first_hyper:, :Vv] - l_off[:, first_hyper:, :Vv]
                         ).abs().max())
    check("B5_causal_prefix_identical", bool(prefix_same),
          "rows before the first hypertoken see identical geometry")
    check("B5_feature_reaches_attention", suffix_diff > 1e-4,
          f"max base-logit |d| from first hypertoken on = {suffix_diff:.3e}")

    # ---- B6: inference span buffer agrees with codebook derivation ----
    mon.reset_inference_cache()
    K = cfg.max_codebook_size
    updates = torch.full((1, K, cfg.max_subtokens), cfg.pad_token_id, device=dev)
    updates[0, 0, :3] = torch.tensor([11, 12, 13], device=dev)
    updates[0, 1, :2] = torch.tensor([21, 22], device=dev)
    with torch.no_grad():
        out_inf = mon(toks, codebook_updates=updates,
                      codebook_updates_indices=[[0, 1]])
    out_inf = (out_inf[0] if isinstance(out_inf, tuple) else out_inf).float()
    pos_inf = mon._base_token_positions(toks, codebook=None)
    check("B6_span_buffer_matches_codebook",
          bool(torch.equal(pos_inf, pos)),
          f"inference {pos_inf[0, :6].tolist()} vs train {pos[0, :6].tolist()}")
    check("B6_inference_forward_wellformed",
          bool(torch.isfinite(out_inf[..., :Vv]).all()),
          "flag-on incremental forward produces finite base logits")

    # ---- B7: fully merged window touches the cache bound exactly ----
    toks_full = torch.full((1, T), cfg.vocab_size + 0, device=dev)
    cb_full = torch.full((1, K, cfg.max_subtokens), cfg.pad_token_id, device=dev)
    cb_full[0, 0] = torch.tensor([31, 32, 33, 34], device=dev)  # span = MS
    pos_full = mon._base_token_positions(toks_full, cb_full)
    check("B7_max_position_is_bound_minus_one",
          int(pos_full.max()) == T * MS - 1,
          f"max pos {int(pos_full.max())} == {T * MS - 1}")
    logits_full = fwd(mon, toks_full, cb_full)
    check("B7_forward_at_cache_bound",
          bool(torch.isfinite(logits_full[..., :Vv]).all()),
          "rope cache sized seq_len*ms is sufficient")

    # ---- B8: resume guard hard key ----
    import zip2zip_core.train as train_mod
    saved_dist = train_mod.dist
    train_mod.dist = types.SimpleNamespace(get_rank=lambda: 0)
    try:
        with tempfile.TemporaryDirectory() as td:
            args = types.SimpleNamespace(
                disable_digit_ids=True, max_codebook_size=4096,
                tokenizer="microsoft/Phi-3.5", untied_hyper_encoder=True,
                base_token_positions=True, encoder_dim=3072,
                encoder_n_layers=2, encoder_n_heads=32,
                encoder_intermediate_size=12288, max_subtokens=4,
                seq_len=2048, data_dir="/data", warmstart_steps=0,
                allow_resume_mismatch=False,
            )
            recorded = dict(vars(args))
            del recorded["allow_resume_mismatch"]

            def guard_raises(meta_args):
                torch.save({"args": meta_args}, os.path.join(td, "meta.pt"))
                try:
                    train_mod.validate_resume_args(td, args)
                    return False
                except ValueError:
                    return True

            check("B8_flag_mismatch_hard_errors",
                  guard_raises({**recorded, "base_token_positions": False}),
                  "recorded off vs requested on raises ValueError")
            check("B8_matching_args_pass", not guard_raises(recorded),
                  "identical recorded args validate cleanly")
    finally:
        train_mod.dist = saved_dist

    return summarize(results)


def test_base_token_positions_invariants():
    failed = main()
    assert not failed, f"failed invariants: {failed}"


if __name__ == "__main__":
    sys.exit(1 if main() else 0)
