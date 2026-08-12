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
  B3  flag-on equals flag-off on uncompressed input BOTH through the
      all-pad plain path (bit-identical) and through the real train path
      (all-base tokens with a live codebook -> spans all 1 -> arange).
  B4  passing explicit positions=arange equals the default no-positions
      forward (pins the torchtitan freqs_cis gather contract).
  B5  with a hypertoken present: logits are IDENTICAL before the first
      hypertoken (causal prefix untouched), DIFFERENT from it on, and the
      flag-on forward EXACTLY equals the flag-off forward fed the helper's
      positions explicitly (the attention consumed the helper's values).
  B6  incremental inference: the span buffer agrees with the training-path
      codebook derivation under a NON-identity update index mapping and
      across two accumulating update calls, and the updates-path positions
      hook has an effect (flag-on differs from flag-off on the same
      updates-path input from the first hypertoken on).
  B7  a fully merged window reaches base position seq_len*ms - 1 without
      overrunning a rope cache sized by train.py's rope_cache_len rule
      (imported, not re-derived).
  B8  resume guard: base_token_positions is a hard key — mismatch raises in
      BOTH directions, a legacy meta.pt that predates the flag counts as
      recorded-off (requesting the flag on such a lineage raises), and
      matching args pass.
  B9  activation checkpointing (train mode) produces the same flag-on
      logits as the plain eval forward (the checkpointed layer call still
      receives positions).
  B10 batched B=2 rows with different span layouts each match their own
      B=1 forward (pins the per-sample freqs_cis gather branch).
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
        # sized like train.py sizes it with the flag on (asserted in B7)
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


def raw_fwd(model, toks, cb, **kw):
    with torch.no_grad():
        out = model(toks, codebook=cb, hyper_causal_mask=True, **kw)
    return (out[0] if isinstance(out, tuple) else out).float()


def main():
    results, check = make_harness()

    # ---- B1: default off ----
    base = zip2zip_llama_configs["Phi3.5-mini"]
    check("B1_config_default_off", base.base_token_positions is False,
          "code default keeps v0.5 semantics")

    mon, cfg = build(base_positions=True, seed=1)
    dev = next(mon.parameters()).device
    toks, cb = spanned_input(cfg, dev)
    Vv = cfg.vocab_size
    K = cfg.max_codebook_size

    # ---- B2: helper arithmetic ----
    pos = mon._base_token_positions(toks, cb)
    # spans: [1,1,3,1,2,1,1,...] -> cumsum-1 = [0,1,4,5,7,8,9,...]
    expected = torch.tensor([0, 1, 4, 5, 7, 8], device=dev)
    check("B2_end_anchored_cumsum", bool(torch.equal(pos[0, :6], expected)),
          f"got {pos[0, :6].tolist()} expected {expected.tolist()}")
    all_base = torch.randint(1, Vv, (1, T), device=dev)
    pos_base = mon._base_token_positions(all_base, cb)
    check("B2_all_base_reduces_to_arange",
          bool(torch.equal(pos_base[0], torch.arange(T, device=dev))),
          "uncompressed rows get arange positions")

    # ---- B3: flag-on == flag-off on uncompressed input, BOTH paths ----
    moff, _ = build(base_positions=False, seed=1)  # same seed -> same weights
    cb_allpad = torch.full_like(cb, cfg.pad_token_id)
    check("B3_plain_path_bit_identical",
          bool(torch.equal(fwd(mon, all_base, cb_allpad), fwd(moff, all_base, cb_allpad))),
          "all-pad codebook takes the plain path in both")
    # live codebook, all-base tokens: train path runs, spans are all 1
    check("B3_train_path_all_base_matches",
          torch.allclose(fwd(mon, all_base, cb), fwd(moff, all_base, cb),
                         atol=1e-5, rtol=1e-5),
          "arange-equivalent positions through the real gather path")

    # ---- B4: explicit arange positions == default ----
    out_default = raw_fwd(moff, all_base, cb_allpad)
    out_explicit = raw_fwd(moff, all_base, cb_allpad,
                           positions=torch.arange(T, device=dev).unsqueeze(0))
    check("B4_explicit_arange_matches_default",
          torch.allclose(out_default, out_explicit, atol=1e-5, rtol=1e-5),
          f"max|d|={float((out_default - out_explicit).abs().max()):.2e}")

    # ---- B5: prefix identical, suffix differs, values reach attention ----
    l_on = fwd(mon, toks, cb)
    l_off = fwd(moff, toks, cb)
    first_hyper = 2
    check("B5_causal_prefix_identical",
          torch.allclose(l_on[:, :first_hyper], l_off[:, :first_hyper],
                         atol=1e-5, rtol=1e-5),
          "rows before the first hypertoken see identical geometry")
    suffix_diff = float((l_on[:, first_hyper:, :Vv] - l_off[:, first_hyper:, :Vv]
                         ).abs().max())
    check("B5_feature_reaches_attention", suffix_diff > 1e-4,
          f"max base-logit |d| from first hypertoken on = {suffix_diff:.3e}")
    # the exact-geometry pin: flag-on forward == flag-off forward given the
    # helper's positions explicitly (same weights, so any difference means the
    # attention did NOT consume the helper's values)
    l_off_explicit = raw_fwd(moff, toks, cb, positions=pos)
    check("B5_attention_consumed_helper_values",
          torch.allclose(l_on, l_off_explicit, atol=1e-5, rtol=1e-5),
          f"max|d|={float((l_on[..., :Vv] - l_off_explicit[..., :Vv]).abs().max()):.2e}")

    # ---- B6: incremental inference — mapping, accumulation, and effect ----
    # NON-identity index mapping: update row 0 -> buffer slot 3 (span 3),
    # update row 1 -> buffer slot 0 (span 2); tokens reference slots 3 and 0.
    toks_inf = toks.clone()
    toks_inf[0, 2] = Vv + 3
    toks_inf[0, 4] = Vv + 0
    updates = torch.full((1, 2, cfg.max_subtokens), cfg.pad_token_id, device=dev)
    updates[0, 0, :3] = torch.tensor([11, 12, 13], device=dev)
    updates[0, 1, :2] = torch.tensor([21, 22], device=dev)

    cb_mapped = torch.full((1, K, cfg.max_subtokens), cfg.pad_token_id, device=dev)
    cb_mapped[0, 3, :3] = torch.tensor([11, 12, 13], device=dev)
    cb_mapped[0, 0, :2] = torch.tensor([21, 22], device=dev)
    pos_train_mapped = mon._base_token_positions(toks_inf, cb_mapped)

    mon.reset_inference_cache()
    with torch.no_grad():
        out_inf_on = mon(toks_inf, codebook_updates=updates,
                         codebook_updates_indices=[[3, 0]])
    out_inf_on = (out_inf_on[0] if isinstance(out_inf_on, tuple) else out_inf_on).float()
    pos_inf = mon._base_token_positions(toks_inf, codebook=None)
    check("B6_span_buffer_matches_codebook_nonidentity_mapping",
          bool(torch.equal(pos_inf, pos_train_mapped)),
          f"inference {pos_inf[0, :6].tolist()} vs train {pos_train_mapped[0, :6].tolist()}")
    # accumulation: a second call writes another entry; earlier spans persist
    updates2 = torch.full((1, 1, cfg.max_subtokens), cfg.pad_token_id, device=dev)
    updates2[0, 0] = torch.tensor([41, 42, 43, 44], device=dev)  # span 4 -> slot 5
    with torch.no_grad():
        mon(toks_inf, codebook_updates=updates2, codebook_updates_indices=[[5]])
    spans_now = mon._hyper_span_buf[0, [3, 0, 5]].tolist()
    check("B6_span_buffer_accumulates_across_calls", spans_now == [3, 2, 4],
          f"slots 3,0,5 hold spans {spans_now}")
    # effect: the updates-path hook must change geometry vs a flag-off model
    moff.reset_inference_cache()
    with torch.no_grad():
        out_inf_off = moff(toks_inf, codebook_updates=updates,
                           codebook_updates_indices=[[3, 0]])
    out_inf_off = (out_inf_off[0] if isinstance(out_inf_off, tuple) else out_inf_off).float()
    inf_prefix_same = torch.allclose(out_inf_on[:, :first_hyper, :Vv],
                                     out_inf_off[:, :first_hyper, :Vv],
                                     atol=1e-5, rtol=1e-5)
    inf_suffix_diff = float((out_inf_on[:, first_hyper:, :Vv]
                             - out_inf_off[:, first_hyper:, :Vv]).abs().max())
    check("B6_updates_path_hook_has_effect",
          bool(inf_prefix_same) and inf_suffix_diff > 1e-4,
          f"prefix_same={bool(inf_prefix_same)} suffix|d|={inf_suffix_diff:.3e}")

    # ---- B7: fully merged window touches the train.py sizing rule exactly ----
    from zip2zip_core.train import rope_cache_len
    check("B7_cfg_uses_train_sizing_rule",
          cfg.rope.max_seq_len == rope_cache_len(T, MS, True)
          and rope_cache_len(T, MS, False) == T,
          f"cache rows {cfg.rope.max_seq_len} == rope_cache_len(T, ms, on) "
          f"= {rope_cache_len(T, MS, True)}; flag-off rule = seq_len")
    toks_full = torch.full((1, T), Vv + 0, device=dev)
    cb_full = torch.full((1, K, cfg.max_subtokens), cfg.pad_token_id, device=dev)
    cb_full[0, 0] = torch.tensor([31, 32, 33, 34], device=dev)  # span = MS
    pos_full = mon._base_token_positions(toks_full, cb_full)
    check("B7_max_position_is_bound_minus_one",
          int(pos_full.max()) == T * MS - 1,
          f"max pos {int(pos_full.max())} == {T * MS - 1}")
    logits_full = fwd(mon, toks_full, cb_full)
    check("B7_forward_at_cache_bound",
          bool(torch.isfinite(logits_full[..., :Vv]).all()),
          "rope cache sized by the rule is sufficient")

    # ---- B8: resume guard hard key, both directions + legacy meta ----
    import zip2zip_core.train as train_mod
    saved_dist = train_mod.dist
    train_mod.dist = types.SimpleNamespace(get_rank=lambda: 0)
    try:
        with tempfile.TemporaryDirectory() as td:
            def make_args(**over):
                base_args = dict(
                    disable_digit_ids=True, max_codebook_size=4096,
                    tokenizer="microsoft/Phi-3.5", untied_hyper_encoder=True,
                    base_token_positions=True, token_type_loss_weight=0.0,
                    encoder_dim=3072,
                    encoder_n_layers=2, encoder_n_heads=32,
                    encoder_intermediate_size=12288, max_subtokens=4,
                    seq_len=2048, data_dir="/data", warmstart_steps=0,
                    allow_resume_mismatch=False,
                )
                base_args.update(over)
                return types.SimpleNamespace(**base_args)

            def guard_raises(meta_args, args):
                torch.save({"args": meta_args}, os.path.join(td, "meta.pt"))
                try:
                    train_mod.validate_resume_args(td, args)
                    return False
                except ValueError:
                    return True

            args_on = make_args()
            recorded_on = dict(vars(args_on))
            del recorded_on["allow_resume_mismatch"]
            legacy_meta = {k: v for k, v in recorded_on.items()
                           if k != "base_token_positions"}

            check("B8_flag_mismatch_hard_errors",
                  guard_raises({**recorded_on, "base_token_positions": False}, args_on),
                  "recorded off vs requested on raises ValueError")
            check("B8_reverse_direction_hard_errors",
                  guard_raises(recorded_on, make_args(base_token_positions=False)),
                  "recorded on vs requested off raises ValueError")
            check("B8_legacy_meta_counts_as_off",
                  guard_raises(legacy_meta, args_on),
                  "pre-flag meta.pt + --base_token_positions raises (absence = off)")
            check("B8_matching_args_pass",
                  not guard_raises(recorded_on, args_on),
                  "identical recorded args validate cleanly")
            check("B8_legacy_meta_flag_off_passes",
                  not guard_raises(legacy_meta, make_args(base_token_positions=False)),
                  "pre-flag meta.pt resumes fine without the flag")
    finally:
        train_mod.dist = saved_dist

    # ---- B9: activation checkpointing keeps positions ----
    mon.gradient_checkpointing = True
    mon.train()
    out_ac = mon(toks, codebook=cb, hyper_causal_mask=True)
    out_ac = (out_ac[0] if isinstance(out_ac, tuple) else out_ac).float()
    mon.eval()
    mon.gradient_checkpointing = False
    check("B9_activation_checkpoint_keeps_positions",
          torch.allclose(out_ac.detach(), l_on, atol=1e-5, rtol=1e-5),
          f"max|d|={float((out_ac.detach()[..., :Vv] - l_on[..., :Vv]).abs().max()):.2e}")

    # ---- B10: batched rows match their own B=1 forwards ----
    toksB = torch.cat([toks, toks.clone()], dim=0)
    toksB[1, 2] = Vv + 1  # row 1 swaps the span-3/span-2 layout
    toksB[1, 4] = Vv + 0
    cbB = torch.cat([cb, cb.clone()], dim=0)
    lB = fwd(mon, toksB, cbB)
    l_row1 = fwd(mon, toksB[1:2], cbB[1:2])
    check("B10_batched_rows_match_single",
          torch.allclose(lB[0:1], l_on, atol=1e-5, rtol=1e-5)
          and torch.allclose(lB[1:2], l_row1, atol=1e-5, rtol=1e-5),
          "per-sample freqs gather branch agrees with B=1")

    return summarize(results)


def test_base_token_positions_invariants():
    failed = main()
    assert not failed, f"failed invariants: {failed}"


if __name__ == "__main__":
    sys.exit(1 if main() else 0)
