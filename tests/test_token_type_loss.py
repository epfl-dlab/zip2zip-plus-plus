"""Correctness invariants for the token-type auxiliary loss (v0.6.3).

v0.6.3 = the v0.6.2 recipe (untied + digit-protected + base-token positions)
plus TOKEN_TYPE_LOSS_WEIGHT=w: a head on the final hidden state predicts
whether the NEXT token is a hypertoken (BCE, masked like the LM loss, added
to the backward loss with weight w). The machinery predates this recipe but
has never been trained with; these tests pin it before the first real run.

Runs on CPU with the fully shrunken config. Runnable as a script and under
pytest. Shared mechanics live in tests/_hyper_common.py.

Invariants proven:
  TT1 the config default is 0.0: no head, forward returns a plain tensor —
      v0.6.2 checkpoints and behavior untouched.
  TT2 weight>0 builds the head and forward returns (logits, type_logits)
      with type_logits shaped (B, T) and finite.
  TT3 the train.py loss composition is well-formed on this model: BCE of
      type_logits vs (labels >= vocab_size) under the LM valid mask is
      finite, and backward through lm_loss + w*type_loss puts nonzero grads
      on BOTH the head and the hyper-encoder.
  TT4 state dict: head-ful checkpoints roundtrip strict=True; loading one
      into a head-less model leaves loud unexpected token_type_head keys.
  TT5 resume guard: token_type_loss_weight is a hard key; a legacy meta.pt
      without it counts as 0.0 (requesting 0.05 on such a lineage raises);
      mismatches raise in both directions; matching args pass.
  TT6 composition with base_token_positions: both features on -> tuple
      return AND the logits exactly equal a positions-off model fed the
      position helper's output explicitly (features do not interfere).
  TT7 the head's parameter names fall in the hyper optimizer group / stay
      excluded from decoder freezing (same name predicates train.py uses).
"""
import dataclasses
import os
import sys
import tempfile
import types

from _hyper_common import build_model, fwd, make_harness, summarize

import torch
import torch.nn.functional as F
from zip2zip_core.configs import zip2zip_llama_configs

MS = 4
T = 16


def small_cfg(tt_weight: float, base_positions: bool = True):
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
        token_type_loss_weight=tt_weight,
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
        rope=dataclasses.replace(base.rope, dim=256 // 8, max_seq_len=T * MS),
    )


def build(tt_weight: float, base_positions: bool = True, seed: int = 0):
    cfg = small_cfg(tt_weight, base_positions)
    return build_model(cfg, seed=seed), cfg


def spanned_input(cfg, dev):
    V = cfg.vocab_size
    pad = cfg.pad_token_id
    K = cfg.max_codebook_size
    torch.manual_seed(7)
    toks = torch.randint(1, V, (1, T), device=dev)
    toks[0, 2] = V + 0
    toks[0, 4] = V + 1
    cb = torch.full((1, K, cfg.max_subtokens), pad, device=dev)
    cb[0, 0, :3] = torch.tensor([11, 12, 13], device=dev)
    cb[0, 1, :2] = torch.tensor([21, 22], device=dev)
    return toks, cb


def main():
    results, check = make_harness()

    # ---- TT1: default off, plain return ----
    base = zip2zip_llama_configs["Phi3.5-mini"]
    check("TT1_config_default_zero", base.token_type_loss_weight == 0.0,
          "code default keeps v0.6.2 semantics")
    m0, cfg0 = build(tt_weight=0.0, seed=1)
    dev = next(m0.parameters()).device
    toks, cb = spanned_input(cfg0, dev)
    check("TT1_no_head_when_zero", m0.token_type_head is None,
          "weight 0 builds no head")
    with torch.no_grad():
        out0 = m0(toks, codebook=cb, hyper_causal_mask=True)
    check("TT1_plain_tensor_return", not isinstance(out0, tuple),
          "no tuple without the head")

    # ---- TT2: weight>0 builds head, tuple return, well-formed ----
    mh, cfgh = build(tt_weight=0.05, seed=1)
    check("TT2_head_built", mh.token_type_head is not None, "head exists")
    with torch.no_grad():
        outh = mh(toks, codebook=cb, hyper_causal_mask=True)
    check("TT2_tuple_return", isinstance(outh, tuple) and len(outh) == 2,
          "(logits, type_logits)")
    logits_h, type_logits = outh
    check("TT2_type_logits_shape_finite",
          tuple(type_logits.shape) == (1, T) and bool(torch.isfinite(type_logits).all()),
          f"shape {tuple(type_logits.shape)}, finite")

    # ---- TT3: train.py loss composition backward ----
    V = cfgh.vocab_size
    mh.train()
    out = mh(toks, codebook=cb, hyper_causal_mask=True)
    lg, tt = out
    # labels: next compressed token, with two masked (-100) positions like SFT
    labels = torch.roll(toks, shifts=-1, dims=1).clone()
    labels[0, -1] = -100
    labels[0, 0] = -100
    flat_logits = lg.flatten(0, 1).float()
    flat_labels = labels.flatten(0, 1)
    per_token = F.cross_entropy(flat_logits, flat_labels, reduction="none",
                                ignore_index=-100)
    valid_mask = flat_labels != -100
    lm_loss = per_token.sum() / valid_mask.sum()
    type_targets = (labels >= V).float().flatten(0, 1)[valid_mask]
    type_logits_v = tt.flatten(0, 1).float()[valid_mask]
    type_loss = F.binary_cross_entropy_with_logits(type_logits_v, type_targets,
                                                   reduction="mean")
    total = lm_loss + 0.05 * type_loss
    check("TT3_losses_finite",
          bool(torch.isfinite(lm_loss)) and bool(torch.isfinite(type_loss)),
          f"lm={float(lm_loss):.3f} type={float(type_loss):.3f}")
    total.backward()
    head_grad = max(float(p.grad.abs().max()) for p in mh.token_type_head.parameters())
    henc_grad = max(float(p.grad.abs().max()) for p in mh.hyper_encoder.parameters()
                    if p.grad is not None)
    check("TT3_head_receives_grads", head_grad > 0, f"max|g|={head_grad:.2e}")
    check("TT3_hyper_encoder_receives_grads", henc_grad > 0, f"max|g|={henc_grad:.2e}")
    mh.zero_grad(set_to_none=True)
    mh.eval()

    # ---- TT4: state dict with head ----
    mh2, _ = build(tt_weight=0.05, seed=2)
    mh2.load_state_dict(mh.state_dict(), strict=True)
    with torch.no_grad():
        reload_logits = mh2(toks, codebook=cb, hyper_causal_mask=True)[0]
    check("TT4_strict_roundtrip",
          torch.allclose(reload_logits.float()[..., :V], logits_h.float()[..., :V],
                         atol=1e-5, rtol=1e-5),
          "head-ful checkpoint loads strict and matches")
    miss, unexp = m0.load_state_dict(mh.state_dict(), strict=False)
    tt_keys = [k for k in unexp if k.startswith("token_type_head")]
    check("TT4_headless_load_is_loud", len(tt_keys) > 0,
          f"{len(tt_keys)} token_type_head keys unexpected in a head-less model")

    # ---- TT5: resume guard hard key + legacy absence = 0.0 ----
    import zip2zip_core.train as train_mod
    saved_dist = train_mod.dist
    train_mod.dist = types.SimpleNamespace(get_rank=lambda: 0)
    try:
        with tempfile.TemporaryDirectory() as td:
            def make_args(**over):
                a = dict(
                    disable_digit_ids=True, max_codebook_size=4096,
                    tokenizer="microsoft/Phi-3.5", untied_hyper_encoder=True,
                    base_token_positions=True, token_type_loss_weight=0.05,
                    encoder_dim=3072, encoder_n_layers=2, encoder_n_heads=32,
                    encoder_intermediate_size=12288, max_subtokens=4,
                    seq_len=2048, data_dir="/data", warmstart_steps=0,
                    allow_resume_mismatch=False,
                )
                a.update(over)
                return types.SimpleNamespace(**a)

            def guard_raises(meta_args, args):
                torch.save({"args": meta_args}, os.path.join(td, "meta.pt"))
                try:
                    train_mod.validate_resume_args(td, args)
                    return False
                except ValueError:
                    return True

            args_on = make_args()
            rec = dict(vars(args_on))
            del rec["allow_resume_mismatch"]
            legacy = {k: v for k, v in rec.items() if k != "token_type_loss_weight"}

            check("TT5_weight_mismatch_hard_errors",
                  guard_raises({**rec, "token_type_loss_weight": 0.0}, args_on),
                  "recorded 0.0 vs requested 0.05 raises")
            check("TT5_reverse_direction_hard_errors",
                  guard_raises(rec, make_args(token_type_loss_weight=0.0)),
                  "recorded 0.05 vs requested 0.0 raises")
            check("TT5_legacy_meta_counts_as_zero",
                  guard_raises(legacy, args_on),
                  "pre-flag meta.pt + weight 0.05 raises (absence = 0)")
            check("TT5_matching_args_pass",
                  not guard_raises(rec, args_on),
                  "identical recorded args validate cleanly")
    finally:
        train_mod.dist = saved_dist

    # ---- TT6: composition with base_token_positions ----
    pos = mh._base_token_positions(toks, cb)
    m_nopos, _ = build(tt_weight=0.05, base_positions=False, seed=1)
    with torch.no_grad():
        out_explicit = m_nopos(toks, codebook=cb, hyper_causal_mask=True,
                               positions=pos)
    check("TT6_composes_with_base_positions",
          torch.allclose(logits_h.float()[..., :V],
                         out_explicit[0].float()[..., :V], atol=1e-5, rtol=1e-5),
          "head + positions == head + explicit helper positions")

    # ---- TT7: optimizer-group / freeze name predicates ----
    head_names = [n for n, _ in mh.named_parameters() if "token_type_head" in n]
    in_hyper_group = all(
        ("hyper_encoder" in n or "hyper_output" in n or "token_type_head" in n)
        for n in head_names
    )
    excluded_from_freeze = all(n.startswith("token_type_head") for n in head_names)
    check("TT7_head_in_hyper_group_and_unfrozen",
          bool(head_names) and in_hyper_group and excluded_from_freeze,
          f"{len(head_names)} head params routed correctly")

    return summarize(results)


def test_token_type_loss_invariants():
    failed = main()
    assert not failed, f"failed invariants: {failed}"


if __name__ == "__main__":
    sys.exit(1 if main() else 0)
