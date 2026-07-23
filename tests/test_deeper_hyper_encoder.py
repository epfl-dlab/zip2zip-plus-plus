"""Correctness invariants for the deeper hyper-encoder recipe (v0.6.1).

v0.6.1 = the v0.5 canonical recipe (untied + digit-protected) with both flat
hyper-encoders deepened from 2 to 4 layers via ENCODER_N_LAYERS=4
(--encoder_n_layers). These tests pin the plumbing that makes that a
single-env-var change: config -> model depth, state-dict integrity, and the
meta.pt contracts the eval adapter and resume guard rely on.

Runs on CPU with a FULLY SHRUNKEN config (decoder dim 256 / vocab 512 /
2 layers, encoder dim 128) — the depth invariants are size-independent and
this keeps every build MB-scale. fp32, varlen disabled.

Runnable both as a script (python tests/test_deeper_hyper_encoder.py) and
under pytest. Shared mechanics live in tests/_hyper_common.py.

Invariants proven:
  D1  --encoder_n_layers reaches BOTH untied encoders: a 4-layer config builds
      4 layers in hyper_encoder and hyper_output; the config default stays 2.
  D2  the 4-layer untied model forwards correctly: real hyper-logit columns
      finite, pad rows -inf, all-pad codebook still skips both encoders
      (the FSDP rank-uniformity guarantee is depth-independent).
  D3  4-layer untied state_dict roundtrips strict=True into a fresh 4-layer
      model and the forward matches (checkpoint save/load holds at depth 4).
  D3b a depth mismatch is visible in plain nn.Module strict=False loading
      (unexpected layer keys), never silently absorbed. (The training resume
      path itself uses set_model_state_dict, strict by default — the guard for
      that path is pinned behaviorally in D5.)
  D4  parameter accounting: doubling depth exactly doubles the per-layer
      parameter block of each encoder (no hidden sharing across layers).
  D5  validate_resume_args behavior: a recorded-vs-requested encoder depth
      mismatch hard-errors; a legacy meta recording None for encoder args is
      exempt; the None exemption does NOT extend to non-encoder hard keys.
"""
import dataclasses
import os
import sys
import tempfile
import types

from _hyper_common import build_model, make_input, fwd, make_harness, summarize

import torch
from zip2zip_core.configs import zip2zip_llama_configs


def small_cfg(n_layers_enc: int):
    base = zip2zip_llama_configs["Phi3.5-mini"]
    return dataclasses.replace(
        base,
        dim=256,
        n_layers=2,
        vocab_size=512,
        pad_token_id=0,
        max_codebook_size=8,
        max_subtokens=4,
        tie_hyper_encoder=False,
        encoder_dim=128,
        encoder_n_layers=n_layers_enc,
        encoder_n_heads=4,
        encoder_intermediate_size=256,
        tok_embeddings=dataclasses.replace(base.tok_embeddings, init_std=256 ** -0.5),
        layer=dataclasses.replace(
            base.layer,
            feed_forward=dataclasses.replace(base.layer.feed_forward, hidden_dim=512),
            attention=dataclasses.replace(base.layer.attention, n_heads=8, n_kv_heads=8),
        ),
        rope=dataclasses.replace(base.rope, dim=256 // 8, max_seq_len=64),
    )


def build(n_layers_enc: int, seed: int = 0):
    cfg = small_cfg(n_layers_enc)
    return build_model(cfg, seed=seed), cfg


def layer_param_count(encoder):
    return sum(p.numel() for p in encoder.layers.parameters())


def _resume_args(**over):
    """An args namespace covering every key validate_resume_args touches."""
    base = dict(
        disable_digit_ids=True, max_codebook_size=4096, tokenizer="microsoft/Phi-3.5",
        untied_hyper_encoder=True, base_token_positions=False,
        encoder_dim=3072, encoder_n_layers=4,
        encoder_n_heads=32, encoder_intermediate_size=12288,
        max_subtokens=4, seq_len=2048, data_dir="/data", warmstart_steps=0,
        allow_resume_mismatch=False,
    )
    base.update(over)
    return types.SimpleNamespace(**base)


def _guard_raises(train_mod, tmpdir, meta_args, args):
    torch.save({"args": meta_args}, os.path.join(tmpdir, "meta.pt"))
    try:
        train_mod.validate_resume_args(tmpdir, args)
        return False
    except ValueError:
        return True


def main():
    results, check = make_harness()

    # ---- D1: depth plumbs to BOTH encoders; default stays 2 ----
    m4, cfg4 = build(n_layers_enc=4, seed=1)
    check("D1_depth4_input_encoder", len(m4.hyper_encoder.layers) == 4,
          f"hyper_encoder layers={len(m4.hyper_encoder.layers)}")
    check("D1_depth4_output_encoder", len(m4.hyper_output.layers) == 4,
          f"hyper_output layers={len(m4.hyper_output.layers)}")
    base = zip2zip_llama_configs["Phi3.5-mini"]
    check("D1_config_default_is_2", base.encoder_n_layers == 2,
          f"Phi3.5-mini default={base.encoder_n_layers} (v0.5 unchanged)")

    # ---- D2: 4-layer untied forward is well-formed ----
    toks, cb = make_input(cfg4)
    logits = fwd(m4, toks, cb)
    V = cfg4.vocab_size
    hyper_cols = logits[0, :, V:]
    check("D2_real_hyper_logits_finite", bool(torch.isfinite(hyper_cols[:, :3]).any()),
          "real entries produce finite hyper logits at depth 4")
    check("D2_pad_rows_neg_inf", bool(torch.isinf(hyper_cols[:, 3:]).all()),
          "pad codebook rows stay -inf at depth 4")
    m4.reset_inference_cache()
    cb_allpad = torch.full_like(cb, cfg4.pad_token_id)
    toks_base = torch.randint(0, V, (1, 16), device=toks.device)
    logits_allpad = fwd(m4, toks_base, cb_allpad)
    check("D2_allpad_base_only", logits_allpad.shape[-1] == V,
          "all-pad codebook skips both 4-layer encoders (rank-uniform)")

    # ---- D3: strict state-dict roundtrip at depth 4 ----
    m4b, _ = build(n_layers_enc=4, seed=2)
    m4b.load_state_dict(m4.state_dict(), strict=True)
    logits_reload = fwd(m4b, toks, cb)
    check("D3_strict_roundtrip_forward_matches",
          torch.allclose(logits_reload, logits, atol=1e-4, rtol=1e-4),
          "strict load at depth 4, forward identical")
    del m4b, logits_reload

    # ---- D3b: depth mismatch is visible, never silent ----
    m2, _ = build(n_layers_enc=2, seed=3)
    miss, unexp = m2.load_state_dict(m4.state_dict(), strict=False)
    deep_keys = [k for k in unexp
                 if ".layers.2." in k or ".layers.3." in k]
    check("D3b_depth_mismatch_leaves_unexpected_keys", len(deep_keys) > 0,
          f"{len(deep_keys)} layer-2/3 keys unexpected in a 2-layer model")

    # ---- D4: depth doubling exactly doubles per-encoder layer params ----
    p2_in = layer_param_count(m2.hyper_encoder)
    p4_in = layer_param_count(m4.hyper_encoder)
    p4_out = layer_param_count(m4.hyper_output)
    check("D4_layers_params_double", p4_in == 2 * p2_in,
          f"layers params 2L={p2_in:,} 4L={p4_in:,}")
    check("D4_both_encoders_same_size", p4_in == p4_out,
          "untied pair stays symmetric at depth 4")

    # ---- D5: resume guard behavior (validate_resume_args, real code path) ----
    import zip2zip_core.train as train_mod
    saved_dist = train_mod.dist
    train_mod.dist = types.SimpleNamespace(get_rank=lambda: 0)
    try:
        with tempfile.TemporaryDirectory() as td:
            args = _resume_args()
            recorded = dict(vars(args))
            del recorded["allow_resume_mismatch"]

            check("D5a_depth_mismatch_hard_errors",
                  _guard_raises(train_mod, td, {**recorded, "encoder_n_layers": 2}, args),
                  "recorded depth 2 vs requested 4 raises ValueError")
            check("D5b_legacy_none_encoder_args_exempt",
                  not _guard_raises(train_mod, td, {**recorded, "encoder_n_layers": None,
                                                    "encoder_dim": None}, args),
                  "meta.pt predating effective-value recording resumes fine")
            check("D5c_none_exemption_scoped_to_encoder_keys",
                  _guard_raises(train_mod, td, {**recorded, "tokenizer": None}, args),
                  "a recorded None tokenizer is still a hard mismatch")
            check("D5d_matching_args_pass",
                  not _guard_raises(train_mod, td, recorded, args),
                  "identical recorded args validate cleanly")
    finally:
        train_mod.dist = saved_dist

    return summarize(results)


def test_deeper_hyper_encoder_invariants():
    failed = main()
    assert not failed, f"failed invariants: {failed}"


if __name__ == "__main__":
    sys.exit(1 if main() else 0)
