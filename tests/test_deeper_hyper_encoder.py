"""Correctness invariants for the deeper hyper-encoder recipe (v0.6.1).

v0.6.1 = the v0.5 canonical recipe (untied + digit-protected) with both flat
hyper-encoders deepened from 2 to 4 layers via ENCODER_N_LAYERS=4
(--encoder_n_layers). These tests pin the plumbing that makes that a
single-env-var change: config -> model depth, state-dict integrity, and the
meta.pt contracts the eval adapter and resume guard rely on.

Runs on CPU with a small Phi3.5-mini-shaped config (2 decoder layers, tiny
codebook) and a SHRUNKEN encoder (dim 256) so depth-doubling stays fast;
depth invariants are size-independent. fp32, varlen disabled.

Runnable both as a script (python tests/test_deeper_hyper_encoder.py) and
under pytest (collection imports the module without executing the checks).

Invariants proven:
  D1  --encoder_n_layers reaches BOTH untied encoders: a 4-layer config builds
      4 layers in hyper_encoder and hyper_output; the config default stays 2.
  D2  the 4-layer untied model forwards correctly: real hyper-logit columns
      finite, pad rows -inf, all-pad codebook still skips both encoders
      (the FSDP rank-uniformity guarantee is depth-independent).
  D3  4-layer untied state_dict roundtrips strict=True into a fresh 4-layer
      model and the forward matches (checkpoint save/load holds at depth 4).
  D3b a depth mismatch cannot load silently: 4-layer weights into a 2-layer
      model leave unexpected layer keys (the backstop behind the resume guard).
  D4  parameter accounting: doubling depth exactly doubles the per-layer
      parameter block of each encoder (no hidden sharing across layers).
  D5  source contracts: the eval adapter's meta.pt override list and train.py's
      resume hard-guard both still name encoder_n_layers.
"""
import dataclasses
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ext", "torchtitan"))

import torch
from zip2zip_core.configs import zip2zip_llama_configs

DEV = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)


def small_cfg(n_layers_enc: int):
    base = zip2zip_llama_configs["Phi3.5-mini"]
    return dataclasses.replace(
        base,
        n_layers=2,
        max_codebook_size=8,
        max_subtokens=4,
        tie_hyper_encoder=False,
        encoder_dim=256,
        encoder_n_heads=8,
        encoder_intermediate_size=512,
        encoder_n_layers=n_layers_enc,
        rope=dataclasses.replace(base.rope, max_seq_len=64),
    )


def build(n_layers_enc: int, seed: int = 0):
    torch.manual_seed(seed)
    cfg = small_cfg(n_layers_enc)
    model = cfg.build()
    with torch.no_grad():
        model.init_weights()
    model = model.to(device=DEV, dtype=torch.float32).eval()
    model.hyper_encoder.disable_varlen = True
    model.hyper_output.disable_varlen = True
    return model, cfg


def make_input(cfg, n_hyper=3, T=16):
    V = cfg.vocab_size
    torch.manual_seed(42)
    toks = torch.randint(0, V, (1, T), device=DEV)
    for i in range(n_hyper):
        toks[0, 2 + i] = V + i
    pad = cfg.pad_token_id
    K = cfg.max_codebook_size
    cb = torch.full((1, K, cfg.max_subtokens), pad, device=DEV)
    for i in range(n_hyper):
        cb[0, i, 0] = 10 + i
        cb[0, i, 1] = 20 + i
    return toks, cb


def fwd(model, toks, cb):
    with torch.no_grad():
        out = model(toks, codebook=cb, hyper_causal_mask=True)
    return (out[0] if isinstance(out, tuple) else out).float()


def layer_param_count(encoder):
    return sum(p.numel() for p in encoder.layers.parameters())


results = {}

def check(name, cond, detail=""):
    results[name] = bool(cond)
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")


def main():
    results.clear()

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
    toks_base = torch.randint(0, V, (1, 16), device=DEV)
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

    # ---- D5: source contracts (eval adapter override + resume hard-guard) ----
    root = os.path.join(os.path.dirname(__file__), "..", "src", "zip2zip_core")
    with open(os.path.join(root, "lm_eval_adapter.py")) as f:
        adapter_src = f.read()
    with open(os.path.join(root, "train.py")) as f:
        train_src = f.read()
    check("D5_adapter_overrides_encoder_n_layers",
          '"encoder_n_layers"' in adapter_src,
          "eval adapter rebuilds depth from meta.pt")
    hard_block = train_src.split("hard = (", 1)[-1].split(")", 1)[0]
    check("D5_resume_hard_guard_has_encoder_n_layers",
          "encoder_n_layers" in hard_block,
          "resume depth mismatch is a hard error")

    print("\n==== SUMMARY ====")
    n_pass = sum(results.values())
    print(f"{n_pass}/{len(results)} passed")
    for k, v in results.items():
        if not v:
            print(f"  FAILED: {k}")
    return 0 if n_pass == len(results) else 1


def test_deeper_hyper_encoder_invariants():
    assert main() == 0


if __name__ == "__main__":
    sys.exit(main())
