"""Correctness invariants for the untied hyper-encoder (v0.5).

Runs on CPU with a small real Phi3.5-mini-shaped config (n_layers=2, tiny
codebook) so it is fast and CI-friendly; the invariants are dtype/size
independent. fp32, varlen disabled (flash-attn is CUDA-only).

Runnable both as a script (python tests/test_untied_hyper_encoder.py) and
under pytest. Shared mechanics live in tests/_hyper_common.py.

Invariants proven:
  T1  threading: an untied model whose two encoders are identical AND whose
      lm_head equals tok_embeddings reduces EXACTLY to the tied model.
  T2  the output encoder's tensor is actually what reaches the logits: a
      genuinely-untied model differs from forcing tied behavior.
  T3b untied state_dict loads into an untied model with NO missing hyper_output
      keys (the adapter hard-check would not spuriously fire), forward matches.
  T3c documents the one asymmetry: untied weights loaded into a TIED config land
      in UNEXPECTED (silently dropped) — flagged, not silently trusted.
  T4  all-pad codebook (zero-hypertoken / control regime): NEITHER encoder runs,
      logits are base-vocab only, no crash, no output buffer allocated. This is
      the rank-uniformity guarantee for the FSDP/NCCL deadlock class.
  T5  pad codebook rows are masked to -inf in the hyper logits; real rows finite.
  T6  inference (generation) path: the untied output buffer is populated from the
      OUTPUT encoder and equals the training-path output encoding of the same
      entries; tied path leaves no output buffer.
  T8  a real codebook with ZERO hypertokens in the input still puts the INPUT
      encoder in the autograd graph (grads exist, all-zero). This is the FSDP
      rank-invariance guarantee for the pt20B step-95231 deadlock: the gated
      embedding mix orphaned hyper_encoder on a rank whose window had its only
      hypertoken in the labels, so that rank skipped the group's backward
      collectives and NCCL deadlocked.
"""
import copy
import dataclasses
import sys

from _hyper_common import build_model, make_input, fwd, make_harness, summarize

import torch
from zip2zip_core.configs import zip2zip_llama_configs


def small_cfg(untied: bool, share: bool = False):
    base = zip2zip_llama_configs["Phi3.5-mini"]
    return dataclasses.replace(
        base,
        n_layers=2,
        max_codebook_size=8,
        max_subtokens=4,
        tie_hyper_encoder=not untied,
        share_hyper_encoder_weights=share,
        rope=dataclasses.replace(base.rope, max_seq_len=64),
    )


def build(untied: bool, share: bool = False, seed: int = 0):
    cfg = small_cfg(untied, share)
    return build_model(cfg, seed=seed), cfg


def maxd(a, b):
    """Max abs diff over finite positions (hyper logits carry -inf pad rows;
    (-inf)-(-inf)=nan would otherwise pollute the metric). allclose handles
    matching -inf correctly, so this is diagnostic only."""
    d = (a - b).abs()
    d = d[torch.isfinite(d)]
    return float(d.max()) if d.numel() else 0.0


def main():
    results, check = make_harness()

    # ---- T1: untied-equiv reduces to tied ----
    mu, cfg = build(untied=True, seed=1)
    toks, cb = make_input(cfg)
    # copy input encoder weights into output encoder, and make lm_head == tok_emb,
    # so the output role is numerically identical to the tied role.
    mu.hyper_output.load_state_dict(mu.hyper_encoder.state_dict())
    orig_out_w = mu.output.weight.data.clone()
    mu.output.weight.data.copy_(mu.tok_embeddings.weight.data)
    logits_untied_equiv = fwd(mu, toks, cb)
    saved_ho = mu.hyper_output
    mu.hyper_output = None  # behave tied (bmm uses input hyper_embeds)
    logits_tied = fwd(mu, toks, cb)
    mu.hyper_output = saved_ho
    mu.output.weight.data.copy_(orig_out_w)
    check("T1_untied_equiv_reduces_to_tied",
          torch.allclose(logits_untied_equiv, logits_tied, atol=1e-4, rtol=1e-4),
          f"max|d|={maxd(logits_untied_equiv, logits_tied):.2e}")

    # ---- T2: genuinely-untied uses hyper_output in the logits ----
    logits_untied_real = fwd(mu, toks, cb)
    saved_out_sd = copy.deepcopy(mu.hyper_output.state_dict())
    with torch.no_grad():
        for p in mu.hyper_output.parameters():
            p.add_(torch.randn_like(p) * 0.1)
    logits_perturbed = fwd(mu, toks, cb)
    mu.hyper_output.load_state_dict(saved_out_sd)
    # only the hyper-logit columns should differ; compare over FINITE positions of
    # the 3 real entries (both pad rows AND causal-masked early positions are -inf,
    # and (-inf)-(-inf)=nan pollutes a naive max).
    V = cfg.vocab_size
    hd = maxd(logits_untied_real[..., V:V + 3], logits_perturbed[..., V:V + 3])
    check("T2_untied_output_tensor_reaches_logits", hd > 1e-3,
          f"max finite real-hyper-logit |d|={hd:.3e} (must be >0)")

    # ---- T5: pad rows masked to -inf, real rows finite (on the untied logits) ----
    hyper_cols = logits_untied_real[0, :, V:]  # (T, K)
    real_ok = torch.isfinite(hyper_cols[:, :3]).any()
    pad_masked = torch.isinf(hyper_cols[:, 3:]).all()
    check("T5_pad_rows_neg_inf_real_finite", bool(real_ok and pad_masked),
          f"real_finite={bool(real_ok)} pad_all_inf={bool(pad_masked)}")

    # ---- T4: all-pad codebook -> base-only logits, no crash, both encoders skip ----
    mu.reset_inference_cache()
    cb_allpad = torch.full_like(cb, cfg.pad_token_id)
    toks_base = torch.randint(0, V, (1, 16), device=toks.device)
    logits_allpad = fwd(mu, toks_base, cb_allpad)
    check("T4_allpad_base_only_logits", logits_allpad.shape[-1] == V,
          f"shape[-1]={logits_allpad.shape[-1]} == vocab {V}")
    check("T4_allpad_no_output_buffer", mu._hyper_out_embeds_buf is None,
          "(neither encoder ran; rank-uniform skip)")

    # ---- T3b: untied sd loads into fresh untied model, no missing hyper_output ----
    mu2, _ = build(untied=True, seed=2)
    miss, unexp = mu2.load_state_dict(mu.state_dict(), strict=False)
    miss_ho = [k for k in miss if k.startswith("hyper_output")]
    check("T3b_untied_load_no_missing_hyper_output", len(miss_ho) == 0,
          f"missing hyper_output keys={len(miss_ho)} (hard-check would not fire)")
    logits_reload = fwd(mu2, toks, cb)
    check("T3b_reload_forward_matches",
          torch.allclose(logits_reload, logits_untied_real, atol=1e-4, rtol=1e-4),
          f"max|d|={maxd(logits_reload, logits_untied_real):.2e}")

    # ---- T3c: untied weights into a TIED model -> hyper_output.* are UNEXPECTED ----
    mt, _ = build(untied=False, seed=3)
    miss3, unexp3 = mt.load_state_dict(mu.state_dict(), strict=False)
    unexp_ho = [k for k in unexp3 if k.startswith("hyper_output")]
    check("T3c_tied_config_drops_untied_output_as_unexpected", len(unexp_ho) > 0,
          f"unexpected hyper_output keys={len(unexp_ho)} (documented asymmetry; "
          f"not reachable via meta.pt auto-config, only manual tampering)")

    # ---- T6: inference path uses OUTPUT encoder for the output buffer ----
    mu.reset_inference_cache()
    K = cfg.max_codebook_size
    updates = torch.full((1, K, cfg.max_subtokens), cfg.pad_token_id, device=toks.device)
    for i in range(3):
        updates[0, i, 0] = 10 + i
        updates[0, i, 1] = 20 + i
    idx = [[0, 1, 2]]
    with torch.no_grad():
        h, hemb, hout = mu._embed_tokens_inference(toks, updates, idx)
    # the output buffer for allocated rows must equal the OUTPUT encoder's encoding
    with torch.no_grad():
        ref_out = mu._encode_codebook_with_weights(updates, mu.output.weight, encoder=mu.hyper_output)
    check("T6_infer_output_buffer_from_output_encoder",
          torch.allclose(hout[0, :3], ref_out[0, :3], atol=1e-4, rtol=1e-4),
          f"max|d|={float((hout[0,:3]-ref_out[0,:3]).abs().max()):.2e}")
    # and it must DIFFER from the input encoder's encoding (genuinely untied)
    with torch.no_grad():
        ref_in = mu._encode_codebook_with_weights(updates, mu.tok_embeddings.weight, encoder=mu.hyper_encoder)
    check("T6_infer_output_differs_from_input",
          float((hout[0, :3] - ref_in[0, :3]).abs().max()) > 1e-3,
          f"max|d|={float((hout[0,:3]-ref_in[0,:3]).abs().max()):.3e}")

    # tied model allocates no output buffer during inference
    mt.reset_inference_cache()
    with torch.no_grad():
        mt._embed_tokens_inference(toks, updates, idx)
    check("T6_tied_no_output_buffer", mt._hyper_out_embeds_buf is None)

    # ---- T8: zero-hypertoken input + real codebook -> input encoder in graph ----
    # Regression for the gated embedding mix: grads must EXIST for every
    # hyper_encoder parameter (zeros are correct; None means the rank would
    # skip the group's FSDP backward collectives and deadlock).
    mu.reset_inference_cache()
    out8 = mu(toks_base, codebook=cb, hyper_causal_mask=True)
    logits8 = out8[0] if isinstance(out8, tuple) else out8
    mu.zero_grad(set_to_none=True)
    logits8[..., :V].float().sum().backward()
    grads8 = [p.grad for p in mu.hyper_encoder.parameters()]
    check("T8_zero_hypertoken_input_encoder_gets_grad",
          len(grads8) > 0 and all(g is not None for g in grads8),
          f"{sum(g is not None for g in grads8)}/{len(grads8)} params got grad")
    mu.zero_grad(set_to_none=True)

    # ---- T7: shared-weight untied mode has no hyper_output but re-encodes output role ----
    ms, cfgs = build(untied=True, share=True, seed=4)
    toks_s, cb_s = make_input(cfgs)
    logits_shared = fwd(ms, toks_s, cb_s)
    check("T7_shared_weight_no_hyper_output_module", ms.hyper_output is None)
    check("T7_shared_weight_no_hyper_output_state", not any(
          k.startswith("hyper_output") for k in ms.state_dict()))
    old_tie = ms.zip2zip_config.tie_hyper_encoder
    ms.zip2zip_config.tie_hyper_encoder = True
    logits_reuse = fwd(ms, toks_s, cb_s)
    ms.zip2zip_config.tie_hyper_encoder = old_tie
    hd_shared = maxd(logits_shared[..., cfgs.vocab_size:cfgs.vocab_size + 3],
                     logits_reuse[..., cfgs.vocab_size:cfgs.vocab_size + 3])
    check("T7_shared_weight_reencodes_output_role", hd_shared > 1e-3,
          f"max finite real-hyper-logit |d|={hd_shared:.3e}")

    return summarize(results)


def test_untied_hyper_encoder_invariants():
    failed = main()
    assert not failed, f"failed invariants: {failed}"


if __name__ == "__main__":
    sys.exit(1 if main() else 0)
