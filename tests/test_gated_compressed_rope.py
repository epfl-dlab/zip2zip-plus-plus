"""Correctness invariants for the v0.7.1 gated compressed-coordinate RoPE.

Runnable directly and under pytest. The shrunken model exercises the real
decoder path without downloading weights or data.
"""

import dataclasses
import os
import sys

import torch

from _hyper_common import build_model, make_harness, summarize
from zip2zip_core.configs import zip2zip_llama_configs
import zip2zip_core.model as model_mod
from zip2zip_core.model import build_gated_rope_inputs


T = 10
MS = 4


def small_cfg(*, gated: bool, start: int = 2, two_axis: bool = False):
    base = zip2zip_llama_configs["Phi3.5-mini"]
    return dataclasses.replace(
        base,
        dim=128,
        n_layers=4,
        vocab_size=128,
        pad_token_id=0,
        max_codebook_size=6,
        max_subtokens=MS,
        tie_hyper_encoder=True,
        base_token_positions=True,
        two_axis_rope=two_axis,
        gated_compressed_rope=gated,
        gated_rope_start_layer=start,
        zero_init_encoder_output=False,
        token_type_loss_weight=0.0,
        encoder_dim=64,
        encoder_n_layers=1,
        encoder_n_heads=4,
        encoder_intermediate_size=128,
        tok_embeddings=dataclasses.replace(
            base.tok_embeddings, init_std=128**-0.5
        ),
        layer=dataclasses.replace(
            base.layer,
            feed_forward=dataclasses.replace(
                base.layer.feed_forward, hidden_dim=256
            ),
            attention=dataclasses.replace(
                base.layer.attention, n_heads=4, n_kv_heads=4
            ),
        ),
        rope=dataclasses.replace(base.rope, dim=32, max_seq_len=T * MS),
    )


def spanned_input(cfg, device):
    torch.manual_seed(9)
    tokens = torch.randint(1, cfg.vocab_size, (1, T), device=device)
    tokens[0, 2] = cfg.vocab_size
    tokens[0, 6] = cfg.vocab_size + 1
    codebook = torch.full(
        (1, cfg.max_codebook_size, cfg.max_subtokens),
        cfg.pad_token_id,
        device=device,
    )
    codebook[0, 0, :3] = torch.tensor([11, 12, 13], device=device)
    codebook[0, 1, :2] = torch.tensor([21, 22], device=device)
    return tokens, codebook


def logits(model, tokens, codebook):
    out = model(tokens, codebook=codebook, hyper_causal_mask=True)
    return out[0] if isinstance(out, tuple) else out


def raises(fn, text):
    try:
        fn()
    except (ValueError, RuntimeError) as exc:
        return text in str(exc)
    return False


def main():
    results, check = make_harness()

    base_cfg = zip2zip_llama_configs["Phi3.5-mini"]
    check(
        "GR1_default_off",
        base_cfg.gated_compressed_rope is False,
        "legacy configs do not add a gate parameter",
    )
    check(
        "GR1_mutually_exclusive_with_two_axis",
        raises(
            lambda: small_cfg(gated=True, two_axis=True).build(),
            "mutually exclusive",
        ),
        "v0.7.1 branches from v0.6.4, not the rejected v0.7 geometry",
    )
    check(
        "GR1_start_layer_bounds",
        raises(lambda: small_cfg(gated=True, start=4).build(), "must be in"),
        "an empty gated layer range fails before training",
    )

    gated = build_model(small_cfg(gated=True), seed=3)
    plain = build_model(small_cfg(gated=False), seed=3)
    device = next(gated.parameters()).device
    tokens, codebook = spanned_input(gated.zip2zip_config, device)
    gate = gated.compressed_rope_gate

    check(
        "GR2_phi_shape_and_zero_init",
        gate.shape == (2, 16) and torch.count_nonzero(gate).item() == 0,
        f"small gate shape={tuple(gate.shape)}; Phi production shape is (16, 48)",
    )
    plain_state = plain.state_dict()
    gated_state = gated.state_dict()
    shared_keys = set(gated_state) - {"compressed_rope_gate"}
    check(
        "GR2_gate_consumes_no_init_rng",
        shared_keys == set(plain_state)
        and all(torch.equal(gated_state[k], plain_state[k]) for k in shared_keys),
        "all pre-existing parameters initialize identically",
    )

    positions = gated._base_token_positions(tokens, codebook)
    zero_cache, lookup = build_gated_rope_inputs(
        gated.freqs_cis,
        positions,
        torch.zeros(gated.freqs_cis.shape[-1], device=device),
        batch_size=1,
        seq_len=T,
    )
    one_cache, one_lookup = build_gated_rope_inputs(
        gated.freqs_cis,
        positions,
        torch.ones(gated.freqs_cis.shape[-1], device=device),
        batch_size=1,
        seq_len=T,
    )
    check(
        "GR3_zero_is_exact_base_geometry",
        torch.equal(zero_cache[lookup], gated.freqs_cis[positions]),
        "gate=0 preserves every pretrained base-stream complex pair",
    )
    compressed = gated.freqs_cis[:T].unsqueeze(0)
    check(
        "GR3_one_reaches_compressed_geometry",
        torch.allclose(
            one_cache[one_lookup], compressed, atol=2e-6, rtol=2e-6
        ),
        "gate=1 reconstructs compressed-index RoPE",
    )

    with torch.no_grad():
        plain_logits = logits(plain, tokens, codebook).float()
        gated_logits = logits(gated, tokens, codebook).float()
    check(
        "GR4_zero_gate_matches_v064_forward",
        torch.allclose(gated_logits, plain_logits, atol=2e-5, rtol=2e-5),
        f"max|delta|={float((gated_logits - plain_logits).abs().max()):.3e}",
    )

    calls = []
    original = model_mod.build_gated_rope_inputs

    def recording_helper(freqs_cis, base_positions, row, **kwargs):
        calls.append(row.detach().clone())
        return original(freqs_cis, base_positions, row, **kwargs)

    model_mod.build_gated_rope_inputs = recording_helper
    try:
        with torch.no_grad():
            logits(gated, tokens, codebook)
    finally:
        model_mod.build_gated_rope_inputs = original
    check(
        "GR5_only_upper_layers_build_gated_cache",
        len(calls) == 2,
        f"called once for each of layers 2-3: {len(calls)} calls",
    )

    gated.train()
    gated.zero_grad(set_to_none=True)
    loss = logits(gated, tokens, codebook)[..., : gated.zip2zip_config.vocab_size].square().mean()
    loss.backward()
    grad = gated.compressed_rope_gate.grad
    check(
        "GR6_gate_gets_finite_nonzero_gradient",
        grad is not None
        and torch.isfinite(grad).all()
        and torch.count_nonzero(grad).item() > 0,
        "the compressed-coordinate delta is trainable from the real LM path",
    )

    # A compressed sample can have its only hypertoken in y=compressed[1:],
    # leaving x all-base while its codebook is still non-empty. The gate must
    # remain in that rank's final-microstep graph so FSDP gradient sync cannot
    # diverge across ranks. Base positions equal arange here, so the gradient is
    # correctly zero-valued but must not be None.
    label_only_tokens = torch.randint(
        1, gated.zip2zip_config.vocab_size, (1, T), device=device
    )
    label_only_codebook = codebook.clone()
    gated.zero_grad(set_to_none=True)
    before = gated.gated_rope_forwards
    label_only_loss = logits(
        gated, label_only_tokens, label_only_codebook
    ).square().mean()
    label_only_loss.backward()
    label_only_grad = gated.compressed_rope_gate.grad
    check(
        "GR7_label_only_hypertoken_keeps_gate_in_sync_graph",
        gated.gated_rope_forwards == before + 1
        and label_only_grad is not None
        and torch.isfinite(label_only_grad).all(),
        "non-empty compressed codebook activates the gate even when x is all base",
    )

    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    train_source = open(
        os.path.join(repo, "src", "zip2zip_core", "train.py")
    ).read()
    check(
        "GR8_lora_freeze_keeps_gate_trainable",
        'name != "compressed_rope_gate"' in train_source,
        "the v0.7.1 gate survives the recipe's decoder freeze/LoRA setup",
    )

    return summarize(results)


def test_gated_compressed_rope_invariants():
    failed = main()
    assert not failed, f"failed invariants: {failed}"


if __name__ == "__main__":
    sys.exit(1 if main() else 0)
