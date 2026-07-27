"""Correctness invariants for the v0.7.1 gated compressed-coordinate RoPE.

Runnable directly and under pytest. The shrunken model exercises the real
decoder path without downloading weights or data.
"""

import dataclasses
import importlib
import importlib.util
import os
import sys
import tempfile
import types

import torch

from _hyper_common import build_model, make_harness, summarize
from zip2zip_core.checkpoint import resolve_gated_rope_start_pair
from zip2zip_core.configs import zip2zip_llama_configs
import zip2zip_core.model as model_mod
from zip2zip_core.model import build_gated_rope_inputs


T = 10
MS = 4


def small_cfg(
    *,
    gated: bool,
    layer_start: int = 0,
    pair_start: int = 10,
    two_axis: bool = False,
):
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
        gated_rope_start_layer=layer_start,
        gated_rope_start_pair=pair_start,
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


def raw_fwd(model, tokens, codebook=None, **kwargs):
    with torch.no_grad():
        out = model(
            tokens,
            codebook=codebook,
            hyper_causal_mask=True,
            **kwargs,
        )
    return (out[0] if isinstance(out, tuple) else out).float()


def import_eval_adapter():
    """Import the real adapter, stubbing only absent lm-eval registration APIs."""
    created = []
    try:
        return importlib.import_module("zip2zip_core.lm_eval_adapter"), created
    except ModuleNotFoundError as exc:
        if exc.name != "lm_eval":
            raise

    lm_eval = types.ModuleType("lm_eval")
    lm_eval.__path__ = []
    api = types.ModuleType("lm_eval.api")
    api.__path__ = []
    api_model = types.ModuleType("lm_eval.api.model")
    api_registry = types.ModuleType("lm_eval.api.registry")
    utils = types.ModuleType("lm_eval.utils")
    api_model.LM = object
    api_registry.register_model = lambda _name: lambda cls: cls
    utils.get_rolling_token_windows = lambda *args, **kwargs: ()
    utils.make_disjoint_window = lambda window: window
    stubs = {
        "lm_eval": lm_eval,
        "lm_eval.api": api,
        "lm_eval.api.model": api_model,
        "lm_eval.api.registry": api_registry,
        "lm_eval.utils": utils,
    }
    for name, module in stubs.items():
        sys.modules[name] = module
        created.append(name)
    return importlib.import_module("zip2zip_core.lm_eval_adapter"), created


def main():
    results, check = make_harness()

    base_cfg = zip2zip_llama_configs["Phi3.5-mini"]
    check(
        "GR1_default_off",
        base_cfg.gated_compressed_rope is False
        and base_cfg.gated_rope_start_pair == 0,
        "legacy configs add no parameter; manual gating is generic/full-width",
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
        raises(
            lambda: small_cfg(gated=True, layer_start=4).build(),
            "must be in",
        ),
        "an empty gated layer range fails before training",
    )
    check(
        "GR1_start_pair_bounds",
        raises(
            lambda: small_cfg(gated=True, pair_start=16).build(),
            "must be in",
        ),
        "an empty gated frequency suffix fails before training",
    )

    gated = build_model(small_cfg(gated=True), seed=3)
    plain = build_model(small_cfg(gated=False), seed=3)
    device = next(gated.parameters()).device
    tokens, codebook = spanned_input(gated.zip2zip_config, device)
    gate = gated.compressed_rope_gate

    check(
        "GR2_phi_shape_and_zero_init",
        gate.shape == (4, 6)
        and torch.count_nonzero(gate).item() == 0
        and (
            (base_cfg.n_layers - base_cfg.gated_rope_start_layer)
            * (base_cfg.rope.dim // 2 - 32)
            == 512
        ),
        f"small gate shape={tuple(gate.shape)}; Phi production shape is (32, 16)",
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
        torch.zeros(gate.shape[-1], device=device),
        start_pair=10,
        batch_size=1,
        seq_len=T,
    )
    one_cache, one_lookup = build_gated_rope_inputs(
        gated.freqs_cis,
        positions,
        torch.ones(gate.shape[-1], device=device),
        start_pair=10,
        batch_size=1,
        seq_len=T,
    )
    check(
        "GR3_zero_is_exact_base_geometry",
        torch.equal(zero_cache[lookup], gated.freqs_cis[positions]),
        "gate=0 preserves every pretrained base-stream complex pair",
    )
    base = gated.freqs_cis[positions]
    compressed = gated.freqs_cis[:T].unsqueeze(0)
    check(
        "GR3_one_only_moves_low_frequency_suffix",
        torch.equal(one_cache[one_lookup][..., :10], base[..., :10])
        and torch.allclose(
            one_cache[one_lookup][..., 10:],
            compressed[..., 10:],
            atol=2e-6,
            rtol=2e-6,
        ),
        "gate=1 keeps pairs 0-9 on base RoPE and moves only pairs 10-15",
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
        "GR5_all_layers_build_low_frequency_gated_cache",
        len(calls) == 4 and all(row.shape == (6,) for row in calls),
        f"called once for all four layers with six pairs: {len(calls)} calls",
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

    gated.zero_grad(set_to_none=True)
    plain_input = tokens.clamp_max(gated.zip2zip_config.vocab_size - 1)
    plain_loss = logits(gated, plain_input, None).square().mean()
    plain_loss.backward()
    check(
        "GR6_base_view_bypasses_gate",
        gated.compressed_rope_gate.grad is None,
        "uncompressed microbatches do not train the positional gate",
    )

    # The replay concern is architectural, not just a logging artifact: the
    # plain view has no graph through either hyper-encoder, while the decoder
    # and token-type head still train.
    replay_cfg = dataclasses.replace(
        small_cfg(gated=True),
        tie_hyper_encoder=False,
        token_type_loss_weight=0.05,
    )
    replay_model = build_model(replay_cfg, seed=13)
    replay_tokens, replay_codebook = spanned_input(replay_cfg, device)
    replay_plain = replay_tokens.clamp_max(replay_cfg.vocab_size - 1)

    def module_has_grad(module):
        return any(p.grad is not None for p in module.parameters())

    replay_model.zero_grad(set_to_none=True)
    plain_logits, plain_type_logits = replay_model(
        replay_plain, codebook=None
    )
    (plain_logits.square().mean() + plain_type_logits.square().mean()).backward()
    check(
        "GR6_base_view_gradient_scope",
        not module_has_grad(replay_model.hyper_encoder)
        and not module_has_grad(replay_model.hyper_output)
        and replay_model.compressed_rope_gate.grad is None
        and module_has_grad(replay_model.token_type_head)
        and module_has_grad(replay_model.layers),
        "base replay trains decoder/type head but neither hyper-encoder nor gate",
    )

    replay_model.zero_grad(set_to_none=True)
    compressed_logits, compressed_type_logits = replay_model(
        replay_tokens, codebook=replay_codebook, hyper_causal_mask=True
    )
    (
        compressed_logits.square().mean()
        + compressed_type_logits.square().mean()
    ).backward()
    check(
        "GR6_compressed_view_gradient_scope",
        module_has_grad(replay_model.hyper_encoder)
        and module_has_grad(replay_model.hyper_output)
        and replay_model.compressed_rope_gate.grad is not None
        and torch.count_nonzero(
            replay_model.compressed_rope_gate.grad
        ).item()
        > 0,
        "compressed microbatches train both hyper-encoders and the gate",
    )

    with torch.no_grad():
        zero_logits = logits(gated, tokens, codebook).float()
        gated.compressed_rope_gate.fill_(0.1)
        moved_logits = logits(gated, tokens, codebook).float()
        gated.compressed_rope_gate.zero_()
    check(
        "GR6_controlled_gate_perturbation_changes_compressed_forward",
        not torch.allclose(zero_logits, moved_logits),
        "the real compressed decoder path consumes the learned gate",
    )

    # Activation checkpointing must close over each layer's own gate row.  A
    # late-bound loop variable would make the ordinary forward look plausible
    # but recompute every layer with the final row during backward.
    with torch.no_grad():
        gated.compressed_rope_gate.copy_(
            torch.linspace(
                0.02,
                0.20,
                gated.compressed_rope_gate.numel(),
                device=device,
            ).view_as(gated.compressed_rope_gate)
        )
    gated.eval()
    checkpoint_reference = raw_fwd(gated, tokens, codebook)
    checkpoint_positions = gated._base_token_positions(tokens, codebook)
    checkpoint_expected = [
        build_gated_rope_inputs(
            gated.freqs_cis,
            checkpoint_positions,
            gated.compressed_rope_gate[layer_id].detach(),
            start_pair=gated.zip2zip_config.gated_rope_start_pair,
            batch_size=1,
            seq_len=T,
        )
        for layer_id in range(gated.zip2zip_config.n_layers)
    ]
    checkpoint_seen = {
        layer_id: [] for layer_id in range(gated.zip2zip_config.n_layers)
    }
    handles = []
    for layer_id, layer in enumerate(gated.layers.values()):
        def capture_checkpoint(_module, args, layer_id=layer_id):
            expected_cache, expected_lookup = checkpoint_expected[layer_id]
            checkpoint_seen[layer_id].append(
                torch.equal(args[1], expected_cache)
                and torch.equal(args[3], expected_lookup)
            )

        handles.append(
            layer.attention.register_forward_pre_hook(capture_checkpoint)
        )
    gated.zero_grad(set_to_none=True)
    gated.gradient_checkpointing = True
    gated.train()
    checkpoint_out = gated(
        tokens, codebook=codebook, hyper_causal_mask=True
    )
    checkpoint_out = (
        checkpoint_out[0]
        if isinstance(checkpoint_out, tuple)
        else checkpoint_out
    )
    checkpoint_forward_matches = torch.allclose(
        checkpoint_out.detach(),
        checkpoint_reference,
        atol=1e-5,
        rtol=1e-5,
    )
    checkpoint_out[..., : gated.zip2zip_config.vocab_size].square().mean().backward()
    checkpoint_grads = [
        parameter.grad
        for parameter in gated.parameters()
        if parameter.grad is not None
    ]
    for handle in handles:
        handle.remove()
    check(
        "GR7_activation_checkpoint_forward_backward",
        checkpoint_forward_matches
        and checkpoint_grads
        and all(torch.isfinite(grad).all() for grad in checkpoint_grads)
        and gated.compressed_rope_gate.grad is not None
        and all(
            len(observations) >= 2 and all(observations)
            for observations in checkpoint_seen.values()
        ),
        "per-layer gated caches observed in forward and recompute: "
        f"{ {key: len(value) for key, value in checkpoint_seen.items()} }",
    )
    gated.zero_grad(set_to_none=True)
    gated.gradient_checkpointing = False
    gated.eval()

    # Incremental generation recovers base positions from codebook updates.
    # Compare nonzero and zero gates on the same weights to prove that the
    # generation path consumes the recovered gated geometry.
    tokens_inf = tokens.clone()
    tokens_inf[0, 2] = gated.zip2zip_config.vocab_size + 3
    tokens_inf[0, 6] = gated.zip2zip_config.vocab_size
    updates = torch.full(
        (1, 2, gated.zip2zip_config.max_subtokens),
        gated.zip2zip_config.pad_token_id,
        device=device,
    )
    updates[0, 0, :3] = torch.tensor([11, 12, 13], device=device)
    updates[0, 1, :2] = torch.tensor([21, 22], device=device)
    mapped_codebook = torch.full_like(codebook, gated.zip2zip_config.pad_token_id)
    mapped_codebook[0, 3, :3] = updates[0, 0, :3]
    mapped_codebook[0, 0, :2] = updates[0, 1, :2]
    train_positions = gated._base_token_positions(tokens_inf, mapped_codebook)
    gated.reset_inference_cache()
    incremental_before = gated.gated_rope_forwards
    incremental_gated = raw_fwd(
        gated,
        tokens_inf,
        codebook_updates=updates,
        codebook_updates_indices=[[3, 0]],
    )
    incremental_positions = gated._base_token_positions(
        tokens_inf, codebook=None
    )
    with torch.no_grad():
        gated.compressed_rope_gate.zero_()
    gated.reset_inference_cache()
    incremental_zero = raw_fwd(
        gated,
        tokens_inf,
        codebook_updates=updates,
        codebook_updates_indices=[[3, 0]],
    )
    incremental_delta = (
        incremental_gated[..., : gated.zip2zip_config.vocab_size]
        - incremental_zero[..., : gated.zip2zip_config.vocab_size]
    ).abs().max()
    check(
        "GR8_incremental_positions_and_gate_effect",
        torch.equal(incremental_positions, train_positions)
        and gated.gated_rope_forwards == incremental_before + 2
        and float(incremental_delta) > 1e-6,
        f"positions match={torch.equal(incremental_positions, train_positions)} "
        f"|delta|={float(incremental_delta):.3e}",
    )
    gated.reset_inference_cache()
    check(
        "GR8_reset_clears_span_state",
        gated._hyper_span_buf is None,
        "generation sequences cannot inherit prior base coordinates",
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
        "GR9_label_only_hypertoken_keeps_gate_in_sync_graph",
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
        "GR10_lora_freeze_keeps_gate_trainable",
        'name != "compressed_rope_gate"' in train_source,
        "the v0.7.1 gate survives the recipe's decoder freeze/LoRA setup",
    )

    # Metadata written before the low-frequency suffix field existed means
    # pair zero.  When a saved tensor is available its width is stronger
    # evidence, which is essential for non-Phi and shrunken test configs.
    inferred_pair = resolve_gated_rope_start_pair(
        {"gated_compressed_rope": True},
        {"_fsdp_wrapped_module.compressed_rope_gate": torch.zeros(4, 6)},
        n_complex_pairs=16,
        default=3,
    )
    fallback_pair = resolve_gated_rope_start_pair(
        {"gated_compressed_rope": True},
        None,
        n_complex_pairs=16,
        default=3,
    )
    check(
        "GR11_legacy_pair_geometry_resolution",
        inferred_pair == 10 and fallback_pair == 0,
        f"tensor-inferred pair={inferred_pair}; historical fallback={fallback_pair}",
    )

    # Strict eval and incremental-inference loaders must rebuild the exact gate
    # shape before loading.  Deliberately give the registered base config a
    # different default and omit the field from meta.pt: only tensor-width
    # inference can reproduce this checkpoint.
    loader_key = "_gated_rope_restore_test"
    loader_cfg = small_cfg(gated=False, pair_start=3)
    loader_source = build_model(
        small_cfg(gated=True, pair_start=10), seed=19
    )
    with torch.no_grad():
        loader_source.compressed_rope_gate.copy_(
            torch.linspace(
                0.01,
                0.12,
                loader_source.compressed_rope_gate.numel(),
                device=device,
            ).view_as(loader_source.compressed_rope_gate)
        )
    adapter, lm_eval_stubs = import_eval_adapter()
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            meta_args = {
                "model_config": loader_key,
                "max_subtokens": loader_cfg.max_subtokens,
                "max_codebook_size": loader_cfg.max_codebook_size,
                "hyper_encoder_type": loader_cfg.hyper_encoder_type,
                "encoder_dim": loader_cfg.encoder_dim,
                "encoder_n_layers": loader_cfg.encoder_n_layers,
                "encoder_n_heads": loader_cfg.encoder_n_heads,
                "encoder_intermediate_size":
                    loader_cfg.encoder_intermediate_size,
                "base_token_positions": True,
                "gated_compressed_rope": True,
                "gated_rope_start_layer": 0,
                # Intentionally no gated_rope_start_pair: legacy metadata.
            }
            torch.save(
                loader_source.state_dict(),
                os.path.join(tmpdir, "model.pt"),
            )
            torch.save(
                {"args": meta_args},
                os.path.join(tmpdir, "meta.pt"),
            )
            adapter.zip2zip_llama_configs[loader_key] = loader_cfg
            loaded_eval, loaded_cfg, _ = adapter._load_zip2zip_checkpoint(
                tmpdir, device, torch.float32
            )
            loaded_eval.hyper_encoder.disable_varlen = True

            repo = os.path.dirname(
                os.path.dirname(os.path.abspath(__file__))
            )
            inference_path = os.path.join(repo, "scripts", "inference.py")
            spec = importlib.util.spec_from_file_location(
                "_gated_rope_inference_test", inference_path
            )
            inference_mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(inference_mod)
            inference_mod.zip2zip_llama_configs[loader_key] = loader_cfg
            loaded_inference, _ = inference_mod.load_model(
                tmpdir, str(device)
            )
            loaded_inference.hyper_encoder.disable_varlen = True

            loader_tokens, loader_codebook = spanned_input(
                loader_source.zip2zip_config, device
            )
            source_logits = raw_fwd(
                loader_source, loader_tokens, loader_codebook
            )
            eval_logits = raw_fwd(
                loaded_eval, loader_tokens, loader_codebook
            )
            inference_logits = raw_fwd(
                loaded_inference, loader_tokens, loader_codebook
            )
            check(
                "GR12_eval_and_inference_restore_legacy_gate",
                loaded_cfg.gated_rope_start_pair == 10
                and loaded_inference.zip2zip_config.gated_rope_start_pair == 10
                and tuple(loaded_eval.compressed_rope_gate.shape) == (4, 6)
                and tuple(
                    loaded_inference.compressed_rope_gate.shape
                ) == (4, 6)
                and torch.equal(source_logits, eval_logits)
                and torch.equal(source_logits, inference_logits)
                and loaded_eval.gated_rope_forwards == 1
                and loaded_inference.gated_rope_forwards == 1,
                "missing metadata is recovered from the saved gate width and "
                "both native runtime paths execute it",
            )
            inference_mod.zip2zip_llama_configs.pop(loader_key, None)
    finally:
        adapter.zip2zip_llama_configs.pop(loader_key, None)
        if lm_eval_stubs:
            sys.modules.pop("zip2zip_core.lm_eval_adapter", None)
            for name in reversed(lm_eval_stubs):
                sys.modules.pop(name, None)

    return summarize(results)


def test_gated_compressed_rope_invariants():
    failed = main()
    assert not failed, f"failed invariants: {failed}"


if __name__ == "__main__":
    sys.exit(1 if main() else 0)
