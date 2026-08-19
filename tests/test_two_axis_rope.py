"""Correctness invariants for the v0.7 two-axis RoPE candidate.

v0.7 = v0.6.4 + two_axis_rope. Every decoder layer uses an interleaved
complex RoPE cache: even complex pairs take end-anchored base-stream positions,
odd pairs take compressed-token indices. The flag is opt-in and requires the
v0.6.4 base-position path.

Runnable both as a script and under pytest. The model is deliberately shrunken
and uses the CPU/GPU-safe padded hyper-encoder path.
"""

import contextlib
import dataclasses
import importlib
import importlib.util
import io
import os
import sys
import tempfile
import types

import torch

from _hyper_common import build_model, make_harness, summarize
from zip2zip_core.configs import zip2zip_llama_configs
from zip2zip_core.model import build_two_axis_rope_inputs


MS = 4
T = 12


def small_cfg(two_axis: bool, base_positions: bool = True):
    base = zip2zip_llama_configs["Phi3.5-mini"]
    return dataclasses.replace(
        base,
        dim=128,
        n_layers=2,
        vocab_size=128,
        pad_token_id=0,
        max_codebook_size=6,
        max_subtokens=MS,
        tie_hyper_encoder=True,
        base_token_positions=base_positions,
        two_axis_rope=two_axis,
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


def build(two_axis: bool, seed: int = 1):
    cfg = small_cfg(two_axis)
    return build_model(cfg, seed=seed), cfg


def spanned_input(cfg, device):
    """Two hypertokens with spans 3 and 2, first appearing at position 2."""
    torch.manual_seed(7)
    tokens = torch.randint(1, cfg.vocab_size, (1, T), device=device)
    tokens[0, 2] = cfg.vocab_size
    tokens[0, 5] = cfg.vocab_size + 1
    codebook = torch.full(
        (1, cfg.max_codebook_size, cfg.max_subtokens),
        cfg.pad_token_id,
        device=device,
    )
    codebook[0, 0, :3] = torch.tensor([11, 12, 13], device=device)
    codebook[0, 1, :2] = torch.tensor([21, 22], device=device)
    return tokens, codebook


def raw_fwd(model, tokens, codebook=None, **kwargs):
    with torch.no_grad():
        out = model(
            tokens,
            codebook=codebook,
            hyper_causal_mask=True,
            **kwargs,
        )
    return (out[0] if isinstance(out, tuple) else out).float()


def raises(fn, text: str) -> bool:
    try:
        fn()
    except (ValueError, RuntimeError) as exc:
        return text in str(exc)
    return False


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

    # ---- TA1: opt-in only; no parameters or initialization change ----
    default = zip2zip_llama_configs["Phi3.5-mini"]
    check(
        "TA1_default_off",
        default.two_axis_rope is False,
        "the base model config keeps historical single-axis RoPE",
    )
    base_model, cfg = build(two_axis=False)
    dual_model, _ = build(two_axis=True)
    base_state = base_model.state_dict()
    dual_state = dual_model.state_dict()
    same_state = base_state.keys() == dual_state.keys() and all(
        torch.equal(base_state[key], dual_state[key]) for key in base_state
    )
    check(
        "TA1_state_dict_and_init_identical",
        same_state,
        "the behavior flag adds no weights and consumes no initialization RNG",
    )

    # ---- TA2: fail-loud configuration and CLI contracts ----
    check(
        "TA2_requires_base_positions",
        raises(
            lambda: small_cfg(True, base_positions=False).build(),
            "requires base_token_positions=True",
        ),
        "two-axis cannot silently run with a short compressed-only cache",
    )
    cos_cfg = dataclasses.replace(
        small_cfg(True),
        rope=dataclasses.replace(
            small_cfg(True).rope, backend="cos_sin", scaling="none"
        ),
        layer=dataclasses.replace(
            small_cfg(True).layer,
            attention=dataclasses.replace(
                small_cfg(True).layer.attention, rope_backend="cos_sin"
            ),
        ),
    )
    check(
        "TA2_requires_complex_rope",
        raises(lambda: cos_cfg.build(), "requires complex RoPE"),
        "unsupported backends fail before model construction",
    )
    odd_head_cfg = dataclasses.replace(
        small_cfg(True),
        dim=120,
        rope=dataclasses.replace(small_cfg(True).rope, dim=30),
        tok_embeddings=dataclasses.replace(
            small_cfg(True).tok_embeddings, init_std=120**-0.5
        ),
        layer=dataclasses.replace(
            small_cfg(True).layer,
            feed_forward=dataclasses.replace(
                small_cfg(True).layer.feed_forward, hidden_dim=240
            ),
        ),
    )
    check(
        "TA2_requires_head_dim_divisible_by_four",
        raises(lambda: odd_head_cfg.build(), "head_dim divisible by 4"),
        "an exact 50/50 complex-pair split is enforced",
    )
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    import zip2zip_core.train as train_mod

    saved_argv = sys.argv
    cli_stderr = io.StringIO()
    cli_exit = None
    try:
        sys.argv = [
            "zip2zip_core.train",
            "--data_dir",
            "/nonexistent",
            "--output_dir",
            "/tmp/zip2zip-two-axis-cli-guard",
            "--two_axis_rope",
        ]
        with contextlib.redirect_stderr(cli_stderr):
            try:
                train_mod.main()
            except SystemExit as exc:
                cli_exit = exc.code
    finally:
        sys.argv = saved_argv
    check(
        "TA2_cli_requires_base_positions",
        cli_exit == 2
        and "--two_axis_rope requires --base_token_positions"
        in cli_stderr.getvalue(),
        "the invalid combination exits before distributed/GPU initialization",
    )

    # ---- TA3: exact pair routing and canonical-cache immutability ----
    n_rows, n_pairs = 16, 48
    real = torch.arange(n_rows * n_pairs, dtype=torch.float32).view(
        n_rows, n_pairs
    )
    synthetic_cache = torch.complex(real, -real)
    untouched = synthetic_cache.clone()
    base_positions = torch.tensor([[0, 2, 5, 7], [1, 4, 6, 9]])
    mixed, lookup = build_two_axis_rope_inputs(
        synthetic_cache,
        base_positions,
        batch_size=2,
        seq_len=4,
    )
    routed = mixed[lookup]
    compressed = synthetic_cache[:4].unsqueeze(0).expand(2, -1, -1)
    check(
        "TA3_shapes_and_synthetic_lookup",
        mixed.shape == (8, n_pairs)
        and torch.equal(lookup, torch.arange(8).view(2, 4)),
        f"mixed={tuple(mixed.shape)} lookup={lookup.tolist()}",
    )
    check(
        "TA3_even_base_odd_compressed_exact",
        torch.equal(routed[..., 0::2], synthetic_cache[base_positions][..., 0::2])
        and torch.equal(routed[..., 1::2], compressed[..., 1::2]),
        "Phi head_dim=96 gives 24 pairs per axis across the full frequency range",
    )
    check(
        "TA3_canonical_cache_unchanged",
        torch.equal(synthetic_cache, untouched),
        "the helper only creates a per-forward derived cache",
    )

    device = next(dual_model.parameters()).device
    tokens, codebook = spanned_input(cfg, device)
    all_base = torch.randint(1, cfg.vocab_size, (1, T), device=device)
    all_pad = torch.full_like(codebook, cfg.pad_token_id)

    # ---- TA4: ordinary all-base inputs are bit-identical to v0.6.4 ----
    # Bit-identity ALONE cannot fail here: mixing arange base positions against
    # arange compressed positions reproduces the canonical rows exactly, so the
    # comparison passes whether the bypass was taken or not. Each check therefore
    # also pins what attention actually received — the canonical cache OBJECT and
    # the historical positions argument — plus the two_axis_forwards counter.
    def capture_first_layer(model, *forward_args, **forward_kwargs):
        """Run a forward, returning (logits, (cache, positions)) seen by layer 0."""
        seen = []
        handle = next(
            iter(model.layers.values())
        ).attention.register_forward_pre_hook(
            lambda _module, args: seen.append((args[1], args[3]))
        )
        try:
            out = raw_fwd(model, *forward_args, **forward_kwargs)
        finally:
            handle.remove()
        return out, seen[0]

    plain_logits, (plain_cache, plain_positions) = capture_first_layer(
        dual_model, all_base, all_pad
    )
    check(
        "TA4_plain_all_base_bit_identical",
        torch.equal(plain_logits, raw_fwd(base_model, all_base, all_pad))
        and plain_cache is dual_model.freqs_cis
        and plain_positions is None,
        "plain path keeps positions=None and bypasses cache mixing",
    )
    live_logits, (live_cache, live_positions) = capture_first_layer(
        dual_model, all_base, codebook
    )
    check(
        "TA4_live_codebook_all_base_bit_identical",
        torch.equal(live_logits, raw_fwd(base_model, all_base, codebook))
        and live_cache is dual_model.freqs_cis
        and live_positions is not None
        and torch.equal(
            live_positions, torch.arange(T, device=device).unsqueeze(0)
        ),
        "train path also bypasses mixing when no token is compressed",
    )
    check(
        "TA4_bypass_did_not_build_a_mixed_cache",
        dual_model.two_axis_forwards == 0,
        "the counter proves no all-base forward paid for cache mixing",
    )

    # ---- TA5: compressed input changes geometry in every decoder layer ----
    base_positions = dual_model._base_token_positions(tokens, codebook)
    expected_cache, expected_lookup = build_two_axis_rope_inputs(
        dual_model.freqs_cis,
        base_positions,
        batch_size=1,
        seq_len=T,
    )
    seen = {}
    handles = []
    for layer_id, layer in enumerate(dual_model.layers.values()):
        def capture(_module, args, layer_id=layer_id):
            seen[layer_id] = (args[1], args[3])

        handles.append(layer.attention.register_forward_pre_hook(capture))
    dual_logits = raw_fwd(dual_model, tokens, codebook)
    for handle in handles:
        handle.remove()
    base_logits = raw_fwd(base_model, tokens, codebook)
    every_layer_mixed = len(seen) == cfg.n_layers and all(
        torch.equal(cache, expected_cache)
        and torch.equal(positions, expected_lookup)
        for cache, positions in seen.values()
    )
    check(
        "TA5_every_decoder_layer_receives_two_axes",
        every_layer_mixed,
        f"captured {len(seen)}/{cfg.n_layers} layers",
    )
    check(
        "TA5_mixed_cache_fired_once_and_stayed_complex",
        dual_model.two_axis_forwards == 1
        and torch.is_complex(expected_cache)
        and expected_cache.dtype == dual_model.freqs_cis.dtype,
        f"forwards={dual_model.two_axis_forwards} "
        f"dtype={expected_cache.dtype} (canonical {dual_model.freqs_cis.dtype})",
    )
    prefix = 2
    suffix_delta = (
        dual_logits[:, prefix:, : cfg.vocab_size]
        - base_logits[:, prefix:, : cfg.vocab_size]
    ).abs().max()
    check(
        "TA5_causal_prefix_same_suffix_changes",
        torch.equal(
            dual_logits[:, :prefix, : cfg.vocab_size],
            base_logits[:, :prefix, : cfg.vocab_size],
        )
        and float(suffix_delta) > 1e-5,
        f"max base-logit suffix |delta|={float(suffix_delta):.3e}",
    )

    # Explicit positions define the base axis, even on otherwise all-base input.
    explicit_base = torch.arange(T, device=device).unsqueeze(0) * 2
    explicit_expected, explicit_lookup = build_two_axis_rope_inputs(
        dual_model.freqs_cis,
        explicit_base,
        batch_size=1,
        seq_len=T,
    )
    explicit_seen = []
    handle = next(iter(dual_model.layers.values())).attention.register_forward_pre_hook(
        lambda _module, args: explicit_seen.append((args[1], args[3]))
    )
    raw_fwd(
        dual_model,
        all_base,
        all_pad,
        positions=explicit_base,
    )
    handle.remove()
    check(
        "TA5_explicit_positions_are_the_base_axis",
        len(explicit_seen) == 1
        and torch.equal(explicit_seen[0][0], explicit_expected)
        and torch.equal(explicit_seen[0][1], explicit_lookup),
        "odd pairs remain compressed arange instead of inheriting explicit positions",
    )

    # ---- TA6: batched layouts do not bleed into each other ----
    batch_tokens = torch.cat([all_base, tokens], dim=0)
    batch_codebook = torch.cat([codebook, codebook], dim=0)
    dual_batch = raw_fwd(dual_model, batch_tokens, batch_codebook)
    dual_row0 = raw_fwd(dual_model, all_base, codebook)
    dual_row1 = raw_fwd(dual_model, tokens, codebook)
    base_batch = raw_fwd(base_model, batch_tokens, batch_codebook)
    check(
        "TA6_batched_rows_match_individual_forwards",
        torch.allclose(dual_batch[0:1], dual_row0, atol=1e-5, rtol=1e-5)
        and torch.allclose(dual_batch[1:2], dual_row1, atol=1e-5, rtol=1e-5),
        "B=2 synthetic lookup preserves each sample's base positions",
    )
    check(
        "TA6_all_base_row_remains_v064_exact_in_mixed_batch",
        torch.equal(
            dual_batch[0:1, :, : cfg.vocab_size],
            base_batch[0:1, :, : cfg.vocab_size],
        ),
        "a compressed neighbor cannot alter the all-base row",
    )

    # ---- TA7: activation checkpoint recomputes the same mixed geometry ----
    checkpoint_seen = {layer_id: [] for layer_id in range(cfg.n_layers)}
    handles = []
    for layer_id, layer in enumerate(dual_model.layers.values()):
        def capture_checkpoint(_module, args, layer_id=layer_id):
            checkpoint_seen[layer_id].append(
                torch.equal(args[1], expected_cache)
                and torch.equal(args[3], expected_lookup)
            )

        handles.append(
            layer.attention.register_forward_pre_hook(capture_checkpoint)
        )
    dual_model.gradient_checkpointing = True
    dual_model.train()
    out = dual_model(tokens, codebook=codebook, hyper_causal_mask=True)
    out = out[0] if isinstance(out, tuple) else out
    forward_matches = torch.allclose(
        out.detach(), dual_logits, atol=1e-5, rtol=1e-5
    )
    out[..., : cfg.vocab_size].square().mean().backward()
    grads = [
        parameter.grad
        for parameter in dual_model.parameters()
        if parameter.grad is not None
    ]
    finite_grad = bool(grads) and all(
        bool(torch.isfinite(grad).all()) for grad in grads
    )
    for handle in handles:
        handle.remove()
    check(
        "TA7_checkpoint_forward_backward",
        forward_matches
        and finite_grad
        and all(
            len(observations) >= 2 and all(observations)
            for observations in checkpoint_seen.values()
        ),
        "layer mixed-cache observations including recompute="
        f"{ {key: len(value) for key, value in checkpoint_seen.items()} }",
    )
    dual_model.zero_grad(set_to_none=True)
    dual_model.gradient_checkpointing = False
    dual_model.eval()

    # ---- TA8: incremental inference derives the same two-axis positions ----
    tokens_inf = tokens.clone()
    tokens_inf[0, 2] = cfg.vocab_size + 3
    tokens_inf[0, 5] = cfg.vocab_size
    updates = torch.full(
        (1, 2, cfg.max_subtokens), cfg.pad_token_id, device=device
    )
    updates[0, 0, :3] = torch.tensor([11, 12, 13], device=device)
    updates[0, 1, :2] = torch.tensor([21, 22], device=device)
    mapped_codebook = torch.full_like(codebook, cfg.pad_token_id)
    mapped_codebook[0, 3, :3] = updates[0, 0, :3]
    mapped_codebook[0, 0, :2] = updates[0, 1, :2]
    train_positions = dual_model._base_token_positions(
        tokens_inf, mapped_codebook
    )
    dual_model.reset_inference_cache()
    with torch.no_grad():
        inf_out = dual_model(
            tokens_inf,
            codebook_updates=updates,
            codebook_updates_indices=[[3, 0]],
        )
    inf_out = (inf_out[0] if isinstance(inf_out, tuple) else inf_out).float()
    inf_positions = dual_model._base_token_positions(tokens_inf, codebook=None)
    base_model.reset_inference_cache()
    with torch.no_grad():
        inf_base = base_model(
            tokens_inf,
            codebook_updates=updates,
            codebook_updates_indices=[[3, 0]],
        )
    inf_base = (inf_base[0] if isinstance(inf_base, tuple) else inf_base).float()
    inf_delta = (
        inf_out[:, 2:, : cfg.vocab_size]
        - inf_base[:, 2:, : cfg.vocab_size]
    ).abs().max()
    check(
        "TA8_incremental_positions_and_effect",
        torch.equal(inf_positions, train_positions) and float(inf_delta) > 1e-5,
        f"positions match={torch.equal(inf_positions, train_positions)} "
        f"suffix |delta|={float(inf_delta):.3e}",
    )
    dual_model.reset_inference_cache()
    check(
        "TA8_reset_clears_span_state",
        dual_model._hyper_span_buf is None,
        "generation sequences cannot inherit prior base coordinates",
    )

    # ---- TA9: fully merged input reaches the cache bound without mutation ----
    canonical_before = dual_model.freqs_cis.clone()
    full_tokens = torch.full(
        (1, T), cfg.vocab_size, dtype=torch.long, device=device
    )
    full_codebook = torch.full_like(codebook, cfg.pad_token_id)
    full_codebook[0, 0] = torch.tensor([31, 32, 33, 34], device=device)
    max_position = dual_model._base_token_positions(
        full_tokens, full_codebook
    ).max()
    full_logits = raw_fwd(dual_model, full_tokens, full_codebook)
    check(
        "TA9_cache_bound_and_immutability",
        int(max_position) == T * MS - 1
        and bool(torch.isfinite(full_logits[..., : cfg.vocab_size]).all())
        and torch.equal(dual_model.freqs_cis, canonical_before),
        f"max position={int(max_position)}, cache rows={len(canonical_before)}",
    )

    # ---- TA10: resume guard treats this behavior flag as recipe-critical ----
    saved_dist = train_mod.dist
    train_mod.dist = types.SimpleNamespace(get_rank=lambda: 0)
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            def make_args(**overrides):
                values = dict(
                    disable_digit_ids=True,
                    max_codebook_size=4096,
                    max_active_codebook_size=4096,
                    tokenizer="microsoft/Phi-3.5",
                    untied_hyper_encoder=True,
                    base_token_positions=True,
                    two_axis_rope=True,
                    zero_init_encoder_output=True,
                    no_encoder_residual=False,
                    token_type_loss_weight=0.05,
                    online_codebook_mask=False,
                    encoder_dim=3072,
                    encoder_n_layers=2,
                    encoder_n_heads=32,
                    encoder_intermediate_size=12288,
                    max_subtokens=3,
                    seq_len=2048,
                    data_dir="/data",
                    warmstart_steps=0,
                    allow_resume_mismatch=False,
                )
                values.update(overrides)
                return types.SimpleNamespace(**values)

            def guard_raises(meta_args, args):
                torch.save(
                    {"args": meta_args}, os.path.join(tmpdir, "meta.pt")
                )
                try:
                    train_mod.validate_resume_args(tmpdir, args)
                    return False
                except ValueError:
                    return True

            args_on = make_args()
            recorded_on = dict(vars(args_on))
            recorded_on.pop("allow_resume_mismatch")
            legacy = {
                key: value
                for key, value in recorded_on.items()
                if key != "two_axis_rope"
            }
            resume_ok = (
                guard_raises(
                    {**recorded_on, "two_axis_rope": False}, args_on
                )
                and guard_raises(
                    recorded_on, make_args(two_axis_rope=False)
                )
                and guard_raises(legacy, args_on)
                and not guard_raises(recorded_on, args_on)
                and not guard_raises(
                    legacy, make_args(two_axis_rope=False)
                )
            )
            check(
                "TA10_resume_hard_both_directions_and_legacy_off",
                resume_ok,
                "missing pre-v0.7 metadata has the unambiguous value False",
            )
    finally:
        train_mod.dist = saved_dist

    # ---- TA11: eval/inference loaders restore the behavior-only geometry ----
    loader_cfg = small_cfg(False, base_positions=False)
    loader_key = "_two_axis_rope_restore_test"
    adapter, lm_eval_stubs = import_eval_adapter()
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            torch.save(dual_model.state_dict(), os.path.join(tmpdir, "model.pt"))
            meta_args = {
                "model_config": loader_key,
                "max_subtokens": loader_cfg.max_subtokens,
                "max_codebook_size": loader_cfg.max_codebook_size,
                "encoder_dim": loader_cfg.encoder_dim,
                "encoder_n_layers": loader_cfg.encoder_n_layers,
                "encoder_n_heads": loader_cfg.encoder_n_heads,
                "encoder_intermediate_size": loader_cfg.encoder_intermediate_size,
                "base_token_positions": True,
                "two_axis_rope": True,
            }
            torch.save({"args": meta_args}, os.path.join(tmpdir, "meta.pt"))
            adapter.zip2zip_llama_configs[loader_key] = loader_cfg
            loaded_eval, loaded_cfg, _ = adapter._load_zip2zip_checkpoint(
                tmpdir, device, torch.float32
            )
            loaded_eval.hyper_encoder.disable_varlen = True

            inference_path = os.path.join(repo, "scripts", "inference.py")
            spec = importlib.util.spec_from_file_location(
                "_two_axis_rope_inference_test", inference_path
            )
            inference_mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(inference_mod)
            inference_mod.zip2zip_llama_configs[loader_key] = loader_cfg
            loaded_inference, _ = inference_mod.load_model(
                tmpdir, str(device)
            )
            loaded_inference.hyper_encoder.disable_varlen = True

            source_logits = raw_fwd(dual_model, tokens, codebook)
            eval_logits = raw_fwd(loaded_eval, tokens, codebook)
            inference_logits = raw_fwd(
                loaded_inference, tokens, codebook
            )
            loaders_ok = (
                loaded_cfg.base_token_positions
                and loaded_cfg.two_axis_rope
                and loaded_inference.zip2zip_config.base_token_positions
                and loaded_inference.zip2zip_config.two_axis_rope
                and torch.equal(source_logits, eval_logits)
                and torch.equal(source_logits, inference_logits)
            )
            check(
                "TA11_eval_and_inference_restore_two_axis",
                loaders_ok,
                "meta.pt restores geometry and reproduces the source forward",
            )

            # ---- TA11b: eval/inference merge-size transfer for hierarchical composer ----
            hier_key = "_eval_max_subtokens_hier_test"
            hier_cfg = dataclasses.replace(
                loader_cfg, max_subtokens=3, hyper_encoder_type="hierarchical"
            )
            hier_model = build_model(hier_cfg, seed=11)
            with tempfile.TemporaryDirectory() as hier_tmp:
                torch.save(hier_model.state_dict(), os.path.join(hier_tmp, "model.pt"))
                torch.save(
                    {
                        "args": {
                            "model_config": hier_key,
                            "max_subtokens": 3,
                            "max_codebook_size": hier_cfg.max_codebook_size,
                            "encoder_dim": hier_cfg.encoder_dim,
                            "encoder_n_layers": hier_cfg.encoder_n_layers,
                            "encoder_n_heads": hier_cfg.encoder_n_heads,
                            "encoder_intermediate_size": hier_cfg.encoder_intermediate_size,
                            "hyper_encoder_type": "hierarchical",
                        }
                    },
                    os.path.join(hier_tmp, "meta.pt"),
                )
                adapter.zip2zip_llama_configs[hier_key] = hier_cfg
                inference_mod.zip2zip_llama_configs[hier_key] = hier_cfg
                loaded_hier, loaded_hier_cfg, _ = adapter._load_zip2zip_checkpoint(
                    hier_tmp, device, torch.float32, eval_max_subtokens=4
                )
                loaded_hier_inference, _ = inference_mod.load_model(
                    hier_tmp, str(device), max_merge_size=4
                )
                check(
                    "TA11b_hier_eval_max_subtokens_override",
                    loaded_hier_cfg.max_subtokens == 4
                    and loaded_hier.zip2zip_config.max_subtokens == 4
                    and loaded_hier_inference.zip2zip_config.max_subtokens == 4,
                    "hierarchical checkpoints can widen the eval merge-size loop",
                )
                adapter.zip2zip_llama_configs.pop(hier_key, None)
                inference_mod.zip2zip_llama_configs.pop(hier_key, None)

            flat_key = "_eval_max_subtokens_flat_test"
            flat_cfg = dataclasses.replace(
                loader_cfg, max_subtokens=3, hyper_encoder_type="flat"
            )
            flat_model = build_model(flat_cfg, seed=12)
            with tempfile.TemporaryDirectory() as flat_tmp:
                torch.save(flat_model.state_dict(), os.path.join(flat_tmp, "model.pt"))
                torch.save({"args": {
                    "model_config": flat_key,
                    "max_subtokens": 3,
                    "max_codebook_size": flat_cfg.max_codebook_size,
                    "encoder_dim": flat_cfg.encoder_dim,
                    "encoder_n_layers": flat_cfg.encoder_n_layers,
                    "encoder_n_heads": flat_cfg.encoder_n_heads,
                    "encoder_intermediate_size": flat_cfg.encoder_intermediate_size,
                    "hyper_encoder_type": "flat",
                }}, os.path.join(flat_tmp, "meta.pt"))
                adapter.zip2zip_llama_configs[flat_key] = flat_cfg
                inference_mod.zip2zip_llama_configs[flat_key] = flat_cfg
                flat_refused = raises(
                    lambda: adapter._load_zip2zip_checkpoint(
                        flat_tmp, device, torch.float32, eval_max_subtokens=4
                    ),
                    "Flat hyper-encoders",
                )
                flat_inference_refused = raises(
                    lambda: inference_mod.load_model(
                        flat_tmp, str(device), max_merge_size=4
                    ),
                    "--max-merge-size",
                )
                check(
                    "TA11b_flat_eval_max_subtokens_refused",
                    flat_refused and flat_inference_refused,
                    "flat checkpoints have length-shaped positional embeddings",
                )
                adapter.zip2zip_llama_configs.pop(flat_key, None)
                inference_mod.zip2zip_llama_configs.pop(flat_key, None)

            inference_mod.zip2zip_llama_configs.pop(loader_key, None)
    finally:
        adapter.zip2zip_llama_configs.pop(loader_key, None)
        if lm_eval_stubs:
            sys.modules.pop("zip2zip_core.lm_eval_adapter", None)
            for name in reversed(lm_eval_stubs):
                sys.modules.pop(name, None)

    # ---- TA12: unsupported HF export fails explicitly ----
    from zip2zip_core.export import refuse_base_token_positions

    with tempfile.TemporaryDirectory() as tmpdir:
        torch.save(
            {
                "args": {
                    "base_token_positions": True,
                    "two_axis_rope": True,
                }
            },
            os.path.join(tmpdir, "meta.pt"),
        )
        export_refused = raises(
            lambda: refuse_base_token_positions(tmpdir),
            "--two_axis_rope",
        )
    check(
        "TA12_export_refuses_wrong_runtime_geometry",
        export_refused,
        "the external runtime cannot silently fall back to one compressed axis",
    )

    return summarize(results)


def test_two_axis_rope_invariants():
    failed = main()
    assert not failed, f"failed invariants: {failed}"


if __name__ == "__main__":
    sys.exit(1 if main() else 0)
