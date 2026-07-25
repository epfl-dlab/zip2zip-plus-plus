"""Correctness invariants for the hyper-encoder zero-init fix (v0.6.4).

The encoder_residual design says the first hypertoken embedding should START as
its first base token's embedding: `hyper = first_token_embed + encoder_out`,
with encoder_out zeroed at init. The zero-init only ever touched `proj_out`,
which exists ONLY when encoder_dim != model_dim — so with the released Phi
recipe (encoder_dim == dim == 3072) it matched nothing and every v0.1-v0.6.3
run started with encoder_out ~54x the norm of the embedding it should nudge.
`zero_init_encoder_output=True` (v0.6.4) zeroes the final LayerNorm instead:
same guarantee, no new parameters, no state-dict change.

Runs on CPU with the shrunken config. Runnable as a script and under pytest.
Shared mechanics live in tests/_hyper_common.py.

Invariants proven:
  Z1  default off, and flag-off REPRODUCES the bug (encoder_out >> embedding) —
      a regression guard that documents what v0.1-v0.6.3 trained with.
  Z2  flag on: encoder_out is EXACTLY zero and the composed hypertoken
      embedding is bit-identical to its first base token's embedding, measured
      end-to-end through _encode_codebook_with_weights.
  Z3  encoder_dim != model_dim (proj_out exists): flag on == flag off, and the
      report names proj_out — the configs where the original code worked are
      untouched.
  Z4  untied: BOTH encoders are zeroed (report carries both roles), and the
      output-role encoder satisfies the same guarantee.
  Z5  state_dict keys and shapes are identical flag on vs off, and every tensor
      except the intended output-gate weights is bit-identical.
  Z6  the zero is escapable: the first backward trains only the zeroed output
      gate; after one update opens it, the next backward reaches upstream
      encoder parameters.
  Z7  fail-loud: an encoder type with neither proj_out nor a final LayerNorm
      raises instead of silently not applying (the original bug's failure mode).
  Z8  the nested-composer type (hierarchical) is covered via pair_encoder.norm.
  Z9  the model-level guard does not zero an encoder used without the residual.
  Z10 the CLI rejects zero-init together with --no_encoder_residual.
  Z11 zero-init and residual mode are hard resume-lineage keys, including when
      absent from legacy meta.pt (absence means false).
  Z12 the real lm-eval checkpoint loader restores residual mode from meta.pt,
      and the setting changes encoder composition by exactly the residual term.
  Z13 the compressed-generation checkpoint loader restores the same flag.
"""
import dataclasses
import contextlib
import importlib
import importlib.util
import io
import os
import sys
import tempfile
import types

from _hyper_common import build_model, make_harness, summarize

import torch
import torch.nn as nn
from zip2zip_core.configs import zip2zip_llama_configs

DIM = 256
MS = 4
K = 8


def small_cfg(zero_init: bool, encoder_dim: int = DIM, untied: bool = True,
              enc_type: str = "flat"):
    """encoder_dim == DIM reproduces the Phi situation (no proj_out)."""
    base = zip2zip_llama_configs["Phi3.5-mini"]
    return dataclasses.replace(
        base,
        dim=DIM,
        n_layers=2,
        vocab_size=512,
        pad_token_id=0,
        max_codebook_size=K,
        max_subtokens=MS,
        tie_hyper_encoder=not untied,
        hyper_encoder_type=enc_type,
        zero_init_encoder_output=zero_init,
        encoder_dim=encoder_dim,
        encoder_n_layers=2,
        encoder_n_heads=4,
        encoder_intermediate_size=2 * encoder_dim,
        tok_embeddings=dataclasses.replace(base.tok_embeddings, init_std=DIM ** -0.5),
        layer=dataclasses.replace(
            base.layer,
            feed_forward=dataclasses.replace(base.layer.feed_forward, hidden_dim=512),
            attention=dataclasses.replace(base.layer.attention, n_heads=8, n_kv_heads=8),
        ),
        rope=dataclasses.replace(base.rope, dim=DIM // 8, max_seq_len=64),
    )


def build(zero_init: bool, seed: int = 1, **kw):
    cfg = small_cfg(zero_init, **kw)
    return build_model(cfg, seed=seed), cfg


def codebook(cfg, dev, n_real=5):
    """n_real entries of 2 subtokens each, rest all-pad."""
    cb = torch.full((1, K, cfg.max_subtokens), cfg.pad_token_id, device=dev)
    for i in range(n_real):
        cb[0, i, 0] = 11 + i
        cb[0, i, 1] = 21 + i
    return cb


def encoder_out_norm(model, cfg, dev):
    """Mean ||encoder_out|| and mean ||first_token_embed|| over real entries."""
    cb = codebook(cfg, dev)
    W = model.tok_embeddings.weight
    with torch.no_grad():
        composed = model._encode_codebook_with_weights(cb, W)[0, :5]   # first+enc
        first = W[cb[0, :5, 0]]
    return (composed - first).norm(dim=-1).mean().item(), first.norm(dim=-1).mean().item()


def import_eval_adapter():
    """Import the real adapter while stubbing only absent lm-eval registration APIs."""
    created = []
    try:
        return importlib.import_module("zip2zip_core.lm_eval_adapter"), created
    except ModuleNotFoundError as e:
        if e.name != "lm_eval":
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

    base = zip2zip_llama_configs["Phi3.5-mini"]
    check("Z1_config_default_off", base.zero_init_encoder_output is False,
          "code default keeps v0.1-v0.6.3 behavior")

    # ---- Z1: flag off reproduces the bug ----
    moff, cfg = build(zero_init=False)
    dev = next(moff.parameters()).device
    enc_off, emb_off = encoder_out_norm(moff, cfg, dev)
    check("Z1_flag_off_reproduces_bug", enc_off > 10 * emb_off,
          f"||enc_out||={enc_off:.3f} vs ||emb||={emb_off:.3f} "
          f"({enc_off / max(emb_off, 1e-9):.1f}x — the v0.1-v0.6.3 state)")
    check("Z1_flag_off_report_empty", moff.zero_init_report["hyper_encoder"] == [],
          "nothing was zeroed (the silent no-op, now visible)")

    # ---- Z2: flag on gives an exact identity start ----
    mon, _ = build(zero_init=True)
    enc_on, emb_on = encoder_out_norm(mon, cfg, dev)
    check("Z2_encoder_out_exactly_zero", enc_on == 0.0,
          f"||enc_out||={enc_on} (exactly 0)")
    cb = codebook(cfg, dev)
    W = mon.tok_embeddings.weight
    with torch.no_grad():
        composed = mon._encode_codebook_with_weights(cb, W)[0, :5]
        first = W[cb[0, :5, 0]]
    check("Z2_hyper_embed_bit_equals_first_token", bool(torch.equal(composed, first)),
          "composed hypertoken embedding IS the first base token's embedding")
    check("Z2_report_names_final_norm",
          mon.zero_init_report["hyper_encoder"] == ["norm"],
          f"report={mon.zero_init_report['hyper_encoder']}")

    # ---- Z3: proj_out configs unchanged by the flag ----
    m_projoff, cfg_p = build(zero_init=False, encoder_dim=DIM // 2, seed=3)
    m_projon, _ = build(zero_init=True, encoder_dim=DIM // 2, seed=3)
    e_off, _ = encoder_out_norm(m_projoff, cfg_p, dev)
    e_on, _ = encoder_out_norm(m_projon, cfg_p, dev)
    check("Z3_proj_out_path_already_zero", e_off == 0.0 and e_on == 0.0,
          f"off={e_off} on={e_on} (original zero-init already worked)")
    check("Z3_report_names_proj_out",
          m_projon.zero_init_report["hyper_encoder"] == ["proj_out"],
          f"report={m_projon.zero_init_report['hyper_encoder']}")
    sd_a = m_projoff.state_dict(); sd_b = m_projon.state_dict()
    check("Z3_proj_out_configs_bit_identical",
          all(torch.equal(sd_a[k], sd_b[k]) for k in sd_a),
          "flag is a no-op where proj_out exists")

    # ---- Z4: untied — both encoders zeroed ----
    check("Z4_both_roles_zeroed",
          mon.zero_init_report.get("hyper_encoder") == ["norm"]
          and mon.zero_init_report.get("hyper_output") == ["norm"],
          f"report={mon.zero_init_report}")
    with torch.no_grad():
        out_role = mon._encode_codebook_with_weights(
            cb, mon.output.weight, encoder=mon.hyper_output)[0, :5]
        first_out = mon.output.weight[cb[0, :5, 0]]
    check("Z4_output_role_identity_start", bool(torch.equal(out_role, first_out)),
          "output-role encoder starts at its lm_head row")

    # ---- Z5: init-only — no state-dict structure change or collateral init ----
    k_off = {k: tuple(v.shape) for k, v in moff.state_dict().items()}
    k_on = {k: tuple(v.shape) for k, v in mon.state_dict().items()}
    check("Z5_state_dict_identical_keys_shapes", k_off == k_on,
          f"{len(k_on)} keys, same shapes -> eval/export/inference unchanged")
    sd_off = moff.state_dict()
    sd_on = mon.state_dict()
    changed = {k for k in sd_off if not torch.equal(sd_off[k], sd_on[k])}
    intended = {"hyper_encoder.norm.weight", "hyper_output.norm.weight"}
    check("Z5_only_output_gate_weights_change", changed == intended,
          f"changed={sorted(changed)}")

    # ---- Z6: the zero is escapable, with one update of gate-only gradients ----
    mgrad, cfg_g = build(zero_init=True, seed=5)
    mgrad.train()
    out = mgrad._encode_codebook_with_weights(codebook(cfg_g, dev), mgrad.tok_embeddings.weight)
    out.sum().backward()
    first_nonzero = {
        name for name, p in mgrad.hyper_encoder.named_parameters()
        if p.grad is not None and float(p.grad.abs().max()) > 0
    }
    check("Z6_first_backward_trains_only_gate",
          first_nonzero == {"norm.weight", "norm.bias"},
          f"nonzero grads={sorted(first_nonzero)}")
    with torch.no_grad():
        for p in mgrad.hyper_encoder.parameters():
            if p.grad is not None:
                p.add_(p.grad, alpha=-1e-3)
    mgrad.zero_grad(set_to_none=True)
    out = mgrad._encode_codebook_with_weights(codebook(cfg_g, dev), mgrad.tok_embeddings.weight)
    out.sum().backward()
    upstream_nonzero = {
        name for name, p in mgrad.hyper_encoder.named_parameters()
        if not name.startswith("norm.")
        and p.grad is not None
        and float(p.grad.abs().max()) > 0
    }
    check("Z6_second_backward_reaches_upstream",
          bool(upstream_nonzero),
          f"{len(upstream_nonzero)} upstream tensors now have nonzero grad")
    mgrad.zero_grad(set_to_none=True)

    # ---- Z7: fail-loud on an unsupported encoder type ----
    raised = False
    try:
        build(zero_init=True, enc_type="fast_hierarchical", seed=7)
    except ValueError as e:
        raised = "zero_init_encoder_output" in str(e)
    check("Z7_unsupported_type_raises", raised,
          "gated-MLP composer has no proj_out/norm -> loud error, never silent")

    # ---- Z8: nested composer (hierarchical) covered ----
    mh, cfg_h = build(zero_init=True, enc_type="hierarchical", seed=8)
    e_h, _ = encoder_out_norm(mh, cfg_h, dev)
    check("Z8_hierarchical_zeroed", e_h == 0.0
          and mh.zero_init_report["hyper_encoder"] == ["pair_encoder.norm"],
          f"||enc_out||={e_h} report={mh.zero_init_report['hyper_encoder']}")

    # ---- Z9: direct model use without residual does not zero the full embedding ----
    cfg_nr = small_cfg(zero_init=True)
    torch.manual_seed(9)
    m_nr = cfg_nr.build()
    m_nr.encoder_residual = False
    with torch.no_grad():
        m_nr.init_weights()
    check("Z9_no_residual_skips_zero_init",
          m_nr.zero_init_report["hyper_encoder"] == [],
          "zero output without a residual would be identically-zero embeddings")

    # ---- Z10: the training CLI rejects the contradictory flag pair ----
    import zip2zip_core.train as train_mod
    saved_argv = sys.argv
    err = io.StringIO()
    cli_rejected = False
    try:
        sys.argv = [
            "train.py", "--data_dir", "/tmp/data", "--output_dir", "/tmp/out",
            "--zero_init_encoder_output", "--no_encoder_residual",
        ]
        with contextlib.redirect_stderr(err):
            try:
                train_mod.main()
            except SystemExit as e:
                cli_rejected = e.code == 2
    finally:
        sys.argv = saved_argv
    cli_error = err.getvalue()
    check("Z10_cli_rejects_zero_init_without_residual",
          cli_rejected
          and "--zero_init_encoder_output" in cli_error
          and "--no_encoder_residual" in cli_error,
          "argparse rejects the mutually exclusive flags before model construction")

    # ---- Z11: both behavior/init flags are hard resume-lineage keys ----
    saved_dist = train_mod.dist
    train_mod.dist = types.SimpleNamespace(get_rank=lambda: 0)
    try:
        with tempfile.TemporaryDirectory() as td:
            common = dict(
                disable_digit_ids=True, max_codebook_size=4096,
                tokenizer="microsoft/Phi-3.5", untied_hyper_encoder=True,
                base_token_positions=True, token_type_loss_weight=0.05,
                encoder_n_layers=2, encoder_n_heads=32,
                encoder_intermediate_size=12288, max_subtokens=4,
                seq_len=2048, data_dir="/data", warmstart_steps=0,
                allow_resume_mismatch=False,
            )

            def guard_rejects(args, recorded):
                torch.save({"args": recorded}, os.path.join(td, "meta.pt"))
                try:
                    train_mod.validate_resume_args(td, args)
                    return False
                except ValueError:
                    return True

            args_zero = types.SimpleNamespace(
                **common, zero_init_encoder_output=True,
                no_encoder_residual=False, encoder_dim=3072,
            )
            rec_zero = dict(vars(args_zero))
            del rec_zero["allow_resume_mismatch"]
            rec_zero["zero_init_encoder_output"] = False
            check("Z11_zero_init_is_resume_hard",
                  guard_rejects(args_zero, rec_zero),
                  "a v0.6.3 checkpoint cannot be silently relabeled v0.6.4")

            legacy = dict(rec_zero)
            del legacy["zero_init_encoder_output"]
            del legacy["no_encoder_residual"]
            check("Z11_legacy_absence_means_zero_init_off",
                  guard_rejects(args_zero, legacy),
                  "legacy meta.pt absence is compared as false")

            args_nores = types.SimpleNamespace(
                **common, zero_init_encoder_output=False,
                no_encoder_residual=True, encoder_dim=3072,
            )
            rec_nores = dict(vars(args_nores))
            del rec_nores["allow_resume_mismatch"]
            rec_nores["no_encoder_residual"] = False
            check("Z11_no_residual_is_resume_hard",
                  guard_rejects(args_nores, rec_nores),
                  "behavior-only residual mode cannot change silently")

            rec_match = dict(vars(args_zero))
            del rec_match["allow_resume_mismatch"]
            torch.save({"args": rec_match}, os.path.join(td, "meta.pt"))
            matching_ok = True
            try:
                train_mod.validate_resume_args(td, args_zero)
            except ValueError:
                matching_ok = False
            check("Z11_matching_recipe_resumes",
                  matching_ok,
                  "the guard accepts an unchanged recipe")
    finally:
        train_mod.dist = saved_dist

    # ---- Z12: eval restores the behavior-only residual flag from meta.pt ----
    adapter, lm_eval_stubs = import_eval_adapter()
    eval_key = "_encoder_residual_restore_test"
    cfg_eval = small_cfg(zero_init=False, untied=False)
    source = build_model(cfg_eval, seed=12)
    try:
        with tempfile.TemporaryDirectory() as td:
            adapter.zip2zip_llama_configs[eval_key] = cfg_eval
            torch.save(source.state_dict(), os.path.join(td, "model.pt"))

            def adapter_load(no_residual):
                meta_args = {"model_config": eval_key}
                if no_residual is not None:
                    meta_args["no_encoder_residual"] = no_residual
                torch.save({"args": meta_args}, os.path.join(td, "meta.pt"))
                loaded, _, _ = adapter._load_zip2zip_checkpoint(
                    td, dev, torch.float32
                )
                loaded.hyper_encoder.disable_varlen = True
                return loaded

            default_eval = adapter_load(None)
            nores_eval = adapter_load(True)
            check("Z12_eval_restores_residual_mode",
                  getattr(default_eval, "encoder_residual", True) is True
                  and nores_eval.encoder_residual is False,
                  "legacy absence -> on; recorded no_encoder_residual -> off")

            cb_eval = codebook(cfg_eval, dev)
            with torch.no_grad():
                default_out = default_eval._encode_codebook_with_weights(
                    cb_eval, default_eval.tok_embeddings.weight
                )[0, :5]
                nores_out = nores_eval._encode_codebook_with_weights(
                    cb_eval, nores_eval.tok_embeddings.weight
                )[0, :5]
                first_eval = default_eval.tok_embeddings.weight[
                    cb_eval[0, :5, 0]
                ]
            check("Z12_eval_flag_changes_composition_exactly",
                  torch.equal(default_out, first_eval + nores_out)
                  and not torch.equal(default_out, nores_out),
                  "residual-on output == first-token embedding + residual-off output")
    finally:
        adapter.zip2zip_llama_configs.pop(eval_key, None)
        if lm_eval_stubs:
            sys.modules.pop("zip2zip_core.lm_eval_adapter", None)
            for name in reversed(lm_eval_stubs):
                sys.modules.pop(name, None)

    # ---- Z13: compressed-generation loader uses the same restoration path ----
    inference_path = os.path.join(
        os.path.dirname(__file__), "..", "scripts", "inference.py"
    )
    spec = importlib.util.spec_from_file_location(
        "_encoder_residual_inference_test", inference_path
    )
    inference_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(inference_mod)
    inference_key = "_encoder_residual_inference_test"
    try:
        with tempfile.TemporaryDirectory() as td:
            inference_mod.zip2zip_llama_configs[inference_key] = cfg_eval
            torch.save(source.state_dict(), os.path.join(td, "model.pt"))
            torch.save(
                {"args": {
                    "model_config": inference_key,
                    "max_subtokens": cfg_eval.max_subtokens,
                    "max_codebook_size": cfg_eval.max_codebook_size,
                    "no_encoder_residual": True,
                }},
                os.path.join(td, "meta.pt"),
            )
            inference_model, _ = inference_mod.load_model(td, str(dev))
            inference_model.hyper_encoder.disable_varlen = True
            with torch.no_grad():
                source_out = source._encode_codebook_with_weights(
                    cb_eval, source.tok_embeddings.weight
                )[0, :5]
                inference_out = inference_model._encode_codebook_with_weights(
                    cb_eval, inference_model.tok_embeddings.weight
                )[0, :5]
            check("Z13_inference_restores_residual_mode",
                  inference_model.encoder_residual is False
                  and torch.equal(source_out, first_eval + inference_out),
                  "compressed generation restores no-residual composition")
    finally:
        inference_mod.zip2zip_llama_configs.pop(inference_key, None)

    return summarize(results)


def test_encoder_zero_init_invariants():
    failed = main()
    assert not failed, f"failed invariants: {failed}"


if __name__ == "__main__":
    sys.exit(1 if main() else 0)
