"""Export a zip2zip-core checkpoint to ext/zip2zip HuggingFace format.

Reads model.pt from a zip2zip-core checkpoint directory and writes:

    <output_dir>/
        zip2zip_config.json     # Zip2ZipConfig (ResLatentAttnConfig encoder)
        model.safetensors       # HF Llama decoder weights
        encoders.safetensors    # encoder weights (input_encoder.*)
"""

from __future__ import annotations

import json
import os

import torch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _infer_llama_heads(sd: dict) -> tuple[int, int]:
    """Infer (n_heads, n_kv_heads) from wq/wk weight shapes."""
    wq = sd["layers.0.attention.wq.weight"]  # (n_heads * head_dim, dim)
    wk = sd["layers.0.attention.wk.weight"]  # (n_kv_heads * head_dim, dim)
    dim = wq.shape[1]
    # Try head_dim = 64 first (standard for ≤8B), then 128 (large models)
    for head_dim in (64, 128):
        if wq.shape[0] % head_dim == 0 and wk.shape[0] % head_dim == 0:
            return wq.shape[0] // head_dim, wk.shape[0] // head_dim
    raise ValueError(
        f"Cannot infer n_heads from wq shape {wq.shape} — pass model_config"
    )


def _infer_encoder_config(sd: dict) -> dict:
    """Infer ResLatentAttnConfig fields from hyper_encoder.* weight shapes."""
    prefix = "hyper_encoder."
    he = {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}

    hidden_size = he["pos_embed.weight"].shape[1]
    max_subtokens = he["pos_embed.weight"].shape[0]

    model_hidden_size = None
    if "proj_in.weight" in he:
        model_hidden_size = he["proj_in.weight"].shape[1]

    num_hidden_layers = sum(
        1 for k in he if k.startswith("layers.") and k.endswith(".wq.weight")
    )
    intermediate_size = he["layers.0.w1.weight"].shape[0]

    return dict(
        hidden_size=hidden_size,
        model_hidden_size=model_hidden_size,
        num_hidden_layers=num_hidden_layers,
        intermediate_size=intermediate_size,
        max_subtokens=max_subtokens,
    )


def _split_state_dict(sd: dict) -> tuple[dict, dict]:
    """Split into (decoder_sd, encoder_sd)."""
    decoder, encoder = {}, {}
    skip_prefixes = ("token_type_head.", "hyper_output.")
    for k, v in sd.items():
        if k.startswith("hyper_encoder."):
            encoder[k[len("hyper_encoder."):]] = v
        elif any(k.startswith(p) for p in skip_prefixes):
            pass  # not used by ext/zip2zip
        else:
            decoder[k] = v
    return decoder, encoder


def _make_llama_config(model_config_name: str | None, sd: dict):
    """Build a Llama3Model.Config for the state dict adapter."""
    if model_config_name is not None:
        from zip2zip_core.configs import zip2zip_llama_configs
        return zip2zip_llama_configs[model_config_name]

    # Auto-detect from state dict
    from torchtitan.models.llama3.model import Llama3Model, TransformerBlock
    from torchtitan.models.llama3 import llama3_configs

    n_heads, n_kv_heads = _infer_llama_heads(sd)
    dim = sd["tok_embeddings.weight"].shape[1]
    n_layers = sum(1 for k in sd if k.startswith("layers.") and k.endswith(".attention.wq.weight"))
    vocab_size = sd["tok_embeddings.weight"].shape[0]

    # Find the closest stock config and patch it
    for name, cfg in llama3_configs.items():
        if cfg.dim == dim and cfg.n_layers == n_layers:
            return cfg

    # Fall back: build minimal config just for the adapter
    cfg = llama3_configs["llama3-8b"]   # use as template
    cfg = cfg.__class__(
        dim=dim,
        n_layers=n_layers,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        vocab_size=vocab_size,
        ffn_dim_multiplier=cfg.ffn_dim_multiplier,
        multiple_of=cfg.multiple_of,
        rope_theta=cfg.rope_theta,
        norm_eps=cfg.norm_eps,
        max_seq_len=cfg.max_seq_len,
    )
    return cfg


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def export(
    ckpt_dir: str,
    output_dir: str,
    base_model: str,
    model_config: str | None = None,
    encoder_n_heads: int | None = None,
    max_codebook_size: int = 4096,
    disabled_ids: list[int] | None = None,
    residual: bool = True,
    causal: bool = False,
):
    """Export a zip2zip-core checkpoint to ext/zip2zip HF format.

    Args:
        ckpt_dir: zip2zip-core checkpoint directory (contains model.pt)
        output_dir: Destination directory for ext/zip2zip format
        base_model: HuggingFace base model name (e.g. meta-llama/Llama-3.2-1B-Instruct)
        model_config: zip2zip-core model config key (e.g. '1B'). Auto-detected if None.
        encoder_n_heads: Number of encoder attention heads. Inferred as hidden_size // 64 if None.
        max_codebook_size: Max codebook size (default 4096)
        disabled_ids: Token IDs to disable for LZW. Loaded from tokenizer if None.
        residual: Use residual connection in encoder (default True)
        causal: Use causal masking in encoder (default False)
    """
    os.makedirs(output_dir, exist_ok=True)

    # ---- Load checkpoint -----------------------------------------------
    model_pt = os.path.join(ckpt_dir, "model.pt")
    print(f"Loading {model_pt} ...")
    sd = torch.load(model_pt, map_location="cpu", weights_only=True)

    decoder_sd, encoder_sd = _split_state_dict(sd)
    enc_info = _infer_encoder_config(sd)

    vocab_size = sd["tok_embeddings.weight"].shape[0]
    print(f"  vocab_size={vocab_size}, encoder hidden={enc_info['hidden_size']}, "
          f"model_dim={enc_info['model_hidden_size']}, "
          f"max_subtokens={enc_info['max_subtokens']}, "
          f"n_layers={enc_info['num_hidden_layers']}")

    # ---- Convert decoder weights to HF format --------------------------
    print("Converting decoder weights to HF format ...")
    llama_cfg = _make_llama_config(model_config, decoder_sd)
    from torchtitan.models.llama3.state_dict_adapter import Llama3StateDictAdapter
    adapter = Llama3StateDictAdapter(llama_cfg, None)
    hf_decoder_sd = adapter.to_hf(decoder_sd)

    # ---- Save model.safetensors ----------------------------------------
    try:
        from safetensors.torch import save_file
    except ImportError:
        raise ImportError("pip install safetensors")

    model_out = os.path.join(output_dir, "model.safetensors")
    print(f"Saving {model_out} ...")
    # clone() breaks weight-tying between embed_tokens and lm_head so safetensors
    # doesn't complain about shared memory (they map to separate tensors on disk)
    save_file({k: v.contiguous().clone() for k, v in hf_decoder_sd.items()}, model_out)

    # ---- Save encoders.safetensors -------------------------------------
    encoders_out = os.path.join(output_dir, "encoders.safetensors")
    print(f"Saving {encoders_out} ...")
    # Wrap as input_encoder.*; tie_encoders=True so no output_encoder needed
    enc_prefixed = {f"input_encoder.{k}": v.contiguous() for k, v in encoder_sd.items()}
    save_file(enc_prefixed, encoders_out)

    # ---- Build zip2zip_config.json -------------------------------------
    n_heads = encoder_n_heads or (enc_info["hidden_size"] // 64)

    if disabled_ids is not None:
        initial_vocab_size = vocab_size
    else:
        print(f"Loading tokenizer from {base_model} to compute disabled_ids ...")
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(base_model)
        disabled_ids = list(tok.get_added_vocab().values())
        initial_vocab_size = len(tok)
        print(f"  initial_vocab_size={initial_vocab_size}, disabled_ids count={len(disabled_ids)}")
        print(f"Saving tokenizer to {output_dir} ...")
        tok.save_pretrained(output_dir)

    config = {
        "base_model_name_or_path": base_model,
        "encoder_type": "res_latent_attn",
        "encoder": {
            "hidden_size": enc_info["hidden_size"],
            "model_hidden_size": enc_info["model_hidden_size"],
            "num_hidden_layers": enc_info["num_hidden_layers"],
            "intermediate_size": enc_info["intermediate_size"],
            "num_heads": n_heads,
            "causal": causal,
            "residual": residual,
            "tie_encoders": True,
            "position_encoding": None,
        },
        "compression": {
            "initial_vocab_size": initial_vocab_size,
            "max_codebook_size": max_codebook_size,
            "max_subtokens": enc_info["max_subtokens"],
            "disabled_ids": sorted(disabled_ids),
        },
    }

    config_out = os.path.join(output_dir, "zip2zip_config.json")
    print(f"Saving {config_out} ...")
    with open(config_out, "w") as f:
        json.dump(config, f, indent=2)

    print("\nDone. Output:")
    for fn in ("zip2zip_config.json", "model.safetensors", "encoders.safetensors"):
        path = os.path.join(output_dir, fn)
        size_mb = os.path.getsize(path) / 1e6
        print(f"  {fn:30s} {size_mb:8.1f} MB")
    print(f"\nLoad with:\n  Zip2ZipModel.from_pretrained('{output_dir}')")
