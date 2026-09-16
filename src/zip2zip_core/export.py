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
import shutil
from collections.abc import Mapping

import torch

from zip2zip_core.checkpoint import (
    merge_lora_weights,
    prepare_inference_state_dict,
    strip_wrapper_prefixes,
)
from zip2zip_core.disabled_ids import compute_disabled_ids


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _merge_lora_weights(sd: dict, scaling: float = 1.0) -> dict:
    """Merge LoRA adapters into base weights if the checkpoint was saved with LoRA.

    Keys like  layers.N.attention.wq.{base_layer,lora_A,lora_B}.weight
    are collapsed into  layers.N.attention.wq.weight = base + scaling * (lora_B @ lora_A).
    `scaling` must match the training-time LoRA scaling (alpha / rank); see LoRALinear.forward.
    Non-LoRA checkpoints are returned unchanged.
    """
    normalized = strip_wrapper_prefixes(sd)
    return merge_lora_weights(normalized, scaling=scaling)


def _lora_scaling_from_meta(ckpt_dir: str) -> float:
    """Read LoRA scaling (alpha / rank) from the checkpoint's meta.pt."""
    meta_pt = os.path.join(ckpt_dir, "meta.pt")
    if not os.path.exists(meta_pt):
        raise FileNotFoundError(
            f"required checkpoint metadata is missing: {meta_pt}"
        )
    meta = torch.load(meta_pt, map_location="cpu", weights_only=False)
    train_args = meta.get("args", {}) if isinstance(meta, dict) else {}
    rank = train_args.get("lora_rank")
    alpha = train_args.get("lora_alpha")
    if rank:
        if not isinstance(rank, int) or isinstance(rank, bool) or rank <= 0:
            raise ValueError(f"invalid lora_rank in {meta_pt}: {rank!r}")
        if alpha is None:
            raise ValueError(
                f"LoRA checkpoint metadata {meta_pt} has lora_rank={rank} "
                "but no lora_alpha"
            )
        return float(alpha) / float(rank)
    return 1.0


def _train_args_from_meta(ckpt_dir: str) -> Mapping[str, object]:
    """Load the training arguments needed for strict checkpoint conversion."""
    meta_pt = os.path.join(ckpt_dir, "meta.pt")
    if not os.path.exists(meta_pt):
        raise FileNotFoundError(
            f"required checkpoint metadata is missing: {meta_pt}; export "
            "cannot safely reconstruct behavior-only model settings"
        )
    meta = torch.load(meta_pt, map_location="cpu", weights_only=False)
    if not isinstance(meta, dict):
        raise ValueError(f"checkpoint metadata {meta_pt} must be a dictionary")
    train_args = meta.get("args")
    if not isinstance(train_args, Mapping) or not train_args:
        raise ValueError(
            f"checkpoint args in {meta_pt} must be a non-empty mapping"
        )
    return train_args


def _prepare_export_state_dict(sd: dict, ckpt_dir: str) -> dict:
    """Normalize wrappers and fold LoRA using mandatory checkpoint metadata."""
    return prepare_inference_state_dict(sd, _train_args_from_meta(ckpt_dir))


def _encoder_residual_from_meta(ckpt_dir: str) -> bool:
    """Restore the encoder's behavior-only residual setting."""
    return not bool(
        _train_args_from_meta(ckpt_dir).get("no_encoder_residual", False)
    )


def refuse_base_token_positions(ckpt_dir: str) -> None:
    """Hard-fail on RoPE schemes outside the zip2zip++ release contract."""
    train_args = _train_args_from_meta(ckpt_dir)
    if train_args and train_args.get("gated_compressed_rope"):
        raise NotImplementedError(
            "this checkpoint was trained with --gated_compressed_rope; the "
            "ext/zip2zip HF runtime has no gated compressed-coordinate RoPE "
            "path, so an export would silently compute wrong attention "
            "geometry. Evaluate it through the in-core adapter instead."
        )
    if train_args and train_args.get("two_axis_rope"):
        raise NotImplementedError(
            "this checkpoint was trained with --two_axis_rope; the ext/zip2zip "
            "HF runtime has no two-axis RoPE path, so an export would silently "
            "compute wrong attention geometry. Evaluate it through the in-core "
            "adapter instead."
        )


def _position_mode_from_meta(ckpt_dir: str) -> str:
    refuse_base_token_positions(ckpt_dir)
    return (
        "base_token_end"
        if _train_args_from_meta(ckpt_dir).get("base_token_positions")
        else "compressed"
    )


def _disable_digit_ids_from_meta(ckpt_dir: str) -> bool:
    """Read whether TRAINING protected digits from LZW merges (meta.pt's
    --disable_digit_ids). Exported disabled_ids must match this or the export
    silently merges digits the model never saw as hyper-tokens in training —
    the same mismatch class as the chat-token bug, for digits instead."""
    train_args = _train_args_from_meta(ckpt_dir)
    return bool(train_args.get("disable_digit_ids"))


def _resolve_encoder_n_heads(ckpt_dir: str, hidden_size: int, explicit: int | None) -> int:
    """Determine the encoder head count that TRAINING actually used.

    Priority:
      1. `explicit` — the caller's --encoder_n_heads override
      2. meta.pt train args (`encoder_n_heads`, the value train.py was invoked with)
      3. hidden_size // 64 (legacy guess — last resort)

    Guessing hidden_size // 64 unconditionally is WRONG whenever head_dim != 64.
    E.g. Phi-3.5-mini trains the encoder with encoder_dim=3072, encoder_n_heads=32
    -> head_dim=96, but //64 yields 48, so the exported encoder would reshape QKV
    into 48x64 instead of 32x96 and compute a different attention at inference.
    """
    n_heads, source = None, None
    if explicit is not None:
        n_heads, source = explicit, "explicit --encoder_n_heads"
    else:
        train_args = _train_args_from_meta(ckpt_dir)
        val = train_args.get("encoder_n_heads")
        if isinstance(val, int) and val > 0:
            n_heads, source = val, "meta.pt train args"
        if n_heads is None:
            n_heads, source = hidden_size // 64, "hidden_size // 64 (legacy fallback)"

    if n_heads <= 0 or hidden_size % n_heads != 0:
        raise ValueError(
            f"Resolved encoder num_heads={n_heads} (from {source}) does not divide "
            f"encoder hidden_size={hidden_size}. Pass a correct --encoder_n_heads or "
            f"check meta.pt."
        )
    print(f"  encoder num_heads={n_heads} (head_dim={hidden_size // n_heads}, from {source})")
    return n_heads


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


def _split_state_dict(sd: dict) -> tuple[dict, dict, dict]:
    """Split into (decoder_sd, input_encoder_sd, output_encoder_sd).

    output_encoder_sd is non-empty only for untied checkpoints (hyper_output.*);
    the caller writes output_encoder.* and sets tie_encoders=False accordingly,
    matching the ext/zip2zip runtime which loads input_encoder.*/output_encoder.*.
    """
    decoder, input_enc, output_enc = {}, {}, {}
    for k, v in sd.items():
        if k.startswith("hyper_encoder."):
            input_enc[k[len("hyper_encoder."):]] = v
        elif k.startswith("hyper_output."):
            output_enc[k[len("hyper_output."):]] = v
        elif k.startswith("token_type_head."):
            pass  # not used by ext/zip2zip
        else:
            decoder[k] = v
    return decoder, input_enc, output_enc


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


def _validate_hf_decoder_state_dict(sd: dict, n_layers: int) -> None:
    """Reject incomplete or architecture-mismatched decoder exports."""
    if not isinstance(n_layers, int) or isinstance(n_layers, bool) or n_layers <= 0:
        raise ValueError(f"decoder n_layers must be a positive integer, got {n_layers!r}")

    expected = {
        "model.embed_tokens.weight",
        "model.norm.weight",
        "lm_head.weight",
    }
    per_layer = (
        "self_attn.q_proj.weight",
        "self_attn.k_proj.weight",
        "self_attn.v_proj.weight",
        "self_attn.o_proj.weight",
        "mlp.gate_proj.weight",
        "mlp.up_proj.weight",
        "mlp.down_proj.weight",
        "input_layernorm.weight",
        "post_attention_layernorm.weight",
    )
    for layer_id in range(n_layers):
        expected.update(
            f"model.layers.{layer_id}.{suffix}" for suffix in per_layer
        )

    actual = set(sd)
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    if missing or unexpected:
        details = []
        if missing:
            details.append(f"missing={missing[:8]}")
        if unexpected:
            details.append(f"unexpected={unexpected[:8]}")
        raise ValueError(
            "incomplete or architecture-mismatched decoder checkpoint: "
            + "; ".join(details)
        )


def _pin_pad_token(tok, model_config: str) -> None:
    """Give the exported tokenizer the pad id the checkpoint trained with.

    The ext runtime pads codebook entries with tokenizer.pad_token_id and builds
    the attention mask as (id != pad_token_id); with no pad set it falls back to
    eos. For Llama-3.2-Instruct eos is <|eot_id|>, so every chat turn separator
    would be masked out and shift the base-space positions. Core pads with
    Config.pad_token_id (<|end_of_text|> for Llama, <|endoftext|> for Phi).
    """
    from zip2zip_core.configs import zip2zip_llama_configs

    cfg = zip2zip_llama_configs.get(model_config)
    if cfg is None:
        print(f"  no core config named {model_config!r}; leaving pad_token as is")
        return
    pad_id = cfg.pad_token_id
    if tok.pad_token_id == pad_id:
        return
    pad_token = tok.convert_ids_to_tokens(pad_id)
    if not isinstance(pad_token, str):
        raise ValueError(f"pad id {pad_id} is not a token of {base_model_name(tok)}")
    print(f"  pad_token {tok.pad_token!r} -> {pad_token!r} (id {pad_id}, core Config.pad_token_id)")
    tok.pad_token = pad_token


def base_model_name(tok) -> str:
    return getattr(tok, "name_or_path", tok.__class__.__name__)


def _fuse_llama_to_phi3(hf_llama_sd: dict, n_layers: int) -> dict:
    """Convert HF Llama projection keys to the fused Phi-3 layout."""
    out = {
        key: hf_llama_sd[key]
        for key in ("model.embed_tokens.weight", "model.norm.weight", "lm_head.weight")
    }
    for layer_id in range(n_layers):
        prefix = f"model.layers.{layer_id}."
        out[f"{prefix}self_attn.qkv_proj.weight"] = torch.cat(
            [
                hf_llama_sd[f"{prefix}self_attn.q_proj.weight"],
                hf_llama_sd[f"{prefix}self_attn.k_proj.weight"],
                hf_llama_sd[f"{prefix}self_attn.v_proj.weight"],
            ],
            dim=0,
        )
        out[f"{prefix}self_attn.o_proj.weight"] = hf_llama_sd[
            f"{prefix}self_attn.o_proj.weight"
        ]
        out[f"{prefix}mlp.gate_up_proj.weight"] = torch.cat(
            [
                hf_llama_sd[f"{prefix}mlp.gate_proj.weight"],
                hf_llama_sd[f"{prefix}mlp.up_proj.weight"],
            ],
            dim=0,
        )
        for name in (
            "mlp.down_proj.weight",
            "input_layernorm.weight",
            "post_attention_layernorm.weight",
        ):
            out[f"{prefix}{name}"] = hf_llama_sd[f"{prefix}{name}"]
    return out


def _save_sharded_safetensors(
    state_dict: dict,
    output_dir: str,
    *,
    max_shard_size: str = "5GB",
    dtype: torch.dtype = torch.bfloat16,
) -> None:
    """Write a standard HF checkpoint without materializing a full cast copy."""
    from huggingface_hub import split_torch_state_dict_into_shards
    from safetensors.torch import save_file

    split = split_torch_state_dict_into_shards(
        state_dict,
        filename_pattern="model{suffix}.safetensors",
        max_shard_size=max_shard_size,
    )
    for filename, tensor_names in split.filename_to_tensors.items():
        shard = {
            name: state_dict[name].to(dtype=dtype).contiguous()
            for name in tensor_names
        }
        save_file(shard, os.path.join(output_dir, filename))
        print(f"  saved {filename} ({len(shard)} tensors)")
        del shard

    if split.is_sharded:
        with open(os.path.join(output_dir, "model.safetensors.index.json"), "w") as f:
            json.dump(
                {"metadata": split.metadata, "weight_map": split.tensor_to_filename},
                f,
                indent=2,
                sort_keys=True,
            )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def export(
    ckpt_dir: str,
    output_dir: str,
    base_model: str | None = None,
    model_config: str | None = None,
    encoder_n_heads: int | None = None,
    max_codebook_size: int = 4096,
    disabled_ids: list[int] | None = None,
    disable_digit_ids: bool | None = None,
    residual: bool | None = None,
    causal: bool = False,
    max_shard_size: str = "5GB",
):
    """Export a zip2zip-core checkpoint to ext/zip2zip HF format.

    Args:
        ckpt_dir: zip2zip-core checkpoint directory (contains model.pt)
        output_dir: Destination directory for ext/zip2zip format
        base_model: HuggingFace base model name (e.g. meta-llama/Llama-3.2-1B-Instruct)
        model_config: zip2zip-core model config key (e.g. '1B'). Auto-detected if None.
        encoder_n_heads: Number of encoder attention heads. Resolved from meta.pt train
            args if None, falling back to hidden_size // 64 only as a last resort.
        max_codebook_size: Max codebook size (default 4096)
        disabled_ids: Explicit token IDs to disable for LZW. If None (the normal
            case), derived from the base_model tokenizer via
            zip2zip_core.disabled_ids, honoring disable_digit_ids.
        disable_digit_ids: Whether digits were protected from LZW merges during
            training. If None, auto-detected from meta.pt's --disable_digit_ids
            (self-healing, same pattern as lm_eval_adapter.py) — passing an
            explicit disabled_ids list skips this entirely.
        residual: Use residual connection in encoder. If None, restore the
            checkpoint's behavior-only setting from meta.pt.
        causal: Use causal masking in encoder (default False)
    """
    os.makedirs(output_dir, exist_ok=True)
    train_args = _train_args_from_meta(ckpt_dir)
    base_model = base_model or train_args.get("init_from_hf")
    model_config = model_config or train_args.get("model_config")
    if not base_model:
        raise ValueError(
            "meta.pt does not record init_from_hf; pass base_model explicitly"
        )
    if not model_config:
        raise ValueError(
            "meta.pt does not record model_config; pass model_config explicitly"
        )

    # ---- Load checkpoint -----------------------------------------------
    position_mode = _position_mode_from_meta(ckpt_dir)
    model_pt = os.path.join(ckpt_dir, "model.pt")
    print(f"Loading {model_pt} ...")
    sd = torch.load(model_pt, map_location="cpu", weights_only=True, mmap=True)
    if any("base_layer" in k for k in sd):
        print("  Merging LoRA adapters with scaling=alpha/rank from meta.pt")
    sd = _prepare_export_state_dict(sd, ckpt_dir)
    if residual is None:
        residual = _encoder_residual_from_meta(ckpt_dir)
        print(f"  encoder residual={residual} (from meta.pt)")

    decoder_sd, encoder_sd, output_encoder_sd = _split_state_dict(sd)
    untied = bool(output_encoder_sd)
    enc_info = _infer_encoder_config(sd)
    vocab_size = sd["tok_embeddings.weight"].shape[0]
    del sd

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
    _validate_hf_decoder_state_dict(hf_decoder_sd, llama_cfg.n_layers)

    is_phi = (
        str(model_config).lower().startswith("phi")
        or "phi" in str(base_model).lower()
    )
    if is_phi:
        print("Fusing decoder projections for Phi3ForCausalLM ...")
        hf_decoder_sd = _fuse_llama_to_phi3(hf_decoder_sd, llama_cfg.n_layers)

    # A v2 release is self-contained: standard config/tokenizer + decoder
    # shards live beside zip2zip_config.json, so AutoModel can load them directly.
    from transformers import AutoConfig
    hf_config = AutoConfig.from_pretrained(base_model)
    hf_config.vocab_size = vocab_size
    hf_config.tie_word_embeddings = llama_cfg.tie_word_embeddings
    hf_config.max_position_embeddings = llama_cfg.rope.max_seq_len
    hf_config.rope_theta = llama_cfg.rope.theta
    if is_phi:
        # These core checkpoints were trained with unscaled complex RoPE.
        # In particular, inheriting Phi-3.5-mini's upstream LongRoPE vectors
        # here would silently change every decoder layer after export.
        hf_config.rope_scaling = None
        hf_config.sliding_window = None
        if hasattr(hf_config, "original_max_position_embeddings"):
            hf_config.original_max_position_embeddings = 4096
        # The Phi-3 repos advertise remote code (configuration_phi3.py,
        # modeling_phi3.py) that this export does not ship; the native
        # transformers implementation is what loads it, so do not point
        # trust_remote_code=True users at missing files.
        hf_config.__dict__.pop("auto_map", None)
    hf_config.torch_dtype = torch.bfloat16
    hf_config.save_pretrained(output_dir)

    # Generation defaults live in the base repo's generation_config.json, not in
    # config.json: Phi-3 stops on <|end|> (32007) only through that file. Copy
    # it so the export stops where the base model stops.
    from transformers import GenerationConfig
    try:
        gen_config = GenerationConfig.from_pretrained(base_model)
    except (OSError, ValueError):
        gen_config = GenerationConfig.from_model_config(hf_config)
    gen_config.save_pretrained(output_dir)

    print(f"Saving decoder in <= {max_shard_size} bfloat16 shards ...")
    _save_sharded_safetensors(
        hf_decoder_sd,
        output_dir,
        max_shard_size=max_shard_size,
        dtype=torch.bfloat16,
    )

    # ---- Save encoders.safetensors -------------------------------------
    encoders_out = os.path.join(output_dir, "zip2zip_encoders.safetensors")
    print(f"Saving {encoders_out} ...")
    # input_encoder.* always; output_encoder.* only when untied (tie_encoders
    # is then False and the ext runtime scores logits with the output encoder).
    enc_prefixed = {f"input_encoder.{k}": v.contiguous() for k, v in encoder_sd.items()}
    if untied:
        enc_prefixed.update(
            {f"output_encoder.{k}": v.contiguous() for k, v in output_encoder_sd.items()}
        )
    print(f"  tie_encoders={not untied} (input_encoder + "
          f"{'output_encoder' if untied else 'no output_encoder'})")
    from safetensors.torch import save_file
    save_file(
        {key: value.to(dtype=torch.bfloat16).contiguous() for key, value in enc_prefixed.items()},
        encoders_out,
    )

    # ---- Build zip2zip_config.json -------------------------------------
    n_heads = _resolve_encoder_n_heads(ckpt_dir, enc_info["hidden_size"], encoder_n_heads)

    if disabled_ids is not None:
        initial_vocab_size = vocab_size
    else:
        if disable_digit_ids is None:
            disable_digit_ids = _disable_digit_ids_from_meta(ckpt_dir)
            if disable_digit_ids:
                print("  checkpoint was trained with digit-protected LZW (meta.pt) "
                      "— auto-enabling digit protection for export")
        print(f"Loading tokenizer from {base_model} to compute disabled_ids ...")
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(base_model)
        # The model vocabulary is authoritative. Phi-3 reserves rows up to
        # 32063 while its tokenizer currently exposes only 32011 entries;
        # using len(tokenizer) would shift every exported hypertoken by 53.
        disabled_ids = compute_disabled_ids(
            tok, vocab_size, disable_digit_ids=disable_digit_ids
        )
        initial_vocab_size = vocab_size
        print(f"  initial_vocab_size={initial_vocab_size}, disabled_ids count={len(disabled_ids)}")
        _pin_pad_token(tok, model_config)
        print(f"Saving tokenizer to {output_dir} ...")
        tok.save_pretrained(output_dir)

    config = {
        "format_version": 2,
        "base_model_name_or_path": ".",
        "position_mode": position_mode,
        "encoder_type": "res_latent_attn",
        "encoder": {
            "hidden_size": enc_info["hidden_size"],
            "model_hidden_size": enc_info["model_hidden_size"],
            "num_hidden_layers": enc_info["num_hidden_layers"],
            "intermediate_size": enc_info["intermediate_size"],
            "num_heads": n_heads,
            "causal": causal,
            "residual": residual,
            "tie_encoders": not untied,
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

    # Carry meta.pt along so check_export_consistency.py has train_args to
    # compare against -- without this it silently "passes" with nothing checked.
    meta_src = os.path.join(ckpt_dir, "meta.pt")
    if os.path.exists(meta_src):
        shutil.copy(meta_src, os.path.join(output_dir, "meta.pt"))
        print(f"  copied meta.pt (for check_export_consistency.py)")

    print("\nDone. Output:")
    output_files = [
        "zip2zip_config.json",
        *sorted(
            fn
            for fn in os.listdir(output_dir)
            if fn == "model.safetensors"
            or fn == "model.safetensors.index.json"
            or (fn.startswith("model-") and fn.endswith(".safetensors"))
        ),
        "zip2zip_encoders.safetensors",
    ]
    for fn in output_files:
        path = os.path.join(output_dir, fn)
        size_mb = os.path.getsize(path) / 1e6
        print(f"  {fn:30s} {size_mb:8.1f} MB")
    print(f"\nLoad with:\n  Zip2ZipModel.from_pretrained('{output_dir}')")
