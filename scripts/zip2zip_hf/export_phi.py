"""Export a from-scratch Phi3.5-mini zip2zip-core checkpoint to a *self-contained*
ext/zip2zip HF directory whose base is a real **Phi3ForCausalLM**.

Our decoder is torchtitan-Llama-structured (separate q/k/v and gate/up projections)
with Phi-3.5-mini dimensions. Phi-3 is mathematically the same network but stores
**fused** qkv_proj and gate_up_proj. This script:

  1. converts the decoder to HF-Llama tensors via Llama3StateDictAdapter (this also
     applies the correct rotate_half RoPE permutation — shared by Llama and Phi3);
  2. FUSES q/k/v -> self_attn.qkv_proj and gate/up -> mlp.gate_up_proj to get
     Phi3ForCausalLM weights (model.safetensors);
  3. writes a Phi3Config config.json with our dims (untied embeddings);
  4. exports the hyper-encoder (encoders.safetensors) + saves the Phi tokenizer;
  5. writes zip2zip_config.json (base_model_name_or_path -> the export dir itself,
     initial_vocab_size = 32064 = the value used during training).

Usage:
    python scripts/zip2zip_hf/export_phi.py \
        --ckpt_dir .../candidate-Phi35-MS3-flat-sft-1bt/step_7630 \
        --output_dir .../export/Phi35-sft-hf
Then (with ext/zip2zip):  Zip2ZipModel.from_pretrained("<output_dir>")
"""

from __future__ import annotations

import argparse
import json
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_ROOT, "src"))
sys.path.insert(0, os.path.join(_ROOT, "ext", "torchtitan"))

import torch

from zip2zip_core.configs import zip2zip_llama_configs
from zip2zip_core.export import (
    _split_state_dict, _infer_encoder_config, _merge_lora_weights, _lora_scaling_from_meta,
    refuse_base_token_positions,
)

PHI_TOKENIZER = "microsoft/Phi-3.5-mini-instruct"
MODEL_CONFIG = "Phi3.5-mini"


def _phi_disabled_ids(tok, vocab_size: int) -> list[int]:
    special = set(tok.all_special_ids or [])
    added = set(tok.get_added_vocab().values())
    return sorted(i for i in (special | added) if 0 <= i < vocab_size)


def _fuse_llama_to_phi3(hf_llama_sd: dict, n_layers: int) -> dict:
    """Fuse HF-Llama per-layer q/k/v -> qkv_proj and gate/up -> gate_up_proj (Phi3 layout)."""
    out = {}
    # pass-through tensors (same names in Phi3)
    for k in ("model.embed_tokens.weight", "model.norm.weight", "lm_head.weight"):
        if k in hf_llama_sd:
            out[k] = hf_llama_sd[k]
    for i in range(n_layers):
        p = f"model.layers.{i}."
        q = hf_llama_sd[f"{p}self_attn.q_proj.weight"]
        k = hf_llama_sd[f"{p}self_attn.k_proj.weight"]
        v = hf_llama_sd[f"{p}self_attn.v_proj.weight"]
        # Phi3 splits qkv_proj as [q | k | v] along the output dim.
        out[f"{p}self_attn.qkv_proj.weight"] = torch.cat([q, k, v], dim=0)
        out[f"{p}self_attn.o_proj.weight"] = hf_llama_sd[f"{p}self_attn.o_proj.weight"]
        g = hf_llama_sd[f"{p}mlp.gate_proj.weight"]
        u = hf_llama_sd[f"{p}mlp.up_proj.weight"]
        # Phi3 chunks gate_up_proj as [gate | up].
        out[f"{p}mlp.gate_up_proj.weight"] = torch.cat([g, u], dim=0)
        out[f"{p}mlp.down_proj.weight"] = hf_llama_sd[f"{p}mlp.down_proj.weight"]
        out[f"{p}input_layernorm.weight"] = hf_llama_sd[f"{p}input_layernorm.weight"]
        out[f"{p}post_attention_layernorm.weight"] = hf_llama_sd[f"{p}post_attention_layernorm.weight"]
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt_dir", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--max_codebook_size", type=int, default=4096)
    args = p.parse_args()

    from transformers import AutoTokenizer, Phi3Config
    from safetensors.torch import save_file
    from torchtitan.models.llama3.state_dict_adapter import Llama3StateDictAdapter

    os.makedirs(args.output_dir, exist_ok=True)
    cfg = zip2zip_llama_configs[MODEL_CONFIG]
    attn = cfg.layer.attention
    n_layers = cfg.n_layers
    vocab_size = cfg.vocab_size
    tok = AutoTokenizer.from_pretrained(PHI_TOKENIZER)
    disabled_ids = _phi_disabled_ids(tok, vocab_size)
    print(f"Phi3 export: vocab={vocab_size}, layers={n_layers}, "
          f"heads={attn.n_heads}/{attn.n_kv_heads}, disabled_ids={disabled_ids}")

    # ---- load + split checkpoint ----
    refuse_base_token_positions(args.ckpt_dir)
    model_pt = os.path.join(args.ckpt_dir, "model.pt")
    print(f"Loading {model_pt} ...")
    sd = torch.load(model_pt, map_location="cpu", weights_only=True)
    sd = _merge_lora_weights(sd, _lora_scaling_from_meta(args.ckpt_dir))
    decoder_sd, encoder_sd, output_encoder_sd = _split_state_dict(sd)
    untied = bool(output_encoder_sd)
    enc_info = _infer_encoder_config(sd)

    # ---- decoder -> HF-Llama (correct RoPE permute) -> fuse to Phi3 ----
    print("Converting decoder -> HF-Llama -> fused Phi3 ...")
    adapter = Llama3StateDictAdapter(cfg, None)   # cfg has .layer.attention.n_heads/.dim
    hf_llama_sd = adapter.to_hf(decoder_sd)
    phi3_sd = _fuse_llama_to_phi3(hf_llama_sd, n_layers)
    save_file({k: v.contiguous().clone() for k, v in phi3_sd.items()},
              os.path.join(args.output_dir, "model.safetensors"))
    print(f"  saved model.safetensors ({len(phi3_sd)} tensors, Phi3ForCausalLM)")

    # ---- encoder -> encoders.safetensors (input_encoder.* always; output_encoder.* when untied) ----
    enc_prefixed = {f"input_encoder.{k}": v.contiguous() for k, v in encoder_sd.items()}
    if untied:
        enc_prefixed.update(
            {f"output_encoder.{k}": v.contiguous() for k, v in output_encoder_sd.items()}
        )
    save_file(enc_prefixed, os.path.join(args.output_dir, "encoders.safetensors"))
    print(f"  saved encoders.safetensors ({len(enc_prefixed)} tensors, tie_encoders={not untied})")

    # ---- Phi3 config.json ----
    phi3_config = Phi3Config(
        vocab_size=vocab_size,
        hidden_size=cfg.dim,
        intermediate_size=cfg.layer.feed_forward.hidden_dim,
        num_hidden_layers=n_layers,
        num_attention_heads=attn.n_heads,
        num_key_value_heads=attn.n_kv_heads or attn.n_heads,
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attention_dropout=0.0,
        hidden_act="silu",
        max_position_embeddings=cfg.rope.max_seq_len,
        original_max_position_embeddings=4096,
        rms_norm_eps=cfg.norm_eps,
        rope_theta=cfg.rope.theta,
        rope_scaling=None,
        tie_word_embeddings=cfg.tie_word_embeddings,
        bos_token_id=tok.bos_token_id,
        eos_token_id=tok.eos_token_id,
        pad_token_id=cfg.pad_token_id,
        sliding_window=None,
        torch_dtype="bfloat16",
    )
    phi3_config.save_pretrained(args.output_dir)
    print(f"  wrote config.json (Phi3ForCausalLM, dim={cfg.dim}, layers={n_layers}, "
          f"ffn={cfg.layer.feed_forward.hidden_dim}, theta={cfg.rope.theta}, "
          f"tie={cfg.tie_word_embeddings})")

    # ---- Phi tokenizer ----
    tok.save_pretrained(args.output_dir)
    print("  saved Phi tokenizer")

    # ---- zip2zip_config.json (self-contained base + training-time compression) ----
    n_heads = enc_info["hidden_size"] // 64
    z = {
        "base_model_name_or_path": os.path.abspath(args.output_dir),
        "encoder_type": "res_latent_attn",
        "encoder": {
            "hidden_size": enc_info["hidden_size"],
            "model_hidden_size": enc_info["model_hidden_size"],
            "num_hidden_layers": enc_info["num_hidden_layers"],
            "intermediate_size": enc_info["intermediate_size"],
            "num_heads": n_heads,
            "causal": False,
            "residual": True,
            "tie_encoders": not untied,
            "position_encoding": None,
        },
        "compression": {
            "initial_vocab_size": vocab_size,   # 32064 — value used during training
            "max_codebook_size": args.max_codebook_size,
            "max_subtokens": enc_info["max_subtokens"],
            "disabled_ids": disabled_ids,
        },
    }
    with open(os.path.join(args.output_dir, "zip2zip_config.json"), "w") as f:
        json.dump(z, f, indent=2)
    print(f"  wrote zip2zip_config.json (initial_vocab_size={vocab_size}, "
          f"max_subtokens={enc_info['max_subtokens']})")

    print("\nDone. Self-contained Phi3 zip2zip dir:", os.path.abspath(args.output_dir))
    print("Load with:  Zip2ZipModel.from_pretrained('%s')" % os.path.abspath(args.output_dir))
    print("When pushing to the Hub, set base_model_name_or_path to the repo id.")


if __name__ == "__main__":
    main()
