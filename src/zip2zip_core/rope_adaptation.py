"""Inference-time RoPE adaptations shared by CLI and evaluation."""

from __future__ import annotations

import math

from zip2zip_core.rope import RoPE as LongRoPE


def phi3_longrope_config(rope_cfg, hf_config):
    """Translate an official Phi-3 HF LongRoPE config into the core config."""
    rope_scaling = getattr(hf_config, "rope_scaling", None)
    if not isinstance(rope_scaling, dict):
        raise ValueError("Phi3-LongRoPE source has no rope_scaling dictionary")
    rope_type = rope_scaling.get("rope_type", rope_scaling.get("type"))
    if rope_type != "longrope":
        raise ValueError(
            "Phi3-LongRoPE requires a longrope base config, got "
            f"{rope_type!r}"
        )
    if rope_cfg.backend != "complex":
        raise ValueError("Phi3-LongRoPE requires the complex RoPE backend")

    head_dim = getattr(hf_config, "head_dim", None) or (
        hf_config.hidden_size // hf_config.num_attention_heads
    )
    if int(head_dim) != rope_cfg.dim:
        raise ValueError(
            "Phi3-LongRoPE head dimension mismatch: "
            f"base config has {int(head_dim)}, core expects {rope_cfg.dim}"
        )
    short_factors = tuple(float(value) for value in rope_scaling["short_factor"])
    long_factors = tuple(float(value) for value in rope_scaling["long_factor"])
    expected = rope_cfg.dim // 2
    if len(short_factors) != expected or len(long_factors) != expected:
        raise ValueError(
            "Phi3-LongRoPE factor length mismatch: expected "
            f"{expected}, got short={len(short_factors)} "
            f"long={len(long_factors)}"
        )

    original_max = int(hf_config.original_max_position_embeddings)
    max_position_embeddings = int(hf_config.max_position_embeddings)
    factor = max_position_embeddings / original_max
    attention_factor = rope_scaling.get("attention_factor")
    if attention_factor is None:
        attention_factor = (
            1.0
            if factor <= 1.0
            else math.sqrt(1 + math.log(factor) / math.log(original_max))
        )

    return LongRoPE.Config(
        dim=rope_cfg.dim,
        max_seq_len=max(rope_cfg.max_seq_len, max_position_embeddings),
        theta=float(hf_config.rope_theta),
        backend=rope_cfg.backend,
        scaling="longrope",
        scaling_factor=rope_cfg.scaling_factor,
        low_freq_factor=rope_cfg.low_freq_factor,
        high_freq_factor=rope_cfg.high_freq_factor,
        original_max_position_embeddings=original_max,
        rope_factor=rope_cfg.rope_factor,
        beta_fast=rope_cfg.beta_fast,
        beta_slow=rope_cfg.beta_slow,
        original_seq_len=rope_cfg.original_seq_len,
        mscale=rope_cfg.mscale,
        longrope_short_factors=short_factors,
        longrope_long_factors=long_factors,
        longrope_attention_factor=float(attention_factor),
    )
