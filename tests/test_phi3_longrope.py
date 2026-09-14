"""Phi-3 LongRoPE compatibility and semantic-position tests."""

from __future__ import annotations

import importlib.util
import os
import types

import torch
from transformers import Phi3Config
from transformers.modeling_rope_utils import _compute_longrope_parameters

from zip2zip_core.rope import RoPE


def _load_inference_module():
    path = os.path.join(
        os.path.dirname(__file__), "..", "scripts", "inference.py"
    )
    spec = importlib.util.spec_from_file_location(
        "_phi3_longrope_inference_test", path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _hf_config() -> Phi3Config:
    return Phi3Config(
        hidden_size=8,
        num_attention_heads=1,
        num_key_value_heads=1,
        max_position_embeddings=32,
        original_max_position_embeddings=8,
        rope_theta=10_000.0,
        rope_scaling={
            "type": "longrope",
            "short_factor": [1.0, 1.1, 1.2, 1.3],
            "long_factor": [2.0, 3.0, 4.0, 5.0],
        },
    )


def _expected_cache(inv_freq: torch.Tensor, attention_factor: float):
    positions = torch.arange(32, dtype=inv_freq.dtype)
    phases = torch.outer(positions, inv_freq).float()
    return torch.polar(torch.full_like(phases, attention_factor), phases)


def test_phi3_longrope_caches_match_transformers_reference():
    inference = _load_inference_module()
    hf_config = _hf_config()
    base = RoPE.Config(
        dim=8,
        max_seq_len=32,
        theta=10_000.0,
        backend="complex",
        scaling="none",
    )
    core = inference._phi3_longrope_rope_config(base, hf_config)
    rope = RoPE(core)

    short_inv_freq, attention_factor = _compute_longrope_parameters(
        hf_config, "cpu", seq_len=8
    )
    long_inv_freq, long_attention_factor = _compute_longrope_parameters(
        hf_config, "cpu", seq_len=9
    )

    torch.testing.assert_close(
        rope.cache,
        _expected_cache(short_inv_freq, attention_factor),
    )
    assert rope.longrope_long_cache is not None
    torch.testing.assert_close(
        rope.longrope_long_cache,
        _expected_cache(long_inv_freq, long_attention_factor),
    )


def test_phi3_longrope_switch_uses_max_semantic_position():
    inference = _load_inference_module()
    core = inference._phi3_longrope_rope_config(
        RoPE.Config(dim=8, max_seq_len=32),
        _hf_config(),
    )
    short_cache = torch.tensor([1.0])
    long_cache = torch.tensor([2.0])
    fake_model = types.SimpleNamespace(
        zip2zip_config=types.SimpleNamespace(rope=core),
        freqs_cis=short_cache,
        longrope_long_freqs_cis=long_cache,
        _longrope_regime=None,
    )
    tokens = torch.zeros((1, 2), dtype=torch.long)
    select = inference.Zip2ZipLlama3Model._select_rope_cache

    selected_short = select(
        fake_model,
        tokens,
        torch.tensor([[0, 7]], dtype=torch.long),
    )
    selected_long = select(
        fake_model,
        tokens,
        torch.tensor([[0, 8]], dtype=torch.long),
    )

    assert selected_short is short_cache
    assert selected_long is long_cache
