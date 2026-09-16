"""Regression tests for strict runtime checkpoint normalization."""

from __future__ import annotations

import os
import subprocess

import pytest
import torch

from zip2zip_core.checkpoint import (
    merge_lora_weights,
    prepare_inference_state_dict,
    strip_wrapper_prefixes,
)
from zip2zip_core.export import (
    _encoder_residual_from_meta,
    _prepare_export_state_dict,
    _validate_hf_decoder_state_dict,
    refuse_base_token_positions,
    _position_mode_from_meta,
)
from zip2zip_core.train import _clean_state_dict


def _lora_state() -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    base = torch.tensor(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float32
    )
    a = torch.tensor(
        [[1.0, -1.0, 0.5], [0.25, 2.0, -0.5]], dtype=torch.float32
    )
    b = torch.tensor(
        [[2.0, -1.0], [0.5, 3.0]], dtype=torch.float32
    )
    state = {
        "layers.0._orig_mod.proj.base_layer.weight": base,
        "layers.0._orig_mod.proj.lora_A.weight": a,
        "layers.0._orig_mod.proj.lora_B.weight": b,
        "layers.0._orig_mod.proj.base_layer.bias": torch.tensor([0.5, -0.5]),
        "_fsdp_wrapped_module.norm.weight": torch.ones(2),
    }
    expected = base + (b @ a) * 3.0
    return state, expected


def test_prepare_inference_state_folds_nonunit_lora_and_wrappers():
    state, expected = _lora_state()
    merged = prepare_inference_state_dict(
        state, {"lora_rank": 2, "lora_alpha": 6.0}
    )

    assert torch.equal(merged["layers.0.proj.weight"], expected)
    assert torch.equal(
        merged["layers.0.proj.bias"], torch.tensor([0.5, -0.5])
    )
    assert torch.equal(merged["norm.weight"], torch.ones(2))
    assert not any("lora_" in key or "base_layer" in key for key in merged)


def test_zero_alpha_is_not_replaced_by_default_scaling():
    state, _ = _lora_state()
    merged = prepare_inference_state_dict(
        state, {"lora_rank": 2, "lora_alpha": 0.0}
    )
    normalized = strip_wrapper_prefixes(state)
    assert torch.equal(
        merged["layers.0.proj.weight"],
        normalized["layers.0.proj.base_layer.weight"],
    )


@pytest.mark.parametrize(
    "mutate, match",
    [
        (
            lambda state: state.pop(
                "layers.0._orig_mod.proj.lora_B.weight"
            ),
            "incomplete LoRA state",
        ),
        (
            lambda state: None,
            "positive lora_rank",
        ),
    ],
)
def test_malformed_lora_checkpoint_fails(mutate, match):
    state, _ = _lora_state()
    mutate(state)
    args = (
        {"lora_rank": 2, "lora_alpha": 6.0}
        if "incomplete" in match
        else {"lora_rank": None, "lora_alpha": 6.0}
    )
    with pytest.raises(ValueError, match=match):
        prepare_inference_state_dict(state, args)


def test_rank_and_shape_mismatches_fail():
    state, _ = _lora_state()
    with pytest.raises(ValueError, match="rank mismatch"):
        prepare_inference_state_dict(
            state, {"lora_rank": 3, "lora_alpha": 6.0}
        )

    normalized = strip_wrapper_prefixes(state)
    normalized["layers.0.proj.lora_B.weight"] = torch.ones(3, 3)
    with pytest.raises(ValueError, match="incompatible LoRA shapes"):
        merge_lora_weights(normalized, scaling=1.0)


def test_wrapper_normalization_collision_fails():
    state = {
        "layer.weight": torch.ones(1),
        "_orig_mod.layer.weight": torch.zeros(1),
    }
    with pytest.raises(ValueError, match="key collision"):
        strip_wrapper_prefixes(state)
    with pytest.raises(ValueError, match="key collision"):
        _clean_state_dict(state)


def test_gated_rope_export_is_refused(tmp_path):
    torch.save(
        {"args": {"gated_compressed_rope": True}},
        tmp_path / "meta.pt",
    )
    with pytest.raises(NotImplementedError, match="gated_compressed_rope"):
        refuse_base_token_positions(str(tmp_path))


def test_base_token_positions_export_as_v2_position_mode(tmp_path):
    torch.save(
        {"args": {"base_token_positions": True}},
        tmp_path / "meta.pt",
    )
    assert _position_mode_from_meta(str(tmp_path)) == "base_token_end"


def test_export_lora_requires_and_uses_checkpoint_metadata(tmp_path):
    state, expected = _lora_state()
    with pytest.raises(FileNotFoundError, match="metadata is missing"):
        _prepare_export_state_dict(state, str(tmp_path))

    torch.save({"args": {"model_config": "1B"}}, tmp_path / "meta.pt")
    with pytest.raises(ValueError, match="positive lora_rank"):
        _prepare_export_state_dict(state, str(tmp_path))

    torch.save(
        {
            "args": {
                "model_config": "1B",
                "lora_rank": 2,
                "lora_alpha": 6.0,
            }
        },
        tmp_path / "meta.pt",
    )
    merged = _prepare_export_state_dict(state, str(tmp_path))
    assert torch.equal(merged["layers.0.proj.weight"], expected)


def test_export_rejects_a_wholly_missing_decoder_tensor():
    state = {
        "model.embed_tokens.weight": torch.ones(1),
        "model.norm.weight": torch.ones(1),
        "lm_head.weight": torch.ones(1),
    }
    for suffix in (
        "self_attn.q_proj.weight",
        "self_attn.k_proj.weight",
        "self_attn.v_proj.weight",
        "self_attn.o_proj.weight",
        "mlp.gate_proj.weight",
        "mlp.up_proj.weight",
        "mlp.down_proj.weight",
        "input_layernorm.weight",
        "post_attention_layernorm.weight",
    ):
        state[f"model.layers.0.{suffix}"] = torch.ones(1)

    _validate_hf_decoder_state_dict(state, 1)
    del state["model.layers.0.self_attn.k_proj.weight"]
    with pytest.raises(ValueError, match="missing=.*k_proj"):
        _validate_hf_decoder_state_dict(state, 1)


def test_export_restores_encoder_residual_from_metadata(tmp_path):
    torch.save(
        {"args": {"model_config": "1B", "no_encoder_residual": True}},
        tmp_path / "meta.pt",
    )
    assert _encoder_residual_from_meta(str(tmp_path)) is False


def test_eval_artifact_paths_distinguish_modes(tmp_path):
    script = os.path.join(
        os.path.dirname(__file__), "..", "scripts", "eval_ckpt_rcp.sh"
    )
    common = {
        "Z2Z_SCRATCH": str(tmp_path),
        "CKPT_DIR": "/checkpoint/run/step_8000",
        "PRESET": "default",
        "TASKS": "gsm8k",
        "TIMESTAMP": "20260727_120000",
        "RUNAI_JOB_NAME": "same-job",
        "EVAL_PATHS_ONLY": "1",
    }

    def paths(mode: str) -> str:
        env = dict(os.environ, **common, EVAL_MODE=mode)
        return subprocess.check_output(
            ["bash", script], env=env, text=True
        )

    base = paths("base")
    compressed = paths("compressed")
    assert base != compressed
    assert "mode-base" in base
    assert "mode-compressed" in compressed
    subprocess.run(["bash", "-n", script], check=True)
