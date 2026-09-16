"""Tests for the four-model Zip2Zip++ release contract."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from zip2zip_core.release import (
    INFERENCE_REVISION,
    checkpoint_step,
    publish_release,
    read_release_metadata,
    render_model_card,
)


def _release_args() -> dict:
    return {
        "model_config": "1B",
        "init_from_hf": "meta-llama/Llama-3.2-1B-Instruct",
        "base_token_positions": True,
        "untied_hyper_encoder": True,
        "hyper_encoder_type": "flat",
        "max_subtokens": 4,
        "disable_digit_ids": True,
        "token_type_loss_weight": 0.05,
        "zero_init_encoder_output": True,
        "no_encoder_residual": False,
        "encoder_n_layers": 2,
        "warmstart_steps": 0,
        "max_codebook_size": 4096,
    }


def _checkpoint(tmp_path: Path, args: dict | None = None) -> Path:
    ckpt_dir = tmp_path / "step_8000"
    ckpt_dir.mkdir()
    torch.save({}, ckpt_dir / "model.pt")
    torch.save({"args": args or _release_args()}, ckpt_dir / "meta.pt")
    return ckpt_dir


def test_release_metadata_accepts_supported_recipe(tmp_path):
    ckpt_dir = _checkpoint(tmp_path)
    assert checkpoint_step(ckpt_dir) == 8000
    assert read_release_metadata(ckpt_dir) == _release_args()


@pytest.mark.parametrize(
    ("model_config", "base_model"),
    [
        ("1B", "meta-llama/Llama-3.2-1B-Instruct"),
        ("3B", "meta-llama/Llama-3.2-3B-Instruct"),
        ("Phi3.5-mini", "microsoft/Phi-3.5-mini-instruct"),
        ("Phi3-medium", "microsoft/Phi-3-medium-4k-instruct"),
    ],
)
def test_release_metadata_accepts_all_four_models(
    tmp_path, model_config, base_model
):
    args = _release_args() | {
        "model_config": model_config, "init_from_hf": base_model
    }
    assert read_release_metadata(_checkpoint(tmp_path, args)) == args

def test_release_metadata_rejects_non_zip2zippp_recipe(tmp_path):

    args = _release_args()
    args["base_token_positions"] = False
    ckpt_dir = _checkpoint(tmp_path, args)
    with pytest.raises(ValueError, match="base_token_positions=False"):
        read_release_metadata(ckpt_dir)


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("token_type_loss_weight", 0.0),
        ("zero_init_encoder_output", False),
        ("no_encoder_residual", True),
        ("encoder_n_layers", 4),
        ("warmstart_steps", 200),
        ("share_hyper_encoder_weights", True),
        ("online_codebook_mask", True),
        ("base_view_replay_prob", 0.25),
    ],
)
def test_release_metadata_rejects_non_v064_ablation(tmp_path, key, value):
    args = _release_args()
    args[key] = value
    ckpt_dir = _checkpoint(tmp_path, args)
    with pytest.raises(ValueError, match=key):
        read_release_metadata(ckpt_dir)


def test_model_card_points_users_to_hf_revision():
    card = render_model_card("epfl-dlab/example", 8000, _release_args())
    assert 'revision="hf"' in card
    assert "original training checkpoint on `main`" in card
    assert "self-contained inference export on `hf`" in card


def test_publish_keeps_main_training_and_hf_inference(monkeypatch, tmp_path):
    ckpt_dir = _checkpoint(tmp_path)
    export_dir = tmp_path / "export"
    export_dir.mkdir()
    (export_dir / "README.md").write_text("export", encoding="utf-8")
    uploads = []

    def fake_upload_folder(repo_id, folder_path, **kwargs):
        uploads.append((repo_id, Path(folder_path), kwargs["branch"]))

    class FakeApi:
        def upload_file(self, **kwargs):
            uploads.append(
                (kwargs["repo_id"], Path("README.md"), kwargs["revision"])
            )

    monkeypatch.setattr("zip2zip_core.release.upload_folder", fake_upload_folder)
    publish_release(
        ckpt_dir,
        export_dir,
        "epfl-dlab/example",
        8000,
        _release_args(),
        api=FakeApi(),
    )

    assert uploads == [
        ("epfl-dlab/example", ckpt_dir, "main"),
        ("epfl-dlab/example", Path("README.md"), "main"),
        ("epfl-dlab/example", export_dir, INFERENCE_REVISION),
    ]
