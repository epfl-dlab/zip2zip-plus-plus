"""Validation and publishing helpers for the four Zip2Zip++ releases."""

from __future__ import annotations

import json
import re
import tempfile
from pathlib import Path

import torch

from zip2zip_core.export import export
from zip2zip_core.hub import upload_folder


INFERENCE_REVISION = "hf"

SUPPORTED_MODELS = {
    "1B": "meta-llama/Llama-3.2-1B-Instruct",
    "3B": "meta-llama/Llama-3.2-3B-Instruct",
    "Phi3.5-mini": "microsoft/Phi-3.5-mini-instruct",
    "Phi3-medium": "microsoft/Phi-3-medium-4k-instruct",
}

EXPECTED_VOCAB_SIZES = {
    "1B": 128256,
    "3B": 128256,
    "Phi3.5-mini": 32064,
    "Phi3-medium": 32064,
}


def checkpoint_step(ckpt_dir: Path) -> int:
    """Return the step encoded by a canonical ``step_N`` directory name."""
    match = re.fullmatch(r"step_(\d+)", ckpt_dir.name)
    if match is None:
        raise ValueError(f"checkpoint directory must be named step_N: {ckpt_dir}")
    return int(match.group(1))


def read_release_metadata(ckpt_dir: Path) -> dict:
    """Validate that a training checkpoint is one of the four releases."""
    meta_path = ckpt_dir / "meta.pt"
    model_path = ckpt_dir / "model.pt"
    if not meta_path.is_file() or not model_path.is_file():
        raise FileNotFoundError(
            f"checkpoint must contain model.pt and meta.pt: {ckpt_dir}"
        )

    meta = torch.load(meta_path, map_location="cpu", weights_only=False)
    train_args = meta.get("args", {}) if isinstance(meta, dict) else {}
    if not isinstance(train_args, dict):
        raise ValueError(f"invalid train args in {meta_path}")

    model_config = train_args.get("model_config")
    base_model = train_args.get("init_from_hf")
    expected_base = SUPPORTED_MODELS.get(model_config)
    if expected_base is None or base_model != expected_base:
        raise ValueError(
            "not one of the four Zip2Zip++ release models: "
            f"model_config={model_config!r}, init_from_hf={base_model!r}"
        )

    required = {
        "base_token_positions": True,
        "untied_hyper_encoder": True,
        "hyper_encoder_type": "flat",
        "max_subtokens": 4,
        "disable_digit_ids": True,
    }
    mismatches = [
        f"{key}={train_args.get(key)!r} (expected {value!r})"
        for key, value in required.items()
        if train_args.get(key) != value
    ]
    for unsupported in ("two_axis_rope", "gated_compressed_rope"):
        if train_args.get(unsupported, False):
            mismatches.append(f"{unsupported}=True (unsupported)")
    if not isinstance(train_args.get("max_codebook_size"), int) or train_args[
        "max_codebook_size"
    ] <= 0:
        mismatches.append("max_codebook_size must be a positive integer")
    if mismatches:
        raise ValueError(
            "checkpoint is not in the Zip2Zip++ release contract: "
            + "; ".join(mismatches)
        )
    return train_args


def validate_export(export_dir: Path, train_args: dict) -> None:
    """Fail closed if an export is incomplete or incompatible with zip2zip."""
    with open(export_dir / "zip2zip_config.json", encoding="utf-8") as file:
        config = json.load(file)
    expected = {
        "format_version": 2,
        "base_model_name_or_path": ".",
        "position_mode": "base_token_end",
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(
                f"exported {key}={config.get(key)!r}, expected {value!r}"
            )
    if config["encoder"].get("tie_encoders") is not False:
        raise ValueError("Zip2Zip++ export must contain untied encoders")
    if config["compression"].get("max_subtokens") != 4:
        raise ValueError("Zip2Zip++ export must use max_subtokens=4")
    expected_vocab_size = EXPECTED_VOCAB_SIZES[train_args["model_config"]]
    if config["compression"].get("initial_vocab_size") != expected_vocab_size:
        raise ValueError(
            "exported hypertoken boundary does not match the decoder vocabulary: "
            f"{config['compression'].get('initial_vocab_size')} != "
            f"{expected_vocab_size}"
        )

    decoder = export_dir / "model.safetensors"
    index = export_dir / "model.safetensors.index.json"
    if not decoder.is_file() and not index.is_file():
        raise FileNotFoundError("export has neither decoder weights nor a shard index")
    for required_file in ("config.json", "zip2zip_encoders.safetensors"):
        if not (export_dir / required_file).is_file():
            raise FileNotFoundError(f"export is missing {required_file}")

    from safetensors import safe_open

    if index.is_file():
        with open(index, encoding="utf-8") as file:
            weight_map = json.load(file)["weight_map"]
        decoder_keys = set(weight_map)
        missing_shards = sorted(
            filename
            for filename in set(weight_map.values())
            if not (export_dir / filename).is_file()
        )
        if missing_shards:
            raise FileNotFoundError(
                f"export is missing decoder shards: {missing_shards}"
            )
    else:
        with safe_open(decoder, framework="pt") as file:
            decoder_keys = set(file.keys())

    from transformers import AutoConfig, AutoModelForCausalLM

    hf_config = AutoConfig.from_pretrained(export_dir)
    with torch.device("meta"):
        expected_decoder = AutoModelForCausalLM.from_config(hf_config)
    expected_keys = set(expected_decoder.state_dict())
    missing = sorted(expected_keys - decoder_keys)
    unexpected = sorted(decoder_keys - expected_keys)
    if missing or unexpected:
        raise ValueError(
            "decoder key mismatch: "
            f"missing={missing[:8]}, unexpected={unexpected[:8]}"
        )

    with safe_open(export_dir / "zip2zip_encoders.safetensors", framework="pt") as file:
        keys = set(file.keys())
    if not any(key.startswith("input_encoder.") for key in keys):
        raise ValueError("export is missing input_encoder weights")
    if not any(key.startswith("output_encoder.") for key in keys):
        raise ValueError("export is missing output_encoder weights")


def render_model_card(repo_id: str, step: int, train_args: dict) -> str:
    """Build the model card shared by the training and inference revisions."""
    return f"""---
library_name: zip2zip
base_model: {train_args['init_from_hf']}
tags:
  - zip2zip
  - zip2zip++
  - adaptive-tokenization
---

# {repo_id.split('/')[-1]}

Zip2Zip++ checkpoint based on `{train_args['init_from_hf']}` (training step
{step}). The repository keeps the original training checkpoint on `main` and
the user-facing, self-contained inference export on `{INFERENCE_REVISION}`.

## Usage

```bash
pip install "zip2zip>=0.2.0"
```

```python
from zip2zip import Zip2ZipModel, Zip2ZipTokenizer

repo_id = "{repo_id}"
tokenizer = Zip2ZipTokenizer.from_pretrained(
    repo_id, revision="{INFERENCE_REVISION}"
)
model = Zip2ZipModel.from_pretrained(
    repo_id, revision="{INFERENCE_REVISION}", device_map="auto", dtype="auto"
)
inputs = tokenizer("Hello", return_tensors="pt").to(model.device)
output = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

Use `main` only with zip2zip-core when resuming training or reproducing the
export. It is not a Transformers/zip2zip inference revision.
"""


def export_release(
    ckpt_dir: Path,
    export_dir: Path,
    repo_id: str,
    *,
    max_shard_size: str = "5GB",
) -> tuple[int, dict]:
    """Export and validate one Zip2Zip++ checkpoint without uploading it."""
    ckpt_dir = ckpt_dir.resolve()
    export_dir = export_dir.resolve()
    train_args = read_release_metadata(ckpt_dir)
    step = checkpoint_step(ckpt_dir)
    export_dir.mkdir(parents=True, exist_ok=True)
    if any(export_dir.iterdir()):
        raise ValueError(f"output directory must be empty: {export_dir}")

    export(
        ckpt_dir=str(ckpt_dir),
        output_dir=str(export_dir),
        max_codebook_size=train_args["max_codebook_size"],
        max_shard_size=max_shard_size,
    )
    (export_dir / "README.md").write_text(
        render_model_card(repo_id, step, train_args), encoding="utf-8"
    )
    validate_export(export_dir, train_args)
    return step, train_args


def publish_release(
    ckpt_dir: Path,
    export_dir: Path,
    repo_id: str,
    step: int,
    train_args: dict,
    *,
    api=None,
) -> None:
    """Publish raw training files to ``main`` and inference files to ``hf``."""
    if api is None:
        from huggingface_hub import HfApi

        api = HfApi()

    upload_folder(
        repo_id,
        str(ckpt_dir),
        branch="main",
        step=step,
        label="Zip2Zip++ training checkpoint",
        api=api,
    )

    # Do not modify the local checkpoint just to add public documentation.
    with tempfile.NamedTemporaryFile("w", suffix=".md", encoding="utf-8") as card:
        card.write(render_model_card(repo_id, step, train_args))
        card.flush()
        api.upload_file(
            path_or_fileobj=card.name,
            path_in_repo="README.md",
            repo_id=repo_id,
            revision="main",
            commit_message=f"Step {step} (Zip2Zip++ model card)",
        )

    upload_folder(
        repo_id,
        str(export_dir),
        branch=INFERENCE_REVISION,
        step=step,
        label="Zip2Zip++ inference export",
        api=api,
    )
