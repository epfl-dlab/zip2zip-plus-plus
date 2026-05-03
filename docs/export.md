# Export and Interoperability

## Overview

zip2zip-core (training) and [zip2zip](https://github.com/epfl-dlab/zip2zip) (inference) use different model formats. This document covers the export process and how the two formats relate.

## Format comparison

| | zip2zip-core (torchtitan) | zip2zip (HuggingFace) |
|---|---|---|
| **Decoder weights** | `model.pt` (torchtitan keys) | `model.safetensors` (HF Llama keys) |
| **Encoder weights** | Inside `model.pt` (`hyper_encoder.*`) | `encoders.safetensors` (`input_encoder.*`) |
| **Config** | `meta.pt` (training args) | `zip2zip_config.json` |
| **Optimizer** | `optimizer.pt` | Not included |
| **HF branch** | `main` | `hf` |

## Exporting a checkpoint

### Automatic (recommended)

`push_checkpoint.py` auto-exports to the `hf` branch:

```bash
python scripts/push_checkpoint.py \
    --ckpt_dir /path/to/step_6000 \
    --repo_id epfl-dlab/Llaza-3.2-1B-v0.1
```

This pushes the training checkpoint to `main` and the exported HF format to `hf`. Use `--no_export` to skip auto-export.

### Manual export

```bash
python scripts/zip2zip_hf/export_to_zip2zip.py \
    --ckpt_dir /path/to/step_6000 \
    --output_dir /path/to/export \
    --base_model meta-llama/Llama-3.2-1B-Instruct
```

Or use the Python API:

```python
from zip2zip_core.export import export

export(
    ckpt_dir="/path/to/step_6000",
    output_dir="/path/to/export",
    base_model="meta-llama/Llama-3.2-1B-Instruct",
    model_config="1B",  # optional, auto-detected if omitted
)
```

### Batch export

Export multiple checkpoints at once:

```bash
bash scripts/zip2zip_hf/batch_export.sh
```

## Export process

The `export()` function in `src/zip2zip_core/export.py` performs these steps:

### 1. Load and split state dict

`model.pt` contains both decoder and encoder weights. The state dict is split into:

- **Decoder keys**: `tok_embeddings.*`, `layers.*`, `norm.*`, `output.*`
- **Encoder keys**: `hyper_encoder.*` (prefix stripped)
- **Skipped keys**: `token_type_head.*`, `hyper_output.*` (not used by inference)

### 2. Convert decoder weights

Decoder weights use torchtitan naming conventions. `Llama3StateDictAdapter.to_hf()` converts them to HuggingFace format:

| torchtitan key | HuggingFace key |
|---|---|
| `tok_embeddings.weight` | `model.embed_tokens.weight` |
| `layers.{i}.attention.wq.weight` | `model.layers.{i}.self_attn.q_proj.weight` |
| `layers.{i}.attention.wk.weight` | `model.layers.{i}.self_attn.k_proj.weight` |
| `layers.{i}.attention.wv.weight` | `model.layers.{i}.self_attn.v_proj.weight` |
| `layers.{i}.attention.wo.weight` | `model.layers.{i}.self_attn.o_proj.weight` |
| `layers.{i}.feed_forward.w1.weight` | `model.layers.{i}.mlp.gate_proj.weight` |
| `layers.{i}.feed_forward.w2.weight` | `model.layers.{i}.mlp.down_proj.weight` |
| `layers.{i}.feed_forward.w3.weight` | `model.layers.{i}.mlp.up_proj.weight` |
| `norm.weight` | `model.norm.weight` |
| `output.weight` | `lm_head.weight` |

### 3. Save encoder weights

Encoder weights are prefixed with `input_encoder.` and saved to `encoders.safetensors`. The `tie_encoders=True` config means the input encoder is shared as the output encoder (no separate `output_encoder.*` weights).

### 4. Generate zip2zip_config.json

The config describes the encoder architecture and compression settings:

```json
{
    "base_model_name_or_path": "meta-llama/Llama-3.2-1B-Instruct",
    "encoder_type": "res_latent_attn",
    "encoder": {
        "hidden_size": 512,
        "model_hidden_size": 2048,
        "num_hidden_layers": 2,
        "intermediate_size": 2048,
        "num_heads": 8,
        "causal": false,
        "residual": true,
        "tie_encoders": true,
        "position_encoding": null
    },
    "compression": {
        "initial_vocab_size": 128256,
        "max_codebook_size": 4096,
        "max_subtokens": 4,
        "disabled_ids": [128000, 128001, ...]
    }
}
```

Encoder parameters are inferred from weight shapes. If `--base_model` is provided and no explicit `--disabled_ids`, the tokenizer is loaded to compute disabled IDs (all added/special tokens).

## Output files

```
<output_dir>/
    zip2zip_config.json     # Model + compression config
    model.safetensors       # HF Llama decoder weights
    encoders.safetensors    # Encoder weights (input_encoder.*)
    tokenizer.json          # Copied from base model
    tokenizer_config.json   # Copied from base model
    special_tokens_map.json # Copied from base model
```

## Loading exported models

With the `zip2zip` inference library:

```python
from zip2zip import Zip2ZipModel, Zip2ZipTokenizer

model = Zip2ZipModel.from_pretrained("epfl-dlab/Llaza-3.2-1B-v0.1", revision="hf")
tokenizer = Zip2ZipTokenizer.from_pretrained("epfl-dlab/Llaza-3.2-1B-v0.1", revision="hf")
```

Or from a local directory:

```python
model = Zip2ZipModel.from_pretrained("/path/to/export")
```

## Running evaluation on exported models

```bash
cd ext/zip2zip
pip install -e ".[eval]"

python bench/run_harness_pretrained.py \
    epfl-dlab/Llaza-3.2-1B-v0.1 \
    --revision hf \
    --tasks hellaswag piqa winogrande
```

## Auto-detection

When `--model_config` is not specified, the export function infers the Llama architecture from weight shapes:

- `n_heads` and `n_kv_heads` from `wq.weight` and `wk.weight` dimensions (tries head_dim=64 then 128)
- `dim` from `tok_embeddings.weight` shape
- `n_layers` from the count of `layers.*.attention.wq.weight` keys
- Encoder config (`hidden_size`, `num_hidden_layers`, `intermediate_size`, `max_subtokens`) from `hyper_encoder.*` weight shapes
