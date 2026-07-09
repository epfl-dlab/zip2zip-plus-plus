# Finetuning from Pretrained Weights

## Overview

zip2zip-core supports finetuning from existing HuggingFace Llama checkpoints. The decoder weights are loaded from the pretrained model, while the hyper-encoder and other zip2zip-specific components are randomly initialized and trained from scratch.

## Quick start

```bash
bash scripts/finetune.sh
```

This runs finetuning from `meta-llama/Llama-3.2-1B-Instruct` with `max_subtokens=2`.

## How it works

### `--init_from_hf`

The `--init_from_hf` flag downloads a HuggingFace Llama model and loads its decoder weights:

```bash
uv run torchrun --nproc_per_node=4 -m zip2zip_core.train \
    --data_dir /path/to/tokens \
    --output_dir /path/to/checkpoints \
    --model_config 1B_llama3.2 \
    --init_from_hf meta-llama/Llama-3.2-1B-Instruct \
    --max_subtokens 2 \
    --lr 5e-5 \
    --min_lr 5e-6 \
    --wandb --wandb_name llaza_1b_ms2
```

### Weight loading

`load_hf_pretrained()` in `train.py`:

1. Downloads `.safetensors` files from HuggingFace Hub (rank 0 downloads, path broadcast to other ranks)
2. Converts HF state dict keys to torchtitan format using `Llama3StateDictAdapter.from_hf()`
3. Loads with `strict=False` — missing keys are expected for zip2zip-specific parameters:
   - `hyper_encoder.*` — the HyperEncoder (2-layer transformer)
   - `token_type_head.*` — optional base/hyper classifier
   - `hyper_output.*` — if present

These missing parameters keep their random initialization from `init_weights()`.

### Model config matching

Use `--model_config 1B_llama3.2` to match the Llama 3.2 1B architecture exactly (8192 FFN hidden dim, 131072 max seq len). The standard `1B` config has slightly different FFN dimensions due to `compute_ffn_hidden_dim`.

## Recommended settings

For finetuning, use a lower learning rate than pretraining:

| Parameter | Pretraining | Finetuning |
|-----------|-------------|------------|
| `--lr` | `3e-4` | `5e-5` |
| `--min_lr` | `3e-5` | `5e-6` |
| `--gradient_accumulation_steps` | 2 | 4 |

### `--reset_step`

When resuming a finetuning run from a checkpoint, use `--reset_step` to start the LR cosine schedule from the beginning:

```bash
uv run torchrun --nproc_per_node=4 -m zip2zip_core.train \
    --resume_from /path/to/step_0 \
    --reset_step \
    --steps 6000 \
    ...
```

Without `--reset_step`, training continues from the saved step number, which affects the LR schedule position.

## Token-budget training

Instead of a fixed step count, you can train for a fixed number of tokens:

```bash
uv run torchrun --nproc_per_node=4 -m zip2zip_core.train \
    --init_from_hf meta-llama/Llama-3.2-1B-Instruct \
    --max_tokens 4_000_000_000 \
    ...
```

`--max_tokens` counts base tokens processed (not compressed tokens). Training stops when the cumulative base token count reaches the target.

## Multi-configuration sweeps

`scripts/finetune_w_zip2zip_1b_data.sh` runs finetuning with multiple `max_subtokens` values (1, 2, 3, 4) sequentially:

```bash
bash scripts/finetune_w_zip2zip_1b_data.sh
```

## Loss masking for SFT data

When using SFT datasets where loss should only apply to certain spans (e.g. assistant responses in chat data), prepare loss mask files alongside the token shards (see [Data Pipeline](data.md#loss-masks-optional)). The dataset automatically propagates masks through LZW compression.

## Reproducing a released model's exact recipe

`scripts/finetune_phi35_rcp.sh` reproduces `epfl-dlab/zip2zip-Phi-3.5-mini-instruct-v0.1`
(originally trained with `ozz-main`) on the EPFL RCP cluster (Run:AI). Its defaults are
verified against the released run's config and HF checkpoint metadata — max_subtokens=4,
seq_len=2048, 32,768 tokens/optimizer-step, 8000 steps, frozen decoder + LoRA r=32/α=32,
hyper-encoder 3072-dim/2-layer/32-head. See the script header for accepted deviations
(single tied hyper-encoder, packed-stream compression, assistant-turn loss masking) that
keep it from being bit-identical. `scripts/finetune_phi35_from_hf_instruct.sbatch` is the
CSCS-SLURM counterpart (same `train.py` flags, different job launcher).

`scripts/tokenize_sft_phi_rcp.sh` prepares the required Phi-tokenized `epfl-dlab/zip2zip-1B`
shards on RCP; sanity-checks the max token ID to catch accidentally-Llama-tokenized data.

## Curriculum finetuning

You can combine finetuning with curriculum training — start with `max_subtokens=2` and increase in later phases:

```bash
# Phase 1
torchrun ... --init_from_hf meta-llama/Llama-3.2-1B-Instruct \
    --max_subtokens 2 --max_tokens 4_000_000_000

# Phase 2
torchrun ... --resume_from /path/to/phase1_checkpoint \
    --max_subtokens 3 --max_tokens 8_000_000_000
```

On phase transition, `load_checkpoint()` automatically pads `hyper_encoder.pos_embed` to accommodate the larger window and skips optimizer state loading (fresh optimizer for the new phase).
