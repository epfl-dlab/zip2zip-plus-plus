# zip2zip-core

Pretraining zip2zip language models (codename: **Llaza**) with inference-time adaptive tokenization via LZW compression (hypertokens).


## Quick Start

```bash
git clone --recurse-submodules https://github.com/epfl-dlab/zip2zip-core.git
cd zip2zip-core
uv sync
```

```bash
# Pre-tokenize
uv run python scripts/pretokenize.py --output_dir /path/to/tokens

# Train (single node, 4 GPUs)
uv run torchrun --nproc_per_node=4 -m zip2zip_core.train \
    --data_dir /path/to/tokens \
    --output_dir /path/to/checkpoints \
    --max_subtokens 2 --wandb --wandb_name my-run

# Evaluate
python scripts/eval_harness.py --ckpt_dir /path/to/step_6000 --resume_wandb_id none

# Push to HF Hub (auto-exports to zip2zip format)
python scripts/push_checkpoint.py --ckpt_dir /path/to/step_6000 --repo_id epfl-dlab/Llaza-3.2-1B-v0.1
```

## Llaza and zip2zip

We have two codebases:

- **[zip2zip-core](https://github.com/epfl-dlab/zip2zip-core)** (this repo) — pretraining and finetuning framework (torchtitan-based, distributed training, curriculum learning)
- **[zip2zip](https://github.com/epfl-dlab/zip2zip)** — inference library (`pip install zip2zip`), HuggingFace-compatible API, lm-evaluation-harness integration

Models trained here are exported to zip2zip format via `scripts/zip2zip_hf/export_to_zip2zip.py` (or automatically when using `scripts/push_checkpoint.py`).

## Documentation

- [Installation](docs/installation.md) — prerequisites, setup, extras
- [Data Pipeline](docs/data.md) — pre-tokenization, LZW compression, codebook remapping, training modes
- [Pretraining](docs/pretraining.md) — single-node, multi-node SLURM, curriculum training, W&B logging
- [Finetuning](docs/finetuning.md) — finetuning from pretrained Llama weights, `--init_from_hf`
- [Evaluation](docs/evaluation.md) — lm-evaluation-harness, W&B integration
- [Inference](docs/inference.md) — HF-based inference via `zip2zip`, torchtitan-based inference (in dev)
- [Profiling](docs/profiling.md) — profiling training with `torch.profiler`
- [Export & Interop](docs/export.md) — exporting to zip2zip HF format, state dict mapping, loading
- [Workflow](docs/workflow.md) — end-to-end: train → eval → publish to HF Hub
- [Workspace](docs/workspace.md) — W&B project, HuggingFace Hub, checkpoint management
- [Project Structure](docs/structure.md) — codebase layout and module descriptions
- [Model Inventory](docs/inventory.md) — trained models, checkpoints, datasets
- [Roadmap](docs/roadmap.md) — planned features and next steps
