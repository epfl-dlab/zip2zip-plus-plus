# zip2zip-core

Pretraining zip2zip language models with inference-time adaptive tokenization via LZW compression (hypertokens).

## Installation

### Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/)
- Rust toolchain (for building `zip2zip-compression`)

### Setup

Clone with submodules:

```bash
git clone --recurse-submodules https://github.com/epfl-dlab/zip2zip-core.git
cd zip2zip-core
```

Install everything (creates venv, builds zip2zip-compression from Rust, installs torchtitan from submodule):

```bash
uv sync
```

For data preprocessing (tokenization), include the optional dependencies:

```bash
uv sync --extra data
```

## Usage

### 1. Pre-tokenize data

```bash
uv run python scripts/pretokenize.py \
    --output_dir /path/to/tokens \
    --dataset HuggingFaceFW/fineweb \
    --dataset_name sample-10BT
```

### 2. Train

Local (single node):

```bash
uv run torchrun --nproc_per_node=4 -m zip2zip_core.train \
    --data_dir /path/to/tokens \
    --output_dir /path/to/checkpoints \
    --max_subtokens 2 \
    --steps 6000
```

SLURM (multi-node on CSCS Alps):

```bash
sbatch scripts/train.sbatch
```

Override training parameters via environment variables:

```bash
MAX_SUBTOKENS=3 STEPS=12000 RESUME_FROM=/path/to/step_6000 \
    sbatch scripts/train.sbatch
```

Enable [Weights & Biases](https://wandb.ai) logging:

```bash
WANDB=1 WANDB_PROJECT=zip2zip-core WANDB_RUN_NAME=my-run \
    sbatch scripts/train.sbatch
```

Set your API key in `scripts/train.sbatch` or export it before submitting:

```bash
export WANDB_API_KEY=<your-wandb-api-key>
```

### 3. Evaluation

Evaluate a checkpoint on held-out data (no gradient, outputs loss/ppl/accuracy):

```bash
# Basic usage
bash scripts/eval_lm.sh /path/to/checkpoint

# Specify model config and max_subtokens
bash scripts/eval_lm.sh /path/to/checkpoint 400M 2
```

Arguments: `<checkpoint_dir> [model_config=1B] [max_subtokens=4]`

### 4. Curriculum training

Run phases sequentially, resuming from the previous checkpoint:

```bash
# Phase 1: max_subtokens=2
MAX_SUBTOKENS=2 STEPS=6000 sbatch scripts/train.sbatch

# Phase 2: max_subtokens=3
MAX_SUBTOKENS=3 STEPS=12000 RESUME_FROM=$SCRATCH/zip2zip-outputs/zip2zip-1b/step_6000 \
    sbatch scripts/train.sbatch

# Phase 3: max_subtokens=4
MAX_SUBTOKENS=4 STEPS=19000 RESUME_FROM=$SCRATCH/zip2zip-outputs/zip2zip-1b/step_12000 \
    sbatch scripts/train.sbatch
```

## Project structure

```
zip2zip-core/
├── ext/
│   ├── torchtitan/              # Git submodule (PyTorch training framework)
│   └── zip2zip-compression/     # Git submodule (Rust LZW compression library)
├── src/zip2zip_core/
│   ├── model.py                 # Zip2ZipLlama3Model, HyperEncoder
│   ├── configs.py               # Model configurations (debugmodel, 1B)
│   ├── data.py                  # Dataset, collation, dataloader
│   ├── parallelize.py           # FSDP parallelization strategy
│   └── train.py                 # DDP training loop
└── scripts/
    ├── train.sbatch             # SLURM job script
    └── pretokenize.py           # Data preprocessing
```
