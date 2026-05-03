# Pretraining

## Pre-tokenize data

```bash
uv run python scripts/pretokenize.py \
    --output_dir /path/to/tokens \
    --dataset HuggingFaceFW/fineweb \
    --dataset_name sample-10BT
```

Default dataset is `epfl-dlab/llaza-20B`. Outputs `.npy` shards (~1B tokens each) where all documents are concatenated into a continuous token stream with `<bos>`/`<eos>` boundaries.

## Single-node training

```bash
uv run torchrun --nproc_per_node=4 -m zip2zip_core.train \
    --data_dir /path/to/tokens \
    --output_dir /path/to/checkpoints \
    --max_subtokens 2 \
    --steps 6000
```

## Multi-node SLURM (CSCS Alps)

```bash
sbatch scripts/train.sbatch
```

Override training parameters via environment variables:

```bash
MAX_SUBTOKENS=3 STEPS=12000 RESUME_FROM=/path/to/step_6000 \
    sbatch scripts/train.sbatch
```

## W&B logging

Enable with `--wandb`. Project defaults to `llaza` (see [Workspace](workspace.md)).

```bash
uv run torchrun --nproc_per_node=4 -m zip2zip_core.train \
    --data_dir /path/to/tokens \
    --output_dir /path/to/checkpoints \
    --max_subtokens 2 \
    --wandb --wandb_name my-run
```

When `--wandb` is enabled:
- Run name gets a unique 4-char suffix (e.g. `my-run_a3f2`)
- Checkpoints are auto-uploaded to HF Hub as `epfl-dlab/candidate-{run_name}`
- HF URL is saved to the wandb run summary after upload

## Curriculum training

Run phases sequentially with increasing `max_subtokens`, resuming from the previous checkpoint:

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

On phase transition, `load_checkpoint()` automatically pads `hyper_encoder.pos_embed` to accommodate the larger window.

See also [Finetuning](finetuning.md) for initializing from pretrained Llama weights.
