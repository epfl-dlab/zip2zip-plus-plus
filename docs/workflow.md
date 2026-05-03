# Workflow: Train → Evaluate → Publish

A typical end-to-end workflow for training a model, evaluating it, and publishing to HuggingFace Hub.

## 1. Train

```bash
uv run torchrun --nproc_per_node=4 -m zip2zip_core.train \
    --data_dir /path/to/tokens \
    --output_dir /path/to/checkpoints \
    --model_config 1B \
    --max_subtokens 2 \
    --steps 6000 \
    --wandb --wandb_name llaza_1b_ms2
```

Note the W&B run ID from the output (e.g. `8d11iyds`), or find it on the [W&B dashboard](https://wandb.ai/epfl-dlab/llaza).

## 2. Evaluate (log to the same W&B run)

```bash
python scripts/eval_harness.py \
    --ckpt_dir /path/to/checkpoints/step_6000 \
    --resume_wandb_id 8d11iyds
```

This runs lm-evaluation-harness benchmarks and logs metrics + per-sample results back into the same W&B run, so training and eval data live together.

## 3. Publish to HuggingFace Hub

If the results look good, push the checkpoint. This auto-exports to HF format on the `hf` branch:

```bash
python scripts/push_checkpoint.py \
    --ckpt_dir /path/to/checkpoints/step_6000 \
    --repo_id epfl-dlab/Llaza-3.2-1B-MS2-v0.1
```

What happens:
1. Generates a model card from training args
2. Uploads training checkpoint to `main` branch
3. Auto-exports to HF format (safetensors + zip2zip_config.json)
4. Uploads exported model to `hf` branch

Skip auto-export with `--no_export` if you only want the training checkpoint.

## 4. Verify

```python
from zip2zip import Zip2ZipModel

model = Zip2ZipModel.from_pretrained(
    "epfl-dlab/Llaza-3.2-1B-MS2-v0.1", revision="hf"
)
```

## Summary

```
train (--wandb)
  │
  ├── W&B run created (run_id: 8d11iyds)
  └── checkpoint saved to /path/to/step_6000
          │
          ▼
eval_harness.py --resume_wandb_id 8d11iyds
  │
  └── metrics + samples logged to same W&B run
          │
          ▼
push_checkpoint.py --repo_id epfl-dlab/Llaza-...
  │
  ├── main branch  ← training checkpoint (model.pt)
  └── hf branch    ← exported HF format (safetensors)
```
