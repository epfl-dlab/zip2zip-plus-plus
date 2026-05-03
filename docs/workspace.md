# Workspace

Shared infrastructure for the Llaza/zip2zip project team.

## W&B Project

All training and evaluation runs are logged to a shared Weights & Biases project:

- **Entity**: `epfl-dlab`
- **Project**: [`llaza`](https://wandb.ai/epfl-dlab/llaza)

Defaults are configured in `src/zip2zip_core/project.py`:
```python
WANDB_ENTITY = "epfl-dlab"
WANDB_PROJECT = "llaza"
HF_ORG = "epfl-dlab"
```

### Naming conventions

- **Training runs**: auto-named with a unique 4-char suffix (e.g. `llaza_1b_llama32_instruct_ms2_1bt_a3f2`)
- **Evaluation runs**: prefixed with `eval-` and tagged with `eval` for easy filtering (e.g. `eval-step_6000`)
- Each run's W&B user is automatically tracked — no need for extra tags

### Filtering tips

- Filter by tag `eval` to see only evaluation runs
- Use the `hf_url` summary field to jump to the model on HF Hub

## HuggingFace Hub

Models and checkpoints are hosted under the [`epfl-dlab`](https://huggingface.co/epfl-dlab) organization.

### Repo naming

| Type | Naming pattern | Example |
|------|---------------|---------|
| Candidate (auto from training) | `epfl-dlab/candidate-{run_name}` | `epfl-dlab/candidate-llaza_1b_ms2_a3f2` |
| Production-ready | `epfl-dlab/Llaza-{base}-{config}-{version}` | `epfl-dlab/Llaza-3.2-1B-MS2F-4K-v0.1` |

### Branch layout

Each model repo has two branches:

| Branch | Format | Content |
|--------|--------|---------|
| `main` | torchtitan (model.pt, meta.pt) | Training checkpoint, for resume or export |
| `hf` | HuggingFace (safetensors, zip2zip_config.json) | Exported model, for inference with `zip2zip` |

### Datasets

| Dataset | URL |
|---------|-----|
| llaza-200B | [`epfl-dlab/llaza-200B`](https://huggingface.co/datasets/epfl-dlab/llaza-200B) |
| llaza-20B | [`epfl-dlab/llaza-20B`](https://huggingface.co/datasets/epfl-dlab/llaza-20B) |
| llaza-1B | [`epfl-dlab/llaza-1B`](https://huggingface.co/datasets/epfl-dlab/llaza-1B) |

## Pushing checkpoints

```bash
# Auto-detects format, pushes training ckpt to main + auto-exports to hf branch
python scripts/push_checkpoint.py \
    --ckpt_dir /path/to/step_6000 \
    --repo_id epfl-dlab/Llaza-3.2-1B-v0.1

# Skip auto-export
python scripts/push_checkpoint.py \
    --ckpt_dir /path/to/step_6000 \
    --repo_id epfl-dlab/Llaza-3.2-1B-v0.1 \
    --no_export
```

### What happens during push

1. Reads `meta.pt` to extract training args
2. Generates a model card (`README.md`) with training config
3. Uploads training checkpoint to `main` branch
4. Auto-exports to HF format (safetensors) into a temp directory
5. Uploads exported model to `hf` branch

### Auto-upload during training

When `--wandb` is enabled, the final checkpoint is automatically uploaded to `epfl-dlab/candidate-{run_name}` in a background thread (avoids NCCL barrier timeout on distributed runs). The HF URL is saved to the wandb run summary.
