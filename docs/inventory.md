# Model Inventory

Trained models and checkpoints hosted on [HuggingFace Hub](https://huggingface.co/epfl-dlab).

## Production models

| Model | Base | max_subtokens | HF Repo | Notes |
|-------|------|---------------|---------|-------|
| — | — | — | — | TODO |

## Candidate models (from training runs)

| Model | Base | max_subtokens | HF Repo | W&B Run | Notes |
|-------|------|---------------|---------|---------|-------|
| — | — | — | — | — | TODO |

## Datasets

| Dataset | Tokens | URL |
|---------|--------|-----|
| llaza-200B | 200B | [epfl-dlab/llaza-200B](https://huggingface.co/datasets/epfl-dlab/llaza-200B) |
| llaza-20B | 20B | [epfl-dlab/llaza-20B](https://huggingface.co/datasets/epfl-dlab/llaza-20B) |
| llaza-1B | 1B | [epfl-dlab/llaza-1B](https://huggingface.co/datasets/epfl-dlab/llaza-1B) |

## Naming conventions

- **Production**: `epfl-dlab/Llaza-{base}-{config}-{version}` (e.g. `epfl-dlab/Llaza-3.2-1B-MS2F-4K-v0.1`)
- **Candidate**: `epfl-dlab/candidate-{run_name}` (auto-generated during training)

Each repo has two branches: `main` (torchtitan training checkpoint) and `hf` (exported HF format for inference).
