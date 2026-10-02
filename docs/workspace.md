# Shared project services

## Weights & Biases

Training and evaluation can log to the shared `llaza` project. The historical
project name is retained so existing runs and comparisons remain available.
Configuration defaults live in `src/zip2zip_core/project.py` and can be
overridden through the training CLI or environment used by a launcher.

## Hugging Face

Datasets and model repositories are hosted under the `epfl-dlab` organization.

### Production model layout

Each of the four Zip2Zip++ model repositories has two revisions:

| Revision | Content |
|---|---|
| `main` | Original zip2zip-plus-plus checkpoint for resume/reproduction |
| `hf` | Validated self-contained export for `zip2zip` inference |

Publish only through the guarded command:

```bash
python scripts/push_checkpoint.py \
    --ckpt-dir /path/to/step_8000 \
    --repo-id epfl-dlab/zip2zip-pp-Llama-3.2-1B-Instruct \
    --upload
```

Training's `--hf_repo` option remains available for background uploads of
intermediate checkpoints. Use candidate repositories for that purpose; do not
point an active training job at a production model repository.

### Datasets

The Llaza datasets and preprocessing workflow remain part of the project:

| Dataset | URL |
|---|---|
| llaza-200B | [epfl-dlab/llaza-200B](https://huggingface.co/datasets/epfl-dlab/llaza-200B) |
| llaza-20B | [epfl-dlab/llaza-20B](https://huggingface.co/datasets/epfl-dlab/llaza-20B) |
| llaza-1B | [epfl-dlab/llaza-1B](https://huggingface.co/datasets/epfl-dlab/llaza-1B) |

See [Data Pipeline](data.md) for preprocessing commands.

## Checkpoint safety

- Keep `model.pt` and its matching `meta.pt` together.
- Keep `optimizer.pt` when the checkpoint must support exact training resume.
- Validate a local export before adding `--upload`.
- Load public inference models with `revision="hf"`.
