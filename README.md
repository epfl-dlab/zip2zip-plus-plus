# zip2zip-plus-plus

Training, evaluation, and release tooling for **Zip2Zip++**, a language-model
architecture with inference-time adaptive tokenization via LZW hypertokens.

This repository contains the distributed training implementation and the data
pipeline. The user-facing Hugging Face runtime lives in
[zip2zip](https://github.com/epfl-dlab/zip2zip).

Try the [Zip2Zip++ demo](https://zip2zip-tokenizer-main-zygo.vercel.app/) — it
visualizes the hypertoken codebook and compares original, optimal, and
model-produced tokenizations side by side.

## Released models

All released checkpoints live in the
[epfl-dlab/zip2zip++ collection](https://huggingface.co/collections/epfl-dlab/zip2zip).

Each model repository follows the revision contract described in
[Release a Zip2Zip++ checkpoint](#release-a-zip2zip-checkpoint): load the `hf`
revision for inference.

| Model | Base model | Params |
|---|---|---|
| [`epfl-dlab/zip2zip-pp-Llama-3.2-1B-Instruct`](https://huggingface.co/epfl-dlab/zip2zip-pp-Llama-3.2-1B-Instruct) | `meta-llama/Llama-3.2-1B-Instruct` | 1B |
| [`epfl-dlab/zip2zip-pp-Llama-3.2-3B-Instruct`](https://huggingface.co/epfl-dlab/zip2zip-pp-Llama-3.2-3B-Instruct) | `meta-llama/Llama-3.2-3B-Instruct` | 3B |
| [`epfl-dlab/zip2zip-pp-Phi-3.5-mini-instruct`](https://huggingface.co/epfl-dlab/zip2zip-pp-Phi-3.5-mini-instruct) | `microsoft/Phi-3.5-mini-instruct` | 3.8B |
| [`epfl-dlab/zip2zip-pp-Phi-3-medium-4k-instruct`](https://huggingface.co/epfl-dlab/zip2zip-pp-Phi-3-medium-4k-instruct) | `microsoft/Phi-3-medium-4k-instruct` | 14B |

## Setup

```bash
git clone --recurse-submodules https://github.com/epfl-dlab/zip2zip-plus-plus.git
cd zip2zip-plus-plus
uv sync
```

## Data and training

The Llaza datasets and their preprocessing pipeline remain the canonical input
pipeline for the project:

```bash
uv run python scripts/pretokenize.py \
    --dataset epfl-dlab/llaza-20B \
    --output_dir /path/to/tokens

uv run torchrun --nproc_per_node=4 -m zip2zip_core.train \
    --data_dir /path/to/tokens \
    --output_dir /path/to/checkpoints \
    --model_config 1B \
    --init_from_hf meta-llama/Llama-3.2-1B-Instruct \
    --max_subtokens 4
```

See [Data Pipeline](docs/data.md) and [Finetuning](docs/finetuning.md) for the
full preprocessing and training options. Checkpoint saving, resuming, and the
optional background upload used by training are unchanged.

## Evaluate

```bash
python scripts/eval_harness.py \
    --ckpt_dir /path/to/checkpoints/step_8000 \
    --resume_wandb_id none
```

See [Evaluation](docs/evaluation.md) for the available tasks and modes.

## Release a Zip2Zip++ checkpoint

The release command accepts only the four supported Zip2Zip++ recipes: Llama
3.2 1B/3B and Phi-3 4B/14B. It validates the checkpoint, produces a
self-contained sharded Hugging Face export, and validates the exported key set
before any upload.

Build and inspect an export locally:

```bash
python scripts/push_checkpoint.py \
    --ckpt-dir /path/to/checkpoints/step_8000 \
    --repo-id epfl-dlab/zip2zip-pp-Llama-3.2-1B-Instruct \
    --output-dir /path/to/export
```

Publish after inspection:

```bash
python scripts/push_checkpoint.py \
    --ckpt-dir /path/to/checkpoints/step_8000 \
    --repo-id epfl-dlab/zip2zip-pp-Llama-3.2-1B-Instruct \
    --upload
```

Each model repository uses the same revision contract:

| Revision | Contents | Intended use |
|---|---|---|
| `main` | Original `model.pt`, `meta.pt`, and checkpoint files | Resume and reproduction with zip2zip-plus-plus |
| `hf` | Config, tokenizer, encoders, and sharded safetensors | Inference with `zip2zip` |

Users must load the `hf` revision:

```python
from zip2zip import Zip2ZipModel, Zip2ZipTokenizer

repo_id = "epfl-dlab/zip2zip-pp-Llama-3.2-1B-Instruct"
tokenizer = Zip2ZipTokenizer.from_pretrained(repo_id, revision="hf")
model = Zip2ZipModel.from_pretrained(
    repo_id, revision="hf", device_map="auto", dtype="auto"
)
```

See [Export and interoperability](docs/export.md) for format details.

## Paper reproduction

The hyper-encoder embedding probes (substitution and sequence probes) used in
the paper live in the `ext/zip2zip-hyperenc_probe` submodule. See its
[README](ext/zip2zip-hyperenc_probe/README.md) for reproduction commands.

## Documentation

- [Installation](docs/installation.md)
- [Data pipeline](docs/data.md)
- [Pretraining](docs/pretraining.md)
- [Finetuning](docs/finetuning.md)
- [Evaluation](docs/evaluation.md)
- [Inference](docs/inference.md)
- [Export and interoperability](docs/export.md)
- [Release workflow](docs/workflow.md)
- [Project structure](docs/structure.md)

## License

zip2zip-plus-plus is released under the MIT License (see [LICENSE](LICENSE)). The
vendored submodules keep their own licenses: `ext/zip2zip` (Apache-2.0),
`ext/zip2zip-compression` (MIT), `ext/torchtitan` (BSD-3-Clause).
