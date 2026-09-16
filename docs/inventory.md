# Model and data inventory

## Zip2Zip++ release targets

The guarded release path supports exactly four base-model families. Fill in the
final repository IDs when the releases are created.

| Release | Base model | HF repository |
|---|---|---|
| Zip2Zip++ Llama 1B | `meta-llama/Llama-3.2-1B-Instruct` | [epfl-dlab/zip2zip-pp-Llama-3.2-1B-Instruct](https://huggingface.co/epfl-dlab/zip2zip-pp-Llama-3.2-1B-Instruct) |
| Zip2Zip++ Llama 3B | `meta-llama/Llama-3.2-3B-Instruct` | [epfl-dlab/zip2zip-pp-Llama-3.2-3B-Instruct](https://huggingface.co/epfl-dlab/zip2zip-pp-Llama-3.2-3B-Instruct) |
| Zip2Zip++ Phi 4B | `microsoft/Phi-3.5-mini-instruct` | [epfl-dlab/zip2zip-pp-Phi-3.5-mini-instruct](https://huggingface.co/epfl-dlab/zip2zip-pp-Phi-3.5-mini-instruct) |
| Zip2Zip++ Phi 14B | `microsoft/Phi-3-medium-4k-instruct` | [epfl-dlab/zip2zip-pp-Phi-3-medium-4k-instruct](https://huggingface.co/epfl-dlab/zip2zip-pp-Phi-3-medium-4k-instruct) |

Each repository uses `main` for the original training checkpoint and `hf` for
the inference export.

## Datasets

The Llaza data assets and preprocessing support are intentionally retained.

| Dataset | Tokens | URL |
|---|---:|---|
| llaza-200B | 200B | [epfl-dlab/llaza-200B](https://huggingface.co/datasets/epfl-dlab/llaza-200B) |
| llaza-20B | 20B | [epfl-dlab/llaza-20B](https://huggingface.co/datasets/epfl-dlab/llaza-20B) |
| llaza-1B | 1B | [epfl-dlab/llaza-1B](https://huggingface.co/datasets/epfl-dlab/llaza-1B) |

Candidate repositories produced during training are development artifacts and
are not part of the four-model public release contract.
