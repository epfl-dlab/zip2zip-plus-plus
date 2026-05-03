# Project Structure

```
zip2zip-core/
├── src/zip2zip_core/
│   ├── model.py                 # Zip2ZipLlama3Model, HyperEncoder
│   ├── configs.py               # Model configurations (debugmodel, 1B, 3B, ...)
│   ├── data.py                  # Dataset, collation, dataloader
│   ├── train.py                 # DDP/FSDP training loop
│   ├── export.py                # Export to ext/zip2zip HF format
│   ├── hub.py                   # HuggingFace Hub upload utilities
│   ├── model_card.py            # Auto-generated model cards
│   ├── project.py               # Centralized W&B/HF config
│   ├── lm_eval_adapter.py       # lm-evaluation-harness adapter
│   └── parallelize.py           # FSDP parallelization strategy
├── scripts/
│   ├── pretokenize.py           # Data preprocessing
│   ├── finetune.sh              # Finetuning from Llama weights
│   ├── eval_harness.py          # lm-evaluation-harness with W&B logging
│   ├── eval_lm.sh               # Internal eval (loss/ppl/accuracy)
│   ├── push_checkpoint.py       # Push + auto-export to HF Hub
│   ├── train.sbatch             # SLURM job script
│   ├── test_checkpoint.py       # Quick checkpoint sanity check
│   └── zip2zip_hf/
│       ├── export_to_zip2zip.py # Export CLI wrapper
│       ├── batch_export.sh      # Batch export multiple checkpoints
│       ├── run.py               # Quick inference test
│       └── debug_hyper.py       # Diagnostic: base vs hyper logits
├── ext/
│   ├── torchtitan/              # Git submodule — PyTorch distributed training framework
│   ├── zip2zip/                 # Git submodule — inference library (pip install zip2zip)
│   └── zip2zip-compression/     # Git submodule — Rust LZW compression library
├── docs/                        # Documentation
└── CLAUDE.md                    # AI assistant context
```

## Key modules

### `src/zip2zip_core/`

| Module | Description |
|--------|-------------|
| `model.py` | `Zip2ZipLlama3Model` — extends Llama3 decoder with HyperEncoder + bilinear output head. `HyperEncoder` maps variable-length base-token sequences into single embeddings. |
| `configs.py` | Model size configurations: debugmodel (2M), 20M, 50M, 150M, 400M, 1B, 3B. Llama 3.2 variants included. |
| `data.py` | `Zip2ZipDataset` — pre-tokenized `.npy` shards with on-the-fly LZW compression. Supports `lm` and `compress` modes. |
| `train.py` | Full training loop with FSDP, gradient accumulation, curriculum support, W&B logging, background HF upload. |
| `export.py` | `export()` — converts torchtitan state dict to HF safetensors + zip2zip_config.json. |
| `hub.py` | `upload_folder()` and `push_checkpoint()` — HF Hub upload with model card generation and wandb integration. |
| `project.py` | `WANDB_ENTITY`, `WANDB_PROJECT`, `HF_ORG` — shared project constants. |
| `parallelize.py` | FSDP setup: bfloat16 params, fp32 reduce, activation checkpointing. No tensor parallelism (HyperEncoder requires full model per rank). |

### `ext/`

| Submodule | Description |
|-----------|-------------|
| `torchtitan` | Provides `Decoder`, `TransformerBlock`, `Embedding`, RoPE, attention modules — the base Llama3 architecture. |
| `zip2zip` | Inference library with HF-compatible API (`Zip2ZipModel`, `Zip2ZipTokenizer`), lm-eval integration. |
| `zip2zip-compression` | Rust LZW compression: `LZWCompressor`, `Codebook`, `CodebookManager`. Used in training data pipeline and inference. |
