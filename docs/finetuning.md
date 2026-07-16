# Finetuning from Pretrained Weights

## Overview

zip2zip-core supports finetuning from existing HuggingFace Llama checkpoints. The decoder weights are loaded from the pretrained model, while the hyper-encoder and other zip2zip-specific components are randomly initialized and trained from scratch.

## Quick start

```bash
bash scripts/finetune.sh
```

This runs finetuning from `meta-llama/Llama-3.2-1B-Instruct` with `max_subtokens=2`.

## How it works

### `--init_from_hf`

The `--init_from_hf` flag downloads a HuggingFace Llama model and loads its decoder weights:

```bash
uv run torchrun --nproc_per_node=4 -m zip2zip_core.train \
    --data_dir /path/to/tokens \
    --output_dir /path/to/checkpoints \
    --model_config 1B_llama3.2 \
    --init_from_hf meta-llama/Llama-3.2-1B-Instruct \
    --max_subtokens 2 \
    --lr 5e-5 \
    --min_lr 5e-6 \
    --wandb --wandb_name llaza_1b_ms2
```

### Weight loading

`load_hf_pretrained()` in `train.py`:

1. Downloads `.safetensors` files from HuggingFace Hub (rank 0 downloads, path broadcast to other ranks)
2. Converts HF state dict keys to torchtitan format using `Llama3StateDictAdapter.from_hf()`
3. Loads with `strict=False` — missing keys are expected for zip2zip-specific parameters:
   - `hyper_encoder.*` — the HyperEncoder (2-layer transformer)
   - `token_type_head.*` — optional base/hyper classifier
   - `hyper_output.*` — if present

These missing parameters keep their random initialization from `init_weights()`.

### Model config matching

Use `--model_config 1B_llama3.2` to match the Llama 3.2 1B architecture exactly (8192 FFN hidden dim, 131072 max seq len). The standard `1B` config has slightly different FFN dimensions due to `compute_ffn_hidden_dim`.

## Recommended settings

For finetuning, use a lower learning rate than pretraining:

| Parameter | Pretraining | Finetuning |
|-----------|-------------|------------|
| `--lr` | `3e-4` | `5e-5` |
| `--min_lr` | `3e-5` | `5e-6` |
| `--gradient_accumulation_steps` | 2 | 4 |

### `--reset_step`

When resuming a finetuning run from a checkpoint, use `--reset_step` to start the LR cosine schedule from the beginning:

```bash
uv run torchrun --nproc_per_node=4 -m zip2zip_core.train \
    --resume_from /path/to/step_0 \
    --reset_step \
    --steps 6000 \
    ...
```

Without `--reset_step`, training continues from the saved step number, which affects the LR schedule position.

## Token-budget training

Instead of a fixed step count, you can train for a fixed number of tokens:

```bash
uv run torchrun --nproc_per_node=4 -m zip2zip_core.train \
    --init_from_hf meta-llama/Llama-3.2-1B-Instruct \
    --max_tokens 4_000_000_000 \
    ...
```

`--max_tokens` counts base tokens processed (not compressed tokens). Training stops when the cumulative base token count reaches the target.

## Multi-configuration sweeps

`scripts/finetune_w_zip2zip_1b_data.sh` runs finetuning with multiple `max_subtokens` values (1, 2, 3, 4) sequentially:

```bash
bash scripts/finetune_w_zip2zip_1b_data.sh
```

## Loss masking for SFT data

When using SFT datasets where loss should only apply to certain spans (e.g. assistant responses in chat data), prepare loss mask files alongside the token shards (see [Data Pipeline](data.md#loss-masks-optional)). The dataset automatically propagates masks through LZW compression.

**End-of-text must be in the loss.** The document-final `<|endoftext|>` gets `mask=1`
(since commit `b12efc0`): with it masked out, models never learn to *emit* end-of-text
after plain-text documents and at inference run past their answer into a fabricated
next document (observed on GSM8K with the v0.1-repro checkpoint). Datasets tokenized
before that commit have the old masks — re-tokenize before training new models.

## Reproducing a released model's exact recipe

`scripts/finetune_phi35_rcp.sh` reproduces `epfl-dlab/zip2zip-Phi-3.5-mini-instruct-v0.1`
(originally trained with `ozz-main`) on the EPFL RCP cluster (Run:AI). Its defaults are
verified against the released run's config and HF checkpoint metadata — max_subtokens=4,
seq_len=2048, 32,768 tokens/optimizer-step, 8000 steps, frozen decoder + LoRA r=32/α=32,
hyper-encoder 3072-dim/2-layer/32-head. See the script header for accepted deviations
(single tied hyper-encoder, packed-stream compression, assistant-turn loss masking) that
keep it from being bit-identical. `scripts/finetune_phi35_from_hf_instruct.sbatch` is the
CSCS-SLURM counterpart (same `train.py` flags, different job launcher).

`scripts/tokenize_sft_phi_rcp.sh` prepares the required Phi-tokenized `epfl-dlab/zip2zip-1B`
shards on RCP; sanity-checks the max token ID to catch accidentally-Llama-tokenized data.

## Continual-pretraining control (uncompressed baseline)

The fair baseline for a zip2zip finetune is the same recipe with compression
disabled (the paper's "Cont. pretrain" row), not the raw base model. Run it with
`MAX_CODEBOOK_SIZE=0` and everything else identical: the LZW encoder becomes an
exact identity (insertion is gated on `next_id < initial_vocab_size +
max_codebook_size`), the hyper path never executes (all-pad codebook fails the
`(codebook != pad).any()` gate), logits reduce to the base vocab, and training
is mathematically plain CLM with LoRA as the only trained parameters.

Caveats to state when comparing:
- **Budget**: steps are matched, base tokens are not — the treatment run covers
  ~C× more base tokens per window (C ≈ 1.4). Steps-matched is the convention;
  add a token-matched arm (`STEPS ≈ 8000×C`) only if the control looks weak.
- **Chunk coverage**: the control trains on the first `seq_len+1` base tokens of
  every `2*seq_len` chunk; the treatment covers more per chunk and silently
  drops chunks that compress ≥2×.
- **Eval**: control checkpoints are **base-mode-only** (they carry a random,
  never-trained hyper-encoder). Pass `EVAL_MODE=base` to the pipeline (or
  `--eval_mode base` to eval_harness). Audit `eval_args.eval_mode` in results
  JSONs. gen/input compression health lines read ~1.0 for these runs by design.

## Canonical RCP locations and run conventions

Single source of truth for where things live on the cluster (`$SCRATCH =
/dlabscratch1/gentilin`); update this table in the same commit as any change.

| What | Path | Notes |
|---|---|---|
| Phi SFT data (current) | `$SCRATCH/datasets/phi-1B-sft-8shards-eosfix` | EOS in loss (`b12efc0`); use for all new finetunes |
| Phi SFT data (mathchat, v0.3) | `$SCRATCH/datasets/phi-1B-sft-8shards-mathchat` | eosfix + math docs re-rendered as multi-turn chat via exact NuminaMath problem matching; experimental until v0.3 validates |
| Phi SFT data (legacy) | *deleted 2026-07-15* | was `phi-1B-sft-8shards` (EOS masked out); v0.1-repro trained on it — superseded by eosfix after v0.2 validated |
| Llama SFT data | `$SCRATCH/datasets/zip2zip-1B-sft-8shards` | For a future Llama-3.2-1B reproduction; do not delete |
| Checkpoints | `$SCRATCH/zip2zip-outputs/<RUN_NAME>/step_N` | `train.py` suffixes `(N)` on name collision — always pick a fresh `RUN_NAME`; the verified baseline is `andrea-z2z-phi35-4B-repro-1BData-v0.1-Zip2zipCore(1)/step_8000` |
| Eval logs + results JSON | `$SCRATCH/logs/eval/` | JSONs are the record when `WANDB=0`; backfill with `scripts/log_results_to_wandb.py` |
| Train / tokenize logs | `$SCRATCH/logs/train/`, `$SCRATCH/logs/tokenize/` | |

Standard run sequence for a new finetune: `tokenize_sft_phi_rcp.sh` (only if the
data recipe changed) → **`pipeline_ft_eval_rcp.sh`** — one Run:AI job that trains,
smoke-evals every 1000-step checkpoint (arc_easy/hellaswag/winogrande/gsm8k_boxed
@ 200 samples, curves at `smoke/step` in W&B), runs the full `default` preset +
wikitext perplexity on the final checkpoint (all sample tables in W&B), and appends
every RCP artifact path plus the ready-made 4-corpora-perplexity command to the run
notes. All in ONE W&B run named `RUN_NAME`; hard-fails on an existing output dir
(never suffixes `(1)`). Rehearse pipeline changes first with
`STEPS=20 SAVE_FREQ=10 SMOKE_EVERY=10 SMOKE_LIMIT=8 FINAL_LIMIT=8` (~30 min).
For manual/partial runs the individual pieces remain: `finetune_phi35_rcp.sh`,
`diagnose_ckpt_rcp.sh` (15-min sanity gate: train-style replay must land near the
run's final W&B `loss`, ~1.6 nats/base-token for healthy Phi runs),
`eval_ckpt_rcp.sh`. Compare against the frozen baseline numbers (see
`docs/evaluation.md`), not the paper's.

## Curriculum finetuning

You can combine finetuning with curriculum training — start with `max_subtokens=2` and increase in later phases:

```bash
# Phase 1
torchrun ... --init_from_hf meta-llama/Llama-3.2-1B-Instruct \
    --max_subtokens 2 --max_tokens 4_000_000_000

# Phase 2
torchrun ... --resume_from /path/to/phase1_checkpoint \
    --max_subtokens 3 --max_tokens 8_000_000_000
```

On phase transition, `load_checkpoint()` automatically pads `hyper_encoder.pos_embed` to accommodate the larger window and skips optimizer state loading (fresh optimizer for the new phase).
