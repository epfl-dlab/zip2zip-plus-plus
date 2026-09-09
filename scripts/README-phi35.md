# Phi-3.5-mini from-scratch zip2zip — run guide

From-scratch pretraining of a **Phi-3.5-mini-architecture** (3.8B) zip2zip model,
then SFT, for comparison against the released
[`epfl-dlab/zip2zip-Phi-3.5-mini-instruct-v0.1`](https://huggingface.co/epfl-dlab/zip2zip-Phi-3.5-mini-instruct-v0.1).

Unlike the released model (zip2zip-finetuned on *pretrained* Phi-3.5), this trains
the same architecture **from scratch** on `epfl-dlab/llaza-20B` (Phi-tokenized),
then instruction-tunes on `epfl-dlab/zip2zip-1B`.

## What's the model

- **Base arch = Phi-3.5-mini** (`MODEL_CONFIG=Phi3.5-mini` in `src/zip2zip_core/configs.py`):
  dim 3072, 32 layers, MHA 32/32 heads, head_dim 96, ffn 8192, vocab **32064**,
  RoPE θ=10000 (no scaling), **untied** input/output embeddings.
- **zip2zip components = same as the previous Llama from-scratch run** (core defaults):
  flat hyper-encoder, `encoder_dim=512`, `encoder_intermediate_size=2048`,
  `max_codebook_size=4096`, **`max_subtokens=3`**.

## Data (Phi tokenizer — must re-tokenize, vocab differs from Llama)

Data lives on /capstor/store (mounted in the container); checkpoints/outputs stay
on /capstor/scratch. Both phi datasets are already generated.

```bash
# Pretraining: 20B tokens, Phi tokenizer  (already done -> phi-20B-tokens-512shards)
.venv/bin/python scripts/pretokenize.py \
  --output_dir /capstor/store/cscs/swissai/a0101/mxx/zip2zip-data/phi-20B-tokens-512shards \
  --model_name microsoft/Phi-3.5-mini-instruct \
  --dataset epfl-dlab/llaza-20B --target_tokens 20e9 --tokens_per_shard 39000000

# SFT data: Phi tokenizer + Phi chat template
# (writes to /capstor/store/.../phi-1B-sft-8shards — path set inside the script)
.venv/bin/python scripts/create_sft_dataset_zip2zip1B_phi.py
```

## Run

Batch is fixed at **~1M tokens/step** = `local_bs(1) × seq(4096) × GPUs(128) × grad_accum(2)`.

```bash
cd /capstor/scratch/cscs/mxx/zip2zip-core

# Stage 1 — from-scratch pretraining, full 20B  (32 nodes / ~6-7h)
WANDB=1 WANDB_RUN_NAME=candidate-Phi35-MS3-flat-20BT MAX_TOKENS=20000000000 \
  sbatch scripts/train_phi35_v1.sbatch

# Resume if the 8h wall cut it short (auto-resolves latest ckpt, keeps step/LR):
WANDB=1 WANDB_RUN_NAME=candidate-Phi35-MS3-flat-20BT MAX_TOKENS=20000000000 \
  RESUME_FROM=/capstor/scratch/cscs/mxx/zip2zip-outputs/candidate-Phi35-MS3-flat-20BT \
  sbatch scripts/train_phi35_v1.sbatch

# Stage 2 — SFT on the Phi zip2zip-1B data (continues the Stage-1 checkpoint)
RESUME_FROM=/capstor/scratch/cscs/mxx/zip2zip-outputs/candidate-Phi35-MS3-flat-20BT/<ckpt> \
  sbatch scripts/finetune_phi35_v1.sbatch

# Quick 2-node smoke test (debug partition is broken for this container; use normal):
STEPS=20 SAVE_FREQ=10 LOG_FREQ=1 GRAD_ACCUM=1 NUM_WORKERS=4 \
  OUTPUT_DIR=/capstor/scratch/cscs/mxx/zip2zip-outputs/debug-Phi35 \
  sbatch --partition=normal --nodes=2 --time=00:30:00 scripts/train_phi35_v1.sbatch
```

Healthy start: step 1 `backward_loss ≈ 10.9` (≈ ln(vocab)), grad_norm ~50-200,
`base_token_acc` rising, `compression ≈ 1.3`.

## The 4 gotchas that had to be fixed (and why)

These are all wired into `train_phi35_v1.sbatch` / `finetune_phi35_v1.sbatch` and the
code; documented here so the behavior is understood and reproducible.

1. **Slingshot plugin fatal in container.** The cluster Slurm was upgraded (Nov 2025);
   the Apr-2025 container image lacks `libjson-c.so.5`, so `srun` *run inside the
   container* (via `#SBATCH --environment=`) can't load `switch_hpe_slingshot.so`.
   Fix: pass `--environment` on the **`srun` line** (launcher runs on the host, only
   the task runs in the container). Multi-node still needs the container for OFI/NCCL.

2. **CUDA device-side assert (hyper-token index OOB).** `data.py`'s LZW compressor
   defaulted to the Llama vocab (`initial_vocab_size=128256`, Llama special ids). With
   Phi data the hyper-token ids were mis-offset and indexed out of the codebook. Fix:
   `train.py` now passes `initial_vocab_size=config.vocab_size`, `pad_token_id`, and
   tokenizer-derived `disabled_ids` through `build_dataloader` → dataset → compressor.
   (Auto-derived from the tokenizer; reproduces the old Llama range for Llama runs.)

3. **3.8B can't be trained under DDP — switched to FSDP2.** `train.py` originally
   used DDP, which **replicates** the full model + fp32 Adam on every GPU
   (~60-76GB/GPU regardless of GPU count), so 128 GPUs still OOM'd. The correct fix
   is **FSDP2 (`fully_shard`)**, which shards params/grads/optimizer across all DP
   ranks (~footprint/world_size per GPU) — a 3.8B model becomes trivial. See
   `apply_fsdp()` in `zip2zip_core.train`. Params stay fp32 (no MixedPrecisionPolicy);
   bf16 compute comes from the `torch.autocast` in the loop — same numerics as before,
   no precision loss. Checkpoints are gathered to a full (unsharded) `model.pt` via
   DCP so eval/SFT/export stay compatible. zip2zip-specific gotcha: the hyper-encoder
   reads `tok_embeddings.weight` directly, which is a sharded DTensor under FSDP — it's
   gathered with `.full_tensor()` in `_encode_codebook_with_weights`.
   (`--activation_checkpoint` is available for extra headroom but unnecessary with FSDP.)

4. **Initial loss ~1580 instead of ~10.** torchtitan's `Embedding` inits with
   `std=1.0`. In the tied configs the output init (`std=dim^-0.5`) overwrote the shared
   weight, masking this; with Phi's **untied** embeddings the input embedding stayed at
   std=1.0, blowing up the hyper-token logits (hyper embeds derive from tok_embeddings).
   Fix: the Phi config sets `tok_embeddings=Embedding.Config(init_std=dim**-0.5)`.

All changes are gated/opt-in; existing Llama scripts (`train_1b_v1.sbatch`,
`finetune_v1.sbatch`) are unchanged.

## Eval / comparison

- In-framework: `scripts/eval_harness.py --tokenizer microsoft/Phi-3.5-mini-instruct`.
