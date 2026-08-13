# Compression Modeling (LZW transducer)

Can a Transformer execute the LZW algorithm at all? This experiment isolates that
question from language modeling: the model is trained as a deterministic
transducer that maps a token sequence to its LZW-compressed form, or back. There
is no linguistic uncertainty left — either it reproduces the algorithm's output
or it does not.

This page covers the Phi-3.5-tokenizer version of that sweep. The original Llama
version lives in `scripts/sweep_finemath_merge_size_compress.sh` and produced the
published grokking figure (`plot/plot_compress_loss_grokking.py`).

## The task

`--mode compress` builds each sample as one of

```
<DECOMPRESS> base_tokens  <COMPRESS>   compressed_tokens     # compression
<COMPRESS>   compressed_tokens <DECOMPRESS> base_tokens      # decompression
```

Loss is masked to the second half; the first half is the given input. See
`_iter_compress` in `src/zip2zip_core/data.py`.

**Markers.** `<COMPRESS>` / `<DECOMPRESS>` are the two lowest unused reserved
special-token slots of the active tokenizer, derived by
`data.default_transducer_token_ids`: 128002/128003 for Llama 3
(`<|reserved_special_token_0/1|>`) and 32002/32003 for Phi-3.5
(`<|placeholder1/2|>`). They are in `disabled_ids`, so LZW never merges the
separator into a hyper-token. Override with `--compress_token_id` /
`--decompress_token_id`.

**One direction per model.** `--direction {both,compress,decompress}`. Use
`compress` or `decompress`: a single-direction run's loss *is* that direction's
loss, with nothing blended in. `both` is the historical 50/50 mix that
reproduces the published figure, and it is not a 50/50 average of the two
losses — the decompression half supervises `compression_ratio` times more target
tokens per sample, so the mixed curve is weighted toward decompression and
drifts as the model learns to compress.

A single-direction run does not draw from the RNG when choosing the direction,
so `--direction compress` and `--direction decompress` on the same seed read
**identical source windows**. The two runs differ only in the task.

## Model ladder

`Phi-7M`, `Phi-20M`, `Phi-50M`, `Phi-150M`, `Phi-1B` in
`src/zip2zip_core/configs.py`. All follow Phi-3.5-mini's aspect ratios: head_dim
96, MHA, `ffn = 8/3 * dim`, RoPE theta 10000 unscaled, vocab 32064, untied
embeddings with an explicit `init_std = dim**-0.5`.

| config | dim | layers | heads | total | backbone | Llama namesake's backbone |
|---|---|---|---|---|---|---|
| Phi-7M | 96 | 4 | 1 | 6.7M | 0.44M | — |
| Phi-20M | 192 | 12 | 2 | 17.7M | 5.31M | 1.57M |
| Phi-50M | 384 | 12 | 4 | 46.2M | 21.24M | 10.23M |
| Phi-150M | 768 | 14 | 8 | 149.5M | 99.11M | 60.96M |
| Phi-1B | 1728 | 24 | 18 | 978.9M | 860.05M | 973.15M |

**Read the caveat before plotting a scale axis.** Sizes are set so the TOTAL
lands within ~12% of the name. Because a 32k vocab shrinks the embedding table
~4x, matching the total forces a larger backbone than the Llama config of the
same name — 3.4x at the 20M rung. The embedding table is a lookup and
contributes almost nothing to learning an algorithm, so a figure comparing the
two families should plot **non-embedding parameters**, not the label.

`Phi-7M` exists because of that skew: it sits *below* the Llama size that fails
in the published sweep, to test whether capacity separates the families at all.
At dim 96 a head_dim of 96 leaves a single attention head — the one departure
from Phi's shape. No ~10M rung is possible: the embedding table alone steps
6.2M → 12.3M between dim 96 and 192.

## Data

Phi-tokenized FineMath, 8 shards x 1.25B tokens:

```bash
HF_HOME=$SCRATCH/.cache/huggingface TOKENIZERS_PARALLELISM=false \
python scripts/pretokenize.py \
    --output_dir $SCRATCH/datasets/finemath-10bt-phi35 \
    --model_name microsoft/Phi-3.5-mini-instruct \
    --dataset HuggingFaceTB/finemath --dataset_name finemath-3plus \
    --dataset_split train --column text \
    --target_tokens 10e9 --tokens_per_shard 1250000000 \
    --min_doc_length 50 --num_workers 32
```

Two things worth knowing:

- `.map()` is eager over the whole split, so the HF Arrow cache peaks far above
  the output (~440 GB for this corpus). Delete
  `$HF_HOME/datasets/HuggingFaceTB___finemath` once the shards exist; keep the
  downloaded parquet under `hub/` to avoid re-downloading.
- `--tokens_per_shard` sets how large a Python list of token ids is held in
  memory before a shard is written: 1.25B tokens is ~40 GB resident, with a
  ~80 GB peak at each shard boundary.

A run consumes `steps * tokens_per_step` = 1.57B tokens of the 10B, so nothing
repeats. Note the same text yields ~38% more tokens under Phi than under Llama
(measured on FineMath), so an equal token budget covers ~28% less text.

## Training

```bash
runai submit --name phi-150m-compress \
  --image ghcr.io/jkminder/dlab-runai-images/pytorch:master \
  --gpu 4 --cpu 32 --memory 256Gi \
  --pvc dlab-scratch:/mnt --large-shm --node-pools h100 \
  --environment MODEL_CONFIG=Phi-150M --environment DIRECTION=compress \
  --environment WANDB_API_KEY=$WANDB_API_KEY \
  -- "bash $SCRATCH/code/zip2zip-core/scripts/compress_phi_rcp.sh"
```

One job per (size, direction) — the full sweep is 5 x 2 = 10 jobs.
`scripts/compress_phi_rcp.sh` pins every hyperparameter the Llama sweep left at
train.py defaults (seq_len 4096, lr 3e-4 → 3e-5, warmup 500, wd 0.1, beta2 0.95,
6000 steps, 262,144 tokens/step) so the only variables against the published
figure are the tokenizer and the model shape. `--no_remap_codebook` and
`--hyper_causal_mask` are not incidental: without the causal mask the model can
read codebook rows the decoder has not installed yet at that position.

Useful env vars: `DIRECTION`, `STEPS`, `N_SAMPLES`, `LOCAL_BATCH_SIZE`
(gradient accumulation is re-derived to hold tokens/step at 262,144),
`ACTIVATION_CHECKPOINT=1`, `INIT_FROM_HF` (the pretrained control, which needs
the `Phi3.5-mini` config and a smaller LR).

Runs log to their own W&B project, not the LM one: here `loss` is the LZW
objective rather than language-model NLL, and `compression` is a property of the
sample rather than of the model.

### Reading the training metrics

| key | meaning |
|---|---|
| `loss` | NLL per **base** token (`compat_loss_legacy`), the published figure's y-axis |
| `compressed/loss_per_valid_token` | NLL per **target** token; differs from `loss` by the compression ratio in the compress direction, identical in decompress |
| `acc` | exact match per supervised token, per-microbatch mean |
| `compressed/acc` | the same, pooled over raw counts |
| `compressed/hyper_token_acc` | accuracy where the **target** is a hyper-token — the hard part of the compression direction, and 0 by construction in decompression |
| `compressed/relaxed_hyper_acc` | prefix credit; the gap to `hyper_token_acc` separates "wrong merge length" from "wrong content" |
| `direction/*` | the same numbers split by direction; redundant in a single-direction run, and what lets two runs share a panel |
| `compression` | base tokens per supervised token. A **data** property: 1.00 by construction in decompression |

Two traps:

- Accuracy spikes periodically in small models. They coincide with `compression`
  jumping (1.5 → 2.7): those windows are highly repetitive text where the next
  hyper-token is mechanically predictable. It is a property of the corpus, not
  of learning.
- `acc` is per token. At 500 supervised positions, 99% per-token accuracy means
  ~0.7% of sequences are fully correct. Token accuracy cannot answer RQ1 — hence
  the evaluation below.

## Evaluation

`scripts/eval_transducer.py` scores every checkpoint on held-out data with two
per-sample verdicts:

- **strict** — the generated sequence equals the target token for token.
- **relaxed** — it *expands* to the same base tokens: a valid compression that
  segments differently from canonical LZW. `strict` implies `relaxed`; the
  script asserts it. In the decompression direction the output is already base
  tokens, so relaxed is identical to strict by construction.

```bash
runai submit --name eval-phi-1b \
  --image ghcr.io/jkminder/dlab-runai-images/pytorch:master \
  --gpu 1 --cpu 16 --memory 128Gi \
  --pvc dlab-scratch:/mnt --large-shm --node-pools h100 \
  --environment MODEL_CONFIG=Phi-1B --environment RELAXED=1 \
  --environment N_SAMPLES=32 --environment VERIFY_GENERATION=8 \
  --environment WANDB_API_KEY=$WANDB_API_KEY \
  -- "bash $SCRATCH/code/zip2zip-core/scripts/eval_transducer_rcp.sh"
```

One job covers both directions x every checkpoint of one size. Results land in
`<run_dir>/eval_transducer.json` (per checkpoint) and
`eval_transducer_samples.jsonl` (per sample), and are logged back into each
run's own W&B run under `eval/<direction>/*` against an `eval/step` x-axis.

**Cost.** Strict needs no generation. Under greedy decoding it is equivalent to
"teacher-forced argmax equals the label at every supervised position": while
every token so far is correct the generated prefix *is* the ground-truth prefix,
so both settings feed the model identical inputs, and at the first mismatch both
emit the same wrong token. One forward per batch instead of one per generated
token. `--verify_generation N` checks that equivalence against real decoding —
run it on a size that actually passes strict, or it only exercises the
both-fail branch. `--relaxed` does need real generation (a diverged sequence may
still expand correctly), but only for samples strict already failed, and it
stops as soon as the expansion disagrees in base space. Budget ~10-20 min per
checkpoint for Phi-1B with `--relaxed`, minutes for the small sizes.

**Held-out data** needs no new corpus. Training reads
`shards[rank::world_size]` from offset 0 and reaches at most
`steps * accum * batch * seq_len` tokens into a rank's first shard, so a late
shard at a large offset was never read. The default is the last shard at offset
1e9; `check_unseen()` turns that arithmetic into an assertion.

`pooled_token_acc` is reported only to reconcile against the training curve. If
it disagrees with the training run's final accuracy, the evaluation is wrong —
do not read the exact-match numbers until it lines up.

### Result shape

Phi-1B, compression direction, 32 held-out samples:

```
step 1000:  strict=0.000  relaxed=0.000  pooled_token_acc=0.9525
step 2000:  strict=0.063  relaxed=0.250  pooled_token_acc=0.9964
step 3000:  strict=0.344  relaxed=0.375  pooled_token_acc=0.9988
step 4000:  strict=0.531  relaxed=0.656  pooled_token_acc=0.9995
step 5000:  strict=0.906  relaxed=0.906  pooled_token_acc=0.9999
step 6000:  strict=0.969  relaxed=0.969  pooled_token_acc=1.0000
```

At step 1000 token accuracy is already 0.95 and not one sequence is correct. At
step 2000 the two verdicts differ 4x: the model compresses correctly four times
more often than it compresses *the LZW way*. It learns to compress first, and to
segment canonically second; the gap closes by step 5000.

## Pitfalls

- **Never `.to(dtype=...)` a loaded model.** The RoPE cache is a complex buffer
  and the cast silently discards its imaginary part, scoring a fully trained
  checkpoint at chance. Move the device only and let autocast handle compute
  precision. Same trap is guarded in `lm_eval_adapter.py` and `inference.py`.
- **`WANDB_API_KEY` must be in the job's environment.** Rank 0 otherwise raises
  in `wandb.init()` and takes the other ranks down with it, several minutes in.
- **Do not compare legacy and `compressed/` metrics.** Pick one family.
- **Do not group runs of different directions on a shared metric key.** The mean
  of a compression score and a decompression score is not a quantity; the eval
  keys are direction-scoped for this reason.
