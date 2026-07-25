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
(packed-stream compression, assistant-turn loss masking) that keep it from being
bit-identical. The hyper-encoder can be tied (one encoder for both the input-embedding
and output-logit roles) or untied (`UNTIED_HYPER_ENCODER=1` builds the released model's
separate pair: input reads `tok_embeddings`, output reads `lm_head`) — canonical since
v0.5 (see below). Untied checkpoints export to the released `ext/zip2zip` HF format via
either `export.py` or `scripts/zip2zip_hf/export_phi.py` — both write `output_encoder.*`
and `tie_encoders=False` for untied checkpoints. `scripts/finetune_phi35_from_hf_instruct.sbatch` is the
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

## Digit-protected compression (canonical since v0.4-digitsafe)

Multi-digit numbers are LZW-merged into hypertokens by default (only chat
specials are in `disabled_ids`), and this measurably destroys arithmetic. The
v0.4-digitsafe run (2026-07-17) validated train-time protection: with the 10
digit tokens in `disabled_ids` for training AND eval, GSM8K flexible went
0.356 → 0.610 (+25pt, ~2/3 of zip2zip's entire math deficit vs the
uncompressed control at 0.742), with MC flat (WinoGrande −3.4pt the only move
beyond ~1pt), wiki byte-ppl +0.9%, and essentially no compression loss on
natural text (MC input ratio 1.077 vs 1.079; wikitext window 1.169 vs 1.209).

**Canonical recipe: pass `DISABLE_DIGIT_IDS=1` on every new finetune.** One
lever applies the protection to BOTH training and every eval
(`train.py --disable_digit_ids` + eval passthroughs; per-eval flag audit;
setting the env to literal `0` is a preflight error). It is deliberately NOT a
code default so pre-digitsafe checkpoints and the frozen baselines evaluate
bit-identically; the eval adapter auto-enables digit protection when a
checkpoint's meta.pt records it, so evals always follow the checkpoint's
training distribution even if the flag is forgotten (results JSON records the
effective value).

## Untied hyper-encoder (canonical since v0.5)

The released model uses two hyper-encoders — one embedding hypertokens on the
input side (from `tok_embeddings`), one scoring them on the output side (from
`lm_head`); zip2zip-core originally tied them into one, which a diagnostic showed
forces a single vector to straddle two geometries. The v0.5-untied run
(2026-07-18, on top of v0.4-digitsafe) validated untying: every MC task moved up
toward the uncompressed control (ARC-c acc_norm .540 → .561, now past the
released model's .551; HellaSwag .702 → .725 at ~5σ; WinoGrande recovered
.713 → .743), wiki byte-ppl improved 1.711 → 1.686, and GSM8K flexible edged
0.610 → 0.623 (the remaining math gap to the control is a separate problem, not
tied/untied). Nothing regressed beyond noise.

**Canonical recipe: pass `UNTIED_HYPER_ENCODER=1` on every new finetune** (with
`DISABLE_DIGIT_IDS=1`). The launcher inherits it into training; the eval adapter
auto-configures untied from the checkpoint's meta.pt (logs `untied hyper-encoder:
building separate output encoder` and hard-fails if the `hyper_output.*` weights
are missing), so there is no eval passthrough or `0`-ambiguity. Like digit
protection it is deliberately NOT a code default (`tie_hyper_encoder=True`), so
the frozen tied baselines (v0.1–v0.4) keep loading and evaluating bit-identically.

## Phased warm-start (v0.6 — negative result, do not use)

Every treatment run starts with a large loss/grad-norm transient (~loss 150,
grad_norm ~4500 for the first ~100 steps) as the random hyper-encoder is trained
from scratch — a candidate cause of early LoRA damage. `WARMSTART_STEPS=N` (train
arg `--warmstart_steps`) freezes the decoder-LoRA for the first N optimizer steps
(its grads are nulled after backward+grad-sync, before clip/step, so AdamW skips
it — no update, no momentum) while the hyper-encoder(s) train alone; at step N the
LoRA unfreezes. `requires_grad` never changes, so FSDP's grad reduction stays
rank-uniform (no deadlock class). Training-only — eval is unaffected (the
checkpoint is a normal untied+digit model), so no eval passthrough. Default 0 =
off (v0.5 behavior). Log signature: `[warmstart] decoder-LoRA frozen for first N
steps` then `[warmstart] step N: unfreezing decoder-LoRA`. The v0.6 run
(2026-07-20, N=200, same budget as v0.5-8k) REFUTED the premise: GSM8K flexible
dropped 0.623 → 0.557 with everything else flat — letting the hyper-encoder
settle alone yields a math-inferior joint minimum. Keep `WARMSTART_STEPS` unset;
the code stays as a default-off negative-result artifact.

## Deeper hyper-encoder (v0.6.1 — negative result, do not use)

Next lever on the residual GSM8K gap to the uncompressed control (v0.5 flexible
.623 vs control .742). The base-mode decomposition splits that gap into an
input-compression term and a trained-weights term of similar size, and encoder
architecture is the proven axis for the compression side (tied → untied was
+1.4pt GSM8K plus MC/ppl gains). v0.6.1 deepens both flat untied encoders from
2 layers to 4: `ENCODER_N_LAYERS=4` (train arg `--encoder_n_layers`, config
field `encoder_n_layers`, default 2 = v0.5 exactly).

Properties worth knowing before launching:

- **Compression-specific by construction.** The control has
  `max_codebook_size=0`, so the hyper path never runs and encoder depth cannot
  move the control — this knob can only close the gap, never lift the ceiling.
- **Cost.** The trainable untied encoder pair roughly doubles (~453M → ~906M
  params), and both encoders run over the full active codebook every step;
  expect a tens-of-percent wall-clock increase and watch step-0 memory (the
  output encoder rides the root FSDP unit).
- **Plumbing is complete end-to-end.** Training records the effective encoder
  architecture (dim/layers/heads/intermediate) in meta.pt, the eval adapter and
  `scripts/inference.py` rebuild the 4-layer encoders from it automatically (no
  eval flag), and resuming with different encoder args than the checkpoint's is
  a hard error (`validate_resume_args`), like `untied_hyper_encoder` — except
  legacy metas that record None for them, where the strict checkpoint load is
  the backstop.

Launch = the v0.5 canonical command plus one env var:

```bash
RUN_NAME=<name> DISABLE_DIGIT_IDS=1 UNTIED_HYPER_ENCODER=1 ENCODER_N_LAYERS=4 \
  bash scripts/pipeline_ft_eval_rcp.sh
```

The v0.6.1 run (2026-07-23, same budget as v0.5-8k, single variable = depth)
came back NEGATIVE: GSM8K flexible 0.623 → 0.597, ARC-c/HellaSwag/OpenBookQA
each down ~1pt, PIQA/WinoGrande up ~0.6 (noise), wikitext byte-ppl exactly flat
(1.6857 → 1.6858), train loss slightly BETTER (~1.45 vs 1.51) — the extra
capacity fit the training distribution without transferring. The known
compressed-reading corruption ("sprints" → "sells") persists at depth 4, so the
failure is input-side/positional, not encoder capacity. Keep
`ENCODER_N_LAYERS` unset (default 2 = v0.5 canonical); the plumbing stays as a
default-off negative-result artifact.

## Base-token RoPE positions (v0.6.2)

Attacks the INPUT term of the GSM8K gap, which the paired base-mode
decomposition (2026-07-23) showed is the dominant residual: v0.5 compressed
.630 vs base .695 vs control .742 — 6.5pt lost purely to reading compressed
input at eval, and it is the *entire* MC gap (base mode matches the control on
ARC-c/HellaSwag). Three lines of evidence say the damage is positional, not
merge-content: digit protection helped (v0.4), symbol/LaTeX protection gained
nothing (triage on v0.4), and the canonical failure ("3 sprints" misread) is
cross-hypertoken corruption with all content words preserved as base tokens.

Mechanism: today every compressed token takes ONE RoPE slot, so the geometric
distance between two tokens depends on how much the compressor merged between
them — the same text yields different position geometry depending on local
compression ratio, and eval prompts produce merge patterns never seen in
training. With `BASE_TOKEN_POSITIONS=1` (train arg `--base_token_positions`,
config `base_token_positions`), every token instead sits at the base-space
index of its LAST constituent (spans end-anchored, positions =
cumsum(span)−1): relative distances mean exactly what they meant in
pretraining, and an uncompressed sequence reduces bit-identically to the
default `arange`.

Implementation notes:

- Positions are derived inside the model from what it already receives — the
  full codebook at training/loglikelihood time, or a per-entry span buffer
  maintained next to the embed buffers during incremental generation. No
  dataloader or harness changes; train/eval/generation cannot drift.
- The training rope cache is sized `seq_len * max_subtokens` when the flag is
  on (a fully merged window ends at base position `seq_len*ms − 1`); eval and
  inference keep the config's native 131k Phi cache.
- Base-mode eval is unaffected by construction (uncompressed stream, positions
  == arange with or without the flag), so the base/compressed decomposition
  stays directly comparable across v0.5 and v0.6.2.
- Like `untied_hyper_encoder`: recorded in meta.pt, auto-configured by the
  eval adapter (log line `base-token RoPE positions: enabled from meta.pt`)
  and `scripts/inference.py`, resume-guarded as a hard key, deliberately NOT a
  code default (v0.1–v0.6.1 checkpoints keep evaluating bit-identically).
- NOT exported/pip-compatible: the ext/zip2zip HF runtime has no custom
  position path — export support is a separate task if v0.6.2 wins.

Launch = the v0.5 canonical command plus one env var:

```bash
RUN_NAME=<name> DISABLE_DIGIT_IDS=1 UNTIED_HYPER_ENCODER=1 BASE_TOKEN_POSITIONS=1 \
  bash scripts/pipeline_ft_eval_rcp.sh
```

The v0.6.2 run (2026-07-23, same budget as v0.5-8k, single variable = position
geometry) VALIDATED the mechanism — the first GSM8K gain of the recipe line:
flexible 0.623 → **0.650** (+2.7pt; fresh paired re-run 0.647 replicates it),
strict 0.078 → 0.210, train loss and throughput identical to v0.5 (positions
are free at runtime). The paired decomposition confirms the effect is exactly
where designed: the GSM8K input term shrank 6.5 → 4.3pt (base-mode 0.690 vs
compressed 0.647) while base-mode math stayed flat — the gain is all
compressed-reading, not weights. The canonical "sprints" corruption is
structurally fixed (answer 540, previously 180). Cost: a small weights-side MC
tax — ARC-c acc_norm .561 → .545, HellaSwag .725 → .713, both ≲1.6σ
individually and mostly weights-term per the decomposition; WinoGrande, ARC-e,
and wikitext ppl are flat. With the input term now the SMALLER share of the
remaining 9.5pt gap (4.3 input / 5.2 weights), the next lever should be
weights-side (`--token_type_loss_weight` is wired and untried). Recipe status:
pass `BASE_TOKEN_POSITIONS=1` on math-focused runs alongside the v0.5 flags;
the MC trade is documented above so the choice stays explicit per run.

## Token-type auxiliary loss (experimental, v0.6.3)

Attacks the WEIGHTS term, which the v0.6.2 decomposition left as the larger
share of the remaining gap (5.2pt weights vs 4.3pt input). A small head on the
final hidden state predicts whether the NEXT token is a base token or a
hypertoken (BCE, masked exactly like the LM loss, weighted into the backward
loss). The machinery has existed since early on (`--token_type_loss_weight`,
`token_type_head`, metrics `type_loss=`/`type_acc=` in the step log) but has
never been trained with. `TOKEN_TYPE_LOSS_WEIGHT=0.05` on top of the v0.6.2
recipe; 0 (default) builds no head and is bit-identical to v0.6.2. The head's
weights ride the checkpoint (eval loaders rebuild it from meta.pt; it is
unused at eval), and `token_type_loss_weight` is a resume-hard key — legacy
metas without it count as 0. The v0.6.3 run (2026-07-24, weight 0.05 on top of v0.6.2, same budget)
DELIVERED both halves: GSM8K flexible held at 0.652 (v0.6.2 was 0.650, so the
math win survived) AND the MC tax roughly halved — 5 of 6 MC tasks nudged back
toward v0.5 (ARC-c .545 → .551, ARC-e .819 → .822, HellaSwag .713 → .715,
OpenBookQA .464 → .472, WinoGrande .739 → .745; only PIQA slipped .798 → .793).
Wikitext byte-ppl essentially flat (1.6866 → 1.6918, +0.3%), final train loss
back to v0.5's ~1.52 (v0.6.2 had trained slightly lower). The head genuinely
LEARNED rather than collapsing to the majority class: hyper-token recall
climbed from ~0 early to ~0.50 by end (type_acc 0.86 vs the 0.78 majority
floor), confirming the auxiliary signal reached the shared representation.
Caveat: every individual v0.6.3-vs-v0.6.2 delta is within ~1σ; the case for it
rests on the *consistent direction* across metrics plus the neutral ppl/loss,
not any single significant gain. Recommended champion recipe: v0.6.3
(`BASE_TOKEN_POSITIONS=1 TOKEN_TYPE_LOSS_WEIGHT=0.05` on top of the v0.5 flags)
— it captures the v0.6.2 math gain with less MC collateral. If maximum
parsimony is preferred, v0.6.2 alone keeps ~90% of the benefit with one fewer
knob.

## Hyper-encoder zero-init fix (v0.6.4)

This one is a **bug fix**, not a new idea. The `encoder_residual` design computes
`hyper_embed = first_token_embed + encoder_out` and zero-initializes the encoder
output so a hypertoken *starts* as its first base token's embedding, then learns
a delta. The zero-init only ever touched `proj_out`:

```python
if name.endswith('proj_out') and isinstance(module, nn.Linear):
    nn.init.zeros_(module.weight)
```

…but `proj_out` is built **only when `encoder_dim != model_dim`**. The released
recipe sets `ENCODER_DIM=3072` on a `dim=3072` model, so `proj_out is None`, the
loop matched nothing, and the intent was silently never implemented. Every
v0.1–v0.6.3 run therefore trained from `first_token_embed + a LayerNorm-scaled
random vector`. Measured on the exact run config (`tests/test_encoder_zero_init.py`
pins the same fact on a small config):

| config | zero-init fires? | ‖first_token‖ | ‖encoder_out‖ | ratio |
|---|---|---|---|---|
| Phi runs (`encoder_dim=3072`) | no | 1.00 | 54.0 | **54×** |
| old Llama cfg (`encoder_dim=512`) | yes | 1.00 | 0.00 | 0× |
| Phi + this fix | yes | 1.00 | 0.00 | 0× |

The ratio is `sqrt(dim)` — the final `LayerNorm` forces unit RMS per channel
while Phi's embeddings are initialized at `dim**-0.5` — so the norm mismatch
grows with model size and is largest for the 3072-wide production model. This
is the same failure mode already documented for `tok_embeddings` in
`configs.py` ("~55x too large … ~1500 initial loss"); that instance was fixed,
this one survived.

It is consistent with the transient the v0.6 warm-start tried to treat: initial
loss is ~70 at step 10 **with or without** warm-start (v0.5 70.2, v0.6 69.6),
because freezing the LoRA does not change the random encoder output. This makes
the initialization bug a plausible source of that transient; only the paired
v0.6.4 run can establish how much it mattered to final performance.

`ZERO_INIT_ENCODER_OUTPUT=1` (train arg `--zero_init_encoder_output`, config
`zero_init_encoder_output`) zeroes the final `LayerNorm`'s weight **and** bias
when no `proj_out` exists, which makes the encoder output exactly zero through
both the padded and the varlen pooling paths. Properties:

- **Init-only.** No new parameters, no architecture change, `state_dict` keys and
  shapes are identical to v0.6.3 (pinned by test Z5), so eval, export, and
  `scripts/inference.py` need *nothing*. The flag is nevertheless a **hard**
  resume-lineage key: a normal resume loads checkpoint weights over the new
  initialization, so turning the flag on while resuming v0.6.3 would have no
  effect and would mislabel the run as v0.6.4. Start v0.6.4 fresh from the same
  HF base; `--allow_resume_mismatch` only acknowledges the mismatch and does not
  make the initialization reapply to loaded weights.
- **No-op where the original code worked.** With `encoder_dim != model_dim` the
  `proj_out` branch still runs and the flag changes nothing (test Z3).
- **Fail-loud.** If the flag is on and neither a `proj_out` nor a final
  `LayerNorm` exists (e.g. the gated-MLP `fast_hierarchical` composer), it raises
  instead of silently doing nothing (test Z7). This prevents another invisible
  no-op; it is an unsupported zero-init path, not a claim that the gated-MLP
  composer is already identity-initialized.
- **Observable.** Training prints `[encoder_zero_init] zeroed={...}` naming the
  modules it matched and whether the flag is on. In residual mode, flag-on must
  name an output gate or fail; flag-off legitimately prints an empty list for
  the Phi configuration.
- **Escapable, with a one-update delay upstream.** On the first backward pass,
  the zeroed `LayerNorm` weight and bias receive gradients, while parameters
  before that gate receive zero gradient. After the first optimizer update opens
  the gate, upstream encoder parameters receive gradients on the next backward
  pass. This is normal zero-gated residual-branch behavior, not a permanently
  dead encoder (test Z6).
- **Requires the residual.** `--zero_init_encoder_output` and
  `--no_encoder_residual` are mutually exclusive. Without the residual, an
  exactly zero encoder output would make the complete hypertoken embedding zero.

Note the flag is default-off so the frozen v0.1–v0.6.3 baselines stay exactly
reproducible and v0.6.4 is a clean single-variable A/B. **If it validates, flip
it to a code default** — it is a correctness fix, not a preference.

Launch = the v0.6.3 command plus one env var:

```bash
RUN_NAME=<name> DISABLE_DIGIT_IDS=1 UNTIED_HYPER_ENCODER=1 BASE_TOKEN_POSITIONS=1 \
  TOKEN_TYPE_LOSS_WEIGHT=0.05 ZERO_INIT_ENCODER_OUTPUT=1 \
  bash scripts/pipeline_ft_eval_rcp.sh
```

The v0.6.4 run (2026-07-25, same budget, single variable = the init fix)
VALIDATED, and it is the largest single improvement of the whole line — the only
change so far that improved **every** axis at once:

| | GSM8K flex | strict | ARC-c | ARC-e | HellaSwag | PIQA | WinoGrande | OBQA | wiki byte-ppl |
|---|---|---|---|---|---|---|---|---|---|
| v0.5 | .6232 | .066 | .5606 | .8237 | .7250 | .7900 | .7427 | .4740 | 1.7110 |
| v0.6.2 | .6505 | .210 | .5452 | .8190 | .7127 | .7976 | .7388 | .4640 | 1.6866 |
| v0.6.3 | .6520 | .024 | .5512 | .8224 | .7145 | .7927 | .7451 | .4720 | 1.6918 |
| **v0.6.4** | **.6770** | .215 | **.5700** | **.8304** | .7233 | **.8003** | .7443 | .4660 | **1.6574** |
| control | .7415 | .372 | .5930 | .8443 | .7372 | .8161 | .7514 | .4780 | — |

GSM8K +2.5pt over v0.6.3 (+5.4pt over v0.5), and the GSM8K gap to the control is
down to **6.5pt** from ~11.8pt at v0.5. The MC tax that v0.6.2 introduced is
**gone**: ARC-c, ARC-e and PIQA now sit at or above v0.5, HellaSwag is back within
noise, and byte-perplexity is the best ever measured on this line (−2% vs
v0.6.3). Only OpenBookQA is marginally down (−0.6pt, ≪1σ). Training was healthier
throughout: step-10 loss 3.97 instead of ~70-100, initial grad-norm 50 instead of
~1000-4700, final loss 1.434 (v0.6.3 ~1.52), end grad-norm 0.23. Throughput was
unchanged at ~32.9k tok/s — the fix is free.

Two checks worth recording because they could have invalidated the result:

- **The gain is not bought by compressing less.** `input_compression_ratio` is
  1.0772 for v0.5/v0.6.2/v0.6.3/v0.6.4 alike, and `gen_compression_ratio` is
  1.2536 vs 1.2616/1.2537/1.2593 — v0.6.4 emits hypertokens at the same rate, so
  it is not quietly falling back to the base vocabulary.
- **The identity start does not trap the hyper path.** An exactly-zero encoder
  output makes two entries sharing a first token initially indistinguishable, so
  `hyper_token_acc` starts at 0.000 — but it reaches 0.339 by step 500 and
  0.42–0.49 later, against v0.6.3's 0.061 at step 400. The degeneracy resolves at
  once and then far surpasses the old init. `type_acc` likewise reaches 0.871 by
  step 500 (v0.6.3: 0.830 at step 1000) and ends at 0.867.

Because the control never runs the hyper-encoder (`max_codebook_size=0` gates the
path off) it never had this handicap, so part of the gap we had been attributing
to compression was this bug. **Recommendation: promote `zero_init_encoder_output`
to a code default** — it is a correctness fix, it improves every metric, and the
only reason it is a flag is to keep the frozen v0.1–v0.6.3 baselines reproducible.

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
data recipe changed) → **`pipeline_ft_eval_rcp.sh`** with `DISABLE_DIGIT_IDS=1`
(canonical since v0.4-digitsafe, see above) — one Run:AI job that trains,
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
