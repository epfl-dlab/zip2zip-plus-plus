# Evaluation

## Internal eval (loss/ppl/accuracy)

Evaluate a checkpoint on held-out data (no gradient):

```bash
bash scripts/eval_lm.sh /path/to/checkpoint

# Specify model config and max_subtokens
bash scripts/eval_lm.sh /path/to/checkpoint 400M 2
```

Arguments: `<checkpoint_dir> [model_config=1B] [max_subtokens=4]`

## lm-evaluation-harness

Standard benchmarks (arc, hellaswag, piqa, winogrande, etc.) via [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness):

```bash
python scripts/eval_harness.py \
    --ckpt_dir /path/to/step_6000 \
    --resume_wandb_id none
```

### W&B integration

Results are logged to W&B by default (project `llaza`, tagged with `eval`). Per-sample results are uploaded as W&B Tables.

```bash
# New eval run
python scripts/eval_harness.py --ckpt_dir /path/to/step_6000 --resume_wandb_id none

# Log into an existing wandb run
python scripts/eval_harness.py --ckpt_dir /path/to/step_6000 --resume_wandb_id 8d11iyds

# Disable W&B
python scripts/eval_harness.py --ckpt_dir /path/to/step_6000 --resume_wandb_id none --no_wandb
```

Run names are auto-generated as `eval-{ckpt_name}` or `eval-{repo}-{revision}`.

### From HF Hub

```bash
python scripts/eval_harness.py \
    --hf_repo epfl-dlab/Llaza-3.2-1B-v0.1 \
    --hf_revision step_6000 \
    --resume_wandb_id none
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--tasks` | arc_challenge, arc_easy, hellaswag, openbookqa, piqa, winogrande, commonsense_qa, medqa_4options | Comma-separated task list |
| `--num_fewshot` | 0 | Few-shot examples |
| `--limit` | None | Per-task sample limit (for quick smoke runs) |
| `--batch_size` | 1 | Eval batch size |
| `--eval_mode` | compressed | `compressed` (LZW) or `base` (vanilla LM scoring) |
| `--resume_wandb_id` | (required) | W&B run ID to resume, or `none` for new run |
| `--no_wandb` | false | Disable W&B logging |
| `--no_log_samples` | false | Skip per-sample W&B Tables |

## Presets

`scripts/eval_presets.yaml` centralizes tasks/num_fewshot/flags for four named presets,
loaded via `--preset` (shared by `eval_harness.py` and `eval_hf_model.py` through
`load_preset.py`) so there are no hardcoded duplicates across scripts. CLI flags still
override preset values.

| Preset | Use case |
|--------|----------|
| `default` | Full MC + generation benchmarks, 2-shot, chat template (matches paper Table 3) |
| `perplexity` | Byte-level PPL on wikitext/pile/mc4/dc4, 1024-token rolling window |
| `default_base` | Same as `default` but no chat template (from-scratch/non-instruct models) |
| `smoke` | 20 samples/task, no W&B — quick sanity check |

`TASKS=...` env var (or `--tasks`) restricts a preset to one task without editing the YAML.

### Repeated-window Wikitext stress tests

`zip2zip_wikitext_repeat4` and `zip2zip_wikitext_repeat8` are opt-in
merge-size-transfer stress tests. They are not part of any default preset or
pipeline, and the standard Wikitext task is unchanged.

The local corpora are built with the
`microsoft/Phi-3.5-mini-instruct` tokenizer. They are suitable for Zip2Zip
checkpoints that use that exact tokenizer; forced evaluation at a merge size
not seen during training is supported only by hierarchical hyper-encoders.
Other tokenizers need their own rebuilt corpus because token-aware block
boundaries, single-window guarantees, and LZW compression patterns change.

Prepare the local JSONL files once from the repository root:

```bash
uv sync --extra eval
uv run python scripts/build_repeated_wikitext.py --repeat_n 4
uv run python scripts/build_repeated_wikitext.py --repeat_n 8
```

By default the builder writes outside the Git repository:

```text
../datasets/wikitext-repeat4-phi35/{test.jsonl,manifest.json}
../datasets/wikitext-repeat8-phi35/{test.jsonl,manifest.json}
```

The default build location is resolved from the script rather than the current
working directory. The task YAMLs use the paths above, so run evaluation from
the repository root. Existing data is never overwritten implicitly; rebuild
explicitly with `--overwrite`.

Run one task by overriding the task list of the 1,024-token perplexity preset:

```bash
uv run python scripts/eval_harness.py \
  --ckpt_dir /path/to/checkpoint \
  --tokenizer microsoft/Phi-3.5-mini-instruct \
  --preset perplexity \
  --tasks zip2zip_wikitext_repeat8 \
  --eval_max_subtokens 4 \
  --resume_wandb_id none
```

For the merge-size transfer comparison, run these three settings:

| Train max_subtokens | Eval max_subtokens | Role |
|---:|---:|---|
| 3 | 3 | Native ms3 checkpoint |
| 3 | 4 | Forced-transfer evaluation of the ms3 checkpoint |
| 4 | 4 | Native ms4 baseline |

The `3 -> 4` and `4 -> 4` rows should have identical input compression:
compression depends on the text, tokenizer, and eval-time LZW settings, not on
model weights. Compare perplexity only within the same repeated corpus; later
copies are intentionally easier to predict, so repeat-4/repeat-8 absolute PPL
is not directly comparable to standard Wikitext PPL.

> **Future packaging TODO:** if these stress tests become stable public
> benchmarks, consider publishing versioned corpus artifacts. Supporting
> non-Phi tokenizers will require either one token-aware build per tokenizer or
> a redesigned evaluation that constructs tokenizer-specific windows at run
> time.

## RCP cluster wrappers (Run:AI)

Three `runai submit`-ready wrappers, all under `scripts/`, sharing the `Z2Z_SCRATCH` env var
(default `/dlabscratch1/gentilin`, override per-user) for cache/log/output paths:

- **`eval_ckpt_rcp.sh`** — evaluates a zip2zip *training checkpoint* (`model.pt` + `meta.pt`)
  hosted on HF, via `eval_harness.py` + the `Zip2ZipLM` adapter. Used for MS2/MS3/MS4
  from-scratch candidates. Preset: `default_base` (no chat template) or `perplexity`.
- **`eval_z2z_rcp.sh`** — evaluates a *released HF adapter-format* zip2zip model (e.g.
  `epfl-dlab/zip2zip-Phi-3.5-mini-instruct-v0.1`) via the zip2zip pip package
  (`eval_hf_model.py`). Preset: `default` (MC + chat template) or `perplexity`.
- **`validate_phi35_rcp.sh`** — runs *stock* lm-eval (no zip2zip adapter) on the base
  `microsoft/Phi-3.5-mini-instruct` to confirm task/fewshot settings match the paper's
  Table 3 "Base" row before trusting any zip2zip-adapter numbers.

All three write JSON results + a `tee`'d log to `$Z2Z_SCRATCH/logs/eval`, with the model name
(and task name, if `TASKS` is set) baked into the output filename for traceability.
Training-checkpoint evaluation requires the matching `meta.pt` and fails before
scoring if it is absent; otherwise behavior-only RoPE settings could silently
fall back to the wrong geometry.
`eval_ckpt_rcp.sh` can additionally log to W&B (results, per-sample tables, compression
ratios): set `WANDB=1` + `WANDB_NAME`/`WANDB_PROJECT`, and pass `WANDB_API_KEY` into the
job env. Run names get an `eval-` prefix automatically.

## Compression ratio

`Zip2ZipLM`/`HFAdapterLM` (`lm_eval_adapter.py`) accumulate compression counts
(`n_compressed`, `n_base`) that were already computed for every scored request but
previously discarded. `compression_summary()` now surfaces them as
`input_compression_ratio` (base tokens per compressed token on context+continuation —
deterministic given tokenizer + LZW config + text, so it's the same for MC and perplexity
regardless of model weights) and `gen_compression_ratio` (generated tokens only, model-
dependent — relevant for generative tasks like GSM8K, where the model chooses whether to
emit hypertokens). Both scripts print this next to accuracy and include it in the output
JSON; no extra GPU runs are needed to backfill it for past-configured evals.

## GSM8K for zip2zip models: which number to trust

Models finetuned on the zip2zip-1B mix answer in NuminaMath style (`$\boxed{42}$`),
never `#### 42`, so lm-eval's `strict-match` is a format lottery — ignore it. Use
`flexible-extract` for reported numbers. If a model does not stop after its answer
(models trained on pre-`b12efc0` data run into a fabricated next problem, whose
numbers poison flexible-extract's last-number rule), the diagnostic task
`gsm8k_boxed` (`scripts/lm_eval_tasks/`, run with `TASKS=gsm8k_boxed`) adds a
`boxed-first` filter reading the *first* `\boxed{}`/`####` answer. Use it only to
decompose "format/stopping artifact vs math ability" — never as the headline number,
since it isn't comparable with anyone else's published GSM8K results.

## Eval-log health checks

Grep every eval log before trusting its numbers (each line guards a past bug):

```
disabled_ids (14): [0, 1, 2, 32000, ...]   # chat specials protected from LZW merging
folded LoRA into 224 linear layers          # decoder weights actually loaded
# and the ABSENCE of:
Casting complex values to real              # would mean RoPE destroyed at load
```

For digit-protected evals (`--disable_digit_ids` / `DISABLE_DIGIT_IDS=1`, matching
checkpoints trained with the same flag — the canonical recipe since v0.4) the
first line reads `disabled_ids (24)` instead, preceded by
`digit ids disabled for LZW (10): [...]` — that pair is the expected health
signature, not a bug. The adapter auto-enables digit protection when the
checkpoint's meta.pt records it (and logs that it did), so a forgotten flag
cannot silently evaluate a digitsafe checkpoint in the wrong distribution; the
results JSON records the effective setting.

For an untied-hyper-encoder checkpoint (`UNTIED_HYPER_ENCODER=1`, experimental)
the adapter additionally logs `untied hyper-encoder: building separate output
encoder` — same self-healing pattern: it rebuilds the untied model from the
checkpoint's meta.pt and hard-fails if the `hyper_output.*` weights are missing,
so an untied checkpoint can never be silently scored as tied.

Two more meta.pt-driven lines, both behavior-only settings that no state-dict key
could restore:

- `base-token RoPE positions: enabled from meta.pt` — for a v0.6.2+ checkpoint
  (`BASE_TOKEN_POSITIONS=1`). Expected on every compressed eval of such a
  checkpoint; its absence means the eval positioned tokens by compressed index
  instead of base index, i.e. a geometry the checkpoint never trained on. Base-mode
  evals are unaffected either way (an uncompressed stream is `arange` regardless).
- `two-axis RoPE: enabled from meta.pt` — additionally required for a v0.7
  checkpoint. It confirms that even complex pairs use base-stream positions and
  odd pairs use compressed-token positions in every decoder layer. The results
  JSON records `base_token_positions` and `two_axis_rope`, and the pipeline
  audits the latter when `RECIPE=v0.7`.
- `gated compressed-coordinate RoPE: enabled ... from meta.pt` — required for a v0.7.1
  checkpoint. It confirms that the learned compressed-coordinate delta was
  restored and reports its first active decoder layer and frequency pair (0 and
  32 for the named recipe). Results JSON records `gated_compressed_rope`,
  `gated_rope_start_layer`, and `gated_rope_start_pair`; the pipeline audits all
  three after every smoke, final, and WikiText evaluation.
- `hyper-encoder residual: disabled from meta.pt` — for a checkpoint trained with
  `NO_ENCODER_RESIDUAL=1` (an ablation; no production run uses it). This is the one
  line that appears only in the *non-default* case, so its absence is normal and
  healthy — do not grep for it expecting a hit. It exists because the residual is a
  plain runtime attribute with no weight signature: before it was restored from
  meta.pt, a no-residual checkpoint was silently scored *with* the residual, which
  changes the composed hypertoken embedding by exactly the first-token term.
  `scripts/inference.py` logs the same thing as `[inference] hyper-encoder
  residual: disabled from meta.pt`.
- `decoder-time online codebook mask: enabled from meta.pt (active in this
  eval)` — for a v0.6.5 checkpoint in compressed mode. Teacher-forced scoring
  replays the same incremental Rust decoder as generation and masks every
  uninstalled row. The results JSON records both the checkpoint flag
  (`online_codebook_mask`) and whether it was active in this evaluation
  (`online_codebook_mask_active`); the pipeline audits both. Rare legal
  unknown-next-row targets that the current generator cannot represent are
  excluded only for rolling perplexity and counted as
  `compression.online_skipped_targets`. Request-based `loglikelihood` scoring
  (including MC) fails loudly if one occurs in the continuation: skipping a
  negative term would bias the option score and could also corrupt
  `is_greedy`.

The v0.6.4 encoder zero-init (`ZERO_INIT_ENCODER_OUTPUT=1`) deliberately has **no**
eval health line: it only changes the initial weights, which the checkpoint load
overwrites, so it cannot affect eval. Its counterpart lives in the *training* log
as `[encoder_zero_init] zeroed={...}`.

GSM8K generation already used the incremental dictionary before v0.6.5, so its
paired delta isolates the effect of training with the corrected vocabulary.
Compressed MC and perplexity use teacher forcing: their v0.6.5 deltas combine
the weight change with exact-mask renormalization. For rolling perplexity, the
very rare excluded targets still contribute bytes to lm-eval's external
denominator, so inspect and report `online_skipped_targets`. A successful MC
evaluation has zero skipped continuation targets by construction; otherwise
the adapter aborts before reporting a score.

Exact decoder replay currently performs one Rust manager update per compressed
input token from a Python loop. The results JSON records total task time as
`eval_wall_seconds`, plus replay requests, tokens, total seconds, milliseconds
per request, and microseconds per token under
`compression.online_replay_*`. Before a full v0.6.5 evaluation, time one
complete MC task and compare both wall-clock and these counters with the same
task under the historical mask.

Generation-mode runs should show `gen_compression_ratio ≈ 1.4` (a healthy model
emits hyper-tokens; ~1.0 means it never does). Before any full eval of a new
checkpoint, run the 15-minute sanity gate `scripts/diagnose_ckpt_rcp.sh` (train-style
replay must land near the run's final W&B `loss`).

## Backfilling W&B for runs executed with WANDB=0

`scripts/log_results_to_wandb.py` uploads a results JSON (metrics, compression,
eval args) as a W&B run, with `--notes` for the RCP log paths and `--log` to attach
the printed per-task sample blocks as a table.

## Per-sample logging

`sample_logging.py` prints prompt + gold target + model output + score to the console
(never written to the results JSON) for both the checkpoint and HF-model eval paths, via
`simple_evaluate(..., log_samples=True)`. Useful for inspecting GSM8K failures one by one
(arithmetic vs. extraction-format vs. logic) — see `--no_log_samples` to disable.

## Eval with ext/zip2zip (exported models)

For models exported to HF format, use the eval script in `ext/zip2zip`:

```bash
cd ext/zip2zip
pip install -e ".[eval]"

python bench/run_harness_pretrained.py \
    epfl-dlab/Llaza-3.2-1B-MS2F-4K-v0.1 \
    --revision hf \
    --tasks hellaswag piqa winogrande \
    --limit 200
```
