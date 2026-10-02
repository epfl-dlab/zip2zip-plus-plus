# Evaluation

## Internal eval (loss/ppl/accuracy)

For internal diagnostics with a compatible Llama checkpoint, use a batch size of
one:

```bash
LOCAL_BATCH_SIZE=1 bash scripts/eval_lm.sh /path/to/checkpoint

# Specify model config and max_subtokens
MODEL_CONFIG=400M MAX_SUBTOKENS=2 LOCAL_BATCH_SIZE=1 \
  bash scripts/eval_lm.sh /path/to/checkpoint
```

The checkpoint directory is the only positional argument; model settings are
environment variables. The wrapper uses the default Llama tokenizer and fixed
evaluation settings, so it does not automatically adapt to arbitrary checkpoints.
For Phi and checkpoint-aware benchmark evaluation, use `eval_harness.py` below.

**Known limitation ([#21](https://github.com/epfl-dlab/zip2zip-plus-plus/issues/21)).**
With more than one sequence per batch, the internal relaxed loss/PPL/BPB can use
stale candidate lengths from another sequence's codebook. `LOCAL_BATCH_SIZE=1`
avoids this bug; set it explicitly because the wrapper still defaults to 8.
The batched helper remains unfixed. Training loss, relaxed accuracy, and the
separate multi-view scoring in `eval_harness.py` do not use this faulty calculation.

### Historical research utilities

- `measure_segmentation_shift.py` uses no protected token IDs. For
  training-faithful measurements on shards containing special tokens, adapt its
  compressor settings to the tokenizer/checkpoint first.
- `sweep_finemath_merge_size.sh` retains historical `wo_remap` names but does not
  disable remapping or enable the hyper-causal mask. Inspect the actual flags
  before reusing it; the run name does not define the experimental condition.
- For non-Llama tokenizers, direct callers of `Zip2ZipDataset` must supply the
  tokenizer's `disabled_ids`;
  the backward-compatible `None` default uses Llama special-token IDs. The normal
  training entry point supplies the derived IDs explicitly.

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
    --hf_repo epfl-dlab/zip2zip-pp-Llama-3.2-1B-Instruct \
    --hf_revision main \
    --resume_wandb_id none
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--tasks` | arc_challenge, arc_easy, hellaswag, openbookqa, piqa, winogrande, commonsense_qa, medqa_4options | Comma-separated task list |
| `--num_fewshot` | per-task (`None`); presets may force a global value | Few-shot examples |
| `--limit` | None | Per-task sample limit (for quick smoke runs) |
| `--batch_size` | 1 | Eval batch size |
| `--eval_mode` | compressed | `compressed` (LZW) or `base` (vanilla LM scoring) |
| `--resume_wandb_id` | (required) | W&B run ID to resume, or `none` for new run |
| `--no_wandb` | false | Disable W&B logging |
| `--no_log_samples` | false | Skip per-sample W&B Tables |

## Presets

`scripts/eval_presets.yaml` centralizes tasks/num_fewshot/flags for the named presets,
loaded via `--preset` (shared by `eval_harness.py` and `eval_hf_model.py` through
`load_preset.py`) so there are no hardcoded duplicates across scripts. CLI flags still
override preset values.

| Preset | Use case |
|--------|----------|
| `default` | Full MC + generation benchmarks, 2-shot, chat template (paper Table 3 set + triviaqa) |
| `perplexity` | Byte-level PPL on WikiText, Pile, mC4 (C4/en), and dC4 (C4/en.noblocklist), using the configured validation files; 1024-token rolling window |
| `default_base` | Same as `default` but no chat template (from-scratch/non-instruct models) |
| `perplexity_subset` | Paper's corpus/subset setup: full WikiText + fixed 1,000-doc subsets of Pile, mC4 (C4/en), and dC4 (C4/en.noblocklist); seed 1234, 1024-token window, logged under `subset_ppl/` |
| `postsft` | Paper Table 2 generation set: MATH-500, HumanEval-instruct, IFEval — per-task few-shot, chat template |
| `smoke` | 20 samples/task, no W&B — quick sanity check |

`TASKS=...` env var (or `--tasks`) restricts a preset to one task without editing the YAML.

### Perplexity corpus names and comparison protocol

The paper uses **mC4 (C4/en)** and **dC4 (C4/en.noblocklist)**, the latter
without bad-word filtering. Both tasks load `allenai/c4`, with these exact
validation files:

| Paper name | Task ID | Validation file |
|---|---|---|
| mC4 (C4/en) | `zip2zip_mc4` | `en/c4-validation.00000-of-00008.json.gz` |
| dC4 (C4/en.noblocklist) | `zip2zip_dc4` | `en.noblocklist/c4-validation.00000-of-00008.json.gz` |

The paper's appendix, "Perplexity corpora and the fixed subsets," specifies the
full WikiText-2 (raw) test set and fixed subsets of **1,000 documents per corpus**
for Pile, mC4, and dC4, drawn once with **seed 1234**. Use
`--preset perplexity_subset` for this corpus/subset setup, with a 1024-token
window. Its C4 `_sub1k` tasks select subsets from the files listed above, and
results are logged under `subset_ppl/`. The `perplexity` preset instead scores
the configured validation files without the 1,000-document subsampling.

For checkpoint PPL comparisons, use `eval_harness.py` consistently. The two PPL
presets currently omit `apply_chat_template` and `fewshot_as_multiturn`, while
`eval_harness.py` defaults both to false and the separate `eval_hf_model.py`
defaults both to true. A shared preset name therefore does not specify identical
protocol settings across these entry points; this is not evidence by itself that
rolling-PPL scores differ.

### Post-SFT generation benchmarks (`postsft`, paper Table 2)

`PRESET=postsft` scores the three generation benchmarks Table 2 adds on top of
GSM8K (which stays in `default`): the stock lm-eval 0.4.9 tasks, or a
one-dataset variant of one, run through the zip2zip adapter with the chat
template on, greedy decoding, `max_length 4096`, seed 1234. It runs only through
`eval_harness.py` / `eval_ckpt_rcp.sh`; restrict it with `TASKS=...` while
keeping `PRESET=postsft` (under another preset the global `num_fewshot` would
silently turn MATH-500 into a 2-shot run).

| Task | What runs | Few-shot | Table 2 column |
|------|-----------|----------|----------------|
| `math500` | local YAML: `HuggingFaceH4/MATH-500` test split (500 problems, revision-pinned) scored by lm-eval's own `minerva_math` code — Minerva "Problem:/Solution:" prompt, answer normalisation and both scorers, reached via `scripts/lm_eval_tasks/math500_utils.py`; `max_gen_toks` 1024 (the harness default 256 truncates MATH solutions) | 4 fixed Minerva exemplars | strict = `exact_match`: the answer must sit in the Minerva sentence "Final Answer: The final answer is X. I hope it is correct." and be sympy-equivalent to the gold; flex = `math_verify`: `verify(parse(gold), parse(generation))`, i.e. the rightmost `\boxed{}`/math expression anywhere in the generation, checked symbolically. Gold for both = Minerva-normalised last `\boxed{}` of the reference solution (the dataset's `answer` column is rewritten with it) |
| `humaneval_instruct_fence` | local copy of the stock `humaneval_instruct` (same user turn "Write a solution to the following problem and make sure that it passes the tests:" + fenced prompt, assistant turn pre-filled with the function header, HF `code_eval`) with the closing ``` added to the stop sequences. The stock task stops only on completion-style strings and its filter keeps everything up to the LAST fence, so a chat model's explanation after the code lands inside the tested program: 0/164 on a Phi-3.5 finetune whose code was often correct. | 0 | `pass@1` (greedy, one sample) |
| `ifeval` | stock task, 541 prompts, no stop strings, `max_gen_toks` 1280 | 0 | `prompt_level_strict_acc` (Table 2's "prompt (s)"); loose and instruction-level variants are logged too |

The preset deliberately has **no `num_fewshot` key**: a preset value is a global
override in `simple_evaluate`, and these tasks have different protocols, so
`eval_harness.py` and `eval_hf_model.py` default `--num_fewshot` to *per-task*
(the MC and perplexity presets keep their explicit 2 and 0; `load_preset.py`
exports an empty `NUM_FEWSHOT` for such presets and `validate_phi35_rcp.sh`
then omits the flag).

`max_gen_toks` through the zip2zip adapter counts **compressed** tokens in
compressed mode — each may expand to several base tokens — and base tokens in
`EVAL_MODE=base`. The budgets above are large enough that generations normally
end at eos long before them, but a truncated generation is cut at a different
base-token length in the two modes; state this next to any base/control row.

`humaneval_instruct` executes model-written code. `eval_ckpt_rcp.sh` exports
`HF_ALLOW_CODE_EVAL=1` for `PRESET=postsft` (or a `TASKS` list naming humaneval)
and `eval_harness.py` passes lm-eval's `confirm_run_unsafe_code` from that same
variable; `HF_ALLOW_CODE_EVAL=0` in the job env refuses instead of running. The
code runs inside the eval job's container on the cluster, nowhere else.

**Generation decoding fix that this preset exposed (2026-09-06).** The adapter
used to decode generated ids standalone; SentencePiece tokenizers (Phi-3.5)
then drop the word-boundary marker of the first piece, so a completion starting
with four spaces of indentation came back with three and every HumanEval
function failed with `IndentationError` (4 of 5 smoke generations). Generations
are now decoded behind a fixed newline token and the newline cut back off, which
restores the exact continuation; regex-scored tasks (gsm8k, triviaqa) are
unaffected in their metrics, and BPE tokenizers (Llama-3) never stripped
anything. `LEGACY_STRIPPED_GENERATION=1` (`--legacy_stripped_generation`)
restores the old decode for bit-exact reproduction of pre-2026-09 samples; the
results JSON records `preserve_leading_space`.

**Environment.** These tasks need packages the shared `.venvs/lm-eval` must not
get: lm-eval 0.4.9's `minerva_math.utils` asserts `antlr4-python3-runtime==4.11`
at import time, and 4.11 breaks `omegaconf` (used by `eval_hf_model.py` /
`eval_z2z_rcp.sh` from the same venv). A second venv is not an option either —
the shared one carries a torch nightly a fresh venv would not see — so
`eval_ckpt_rcp.sh` installs the extras with `pip --target` into
`.venvs/lm-eval-postsft-extras` and prepends that directory to `PYTHONPATH` for
postsft runs only: `langdetect==1.0.9`, `immutabledict==4.2.1`,
`math-verify==0.7.0`, `antlr4-python3-runtime==4.11.0`. The shared venv is
byte-identical for every other preset and for the pipeline. ifeval downloads
nltk `punkt_tab` on first use.

Typical use is a follow-up eval of an existing pipeline checkpoint, logged into
its W&B run so the numbers sit next to `final/gsm8k/*`:

```bash
runai submit --name postsft-v064 \
  --image ghcr.io/jkminder/dlab-runai-images/pytorch:master \
  --gpu 1 --cpu 8 --memory 64Gi --pvc dlab-scratch:/mnt --large-shm --node-pools h100 \
  --environment CKPT_DIR=/dlabscratch1/gentilin/zip2zip-outputs/<run>/step_8000 \
  --environment TOKENIZER=microsoft/Phi-3.5-mini-instruct \
  --environment PRESET=postsft \
  --environment RESUME_WANDB_ID=<run id> --environment WANDB_STEP=8000 \
  -- bash /dlabscratch1/gentilin/code/zip2zip-core/scripts/eval_ckpt_rcp.sh
```

W&B keys follow `log_results_to_wandb.py`'s `<task>/<metric>_<filter>` flattening:
`final/math500/exact_match`, `final/math500/math_verify`,
`final/humaneval_instruct_fence/pass@1_create_test` (the task's filter is named
`create_test`) and `final/ifeval/prompt_level_strict_acc`, plus their stderr and
loose/instruction-level variants. Every `eval_harness.py` run with
`--output_path` now also writes `<results stem>_samples.json` with the
per-sample prompts, generations and scores (the W&B sample tables are not
written under `RESUME_WANDB_ID`), so paired per-problem comparisons stay
possible, and the results JSON carries lm-eval's `versions` and `config`
records. For a `MAX_CODEBOOK_SIZE=0` control checkpoint add `EVAL_MODE=base` as
for every other preset. Smoke-test with `LIMIT=5` and no `RESUME_WANDB_ID` first.

### Repeated-window Wikitext stress tests

`zip2zip_wikitext_repeat4` and `zip2zip_wikitext_repeat8` are opt-in
merge-size-transfer stress tests. They are not part of any default preset or
pipeline, and the standard Wikitext task is unchanged.

The released corpora are the `repeat4` and `repeat8` configurations of
[`epfl-dlab/zip2zip-wikitext-repeat-phi35`](https://huggingface.co/datasets/epfl-dlab/zip2zip-wikitext-repeat-phi35).
The task YAMLs pin revision `v1.0.0`, so evaluation downloads the exact
artifacts used for the paper through the normal Hugging Face cache; no local
corpus build is required.

Both configurations were built with the
`microsoft/Phi-3.5-mini-instruct` tokenizer and are suitable only for Zip2Zip
checkpoints using that tokenizer. Forced evaluation at a merge size unseen
during training is supported only by hierarchical hyper-encoders.

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

For dataset maintenance only, `scripts/build_repeated_wikitext.py` reproduces
the JSONL and manifests from the complete WikiText-2 raw test split. Evaluation
does not invoke the builder, and an explicit rebuild requires `--overwrite`.

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

## TriviaQA and stop-string trimming

`triviaqa` (built-in lm-eval task, 17,944 validation questions, closed-book) joined the
`default`/`default_base`/`smoke` presets in 2026-08 as a second generative benchmark. It
probes factual recall rather than reasoning, and its size makes 2-3pt deltas resolvable
in a single run (SE ~0.4pt). Unlike GSM8K it is scored by `exact_match` of the whole
generation against the alias list — no extraction regex — which is why the adapter now
cuts generated text at the first stop-string occurrence (lm-eval's contract; the
pre-2026-08 adapter returned the stop string and, in compressed mode, any hyper-token
expansion overshoot behind it). The tail would systematically depress text-scored
metrics. It was invisible to GSM8K's first-match filters (`strict-match`, `boxed-first`),
but `flexible-extract` takes the LAST number, so a re-run of an old checkpoint can score
a sample differently wherever the discarded tail happened to contain a match (digits are
impossible in the overshoot of digit-protected checkpoints, but `[$.,]` runs still count)
— treat trimmed vs untrimmed `flexible-extract` as a comparability boundary rather than
assuming bit-equality. `--legacy_untrimmed_stops` (env: `LEGACY_UNTRIMMED_STOPS=1` for
`eval_ckpt_rcp.sh`) restores the old per-request returns bit-exactly; reproducing a full
pre-2026-08 results artifact also needs the old 7-task list pinned via `TASKS=...`, since
the presets now include triviaqa and the run-global `eval/*_compression_ratio` aggregates
absorb every task in the run. The results JSON records the effective setting as
`trim_stop_strings` in `args`.

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
signature on Phi-3.5, not a bug. On Llama-3.2 the same recipe reads
`digit ids disabled for LZW (1110)` and `disabled_ids (1366)`: its byte-level
BPE has a piece for every 1-, 2- and 3-digit string, and all of them are
protected so numbers keep their base segmentation (up-to-3-digit groups,
where Phi keeps single digits). The adapter auto-enables digit protection when the
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
    epfl-dlab/zip2zip-pp-Llama-3.2-1B-Instruct \
    --revision hf \
    --tasks hellaswag piqa winogrande \
    --limit 200
```
