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

## RCP cluster wrappers (Run:AI)

Three `runai submit`-ready wrappers, all under `scripts/`, sharing the `SCRATCH` env var
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

All three write JSON results + a `tee`'d log to `$SCRATCH/logs/eval`, with the model name
(and task name, if `TASKS` is set) baked into the output filename for traceability.

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
