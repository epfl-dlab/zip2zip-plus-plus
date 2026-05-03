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
