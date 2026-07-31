# Workspace

Shared infrastructure for the Llaza/zip2zip project team.

## W&B Project

All training and evaluation runs are logged to a shared Weights & Biases project:

- **Entity**: `epfl-dlab`
- **Project**: [`llaza`](https://wandb.ai/epfl-dlab/llaza)

Defaults are configured in `src/zip2zip_core/project.py`:
```python
WANDB_ENTITY = "epfl-dlab"
WANDB_PROJECT = "llaza"
HF_ORG = "epfl-dlab"
```

### Naming conventions

- **Training runs**: auto-named with a unique 4-char suffix (e.g. `llaza_1b_llama32_instruct_ms2_1bt_a3f2`)
- **Evaluation runs**: prefixed with `eval-` and tagged with `eval` for easy filtering (e.g. `eval-step_6000`)
- Each run's W&B user is automatically tracked — no need for extra tags

### Filtering tips

- Filter by tag `eval` to see only evaluation runs
- Use the `hf_url` summary field to jump to the model on HF Hub

### Workspace sections

A finetune+eval pipeline run logs ~124 metric keys from four writers, so the
default workspace is a wall of alphabetically grouped panels. Nothing here
renames or drops a metric: legacy keys stay comparable across the whole v0.6.x
history. The cleanup is split by what W&B actually scopes per run:

- **Duplicate panels are suppressed at log time**, so every pipeline run comes
  out deduplicated on its own. `define_metric(..., hidden=True)` hides a metric's
  auto-panel while keeping its values in the run:
  [`train.py`](../src/zip2zip_core/train.py) for `compressed/backward_loss`, and
  [`eval_harness.py`](../scripts/eval_harness.py) for lm-eval's unprefixed
  `<task>/*` keys — the latter only when `--resume_wandb_id` is set, i.e. on the
  pipeline path where `final/*` already carries the same numbers. A standalone
  `eval_ckpt_rcp.sh` run keeps them visible, being the only copy there.
- **Sections, order and open/collapsed state are project-scoped**: a run cannot
  ship its own panel layout. `scripts/wandb_workspace_view.py` defines them as
  code, and pipeline phase 4 refreshes the view when `WANDB_VIEW_URL` points at
  one (a failure there never fails the pipeline). The eval venv installs
  `wandb-workspaces` next to `wandb` for it, W&B runs only.

Create the view once, then reuse its URL — saving a freshly constructed
`Workspace` always creates a *new* view, so a pipeline that saved without a URL
would leave one saved view behind per run:

```bash
python scripts/wandb_workspace_view.py --apply          # prints the view URL
# then pass that URL on every run:
runai submit ... --environment WANDB_VIEW_URL='<printed URL>'
```

```bash
python scripts/wandb_workspace_view.py                       # print the layout
python scripts/wandb_workspace_view.py --verify epfl-dlab/zip2zip-core/<run_id>
python scripts/wandb_workspace_view.py --apply --view-name "z2z clean"
```

`--verify` reports keys a run logs that no panel covers. `tests/test_wandb_view.py`
enforces the same coverage statically against `train.py`, so adding a metric
without a panel fails a test instead of quietly vanishing from the workspace.

Who writes what, and why the sections are split that way:

| Keys | Writer | Section |
|---|---|---|
| no prefix (`loss`, `acc`, `compression`, `lr`, ...) | `train.py:2236` | legacy training sections (3, 5, 7, 8, 9) |
| `compressed/*` | `train.py:2257` | corrected training sections (4, 6, 7, 8) |
| `objective/backward_loss` | `train.py:2256` | section 3 — the loss actually optimised |
| `base_view/*`, `rope_gate*` | `train.py:2336`, `:2369` | section 11-12, only with replay / gated RoPE |
| `smoke/*`, `final/*` | `log_results_to_wandb.py --prefix` | sections 1-2, on their own `*/step` x-axis |
| `<task>/<metric>`, `evaluation/*`, tables | lm-eval's `WandbLogger`, `eval_harness.py:305` | section 15 |
| `eval/*_compression_ratio` | `eval_harness.py:311` | section 10 |

Legacy and corrected metrics are deliberately kept in different sections, never
overlaid in one panel: they use different denominators and are not comparable
(see the metric rules in `CLAUDE.md` and `docs/finetuning.md`).

Two duplications are known, and both are suppressed at log time:

- `compressed/backward_loss` is byte-for-byte the bare `backward_loss` (both read
  `compat_backward_loss`); its panel is hidden and it has no panel in the layout.
  `objective/backward_loss` is kept, being the only one that differs once
  base-view replay is on.
- pipeline phase 3 logs the final numbers twice: unprefixed by lm-eval and under
  `final/` by `log_results_to_wandb.py`. The unprefixed copies keep an explicit
  collapsed section, because they are the *only* keys a standalone
  `eval_ckpt_rcp.sh` run writes — hidden metrics still plot in a panel that names
  them, they just get no auto-panel.

One trap the view labels explicitly: `final/eval/*_compression_ratio` is written
twice at the same `final/step`, once from the MC JSON and once from the wikitext
JSON. The unprefixed `eval/*` is MC-only.

## HuggingFace Hub

Models and checkpoints are hosted under the [`epfl-dlab`](https://huggingface.co/epfl-dlab) organization.

### Repo naming

| Type | Naming pattern | Example |
|------|---------------|---------|
| Candidate (auto from training) | `epfl-dlab/candidate-{run_name}` | `epfl-dlab/candidate-llaza_1b_ms2_a3f2` |
| Production-ready | `epfl-dlab/Llaza-{base}-{config}-{version}` | `epfl-dlab/Llaza-3.2-1B-MS2F-4K-v0.1` |

### Branch layout

Each model repo has two branches:

| Branch | Format | Content |
|--------|--------|---------|
| `main` | torchtitan (model.pt, meta.pt) | Training checkpoint, for resume or export |
| `hf` | HuggingFace (safetensors, zip2zip_config.json) | Exported model, for inference with `zip2zip` |

### Datasets

| Dataset | URL |
|---------|-----|
| llaza-200B | [`epfl-dlab/llaza-200B`](https://huggingface.co/datasets/epfl-dlab/llaza-200B) |
| llaza-20B | [`epfl-dlab/llaza-20B`](https://huggingface.co/datasets/epfl-dlab/llaza-20B) |
| llaza-1B | [`epfl-dlab/llaza-1B`](https://huggingface.co/datasets/epfl-dlab/llaza-1B) |

## Pushing checkpoints

```bash
# Auto-detects format, pushes training ckpt to main + auto-exports to hf branch
python scripts/push_checkpoint.py \
    --ckpt_dir /path/to/step_6000 \
    --repo_id epfl-dlab/Llaza-3.2-1B-v0.1

# Skip auto-export
python scripts/push_checkpoint.py \
    --ckpt_dir /path/to/step_6000 \
    --repo_id epfl-dlab/Llaza-3.2-1B-v0.1 \
    --no_export
```

### What happens during push

1. Reads `meta.pt` to extract training args
2. Generates a model card (`README.md`) with training config
3. Uploads training checkpoint to `main` branch
4. Auto-exports to HF format (safetensors) into a temp directory
5. Uploads exported model to `hf` branch

### Auto-upload during training

When `--wandb` is enabled, the final checkpoint is automatically uploaded to `epfl-dlab/candidate-{run_name}` in a background thread (avoids NCCL barrier timeout on distributed runs). The HF URL is saved to the wandb run summary.
