# Xinxian Finetuning Experiments

Naming convention:

- `v*` recipes are Andrea's mainline runs. See `docs/finetuning.md` and
  `scripts/recipes.py` for the canonical recipe history.
- `vx<base>.N` recipes are Xinxian's exploratory / interpretability runs:
  the `v<base>` portion before the final `.N` names the mainline recipe they
  branch from, and `.N` is Xinxian's exploratory sequence number. For example,
  `vx0.6.4.1` branches from `v0.6.4`; a future `vx0.7.1` would branch from
  `v0.7`.
- `vx*` runs must not change the `v*` mainline recipe by default.
- Only measured improvements or clear mechanistic findings should be promoted
  back into the `v*` line.

Current anchor: `v0.6.4`, the standard recipe.

## Hyper-Encoder Ablations

### `vx0.6.4.1`: shared hyper-encoder

Comparison: `v0.6.4` vs `vx0.6.4.1`.

Single change: use a shared hyper-encoder instead of the untied input/output
hyper-encoders in `v0.6.4`.

Question: does separating the input-embedding role from the output-logit role
matter in practice?

Interpretation goal: understand whether the input and output hyper-encoders are
learning different functions, rather than only measuring the final benchmark
delta.

Status: to run. (original checkpoints use false initialization)

### `vx0.6.4.2`: no residual initialization

Comparison: `v0.6.4` vs `vx0.6.4.2`.

Single change: remove residual initialization.

Question: does residual initialization actually help in the flat hyper-encoder?

Motivation: although residual initialization really helps early training, current embedding analysis suggests that hypertoken embeddings may
have unusually high similarity to the first base token in the merged span (next section). This
run tests whether that behavior is helpful structure or an initialization
artifact.

Status: to run.

## Embedding Interpretation

After `vx0.6.4.2`, analyze the embedding properties directly (`v0.6.4` vs `vx0.6.4.2`).

Starting point: the previous Claude report.

Goal: explain the measured differences mechanistically, especially:

- whether input and output hyper-encoders specialize differently;
- whether residual initialization mainly helps optimization or imposes a useful
  embedding geometry;
- whether first-token similarity is a feature, a shortcut, or a failure mode.

Owner: Xinxian.

## Hierarchical Hyper-Encoder

### `vx0.6.4.3`: hierarchical hyper-encoder

Comparison: `v0.6.4` vs `vx0.6.4.3`.

Single change: replace the flat hyper-encoder with the hierarchical
hyper-encoder.

Question: does the hierarchical hyper-encoder improve performance under the
standard recipe?

Default setting: keep `MAX_SUBTOKENS=4`, matching the current recipe line.

### `vx0.6.4.4`: hierarchical hyper-encoder without residual initialization

Comparison: `vx0.6.4.3` vs `vx0.6.4.4`.

Single change: remove residual initialization from the hierarchical
hyper-encoder.

Question: does adding the first base-token embedding interfere with the
recurrent pattern that the hierarchical hyper-encoder is supposed to learn?

Decision rule: compare against `vx0.6.4.3` first. If removing residual
initialization helps, use `vx0.6.4.4` as the base for the merge-size transfer
experiment below; otherwise use `vx0.6.4.3`.

### `vx0.6.4.5`: merge-size transfer test

Base: whichever is better between `vx0.6.4.3` and `vx0.6.4.4`.

Single change: train with `MAX_SUBTOKENS=3` instead of `MAX_SUBTOKENS=4`.

Evaluation: evaluate the checkpoint both normally and with forced
`MAX_SUBTOKENS=4`.

Question: does the hierarchical hyper-encoder learn transferable recurrent
patterns across merge sizes?

Success criterion: the `MAX_SUBTOKENS=3` hierarchical run should remain
competitive when evaluated with forced merge size 4. If it does, that would
support the hypothesis that the hierarchical hyper-encoder learns reusable
structure rather than overfitting to one merge-size setting.

Implementation: add an eval-time merge-size override, then launch training and evaluation as separate jobs.