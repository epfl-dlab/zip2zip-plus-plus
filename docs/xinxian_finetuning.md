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

**Question**: does separating the input-embedding role from the output-logit role
matter in practice?

Interpretation goal: understand whether the input and output hyper-encoders are
learning different functions, rather than only measuring the final benchmark
delta.

Result: comparison between `v0.6.4` and `vx0.6.4.1`.

| Version | ARC-c | ARC-e | HellaSwag | OBQA | PIQA | WinoGrande | GSM8K strict \| flexible | Wiki byte_ppl↓ | gen_compression_ratio |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| `v0.6.4` | 0.5700 | **0.8304** | 0.7233 | 0.4660 | **0.8003** | **0.7443** | 0.215 \| **0.6770** | **1.6574** | **1.2593** |
| `vx0.6.4.1` | **0.5751** | 0.8300 | **0.7247** | **0.4680** | 0.7982 | 0.7435 | **0.224** \| 0.5861 | 1.6592 | 1.2406 |

Takeaway: the shared hyper-encoder is roughly neutral on multiple-choice tasks
and Wiki byte-ppl, but it hurts GSM8K flexible extraction sharply (-9.1pt). This
suggests that the input and output hyper-encoders may need to learn different
functions, and that the untied design in `v0.6.4` remains important for
generation.

Note: this drop is larger than the earlier shared-hyper-encoder experiment on
top of `v0.5`, where GSM8K flexible extraction was around 0.636. The current run
also includes the base-token-position fix, so the interaction may be different.
Since that earlier result was compared against `v0.5` rather than `v0.6.4`, it
does not change the conclusion here.


### `vx0.6.4.2`: no residual initialization

Comparison: `v0.6.4` vs `vx0.6.4.2`.

Single change: remove residual initialization.

**Question**: does residual initialization actually help in the flat hyper-encoder?

Motivation: although residual initialization really helps early training,
current embedding analysis suggests that hypertoken embeddings may have
unusually high similarity to the first base token in the merged span. This run
tests whether that behavior is helpful structure or an initialization artifact.

Result: comparison between `v0.6.4` and `vx0.6.4.2`.

| Version | ARC-c | ARC-e | HellaSwag | OBQA | PIQA | WinoGrande | GSM8K strict \| flexible | Wiki byte_ppl↓ | gen_compression_ratio |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| `v0.6.4` | **0.5700** | **0.8304** | **0.7233** | **0.4660** | **0.8003** | **0.7443** | **0.215** \| **0.6770** | **1.6574** | **1.2593** |
| `vx0.6.4.2` | 0.5367 | 0.8089 | 0.7104 | 0.4620 | 0.7976 | 0.7427 | 0.152 \| 0.6384 | 1.7052 | 1.2537 |

Takeaway: removing residual initialization hurts broadly. Multiple-choice scores
fall across the board, Wiki byte-ppl worsens substantially, and GSM8K flexible
extraction drops by 3.9pt. The residual path therefore looks like useful
structure rather than only an early-training optimization aid.

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

**Question**: does the hierarchical hyper-encoder improve performance under the
standard recipe?

Default setting: keep `MAX_SUBTOKENS=4`, matching the current recipe line.

Result: compare the flat baseline against the hierarchical encoder with
residual initialization.

| Version | Encoder | ARC-c | ARC-e | HellaSwag | OBQA | PIQA | WinoGrande | GSM8K strict \| flexible | Wiki byte_ppl↓ | gen_compression_ratio |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| `v0.6.4` | flat | **0.5700** | 0.8304 | 0.7233 | **0.4660** | **0.8003** | **0.7443** | **0.215** \| **0.6770** | **1.6574** | **1.2593** |
| `vx0.6.4.3` | hierarchical | 0.5597 | **0.8359** | **0.7252** | 0.4640 | **0.8003** | 0.7348 | 0.201 \| 0.6308 | 1.6586 | 1.2490 |

Takeaway: the hierarchical encoder slightly improves ARC-e and HellaSwag and
roughly matches PIQA and Wiki byte-ppl, but it is weaker on ARC-c, WinoGrande,
GSM8K, and generation compression.

It is not a broad improvement over the flat
`v0.6.4` baseline, especially once training cost is included: in the local logs,
`vx0.6.4.3` averaged 1.41s/step and took 3h26m for 8k steps, while the flat
`vx0.6.4.2` ablation averaged 1.00s/step and took 2h17m under the same 8k-step
setup.

One possible explanation is that the hierarchical pairwise composer is
simpler and more structured, but less expressive than the flat encoder. Its
main remaining advantage is that the recurrent structure might transfer across
merge sizes, which is what `vx0.6.4.5` will test.

### `vx0.6.4.4`: hierarchical hyper-encoder without residual initialization

Comparison: `vx0.6.4.3` vs `vx0.6.4.4`.

Single change: remove residual initialization from the hierarchical
hyper-encoder.

**Question**: does adding the first base-token embedding interfere with the
recurrent pattern that the hierarchical hyper-encoder is supposed to learn?

Motivation: in the current residual design, the first base-token embedding
reaches the final hypertoken representation through two paths:

`h_hyper = PairwiseHyperEncoder(e_1, ..., e_k) + e_1`

The first token is both an input to the pairwise composer and an external
residual added to the output. This may overemphasize `e_1` and force the
pairwise composer to learn a counter-correction instead of only learning how to
combine subtokens.

Possible interventions:

- **Remove the residual path entirely** (`vx0.6.4.4`):
  `h_hyper = PairwiseHyperEncoder(e_1, ..., e_k)`.
  This is the simplest test and the only one implemented here. The downside is
  that the model loses the stable first-token starting point, which may need
  longer training time.
- **Keep the residual path, but remove the first base token from the
  hyper-encoder input**:
  `h_hyper = PairwiseHyperEncoder(e_2, ..., e_k) + e_1`.
  This avoids double-counting the first token, but may hide useful composition
  information from the composer.
- **Remove the permanent residual path, but initialize the composer to behave
  like a first-token mapping at step 0**:
  `h_hyper = PairwiseHyperEncoder(e_1, ..., e_k)`, with
  `h_hyper^(0) ~= e_1`.
  This keeps the good initialization behavior without forcing a fixed skip
  connection throughout training.

Decision rule: compare against `vx0.6.4.3` first. If removing residual
initialization helps, use `vx0.6.4.4` as the base for the merge-size transfer
experiment below; otherwise use `vx0.6.4.3`.

Result: compare the two hierarchical variants.

| Version | Residual init | ARC-c | ARC-e | HellaSwag | OBQA | PIQA | WinoGrande | GSM8K strict \| flexible | Wiki byte_ppl↓ | gen_compression_ratio |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| `vx0.6.4.3` | yes | **0.5597** | **0.8359** | **0.7252** | **0.4640** | **0.8003** | 0.7348 | **0.201** \| **0.6308** | **1.6586** | 1.2490 |
| `vx0.6.4.4` | no | 0.5478 | 0.8304 | 0.7051 | 0.4620 | 0.7965 | **0.7356** | 0.134 \| 0.6141 | 1.7128 | **1.2567** |

Takeaway: removing the residual path hurts the hierarchical encoder broadly,
especially Wiki byte-ppl and GSM8K. `vx0.6.4.3` is the cleaner base for any
merge-size transfer test.

### `vx0.6.4.5`: merge-size transfer test

Base: `vx0.6.4.3`, the better hierarchical variant.

Single change: train with `MAX_SUBTOKENS=3` instead of `MAX_SUBTOKENS=4`.

Evaluation: evaluate the checkpoint both normally and with forced
`MAX_SUBTOKENS=4`.

**Question**: does the hierarchical hyper-encoder learn transferable recurrent
patterns across merge sizes?

Success criterion: the `MAX_SUBTOKENS=3` hierarchical run should remain
competitive when evaluated with forced merge size 4. If it does, that would
support the hypothesis that the hierarchical hyper-encoder learns reusable
structure rather than overfitting to one merge-size setting.

Implementation: add an eval-time merge-size override, then launch training and evaluation as separate jobs.

Transfer eval: compare the `vx0.6.4.5` checkpoint trained with merge size 3
against the `vx0.6.4.3` merge-size-4 baseline, with both its native merge-size-3
eval and the forced merge-size-4 eval.

| Version | Train max_subtokens | Eval max_subtokens | ARC-c | ARC-e | HellaSwag | OBQA | PIQA | WinoGrande | GSM8K strict \| flexible | Wiki byte_ppl↓ | gen_compression_ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `vx0.6.4.3` | 4 | 4 | **0.5597** | **0.8359** | **0.7252** | 0.4640 | 0.8003 | 0.7348 | 0.201 \| **0.6308** | 1.6586 | 1.2490 |
| `vx0.6.4.5` | 3 | 3 | 0.5529 | 0.8333 | 0.7224 | **0.4680** | 0.8058 | **0.7459** | **0.211** \| 0.6118 | **1.6581** | 1.2527 |
| `vx0.6.4.5` | 3 | 4 | 0.5520 | 0.8321 | 0.7231 | **0.4680** | **0.8063** | **0.7459** | 0.208 \| 0.6179 | 1.6600 | 1.2538 |

Takeaway: the merge-size-3 run appears to transfer cleanly to eval merge
size 4, but this conclusion may be biased by how small the actual compression
difference is between `max_subtokens=3` and `max_subtokens=4`. The `4 -> 4` and
`3 -> 3` results are very close overall, and `4 -> 4` is still better on several
MC tasks.

Compression ratio differences are tiny:

| Setting | ms4 | ms3 | Relative diff |
|---|---:|---:|---:|
| Train summary | 1.38337 | 1.37389 | ~0.69% |
| Default eval input | 1.07716 | 1.07697 | ~0.018% |
| Wiki eval input | 1.16898 | 1.16643 | ~0.22% |

So `max_subtokens=3` does not actually make the eval inputs much less
compressed. Most merges likely already have length <=3, or the length-4 cap only
affects a very small fraction of tokens. This means the smaller merge size is not
exposed strongly enough to produce a stable improvement. Generation is also
short-form here, so the output-side compression difference is limited as well,
as reflected by the similar `gen_compression_ratio` values.

The one place where the setting has more room to matter is Wikitext, the only
long-context task:

| Eval set | Requests / windows | Base tokens | Avg base tokens per request |
|---|---:|---:|---:|
| MC + GSM8K | 63,889 | 13,541,095 | ~212 |
| Wikitext | 114 | 353,256 | ~3,100 |

The compressed-token difference between ms3 and ms4 is also larger on Wikitext:

| Eval set | ms3 vs ms4 compressed-token difference |
|---|---:|
| MC + GSM8K | ~0.0175% |
| Wikitext | ~0.22% |

Notably, the clearest degradation from forced ms4 appears exactly on Wikitext:
byte-ppl worsens from `1.6581` to `1.6600`. This is also the only setting where
native ms3 is clearly better than forced ms4. That suggests the transfer cost may
only become visible when merge size has enough opportunity to actually change the
compression pattern.
