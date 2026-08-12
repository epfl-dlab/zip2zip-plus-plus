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

Result: compare the three hierarchical variants.

| Version | Residual init | ARC-c | ARC-e | HellaSwag | OBQA | PIQA | WinoGrande | GSM8K strict \| flexible | Wiki byte_ppl↓ | gen_compression_ratio |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| `vx0.6.4.3` | yes | 0.5597 | **0.8359** | **0.7252** | 0.4640 | **0.8003** | 0.7348 | **0.201** \| **0.6308** | **1.6586** | 1.2490 |
| `vx0.6.4.4` | no | 0.5478 | 0.8304 | 0.7051 | 0.4620 | 0.7965 | 0.7356 | 0.134 \| 0.6141 | 1.7128 | **1.2567** |
| `vx0.6.4.8` | no | **0.5606** | 0.8354 | 0.7225 | **0.4800** | 0.7922 | **0.7522** | 0.177 \| 0.5315 | 1.6619 | 1.1986 |

Takeaway: the broad `vx0.6.4.4` regression was largely an initialization
confound, not an inherent cost of removing the residual. With the controlled
`gamma=0.05` start, `vx0.6.4.8` recovers the MC and WikiText performance;
GSM8K flexible remains the notable regression.

#### `vx0.6.4.8`: revisit no-residual with controlled LayerNorm initialization

Comparison: `vx0.6.4.4` vs `vx0.6.4.8`.

The original no-residual hierarchical ablation also changed the effective
initialization from the residual recipe's zero-output encoder branch to the
pair encoder's default final LayerNorm (`gamma = 1`, `beta = 0`). That large
start was not controlled in the original interpretation of the result.

`vx0.6.4.8` keeps the hierarchical composer and residual-disabled architecture
from `vx0.6.4.4`, but initializes the pair encoder's final LayerNorm with
`gamma = 0.05`, `beta = 0`, matching the small-start treatment in
`vx0.6.4.7`. The pair composer applies its final LayerNorm at every left-fold
step, so the small scale is reset rather than accumulated with merge depth.
This makes LayerNorm initialization the only change from `vx0.6.4.4`.

Recipe: `RECIPE=vx0.6.4.8`. Use seed 42 for the direct comparison with
`vx0.6.4.4`.

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

#### Repeated-window Wikitext stress test

The standard evaluation above does not activate the distinction between merge
sizes strongly enough: on ordinary Wikitext, ms4 changes the compressed-token
count by only about 0.22% relative to ms3. We therefore added an opt-in
repeated-window Wikitext evaluation. The original Wikitext task and every
default evaluation preset remain unchanged.

The new evaluation is constructed as follows:

1. Start from the Wikitext-2 raw test split and apply the same detokenization as
   the standard lm-eval Wikitext task.
2. Tokenize with the Phi-3.5 tokenizer and split each source document into
   token-aware blocks.
3. Decode each block, repeat its text either four or eight times with `\n\n`
   separators, and re-tokenize it. Source blocks are shortened as necessary so
   every final row contains at most 1,000 base tokens.
4. Score every repeated copy with `loglikelihood_rolling`; no copy is excluded
   from the loss. Because every row fits within the 1,024-token evaluation
   window, it is scored as one window with a freshly initialized LZW state.
5. Compute word and byte denominators from the exact repeated text stored in
   the JSONL. This avoids changing the denominator merely because the corpus is
   represented as repeated blocks.

The two tasks are `zip2zip_wikitext_repeat4` and
`zip2zip_wikitext_repeat8`. They are deliberately absent from the default
pipeline. For each corpus we evaluate the same three settings as above:

- `3 -> 3`: native evaluation of the ms3 checkpoint;
- `3 -> 4`: forced-ms4 evaluation of the ms3 checkpoint;
- `4 -> 4`: native evaluation of the ms4 checkpoint.

The `3 -> 4` and `4 -> 4` rows must have exactly the same compression ratio.
LZW tokenization is determined by the corpus, tokenizer, protected token IDs,
codebook size, and eval-time `max_subtokens`; it does not depend on model
weights. This gives the clean comparison we want: both models score exactly
the same ms4-compressed sequence, and only their learned parameters differ.

The repeated corpora substantially increase both overall compression and the
frequency with which ms4 actually creates length-4 hypertokens:

| Eval corpus | ms3 compression | ms4 compression | Relative ratio gain | Length-4 share of compressed targets | Base-token coverage by length-4 targets |
|---|---:|---:|---:|---:|---:|
| Standard Wikitext | 1.16643 | 1.16898 | ~0.22% | not measured | not measured |
| Repeat-4 | 1.69861 | 1.71488 | 0.96% | 2.81% | 6.54% |
| Repeat-8 | 2.02848 | 2.13734 | 5.37% | 15.99% | 29.90% |

The offline span audit exactly reproduces the aggregate compressor counts in
the evaluation JSONs. Repeat-4 contains 21,221 scored length-4 hypertokens;
repeat-8 contains 195,628. Thus repeat-8 is no longer a nominal merge-size
change: almost 30% of its scored base tokens belong to length-4 spans.

Final repeated-window results (`byte_ppl = 2^(bits/byte)`; lower is better):

<table>
  <thead>
    <tr>
      <th rowspan="2">Train &rarr; eval<br><code>max_subtokens</code></th>
      <th colspan="3">Repeat-4</th>
      <th colspan="3">Repeat-8</th>
    </tr>
    <tr>
      <th>Byte ppl</th>
      <th>Bits/byte</th>
      <th>Input compression</th>
      <th>Byte ppl</th>
      <th>Bits/byte</th>
      <th>Input compression</th>
    </tr>
  </thead>
  <tbody>
    <tr><td><code>3 &rarr; 3</code></td><td><strong>1.18895</strong></td><td><strong>0.24968</strong></td><td>1.69861</td><td><strong>1.10561</strong></td><td><strong>0.14484</strong></td><td>2.02848</td></tr>
    <tr><td><code>3 &rarr; 4</code></td><td>1.20498</td><td>0.26900</td><td>1.71488</td><td>1.15337</td><td>0.20585</td><td>2.13734</td></tr>
    <tr><td><code>4 &rarr; 4</code></td><td>1.19217</td><td>0.25358</td><td>1.71488</td><td>1.11206</td><td>0.15324</td><td>2.13734</td></tr>
  </tbody>
</table>

On repeat-4, forced `3 -> 4` is 1.07% worse in byte-ppl than native
`4 -> 4`, an absolute increase of 0.01542 bits/byte. On repeat-8, where the
compression patterns diverge much more strongly, that gap grows to 3.71% in
byte-ppl and 0.05261 bits/byte. Forced `3 -> 4` is also consistently worse than
the same checkpoint's native `3 -> 3` evaluation.

**Takeaway:** the ms3 checkpoint can execute an unseen ms4 composition without
collapsing, so the hierarchical hyper-encoder exhibits partial merge-size
transfer. However, the transfer is not clean or lossless. Once length-4 spans
are common enough to make the two compression patterns meaningfully different,
the ms3 checkpoint falls clearly behind the native ms4 checkpoint on the same
compressed sequence. The original benchmark understated this transfer cost
because it almost never exercised the additional merge capacity.

The absolute perplexities of repeat-4 and repeat-8 should not be compared as if
they were natural-language benchmark improvements: later copies are
intentionally easier to predict. These tasks isolate transfer under recurrent,
high-compression LZW patterns; they do not by themselves establish general
long-context performance on non-repeated text.

### `vx0.6.4.6` and `vx0.6.4.7`: no-residual initialization

Status: planned.

**Motivation**: the existing no-residual run, `vx0.6.4.2`, starts with
`gamma = 1, beta = 0`. Its initial encoder output is large, and the trained
output develops a large shared ruler component. We hypothesize that this
geometry is driven primarily by the final LayerNorm initialization and the
resulting large-norm hypertoken embeddings, rather than by the removal of the
residual path itself. To test this hypothesis, we keep the residual disabled
and compare two controlled LayerNorm initializations: an exact-zero
initialization matching the residual model's hyper-encoder branch, and a small
nonzero initialization at roughly the scale of ordinary token embeddings. We
then compare both performance and whether the shared ruler disappears.

Both new runs use

`E(H) = F(H)`

with no first-token residual. Only the final LayerNorm initialization changes.

| Version | gamma init | beta init | Step-0 behavior |
|---|---:|---:|---|
| `vx0.6.4.6` | 0 | 0 | Every hypertoken output is exactly zero |
| `vx0.6.4.7` | small | 0 | Token-dependent output with ordinary-token-scale norm |

#### `vx0.6.4.6`: exact zero

Set `gamma = 0, beta = 0`, both trainable. This preserves the controlled
zero-output start, but without the first-token residual. Gamma and beta can
learn immediately; earlier encoder layers receive little or no gradient until
gamma moves away from zero.

#### `vx0.6.4.7`: small meaningful initialization

Use `beta = 0` and a small positive scalar gamma. Exact norm matching is not
necessary for this first scout. Start approximately with:

- input encoder: `gamma_in ~= 0.04`;
- output encoder: `gamma_out ~= 0.05`.

Before launch, run a fixed sample of active-codebook entries and measure the
output norm **after mean pooling**. We only need it to be in roughly the same
range as ordinary embeddings: about 2.04 for input and 2.63 for output. If it is
clearly too large or too small, rescale once:

`g_new = g_old * target_norm / measured_norm`.

The post-pooling check is important: `sqrt(3072) ~= 55` is the rough
per-position LayerNorm scale, not necessarily the final pooled norm. Keep beta
at zero because nonzero beta directly inserts a shared vector and may create a
ruler by construction.

Result (`vx0.6.4.7`): the launched recipe uses a single
`encoder_output_init_scale = 0.05` (beta = 0) on **both** encoders
(`gamma_in = gamma_out = 0.05`). Seed 42, against the `v0.6.4` anchor with the
`gamma = 1` no-residual baseline `vx0.6.4.2` included to isolate the init effect:

| seed 42 | ARC-c | ARC-e | HellaSwag | OBQA | PIQA | WinoGrande | GSM8K strict \| flexible | Wiki byte_ppl↓ | gen_compression_ratio |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| `v0.6.4` (residual, γ=0) | **0.5700** | **0.8304** | 0.7233 | 0.4660 | **0.8003** | **0.7443** | 0.215 \| **0.6770** | **1.6574** | **1.2593** |
| `vx0.6.4.2` (none, γ=1) | 0.5367 | 0.8089 | 0.7104 | 0.4620 | 0.7976 | 0.7427 | 0.152 \| 0.6384 | 1.7052 | 1.2537 |
| `vx0.6.4.7` (none, γ=0.05) | 0.5546 | 0.8224 | **0.7256** | **0.4760** | 0.7971 | 0.7372 | **0.259** \| 0.6315 | 1.6601 | 1.1922 |

Reading `vx0.6.4.7` against `vx0.6.4.2` (same no-residual, init only): the small
init recovers most of what the `gamma = 1` start lost — Wiki byte-ppl
`1.7052 -> 1.6601` (back to `v0.6.4`), and MC rises broadly. So the large-norm
`gamma = 1` init, not removing the residual, drove `vx0.6.4.2`'s degradation.

Seed replication (seed 43): the seed-42 GSM8K gap does **not** hold up. Same-seed
against `v0.6.4-seed43`:

| seed 43 | ARC-c | ARC-e | HellaSwag | OBQA | PIQA | WinoGrande | GSM8K strict \| flexible | Wiki byte_ppl↓ | gen_compression_ratio |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| `v0.6.4` | 0.5597 | 0.8274 | 0.7250 | 0.4740 | 0.8069 | 0.7482 | 0.256 \| 0.6603 | 1.6578 | 1.2496 |
| `vx0.6.4.7` | 0.5708 | 0.8274 | 0.7238 | 0.4740 | 0.8009 | 0.7601 | 0.193 \| 0.6664 | 1.6602 | 1.1973 |

Revised takeaway: across the two seeds, `vx0.6.4.7` matches `v0.6.4` on MC, Wiki
byte-ppl, **and GSM8K** — GSM8K flexible is `0.6315/0.6664` (seed 42/43) vs
`v0.6.4`'s `0.6770/0.6603`, i.e. within seed noise, so the seed-42 -4.6pt drop
was mostly a seed artifact, not a robust residual effect. The one difference that
**does** replicate is `gen_compression_ratio`: `vx0.6.4.7` is consistently lower
(`1.19/1.20` vs `v0.6.4`'s `1.26/1.25`). So a small no-residual init matches the
standard recipe on quality; its only robust cost is a bit less generation-time
compression.

Why compression is lower (mechanism): generation is a per-step argmax between
each hyper-token and the base vocabulary, and `logit = hidden . E_out`, so a
hyper-token competes well only if `||E_out||` is on the base-token scale
(`||W_out|| ~ 2.63`). With the residual, `E_out(H) = W_out[t1] + correction`
structurally pins the norm at base scale (`v0.6.4`: `||E_out|| ~ 2.95`) and ties
the hyper-token's logit to its first base token, so it stays a fair competitor.
The no-residual model has no such anchor: it **starts** at base scale
(`||E_out|| ~ 2.65` at `gamma = 0.05`) but training drifts the norm **down** to
`~2.01` (below base) — teacher-forced CE rewards only relative ranking, not norm,
so nothing holds it up. Under greedy argmax a ~24%-shorter output vector loses
the hyper-vs-base competition more often, so fewer hyper-tokens are emitted. This
is invisible to teacher-forced metrics (loss, `hyper_token_acc`, MC, ppl — all
matched, since those never need to win the argmax) and is why the effect is
robust across seeds while GSM8K accuracy is not.

Ruler probe (from the trained checkpoints, method as in the embedding report):
the shared ruler is **gone**. Output-encoder cos-with-mean drops from
`vx0.6.4.2`'s `0.9984` (γ=1) to `0.59`, at/below the no-residual matched-init
baseline (~0.64), i.e. not training-induced; input is `0.54`. Hyper-token norm
returns to ordinary-token scale (`||E_out|| ~ 2.0`, `||E_in|| ~ 3.5`, vs
`vx0.6.4.2`'s `48.5`). This confirms the shared ruler was driven by the `gamma = 1`
large-norm init, not by removing the residual.

Follow-up lever: since the trained norm drifts below base (`2.65 -> 2.01`), a
larger output init (or a light output-norm regularizer) that lands trained
`||E_out||` back at `~2.63` should recover `gen_compression_ratio` toward
`v0.6.4`'s level — directly testable with `encoder_output_init_scale`.
