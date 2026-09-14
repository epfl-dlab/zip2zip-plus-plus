# Long-context evaluation: RULER

The protocol is inference-only: it restores the official
`microsoft/Phi-3.5-mini-instruct` LongRoPE factors from the
`meta.pt:init_from_hf` model and does not retrain or rewrite the checkpoint.

The full RULER curve uses 4K, 8K, 16K, 32K, 64K, and 128K. Run each task and
context length in a separate process so a late 128K failure loses no earlier
result: lm-eval eagerly materializes 500 long examples per task, so loading the full
group at once has an unnecessarily large memory peak.

```bash
CKPT_DIR=/path/to/step_8000 \
TOKENIZER=microsoft/Phi-3.5-mini-instruct \
PRESET=longcontext_ruler \
TASKS=niah_single_1 \
RULER_LENGTHS=131072 \
bash scripts/eval_ckpt_rcp.sh
```

Repeat with `niah_single_2`, `niah_single_3`, `niah_multikey_1`,
`niah_multikey_2`, `niah_multikey_3`, `niah_multiquery`, `niah_multivalue`,
`ruler_vt`, `ruler_cwe`, `ruler_fwe`, `ruler_qa_squad`, and
`ruler_qa_hotpot`, then repeat the task matrix for each of the six lengths.
Omitting `RULER_LENGTHS` asks one process to run all six lengths; it is valid,
but separate jobs are safer for formal runs. The preset forces greedy decoding,
sets model capacity to 128K, and fails if the adapter would left-truncate a
request.

Run the official Phi-3.5 baseline through the HF entry point. Its Transformers
config already contains the official LongRoPE:

```bash
python scripts/eval_longcontext_hf.py \
  --preset longcontext_ruler \
  --output_path results_phi35_ruler.json
```

Combine the 13 split result files with:

```bash
python scripts/aggregate_longcontext.py \
  --ruler result_niah_single_1.json result_niah_single_2.json \
          result_niah_single_3.json result_niah_multikey_1.json \
          result_niah_multikey_2.json result_niah_multikey_3.json \
          result_niah_multiquery.json result_niah_multivalue.json \
          result_ruler_vt.json result_ruler_cwe.json result_ruler_fwe.json \
          result_ruler_qa_squad.json result_ruler_qa_hotpot.json \
  --output results_ruler_summary.json
```

The aggregate accepts results split by task, by length, or both. Read the six
entries under `ruler.curve` for the context-length curve.

## Generation speed: the KV cache

RULER is a generation benchmark, and until 2026-09 the Zip2Zip adapter decoded
by re-running the model on the whole prefix for every sampled token. At 32K
with 128 generated tokens that is 128 prefills per sample — about 4.2M tokens
pushed through the transformer instead of 32.9K, with each of those forwards
paying quadratic attention on the full prompt. It is the reason the preset felt
unusable rather than merely slow.

Generation now keeps a key/value cache (`zip2zip_core.kv_cache`), so a decode
step costs one token. The prefill also projects only the final position into
the vocabulary, which avoids materializing a 32K x 32K logits tensor that
generation never reads.

Two Zip2Zip details make this more than the textbook cache, and both are
pinned by `tests/test_kv_cache.py`:

- **Positions are base-space.** A hypertoken advances the RoPE position by its
  span, not by one, so the cache carries the running `cumsum(spans)` offset. A
  cache that restarted the cumsum would stack every generated token on top of
  the prompt.
- **Some RoPE decisions are re-made from the whole prefix.** LongRoPE picks
  short or long factors from the maximum position, and two-axis RoPE switches
  on at the first hypertoken. The uncached path re-rotates every key each step,
  so flipping either applies retroactively; cached keys cannot follow. A flip
  raises `RopeRegimeChanged`, and the adapter drops the cache and re-prefills —
  reproducing the uncached result exactly. `kv_cache_reprefills` in the results
  JSON counts how often that happened.

To reproduce a pre-2026-09 number, or to A/B the cache against the reference
implementation, pass `--no_kv_cache` to `scripts/eval_harness.py`. Expect it to
take the better part of a day on the same hardware.
