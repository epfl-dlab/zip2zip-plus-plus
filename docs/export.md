# Export and interoperability

zip2zip-core training checkpoints and user-facing `zip2zip` models use two
different formats. A production model repository intentionally keeps both:

| Revision | Format | Purpose |
|---|---|---|
| `main` | `model.pt`, `meta.pt`, optional `optimizer.pt` | Resume, inspection, and reproducible export |
| `hf` | Config, tokenizer, encoders, sharded safetensors | Inference with the `zip2zip` package |

Both `model.pt` and its matching `meta.pt` are required for export. Behavioral
settings such as base-token positions and encoder residual mode cannot be
inferred safely from weights alone.

## Releasing the four Zip2Zip++ models

Use the guarded release command for the Llama 3.2 1B/3B and Phi-3 4B/14B
checkpoints. It accepts only the agreed Zip2Zip++ recipe: flat untied encoders,
four subtokens, digit-token protection, and base-token-end positions.

Build and validate locally first:

```bash
python scripts/push_checkpoint.py \
    --ckpt-dir /path/to/step_8000 \
    --repo-id epfl-dlab/zip2zip-pp-Llama-3.2-1B-Instruct \
    --output-dir /path/to/export
```

Publish both revisions:

```bash
python scripts/push_checkpoint.py \
    --ckpt-dir /path/to/step_8000 \
    --repo-id epfl-dlab/zip2zip-pp-Llama-3.2-1B-Instruct \
    --upload
```

No network write occurs without `--upload`. When uploading, the command:

1. validates `model.pt`, `meta.pt`, the base model, and the Zip2Zip++ recipe;
2. exports and validates all decoder and encoder keys locally;
3. uploads the original checkpoint directory to `main`;
4. writes a model card explaining the two revisions; and
5. uploads the self-contained inference export to `hf`, deleting any
   `model.pt`/`optimizer.pt` that revision may still carry, so `hf` holds
   exactly the export.

The `hf` branch is created from the repository's root commit, never from
`main`, so it cannot inherit training files. Re-running the command against an
existing repository is safe: unchanged files produce no new commit.

The upload step is separate from training. It does not alter checkpoint saving,
resuming, or the optional background checkpoint upload in `zip2zip_core.train`.

## Export validation

The release path requires:

- `format_version: 2`;
- `position_mode: base_token_end`;
- a self-contained base model (`base_model_name_or_path: "."`);
- both input and output encoder weights;
- the exact Transformers decoder key set;
- a 128256 base vocabulary for Llama or 32064 for Phi;
- `generation_config.json` copied from the base model (Phi-3 stops on `<|end|>`
  only through it);
- a tokenizer with an explicit `pad_token` equal to the core `pad_token_id`
  (`<|end_of_text|>` for Llama, `<|endoftext|>` for Phi), because the runtime
  masks pad ids and would otherwise fall back to eos; and
- all shards referenced by `model.safetensors.index.json`.

This check happens before either revision is uploaded.

## Manual development export

For local experiments that are not one of the four releases, use the lower-level
exporter directly:

```bash
python scripts/zip2zip_hf/export_to_zip2zip.py \
    --ckpt_dir /path/to/step_8000 \
    --output_dir /path/to/export
```

or the Python API:

```python
from zip2zip_core.export import export

export(
    ckpt_dir="/path/to/step_8000",
    output_dir="/path/to/export",
    max_shard_size="5GB",
)
```

The manual exporter supports the same unified Llama/Phi conversion, but it does
not enforce the four-model release contract and does not upload anything.

## Exported files

```text
<export>/
├── README.md
├── config.json
├── generation_config.json
├── zip2zip_config.json
├── zip2zip_encoders.safetensors
├── model-00001-of-0000N.safetensors
├── model.safetensors.index.json
└── tokenizer files
```

Small models may use a single `model.safetensors` instead of shards.

`model.pt` uses TorchTitan key names. During export, decoder tensors are mapped
to the Hugging Face Llama or Phi layout, including Phi's fused QKV and gate/up
projections. `hyper_encoder.*` becomes `input_encoder.*`, while the untied
`hyper_output.*` weights become `output_encoder.*`.

## Loading the published model

The inference files live on `hf`, not the default `main` revision:

```python
from zip2zip import Zip2ZipModel, Zip2ZipTokenizer

repo_id = "epfl-dlab/zip2zip-pp-Llama-3.2-1B-Instruct"
tokenizer = Zip2ZipTokenizer.from_pretrained(repo_id, revision="hf")
model = Zip2ZipModel.from_pretrained(
    repo_id, revision="hf", device_map="auto", dtype="auto"
)
```

Local exports do not need a revision:

```python
model = Zip2ZipModel.from_pretrained("/path/to/export")
tokenizer = Zip2ZipTokenizer.from_pretrained("/path/to/export")
```

Format version 2 requires `zip2zip>=0.2.0`. Beam search is deliberately rejected
until adaptive-codebook beam reordering is implemented; greedy and sampling
generation are supported.
