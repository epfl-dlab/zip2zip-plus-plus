# Inference

There are two ways to run inference with Llaza models.

## 1. HuggingFace-based inference (via `zip2zip` library)

The recommended way for exported models. Uses the [`zip2zip`](https://github.com/epfl-dlab/zip2zip) inference library with a HuggingFace-compatible API.

```bash
pip install zip2zip
```

```python
from zip2zip import Zip2ZipModel, Zip2ZipTokenizer

model = Zip2ZipModel.from_pretrained("epfl-dlab/Llaza-3.2-1B-v0.1", revision="hf")
tokenizer = Zip2ZipTokenizer.from_pretrained("epfl-dlab/Llaza-3.2-1B-v0.1", revision="hf")

output = model.generate(tokenizer.encode("The capital of France is"), max_new_tokens=50)
print(tokenizer.decode(output))
```

This requires models exported to HF format (see [Export & Interop](export.md)).
It does **not** apply to v0.7.1: the external HF runtime cannot represent its
per-layer gated RoPE geometry, so both export paths fail explicitly.

## 2. Torchtitan-based inference (in development)

For running inference directly from training checkpoints (torchtitan format) without exporting. Useful for quick sanity checks during development.

```bash
python scripts/inference.py \
    --prompt "The capital of France is" \
    --max-new-tokens 50 \
    --ckpt-dir /path/to/step_6000
```

This script loads the `Zip2ZipLlama3Model` directly from `model.pt`, folds
training-time LoRA adapters into the decoder, and uses `CodebookManager` (the
Rust LZW state machine) to drive autoregressive generation. Hyper-token outputs
are expanded back to base tokens before updating the dictionary and decoding.
Behavior-only geometry is recovered from `meta.pt`, including base positions,
legacy v0.7 two-axis RoPE, and v0.7.1 gated compressed-position RoPE.

There is also `scripts/generate.py` with a similar interface using `Zip2ZipTokenizer` for encoding/decoding.

> **Note**: The torchtitan inference scripts are development tools, not
> production-ready. Use the HF-based path for export-compatible checkpoints.
> v0.7.1 must instead use this native path for both evaluation and inference
> until the external runtime implements the gated geometry.
