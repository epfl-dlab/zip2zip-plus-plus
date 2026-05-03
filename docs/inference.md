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

## 2. Torchtitan-based inference (in development)

For running inference directly from training checkpoints (torchtitan format) without exporting. Useful for quick sanity checks during development.

```bash
python scripts/inference.py \
    --prompt "The capital of France is" \
    --max-new-tokens 50 \
    --ckpt-dir /path/to/step_6000
```

This script loads the `Zip2ZipLlama3Model` directly from `model.pt` and uses `CodebookManager` (Rust LZW state machine) to drive autoregressive generation.

There is also `scripts/generate.py` with a similar interface using `Zip2ZipTokenizer` for encoding/decoding.

> **Note**: The torchtitan inference scripts are development tools, not production-ready. For benchmarking and deployment, use the HF-based path above.
