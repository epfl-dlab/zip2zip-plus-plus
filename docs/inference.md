# Inference

## Published Zip2Zip++ models

Install the user-facing runtime:

```bash
pip install "zip2zip>=0.2.0"
```

Published repositories keep their inference artifacts on the `hf` revision:

```python
from zip2zip import Zip2ZipModel, Zip2ZipTokenizer

repo_id = "epfl-dlab/zip2zip-pp-Llama-3.2-1B-Instruct"
tokenizer = Zip2ZipTokenizer.from_pretrained(repo_id, revision="hf")
model = Zip2ZipModel.from_pretrained(
    repo_id, revision="hf", device_map="auto", dtype="auto"
)

inputs = tokenizer("The capital of France is", return_tensors="pt").to(
    model.device
)
output = model.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

The `hf` revision is self-contained: it includes the base decoder, Zip2Zip++
encoders, tokenizer, and configuration. See [Export](export.md) for the exact
repository layout.

## Native checkpoint inference

For development, a raw `main`-revision checkpoint can be run through the
training implementation:

```bash
python scripts/inference.py \
    --prompt "The capital of France is" \
    --max-new-tokens 50 \
    --ckpt-dir /path/to/step_8000
```

This path loads `model.pt` and `meta.pt` directly and is useful for checking a
checkpoint before export. It is not the recommended API for downstream users.
