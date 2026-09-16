# Train, evaluate, and publish Zip2Zip++

## 1. Prepare data

The project continues to use the Llaza preprocessing pipeline:

```bash
python scripts/pretokenize.py \
    --dataset epfl-dlab/llaza-20B \
    --output_dir /path/to/tokens
```

See [Data Pipeline](data.md) for sharding, tokenizer, and compression options.

## 2. Train

```bash
uv run torchrun --nproc_per_node=4 -m zip2zip_core.train \
    --data_dir /path/to/tokens \
    --output_dir /path/to/checkpoints \
    --model_config 1B \
    --init_from_hf meta-llama/Llama-3.2-1B-Instruct \
    --max_subtokens 4 \
    --wandb --wandb_name zip2zippp-1b
```

Training still supports local checkpoint save/resume and optional background
uploads of step revisions. The production release command does not replace or
modify those functions.

## 3. Evaluate

```bash
python scripts/eval_harness.py \
    --ckpt_dir /path/to/checkpoints/step_8000 \
    --resume_wandb_id none
```

## 4. Build and inspect the release

```bash
python scripts/push_checkpoint.py \
    --ckpt-dir /path/to/checkpoints/step_8000 \
    --repo-id epfl-dlab/<model-repo> \
    --output-dir /path/to/export
```

The command fails before upload unless the checkpoint matches one of the four
supported Zip2Zip++ recipes and the export is complete.

## 5. Publish

```bash
python scripts/push_checkpoint.py \
    --ckpt-dir /path/to/checkpoints/step_8000 \
    --repo-id epfl-dlab/<model-repo> \
    --upload
```

The resulting revisions are:

```text
main  -> original training checkpoint
hf    -> validated, self-contained inference model
```

## 6. Verify the public revision

```python
from zip2zip import Zip2ZipModel, Zip2ZipTokenizer

repo_id = "epfl-dlab/<model-repo>"
tokenizer = Zip2ZipTokenizer.from_pretrained(repo_id, revision="hf")
model = Zip2ZipModel.from_pretrained(repo_id, revision="hf")
```
