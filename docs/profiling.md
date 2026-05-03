# Profiling

Profile training to measure time spent in the hyper-encoder vs decoder vs output head.

## Usage

```bash
bash scripts/profile_train.sh /path/to/tokens [model_config] [max_subtokens] [profile_steps]
```

Examples:

```bash
# Quick profile with debugmodel
bash scripts/profile_train.sh /scratch/fineweb-tokens debugmodel 2 5

# Profile 1B model with max_subtokens=4
bash scripts/profile_train.sh /scratch/fineweb-tokens 1B 4 3
```

## How it works

The script runs `zip2zip_core.train` with `--profile` enabled and `--no_compile` (to get readable trace names). After 3 warmup steps, `torch.profiler` records the specified number of active steps.

Key `record_function` regions in the code:

| Region | Where |
|--------|-------|
| `hyper_encoder` | Full hyper-encoder pass (embedding + encoding) |
| `hyper_encoder.core` | Core encoder forward (attention layers + pooling) |
| `he.forward_varlen.*` | Varlen attention path (packing, layers, pooling) |
| `he.proj_in` / `he.proj_out` | Dimension projection layers |
| `Main LM` | Decoder transformer layers |
| `lm_head` | Base vocabulary logits |
| `hyper_lm_head` | Hypertoken logits (bilinear) |
| `logit_cat` | Concatenating base + hyper logits |

## Output

- Console table sorted by `device_time_total` — look for the regions above
- Chrome/TensorBoard traces saved to `$OUTPUT_DIR/profiler_traces/`

View traces with:

```bash
# Chrome
# Open chrome://tracing and load the .json trace file

# TensorBoard
tensorboard --logdir $OUTPUT_DIR/profiler_traces
```

## Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OUTPUT_DIR` | `/tmp/zip2zip-profile` | Output directory for traces |
| `NGPU` | `1` | Number of GPUs |
