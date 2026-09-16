# Project structure

```text
zip2zip-core/
├── src/zip2zip_core/
│   ├── model.py          # Training model and hyper-encoders
│   ├── configs.py        # Llama and Phi model configurations
│   ├── data.py           # Llaza/token-shard data pipeline
│   ├── train.py          # Distributed training, save, and resume
│   ├── export.py         # Llama/Phi conversion to zip2zip format
│   ├── release.py        # Four-model validation and main/hf publishing
│   ├── hub.py            # Shared Hugging Face upload helpers
│   └── lm_eval_adapter.py
├── scripts/
│   ├── pretokenize.py
│   ├── push_checkpoint.py # Guarded Zip2Zip++ release CLI
│   ├── inference.py
│   ├── eval_harness.py
│   └── zip2zip_hf/
│       └── export_to_zip2zip.py # Lower-level development exporter
├── tests/
├── docs/
└── ext/
    ├── torchtitan/
    ├── zip2zip/
    ├── zip2zip-compression/
    └── zip2zip-hyperenc_probe/
```

## Responsibility split

- `zip2zip-core` owns data preparation, training, raw checkpoints, evaluation,
  conversion, and release validation.
- `zip2zip` owns the lightweight user-facing tokenizer, model loader, and
  generation runtime.
- `zip2zip-compression` owns the LZW state machine shared by both paths.
- `zip2zip-hyperenc_probe` owns the paper's hyper-encoder embedding probes
  (substitution and sequence probes) and the reproduction of their figures;
  see [ext/zip2zip-hyperenc_probe/README.md](../ext/zip2zip-hyperenc_probe/README.md).

Training checkpoints remain independent from the release layer. The trainer can
save and resume locally or upload step revisions through `hub.py`; the guarded
release CLI reads a completed checkpoint without modifying it.

## Hugging Face layout

Production Zip2Zip++ repositories use `main` for the original training
checkpoint and `hf` for the self-contained inference export. This distinction
is enforced in `release.py` and covered by tests.
