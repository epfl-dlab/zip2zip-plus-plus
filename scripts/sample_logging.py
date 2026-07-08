"""Console logging of per-sample eval results (prompt, target, output, score).

Shared by eval_harness.py and eval_hf_model.py so the checkpoint and HF-model
eval paths print the same per-sample view. Console-only by design: samples are
never written to the results JSON.
"""

import json

_INTERNAL_KEYS = {"doc", "doc_id", "arguments", "resps", "filtered_resps", "target"}


def print_samples(results, max_per_task=10):
    """Print up to max_per_task samples per task from lm-eval results.

    Requires simple_evaluate(..., log_samples=True); no-op otherwise.
    """
    samples = results.get("samples")
    if not samples:
        return
    print("Sample generations (prompt + gold target + model output + score):")
    for task_name, task_samples in samples.items():
        for i, sample in enumerate(task_samples[:max_per_task]):
            print(f"\n--- {task_name} sample {i} ---")
            arguments = sample.get("arguments") or []
            prompt = arguments[0][0] if arguments else sample.get("doc")
            print("PROMPT:")
            print(prompt)
            print("TARGET (gold answer):")
            print(sample.get("target"))
            print("MODEL OUTPUT (raw generation):")
            print(sample.get("resps"))
            print("MODEL OUTPUT (filtered/extracted answer):")
            print(sample.get("filtered_resps"))
            metrics = {k: v for k, v in sample.items() if k not in _INTERNAL_KEYS}
            print("SCORE:")
            print(json.dumps(metrics, default=str))
        if len(task_samples) > max_per_task:
            print(f"\n... {len(task_samples) - max_per_task} more {task_name} samples omitted")
    print("=" * 72)
