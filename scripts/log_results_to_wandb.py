"""Backfill-log an eval results JSON (from eval_harness.py / eval_hf_model.py)
to W&B, for runs that were executed with WANDB=0.

Creates one W&B run per invocation with:
  - all numeric metrics from the results block, flattened as <task>/<metric>
  - the eval args as the run config
  - compression ratios (eval/input_compression_ratio, eval/gen_compression_ratio)
  - free-text notes (use it to point at the RCP log/JSON paths)
  - optionally, the printed per-task sample blocks parsed out of the matching
    .log file and uploaded as a wandb.Table ("samples")

Usage (cluster, where WANDB_API_KEY is set):
    python scripts/log_results_to_wandb.py \
        --json  /dlabscratch1/gentilin/logs/eval/results_<...>.json \
        --name  eval-MC-andrea-z2z-phi35-4B-repro-1BData-v0.1-Zip2zipCore-fixed \
        --notes "RCP log: /dlabscratch1/gentilin/logs/eval/eval_<...>.log" \
        --log   /dlabscratch1/gentilin/logs/eval/eval_<...>.log
"""

from __future__ import annotations

import argparse
import json
import re


def flatten_results(results: dict) -> dict:
    flat = {}
    for task, metrics in (results or {}).items():
        if not isinstance(metrics, dict):
            continue
        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                flat[f"{task}/{k.replace(',none', '').replace(',', '_')}"] = v
    return flat


def parse_samples(log_path: str):
    """Parse the '--- <task> sample N ---' blocks print_samples writes to the log."""
    text = open(log_path, errors="replace").read()
    rows = []
    for m in re.finditer(
        r"--- (\S+) sample (\d+) ---\nPROMPT:\n(.*?)\nTARGET \(gold answer\):\n(.*?)"
        r"\nMODEL OUTPUT \(raw generation\):\n(.*?)\nMODEL OUTPUT \(filtered/extracted answer\):\n(.*?)\nSCORE:\n(\{.*?\})\n",
        text,
        re.S,
    ):
        task, idx, prompt, target, raw, filtered, score = m.groups()
        rows.append([task, int(idx), prompt, target, raw, filtered, score])
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--json", required=True, help="results_*.json from an eval run")
    p.add_argument("--name", required=True, help="W&B run name")
    p.add_argument("--notes", default="", help="Run notes (e.g. RCP log/JSON paths)")
    p.add_argument("--log", default=None,
                   help="Matching eval_*.log; its printed sample blocks are "
                        "uploaded as a wandb.Table")
    p.add_argument("--entity", default="epfl-dlab")
    p.add_argument("--project", default="zip2zip-core")
    p.add_argument("--tags", default="eval,backfill", help="Comma-separated run tags")
    args = p.parse_args()

    data = json.load(open(args.json))
    flat = flatten_results(data.get("results"))
    if not flat:
        raise SystemExit(f"No numeric metrics found in {args.json}")
    compression = data.get("compression") or {}
    for k, v in compression.items():
        if k.endswith("_ratio"):
            flat[f"eval/{k}"] = v

    notes = args.notes or ""
    src = f"results JSON: {args.json}"
    notes = f"{notes}\n{src}" if notes else src

    import wandb

    run = wandb.init(
        entity=args.entity,
        project=args.project,
        name=args.name,
        job_type="eval",
        tags=[t.strip() for t in args.tags.split(",") if t.strip()],
        notes=notes,
        config={
            "eval_args": data.get("args"),
            "ckpt_dir": data.get("ckpt_dir") or data.get("model"),
            "tasks": data.get("tasks"),
            "compression": compression,
        },
    )
    wandb.log(flat)

    if args.log:
        rows = parse_samples(args.log)
        if rows:
            table = wandb.Table(
                columns=["task", "sample", "prompt", "target",
                         "raw_generation", "filtered_answer", "score"],
                data=rows,
            )
            wandb.log({"samples": table})
            print(f"logged {len(rows)} sample rows from {args.log}")
        else:
            print(f"warning: no sample blocks found in {args.log}")

    print(f"logged {len(flat)} metrics to W&B run: {run.url if hasattr(run, 'url') else run.id}")
    run.finish()


if __name__ == "__main__":
    main()
