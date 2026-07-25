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
    p.add_argument("--json", default=None, help="results_*.json from an eval run")
    p.add_argument("--name", default=None,
                   help="W&B run name (creates a NEW run; mutually exclusive with --resume_id)")
    p.add_argument("--resume_id", default=None,
                   help="Existing W&B run id: log the metrics INTO that run instead "
                        "of creating one (used by the finetune->eval pipeline)")
    p.add_argument("--step", type=int, default=None,
                   help="With --resume_id: checkpoint step for these metrics. Logged "
                        "on a dedicated '<prefix>/step' x-axis (via define_metric) so "
                        "it never conflicts with the training run's own step counter.")
    p.add_argument("--prefix", default=None,
                   help="Namespace metrics as <prefix>/<task>/<metric> (e.g. 'smoke', 'final')")
    p.add_argument("--append_notes", default=None,
                   help="With --resume_id: text appended to the run's notes")
    p.add_argument("--resolve_pending", action="store_true",
                   help="With --resume_id: remove the '[pending] ... [/pending]' block "
                        "from the run's notes (the pipeline's ready-made follow-up "
                        "command) — pass it when that follow-up job has now run.")
    p.add_argument("--notes", default="", help="Run notes (e.g. RCP log/JSON paths)")
    p.add_argument("--log", default=None,
                   help="Matching eval_*.log; its printed sample blocks are "
                        "uploaded as a wandb.Table")
    p.add_argument("--entity", default="epfl-dlab")
    p.add_argument("--project", default="zip2zip-core")
    p.add_argument("--tags", default="eval,backfill", help="Comma-separated run tags")
    args = p.parse_args()

    if bool(args.name) == bool(args.resume_id):
        raise SystemExit("Pass exactly one of --name (new run) or --resume_id (existing run).")
    if args.step is not None and not args.prefix:
        raise SystemExit("--step requires --prefix (metrics need their own x-axis namespace).")
    if args.resolve_pending and not args.resume_id:
        raise SystemExit("--resolve_pending requires --resume_id.")

    flat, compression, data = {}, {}, {}
    if args.json:
        data = json.load(open(args.json))
        flat = flatten_results(data.get("results"))
        if not flat:
            raise SystemExit(f"No numeric metrics found in {args.json}")
        compression = data.get("compression") or {}
        for k, v in compression.items():
            if (
                k.endswith("_ratio")
                or k == "online_skipped_targets"
                or k.startswith("online_replay_")
            ):
                flat[f"eval/{k}"] = v
        if isinstance(data.get("eval_wall_seconds"), (int, float)):
            flat["eval/wall_seconds"] = data["eval_wall_seconds"]
    elif not args.append_notes and not args.resolve_pending:
        raise SystemExit("Nothing to do: pass --json, --append_notes and/or --resolve_pending.")

    if args.prefix:
        flat = {f"{args.prefix}/{k}": v for k, v in flat.items()}

    import wandb

    run = None
    if args.resume_id:
        if flat or args.log:
            run = wandb.init(entity=args.entity, project=args.project,
                             id=args.resume_id, resume="must")
    else:
        notes = args.notes or ""
        if args.json:
            src = f"results JSON: {args.json}"
            notes = f"{notes}\n{src}" if notes else src
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

    if flat:
        if args.step is not None:
            # Own x-axis: a resumed training run's global step is already past
            # the checkpoint steps, and wandb drops non-monotonic step= values.
            wandb.define_metric(f"{args.prefix}/step")
            wandb.define_metric(f"{args.prefix}/*", step_metric=f"{args.prefix}/step")
            wandb.log({**flat, f"{args.prefix}/step": args.step})
        else:
            wandb.log(flat)

    if args.log:
        rows = parse_samples(args.log)
        if rows:
            table = wandb.Table(
                columns=["task", "sample", "prompt", "target",
                         "raw_generation", "filtered_answer", "score"],
                data=rows,
            )
            key = f"{args.prefix}/samples" if args.prefix else "samples"
            wandb.log({key: table})
            print(f"logged {len(rows)} sample rows from {args.log}")
        else:
            print(f"warning: no sample blocks found in {args.log}")

    if run is not None:
        print(f"logged {len(flat)} metrics to W&B run: {getattr(run, 'url', None) or run.id}")
        run.finish()

    if args.resume_id and (args.append_notes or args.resolve_pending):
        # Notes via the public API, after finish(): Api.run.notes is the
        # authoritative server-side value, unlike the resumed run object's.
        apirun = wandb.Api().run(f"{args.entity}/{args.project}/{args.resume_id}")
        notes = apirun.notes or ""
        if args.resolve_pending:
            kept, skipping = [], False
            for line in notes.splitlines():
                if not skipping and "[pending]" in line:
                    skipping = True
                elif skipping and "[/pending]" in line:
                    skipping = False
                elif not skipping:
                    kept.append(line)
            notes = "\n".join(kept)
        if args.append_notes:
            notes = (notes.rstrip() + "\n" + args.append_notes).strip()
        apirun.notes = notes
        apirun.update()
        print(f"notes updated on run {args.resume_id}")


if __name__ == "__main__":
    main()
