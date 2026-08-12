"""Copy a pretraining run's scalar history into the main pipeline W&B run.

Why this exists: train.py logs with an explicit ``step=`` on the default
``_step`` axis, so two trainings cannot share one run id directly — the second
one's steps restart at 1, W&B sees a non-monotonic step and silently drops
every point. The pipeline therefore trains stage 1 under its own run id and,
once every explicit-step write is done (train + smoke + final evals), copies
the stage-1 scalars into the main run under ``pt/*`` with a dedicated
``pt/step`` axis — the same define_metric pattern log_results_to_wandb.py uses
for smoke/final. Nothing is dropped because these writes never pass ``step=``.

Rerunning duplicates the pt/ rows (cosmetic overlap in panels, no data loss),
which is why the pipeline calls this non-fatally and it can be rerun by hand:

    python scripts/wandb_merge_pretrain.py \
        --entity epfl-dlab --project zip2zip-core \
        --src <pretrain_run_id> --dst <main_run_id>
"""

import argparse
import numbers


def merge(entity, project, src_id, dst_id, prefix):
    import wandb

    api = wandb.Api()
    src = api.run(f"{entity}/{project}/{src_id}")
    # scan_history: full-fidelity rows, not the sampled history() view.
    rows = list(src.scan_history())
    print(f"[merge] {len(rows)} history rows in {src_id} ({src.name})")

    run = wandb.init(entity=entity, project=project, id=dst_id, resume="allow")
    wandb.define_metric(f"{prefix}/step")
    wandb.define_metric(f"{prefix}/*", step_metric=f"{prefix}/step")

    copied = 0
    for row in rows:
        step = row.get("_step")
        if step is None:
            continue
        payload = {
            f"{prefix}/{k}": v
            for k, v in row.items()
            if not k.startswith("_")
            and isinstance(v, numbers.Number)
            and not isinstance(v, bool)
        }
        if not payload:
            continue
        payload[f"{prefix}/step"] = step
        wandb.log(payload)
        copied += 1
    wandb.finish()
    print(f"[merge] copied {copied} rows into {dst_id} under {prefix}/")
    return copied


def selftest(entity, project):
    """Prove on two throwaway runs that the merge drops nothing.

    Simulates the exact failure this script works around: the destination run
    has already logged with explicit step= (like train.py), then the source
    curves arrive afterwards. Cleans up both runs on success.
    """
    import secrets
    import string

    import wandb

    tag = "".join(secrets.choice(string.ascii_lowercase + string.digits) for _ in range(8))
    src_id, dst_id = f"mgsrc{tag[:3]}", f"mgdst{tag[:3]}"

    run = wandb.init(entity=entity, project=project, id=src_id,
                     name=f"wandb-merge-selftest-src-{tag}")
    for s in range(1, 31):
        wandb.log({"train/loss": 1.0 / s}, step=s)
    wandb.finish()

    run = wandb.init(entity=entity, project=project, id=dst_id,
                     name=f"wandb-merge-selftest-dst-{tag}")
    for s in range(1, 6):
        wandb.log({"train/loss": 2.0 / s}, step=s)
    wandb.finish()

    merge(entity, project, src_id, dst_id, prefix="pt")

    api = wandb.Api()
    dst = api.run(f"{entity}/{project}/{dst_id}")
    pt_rows = [r for r in dst.scan_history() if any(k.startswith("pt/") for k in r)]
    own_rows = [r for r in dst.scan_history() if r.get("train/loss") is not None]
    assert len(pt_rows) == 30, f"expected 30 pt/ rows, found {len(pt_rows)}"
    assert len(own_rows) == 5, f"destination's own 5 rows must survive, found {len(own_rows)}"
    steps = [r["pt/step"] for r in pt_rows]
    assert steps == sorted(steps) and steps[0] == 1 and steps[-1] == 30, f"pt/step axis wrong: {steps[:5]}..."

    api.run(f"{entity}/{project}/{src_id}").delete()
    api.run(f"{entity}/{project}/{dst_id}").delete()
    print("[selftest] PASS: 30/30 rows merged, destination history intact, throwaway runs deleted")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--entity", required=True)
    ap.add_argument("--project", required=True)
    ap.add_argument("--src", help="pretraining run id to copy from")
    ap.add_argument("--dst", help="main run id to copy into")
    ap.add_argument("--prefix", default="pt")
    ap.add_argument("--selftest", action="store_true",
                    help="run an end-to-end proof on two throwaway runs, then delete them")
    args = ap.parse_args()

    if args.selftest:
        selftest(args.entity, args.project)
        return
    if not args.src or not args.dst:
        raise SystemExit("--src and --dst are required (or use --selftest)")
    merge(args.entity, args.project, args.src, args.dst, args.prefix)


if __name__ == "__main__":
    main()
