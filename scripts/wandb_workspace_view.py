"""Build the W&B workspace view for this project: same metrics, organised sections.

This touches the VIEW only. It never changes what train.py / eval_harness.py log,
so every legacy-comparable key (`loss`, `ppl`, `compression`, ...) keeps its name
and every existing run stays comparable.

Sections and panel order are a property of the PROJECT workspace, not of a run,
which is why they live here and not in the logging path: one saved view serves
every run, including the ones already finished. The other half of the cleanup —
suppressing duplicate panels — IS per-run and is done at log time, by
`define_metric(..., hidden=True)` in train.py and eval_harness.py. Pipeline
phase 4 calls this script with --apply --view-url so the layout is refreshed
automatically; the URL is required because saving a fresh Workspace() always
creates a new view rather than updating one.

What it does:
  - groups all ~150 logged keys into named, ordered sections ("windows")
  - keeps legacy keys and corrected `compressed/` keys in SEPARATE sections, so
    the workspace never invites the legacy-vs-corrected comparison that
    CLAUDE.md forbids
  - drops the one always-identical duplicate panel (see DUPLICATES)
  - collapses the sections that are duplicates-by-construction or rarely read
    instead of deleting them, so nothing is lost

Usage:
    # print the layout + the dedup report, no network, no extra deps
    python scripts/wandb_workspace_view.py

    # check the layout against a real run: every key it logs must be covered
    python scripts/wandb_workspace_view.py --verify epfl-dlab/zip2zip-core/<run_id>

    # create the saved view once (needs `pip install wandb-workspaces`)
    python scripts/wandb_workspace_view.py --apply --view-name "z2z clean"

    # update that same view afterwards (this is what the pipeline runs)
    python scripts/wandb_workspace_view.py --apply --view-url "<printed URL>"
"""

from __future__ import annotations

import argparse
from urllib.parse import urlparse

ENTITY = "epfl-dlab"
PROJECT = "zip2zip-core"

# The project's shared saved view, created once on 2026-07-31. Every pipeline run
# refreshes THIS view, for whoever launches it, so the team converges on one
# layout instead of each person needing their own env var. The content is fully
# determined by this file, so concurrent refreshes are harmless.
# Consequence to know: manual panel edits made in the UI on this view are
# reverted by the next pipeline run. Use the UI's "Save as new view" for a
# personal arrangement, or point --view-url at your own view.
VIEW_URL = "https://wandb.ai/epfl-dlab/zip2zip-core?nw=ajtwww7qjwp"

# x-axis of each family. The eval families are logged on their own step metric
# (log_results_to_wandb.py --step) because a resumed training run's global step
# is already past the checkpoint steps.
X_TRAIN = "Step"
X_FINAL = "final/step"
X_SMOKE = "smoke/step"

# Panels removed as exact duplicates: same value, logged twice.
#   train.py:2247 bare `backward_loss`      = compressed_metrics["compat_backward_loss"]
#   train.py:2291 `compressed/backward_loss`= compressed_metrics["compat_backward_loss"]
# The bare one is kept (it is the legacy-comparable key) plus
# `objective/backward_loss`, which is the only one that differs once
# base-view replay is on. train.py also hides it via define_metric, so new runs
# do not auto-create the panel in the first place; this list keeps the two in
# sync (tests/test_wandb_view.py asserts it).
DUPLICATES = ["compressed/backward_loss"]

# Duplicates by construction, kept in a collapsed section rather than dropped:
# pipeline phase 3 logs the final numbers twice (lm-eval's own WandbLogger,
# unprefixed, and log_results_to_wandb.py under final/). The unprefixed keys are
# the ONLY ones a standalone eval run (eval_ckpt_rcp.sh) writes, so deleting
# them would blank the workspace for those runs.
MC_TASKS_ACC_NORM = ["arc_challenge", "arc_easy", "hellaswag", "openbookqa", "piqa"]
SMOKE_TASKS_ACC_NORM = ["arc_easy", "hellaswag"]

# lm-eval logs its sample tables as wandb.Table objects, not metrics, so they
# need a table panel rather than a line plot. Names verified in lm-eval's
# loggers/wandb_logger.py for both 0.4.9 (the eval venv pin) and 0.4.12
# (uv.lock): "evaluation/eval_results" and f"{task_name}_eval_results".
TABLE_PANELS = [
    "evaluation/eval_results",
    "evaluation/group_eval_results",
] + [
    f"{task}_eval_results"
    for task in (
        "arc_challenge", "arc_easy", "hellaswag", "openbookqa", "piqa",
        "winogrande", "gsm8k", "gsm8k_boxed", "triviaqa", "wikitext",
        "zip2zip_pile", "zip2zip_mc4", "zip2zip_dc4",
    )
]

# v0.7.1 logs one gate per (layer, rope pair). Ranges are recipe-dependent;
# override on the command line if a recipe uses a different slice.
GATE_LAYERS = range(0, 32)
GATE_PAIRS = range(32, 48)


def line(title: str, keys: list[str], x: str = X_TRAIN, log_y: bool = False) -> dict:
    return {"title": title, "keys": keys, "x": x, "log_y": log_y}


def build_layout() -> list[dict]:
    """Ordered sections. Each is one 'window' in the W&B workspace."""

    verdict = [
        line("GSM8K strict-match", ["final/gsm8k/exact_match_strict-match"], X_FINAL),
        line("GSM8K flexible-extract", ["final/gsm8k/exact_match_flexible-extract"], X_FINAL),
        line("TriviaQA exact_match", ["final/triviaqa/exact_match_remove_whitespace"], X_FINAL),
    ] + [
        line(f"{task} acc_norm", [f"final/{task}/acc_norm"], X_FINAL)
        for task in MC_TASKS_ACC_NORM
    ] + [
        line("winogrande acc", ["final/winogrande/acc"], X_FINAL),
        line("wikitext word_perplexity", ["final/wikitext/word_perplexity"], X_FINAL),
        line("wikitext byte_perplexity", ["final/wikitext/byte_perplexity"], X_FINAL),
        line("wikitext bits_per_byte", ["final/wikitext/bits_per_byte"], X_FINAL),
    ]

    smoke = [
        line(
            "gsm8k_boxed (3 filters)",
            [
                "smoke/gsm8k_boxed/exact_match_boxed-first",
                "smoke/gsm8k_boxed/exact_match_strict-match",
                "smoke/gsm8k_boxed/exact_match_flexible-extract",
            ],
            X_SMOKE,
        ),
    ] + [
        line(f"{task} acc_norm", [f"smoke/{task}/acc_norm"], X_SMOKE)
        for task in SMOKE_TASKS_ACC_NORM
    ] + [
        line("winogrande acc", ["smoke/winogrande/acc"], X_SMOKE),
        line(
            "compression ratios",
            ["smoke/eval/input_compression_ratio", "smoke/eval/gen_compression_ratio"],
            X_SMOKE,
        ),
        line("eval wall seconds", ["smoke/eval/wall_seconds"], X_SMOKE),
    ]

    # Legacy vs corrected stay in different sections on purpose: the two use
    # different denominators and must not be read off one shared axis.
    loss_legacy = [
        line("loss (legacy base-token denom)", ["loss"]),
        line("ppl (legacy)", ["ppl"]),
        line("loss_target_base_token", ["loss_target_base_token"]),
        line("ppl_target_base_token", ["ppl_target_base_token"]),
        line("backward_loss (compressed view)", ["backward_loss"]),
        line("objective/backward_loss (mixed, what is optimised)", ["objective/backward_loss"]),
    ]

    loss_corrected = [
        line("loss_legacy_base_token", ["compressed/loss_legacy_base_token"]),
        line("ppl_legacy_base_token", ["compressed/ppl_legacy_base_token"]),
        line("loss_target_base_token", ["compressed/loss_target_base_token"]),
        line("ppl_target_base_token", ["compressed/ppl_target_base_token"]),
        line("loss_per_valid_token", ["compressed/loss_per_valid_token"]),
    ]

    acc_legacy = [
        line("acc", ["acc"]),
        line("base_token_acc", ["base_token_acc"]),
        line("hyper_token_acc", ["hyper_token_acc"]),
        line("relaxed_acc", ["relaxed_acc"]),
    ]

    acc_corrected = [
        line("acc", ["compressed/acc"]),
        line("base_token_acc", ["compressed/base_token_acc"]),
        line("hyper_token_acc", ["compressed/hyper_token_acc"]),
        line("relaxed_hyper_acc", ["compressed/relaxed_hyper_acc"]),
        line("relaxed_acc", ["compressed/relaxed_acc"]),
    ]

    compression = [
        line("compression (legacy)", ["compression"]),
        line("compression_target (legacy denom)", ["compression_target"]),
        line("compression_legacy (raw counts)", ["compressed/compression_legacy"]),
        line("compression_target (raw counts)", ["compressed/compression_target"]),
    ]

    type_head = [
        line("type_loss (legacy)", ["type_loss"]),
        line("type_acc (legacy)", ["type_acc"]),
        line("base_type_acc (legacy)", ["base_type_acc"]),
        line("hyper_type_acc (legacy)", ["hyper_type_acc"]),
        line("hyper_ratio (legacy)", ["hyper_ratio"]),
        line("type_loss (raw counts)", ["compressed/type_loss"]),
        line("type_acc (raw counts)", ["compressed/type_acc"]),
        line("base_type_acc (raw counts)", ["compressed/base_type_acc"]),
        line("hyper_type_acc (raw counts)", ["compressed/hyper_type_acc"]),
        line("hyper_ratio (raw counts)", ["compressed/hyper_ratio"]),
    ]

    optimisation = [
        line("learning rates", ["lr", "hyper_lr"]),
        line("grad_norm", ["grad_norm"]),
        line("avg_step_time", ["avg_step_time"]),
        line("tokens_per_sec", ["tokens_per_sec"]),
        line("total_tokens", ["total_tokens"]),
        line("rank_microbatches", ["compressed/rank_microbatches"]),
    ]

    eval_compression = [
        line(
            "MC eval compression (harness)",
            ["eval/input_compression_ratio", "eval/gen_compression_ratio"],
            X_TRAIN,
        ),
        line("MC eval wall seconds", ["eval/wall_seconds"], X_TRAIN),
        # NOTE: pipeline phase 3 writes final/eval/* twice at the same
        # final/step - once from the MC JSON, once from the wikitext JSON.
        # Two points at one x; the unprefixed eval/* above is MC-only.
        line(
            "final/eval compression (MC then wikitext)",
            ["final/eval/input_compression_ratio", "final/eval/gen_compression_ratio"],
            X_FINAL,
        ),
        line("final/eval wall seconds", ["final/eval/wall_seconds"], X_FINAL),
    ]

    experiment_diag = [
        line("online_skipped_target_rate (v0.6.5)", ["online_skipped_target_rate"]),
        line("online_skipped_targets (v0.6.5)", ["online_skipped_targets"]),
        line("base_view_replay_rate (v0.6.6)", ["base_view_replay_rate"]),
        line("base_view loss_target_base_token", ["base_view/loss_target_base_token"]),
        line("base_view ppl_target_base_token", ["base_view/ppl_target_base_token"]),
        line("base_view loss_per_valid_token", ["base_view/loss_per_valid_token"]),
        line("base_view acc", ["base_view/acc"]),
        line("base_view base_token_acc", ["base_view/base_token_acc"]),
        line("base_view rank_microbatches", ["base_view/rank_microbatches"]),
        line("base_view type_loss", ["base_view/type_loss"]),
        line("base_view type_acc", ["base_view/type_acc"]),
        line("base_view base_type_acc", ["base_view/base_type_acc"]),
        line("rope gate summary (v0.7.1)", ["rope_gate_mean", "rope_gate_rms"]),
        line("rope gate range", ["rope_gate_min", "rope_gate_max"]),
    ]

    gate_detail = [
        line("gate mean by layer", [f"rope_gate/layer_{i:02d}_mean" for i in GATE_LAYERS]),
        line("gate rms by layer", [f"rope_gate/layer_{i:02d}_rms" for i in GATE_LAYERS]),
        line("gate mean by pair", [f"rope_gate/pair_{i:02d}_mean" for i in GATE_PAIRS]),
        line("gate rms by pair", [f"rope_gate/pair_{i:02d}_rms" for i in GATE_PAIRS]),
    ]

    stderr = [
        line(
            f"{task} stderr",
            [f"final/{task}/acc_stderr", f"final/{task}/acc_norm_stderr"],
            X_FINAL,
        )
        for task in MC_TASKS_ACC_NORM
    ] + [
        line("winogrande stderr", ["final/winogrande/acc_stderr"], X_FINAL),
        line(
            "GSM8K stderr",
            [
                "final/gsm8k/exact_match_stderr_strict-match",
                "final/gsm8k/exact_match_stderr_flexible-extract",
            ],
            X_FINAL,
        ),
        line(
            "TriviaQA stderr",
            ["final/triviaqa/exact_match_stderr_remove_whitespace"],
            X_FINAL,
        ),
        line(
            "smoke arc_easy / hellaswag stderr",
            [
                "smoke/arc_easy/acc_stderr",
                "smoke/arc_easy/acc_norm_stderr",
                "smoke/hellaswag/acc_stderr",
                "smoke/hellaswag/acc_norm_stderr",
            ],
            X_SMOKE,
        ),
        line("smoke winogrande stderr", ["smoke/winogrande/acc_stderr"], X_SMOKE),
        line(
            "smoke gsm8k_boxed stderr",
            [
                "smoke/gsm8k_boxed/exact_match_stderr_boxed-first",
                "smoke/gsm8k_boxed/exact_match_stderr_strict-match",
                "smoke/gsm8k_boxed/exact_match_stderr_flexible-extract",
            ],
            X_SMOKE,
        ),
    ]

    raw_acc = [
        line(f"final {task} acc (unnormalised)", [f"final/{task}/acc"], X_FINAL)
        for task in MC_TASKS_ACC_NORM
    ] + [
        line(f"smoke {task} acc (unnormalised)", [f"smoke/{task}/acc"], X_SMOKE)
        for task in SMOKE_TASKS_ACC_NORM
    ]

    # lm-eval's own WandbLogger keys. Duplicates of final/* in pipeline runs,
    # but the only source in standalone eval runs -> collapsed, not deleted.
    # Naming is lm-eval's, not ours; --verify confirms the exact spelling.
    unprefixed = [
        line(f"{task} (lm-eval keys)", [f"{task}/acc", f"{task}/acc_norm"], X_TRAIN)
        for task in MC_TASKS_ACC_NORM
    ] + [
        line("winogrande (lm-eval keys)", ["winogrande/acc"], X_TRAIN),
        line(
            "gsm8k (lm-eval keys)",
            ["gsm8k/exact_match,strict-match", "gsm8k/exact_match,flexible-extract"],
            X_TRAIN,
        ),
        line(
            "triviaqa (lm-eval keys)",
            ["triviaqa/exact_match,remove_whitespace"],
            X_TRAIN,
        ),
        line(
            "stderr (lm-eval keys)",
            [f"{task}/acc_stderr" for task in MC_TASKS_ACC_NORM]
            + [f"{task}/acc_norm_stderr" for task in MC_TASKS_ACC_NORM]
            + ["winogrande/acc_stderr"]
            + ["triviaqa/exact_match_stderr,remove_whitespace"],
            X_TRAIN,
        ),
    ]

    return [
        {"name": "1 Verdetto finale", "open": True, "panels": verdict},
        {"name": "2 Curve smoke", "open": True, "panels": smoke},
        {"name": "3 Train - loss/ppl legacy (comparabili v0.6.x)", "open": True, "panels": loss_legacy},
        {"name": "4 Train - loss/ppl corrette (compressed/)", "open": True, "panels": loss_corrected},
        {"name": "5 Train - accuracy legacy", "open": False, "panels": acc_legacy},
        {"name": "6 Train - accuracy corrette (compressed/)", "open": False, "panels": acc_corrected},
        {"name": "7 Compressione (train)", "open": True, "panels": compression},
        {"name": "8 Token-type head", "open": False, "panels": type_head},
        {"name": "9 Ottimizzazione e throughput", "open": False, "panels": optimisation},
        {"name": "10 Compressione a eval-time", "open": False, "panels": eval_compression},
        {"name": "11 Diagnostica per-esperimento (v0.6.5/6.6/7.1)", "open": False, "panels": experiment_diag},
        {"name": "12 rope_gate per layer/pair", "open": False, "panels": gate_detail},
        {"name": "13 Stderr benchmark", "open": False, "panels": stderr},
        {"name": "14 MC acc non normalizzata", "open": False, "panels": raw_acc},
        {"name": "15 Eval non prefissate (lm-eval)", "open": False, "panels": unprefixed},
    ]


def all_keys(layout: list[dict]) -> set[str]:
    """Every key the view puts on screen, tables included."""
    keys = set(TABLE_PANELS)
    for section in layout:
        for panel in section["panels"]:
            keys.update(panel["keys"])
            keys.add(panel["x"])
    return keys


def print_layout(layout: list[dict]) -> None:
    panels = sum(len(s["panels"]) for s in layout)
    print(f"{len(layout)} sections, {panels} panels, {len(all_keys(layout))} keys covered\n")
    for section in layout:
        state = "open" if section["open"] else "collapsed"
        print(f"[{state}] {section['name']}  ({len(section['panels'])} panels)")
        for panel in section["panels"]:
            keys = ", ".join(panel["keys"][:4])
            if len(panel["keys"]) > 4:
                keys += f", ... (+{len(panel['keys']) - 4})"
            print(f"    - {panel['title']:<48} x={panel['x']:<12} {keys}")
        print()
    print("Dropped as exact duplicates:")
    for key in DUPLICATES:
        print(f"    - {key}")


def verify(layout: list[dict], run_path: str) -> int:
    """Every key the run logs must land in some panel. Reports both directions."""
    import wandb

    run = wandb.Api().run(run_path)
    logged = set(run.summary.keys())
    # Summary holds the last value of every key, but a key logged only in
    # history (never in summary) would still deserve a panel.
    rows = run.history(samples=1, pandas=False)
    if rows:
        logged |= set(rows[0].keys())
    logged = {
        k for k in logged
        if not k.startswith(("_", "system", "eval_args", "graph", "gradients"))
    }

    covered = all_keys(layout)
    uncovered = sorted(k for k in logged if k not in covered and k not in DUPLICATES)
    missing = sorted(k for k in covered if k not in logged)

    print(f"run {run_path}: {len(logged)} keys logged, {len(covered)} keys in layout\n")
    print(f"logged but NOT in any panel ({len(uncovered)}) - these would be lost:")
    for key in uncovered:
        print(f"    - {key}")
    print(f"\nin the layout but not logged by this run ({len(missing)}) - empty panels:")
    for key in missing:
        print(f"    - {key}")
    return 1 if uncovered else 0


def build_sections(layout: list[dict], ws, wr) -> list:
    """Layout dicts -> wandb_workspaces objects. No network, no side effects."""
    sections = [
        ws.Section(
            name=section["name"],
            is_open=section["open"],
            panels=[
                wr.LinePlot(
                    title=panel["title"],
                    x=panel["x"],
                    y=panel["keys"],
                    log_y=panel["log_y"],
                )
                for panel in section["panels"]
            ],
        )
        for section in layout
    ]
    # lm-eval's sample tables are not metrics, so they need their own panel type.
    # Declared explicitly because auto_generate_panels stays False: with it True,
    # W&B would add its own panel next to every panel below and re-create the
    # duplicates this whole layout exists to remove.
    sections.append(
        ws.Section(
            name="16 Tabelle sample (lm-eval)",
            is_open=False,
            panels=[
                wr.WeavePanelSummaryTable(table_name=name)
                for name in TABLE_PANELS
            ],
        )
    )
    return sections


def apply(
    layout: list[dict],
    view_name: str,
    entity: str,
    project: str,
    view_url: str | None,
) -> None:
    try:
        import wandb_workspaces.reports.v2 as wr
        import wandb_workspaces.workspaces as ws
    except ImportError:
        raise SystemExit(
            "--apply needs the workspace-as-code package: pip install wandb-workspaces"
        )

    sections = build_sections(layout, ws, wr)

    if view_url:
        # A saved view belongs to one entity/project. Refuse to rewrite a view
        # that lives somewhere other than where this run logged: without this,
        # a run with WANDB_PROJECT=llaza would silently overwrite the
        # zip2zip-core view, because from_url takes the target from the URL.
        target = urlparse(view_url).path.strip("/").split("/")
        if len(target) >= 2 and (target[0], target[1]) != (entity, project):
            raise SystemExit(
                f"--view-url points at {target[0]}/{target[1]} but this run is "
                f"{entity}/{project}. Pass a view URL for that project, or "
                f"drop --view-url to create one."
            )

        # save() on a freshly constructed Workspace creates a NEW saved view
        # every call (interface.py: empty _internal_name -> _generate_view_name).
        # Loading the existing view first is what makes a re-run overwrite it
        # instead of piling up one view per pipeline run.
        workspace = ws.Workspace.from_url(view_url)
        workspace.name = view_name
        workspace.sections = sections
        saved = workspace.save()
        print(f"updated view: {saved.url}")
        return

    workspace = ws.Workspace(
        name=view_name,
        entity=entity,
        project=project,
        sections=sections,
        auto_generate_panels=False,
    )
    saved = workspace.save()
    print(f"created view: {saved.url}")
    print(
        "NOTE: pass this URL back as --view-url (or WANDB_VIEW_URL in the "
        "pipeline) so the next run updates this view instead of creating "
        "another one."
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--entity", default=ENTITY)
    p.add_argument("--project", default=PROJECT)
    p.add_argument("--verify", metavar="ENTITY/PROJECT/RUN_ID", default=None,
                   help="Check the layout against a real run (needs network).")
    p.add_argument("--apply", action="store_true",
                   help="Save the layout as a saved view (never touches the default workspace).")
    p.add_argument("--view-name", default="z2z clean")
    p.add_argument("--view-url", default=VIEW_URL,
                   help="Saved view to overwrite (default: the project's shared "
                        "view). Pass '' to CREATE a new view instead and print "
                        "its URL: a fresh Workspace() always saves to a new view, "
                        "so this must never be the default in a loop.")
    p.add_argument("--gate-layers", default="0:32", help="rope_gate layer range, START:END")
    p.add_argument("--gate-pairs", default="32:48", help="rope_gate pair range, START:END")
    args = p.parse_args()

    global GATE_LAYERS, GATE_PAIRS
    GATE_LAYERS = range(*(int(v) for v in args.gate_layers.split(":")))
    GATE_PAIRS = range(*(int(v) for v in args.gate_pairs.split(":")))

    layout = build_layout()
    if args.verify:
        raise SystemExit(verify(layout, args.verify))
    if args.apply:
        apply(layout, args.view_name, args.entity, args.project, args.view_url)
        return
    print_layout(layout)


if __name__ == "__main__":
    main()
