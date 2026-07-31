"""Static audit of the W&B view wiring.

Two invariants, both text/AST only (no torch, no network):

  1. The duplicate panels are suppressed where the value is logged, not by hand
     in the UI, so every future run comes out deduplicated by itself.
  2. Every metric key train.py logs is covered by a panel in
     scripts/wandb_workspace_view.py. Adding a metric without giving it a panel
     fails here instead of silently disappearing from the workspace.
"""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
TRAIN = REPO / "src" / "zip2zip_core" / "train.py"
HARNESS = REPO / "scripts" / "eval_harness.py"
PIPELINE = REPO / "scripts" / "pipeline_ft_eval_rcp.sh"
LAYOUT = REPO / "scripts" / "wandb_workspace_view.py"


def _load_layout():
    spec = importlib.util.spec_from_file_location("wandb_workspace_view", LAYOUT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _logged_train_keys() -> set[str]:
    """Constant string keys train.py puts into its wandb log_dict.

    f-string keys (per-layer gate diagnostics, ...) are skipped: their names are
    built at runtime, so the layout covers them by explicit range instead.
    """
    tree = ast.parse(TRAIN.read_text())
    keys: set[str] = set()

    def dict_keys(node: ast.AST) -> None:
        if isinstance(node, ast.Dict):
            for key in node.keys:
                if isinstance(key, ast.Constant) and isinstance(key.value, str):
                    keys.add(key.value)

    for node in ast.walk(tree):
        # log_dict = {...} / eval_dict = {...}
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "log_dict":
                    dict_keys(node.value)
                # log_dict["key"] = ...
                if (
                    isinstance(target, ast.Subscript)
                    and isinstance(target.value, ast.Name)
                    and target.value.id == "log_dict"
                    and isinstance(target.slice, ast.Constant)
                    and isinstance(target.slice.value, str)
                ):
                    keys.add(target.slice.value)
        # log_dict.update({...})
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "update"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "log_dict"
        ):
            for arg in node.args:
                dict_keys(arg)
    return keys


def test_duplicate_panels_are_hidden_where_they_are_logged():
    train_source = TRAIN.read_text()
    # Exact duplicate of the bare `backward_loss`: same compat_backward_loss.
    assert 'wandb.define_metric("compressed/backward_loss", hidden=True)' in train_source
    assert '"backward_loss": compressed_metrics[' in train_source, (
        "the bare key must stay logged: it is the legacy-comparable one"
    )
    assert '"objective/backward_loss": mixed_backward_loss' in train_source, (
        "the mixed objective is the one panel that is not a duplicate"
    )

    harness_source = HARNESS.read_text()
    assert 'wandb.define_metric(f"{task_name}/*", hidden=True)' in harness_source
    # Only on the pipeline path: in a standalone eval run those unprefixed keys
    # are the only copy of the result and must stay visible.
    hide_at = harness_source.index('wandb.define_metric(f"{task_name}/*"')
    guard_at = harness_source.rindex(
        'if resume_id and resume_id.lower() != "none":', 0, hide_at
    )
    assert guard_at < hide_at
    assert harness_source.index("log_eval_result()", hide_at) > hide_at, (
        "the metrics must be defined before lm-eval logs them"
    )


def test_pipeline_refreshes_the_saved_view_without_being_able_to_fail():
    source = PIPELINE.read_text()
    assert "scripts/wandb_workspace_view.py --apply" in source
    assert "WANDB_VIEW_URL=${WANDB_VIEW_URL:-}" in source
    # Refresh only against a known view: saving without a URL creates a new view
    # every time, which would leave one saved view per pipeline run behind.
    assert 'if [ -n "$WANDB_VIEW_URL" ]; then' in source
    # Anchor on the guarded block itself, not on the first mention of the script
    # (the env-var documentation in the header mentions it too).
    guard_at = source.index('if [ -n "$WANDB_VIEW_URL" ]; then')
    block = source[guard_at:guard_at + 600]
    assert "scripts/wandb_workspace_view.py --apply" in block
    assert '--view-url "$WANDB_VIEW_URL"' in block
    # Cosmetic step at the very end of a multi-hour job: it must never turn a
    # finished run into a failed one.
    assert "|| echo" in block
    # ... and the package it needs is installed by the eval venv, W&B-only.
    assert "EVAL_DEPS+=(wandb wandb-workspaces)" in source


def test_layout_covers_every_metric_train_py_logs():
    layout = _load_layout()
    covered = layout.all_keys(layout.build_layout()) | set(layout.DUPLICATES)
    logged = _logged_train_keys()
    assert logged, "AST walk found no log_dict keys - the parser drifted"
    missing = sorted(logged - covered)
    assert not missing, f"logged by train.py but no panel: {missing}"


def test_workspace_objects_build_and_serialise():
    """Everything --apply does except the network call.

    Covers the constructors, their validation, and the _to_model() serialisation
    that immediately precedes the upsert - i.e. every way the layout can be
    malformed without a W&B key. Skipped where wandb-workspaces is absent: the
    pipeline's eval venv installs it, the training env does not.
    """
    ws = pytest.importorskip("wandb_workspaces.workspaces")
    wr = pytest.importorskip("wandb_workspaces.reports.v2")
    layout_module = _load_layout()
    layout = layout_module.build_layout()
    sections = layout_module.build_sections(layout, ws, wr)

    assert len(sections) == len(layout) + 1, "the table section must be appended"
    panels = sum(len(s["panels"]) for s in layout) + len(layout_module.TABLE_PANELS)
    assert sum(len(s.panels) for s in sections) == panels

    workspace = ws.Workspace(
        name="z2z view audit",
        entity="epfl-dlab",
        project="zip2zip-core",
        sections=sections,
        auto_generate_panels=False,
    )
    # False on purpose: with auto-generated panels on, W&B would add its own
    # panel for metrics that already have one here, re-creating the duplicates.
    assert workspace.auto_generate_panels is False
    assert workspace._to_model() is not None


def test_dropped_duplicates_have_no_panel():
    layout = _load_layout()
    covered = layout.all_keys(layout.build_layout())
    for key in layout.DUPLICATES:
        assert key not in covered, f"{key} is dropped as a duplicate but has a panel"
