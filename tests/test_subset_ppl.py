"""Tests for the pinned-subset perplexity tasks (*_sub1k) and their wiring.

The whole point of these tasks is that every run scores exactly the same
documents. These tests pin that promise:

  - the seeded index draw must reproduce the sha256 fingerprints recorded in
    utils.py (a drift in random.Random across Python versions fails here
    before it can fail on a GPU job);
  - a corpus whose size no longer matches the pin must be a hard error;
  - the subset YAMLs must point at exactly the same pinned data as their
    parent tasks;
  - the perplexity_subset preset must differ from perplexity only in tasks —
    a different window size or seed would silently change the scoring regime;
  - eval_ckpt_rcp.sh must default subset runs into the subset_ppl/ W&B
    section and refuse final/.
"""

import hashlib
import importlib.util
import json
import re
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "scripts"
TASKS_DIR = SCRIPTS / "lm_eval_tasks"

SUBSET_TASKS = {
    "zip2zip_pile_sub1k": "zip2zip_pile",
    "zip2zip_mc4_sub1k": "zip2zip_mc4",
    "zip2zip_dc4_sub1k": "zip2zip_dc4",
}


def load_task_utils():
    path = TASKS_DIR / "utils.py"
    spec = importlib.util.spec_from_file_location("_subset_task_utils", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class FakeDataset:
    """len() + select(), which is all _fixed_subset uses."""

    def __init__(self, n):
        self.rows = list(range(n))

    def __len__(self):
        return len(self.rows)

    def select(self, indices):
        picked = FakeDataset(0)
        picked.rows = [self.rows[i] for i in indices]
        return picked


# ───────────────────────── pinned index draw ───────────────────────────────


def test_pinned_indices_reproduce_their_fingerprint():
    utils = load_task_utils()
    for task, (expected_n, expected_sha) in utils._SUBSET_SPECS.items():
        # subset_indices itself raises if the digest drifts; recompute anyway
        # so this test does not depend on the guard it is testing.
        indices = utils.subset_indices(task)
        digest = hashlib.sha256(json.dumps(indices).encode()).hexdigest()
        assert digest == expected_sha, f"{task}: index draw drifted"
        assert len(indices) == utils._SUBSET_SIZE
        assert len(set(indices)) == len(indices)
        assert indices == sorted(indices)
        assert 0 <= indices[0] and indices[-1] < expected_n


def test_draw_is_deterministic_across_calls():
    utils = load_task_utils()
    for task in SUBSET_TASKS:
        assert utils.subset_indices(task) == utils.subset_indices(task)


def test_wrong_corpus_size_is_fatal():
    utils = load_task_utils()
    with pytest.raises(ValueError, match="not be comparable"):
        utils.subset_pile_sub1k(FakeDataset(12345))


def test_fixed_subset_selects_exactly_the_pinned_rows():
    utils = load_task_utils()
    for fn_name, task in [
        ("subset_pile_sub1k", "zip2zip_pile_sub1k"),
        ("subset_mc4_sub1k", "zip2zip_mc4_sub1k"),
        ("subset_dc4_sub1k", "zip2zip_dc4_sub1k"),
    ]:
        expected_n, _ = utils._SUBSET_SPECS[task]
        picked = getattr(utils, fn_name)(FakeDataset(expected_n))
        assert picked.rows == utils.subset_indices(task)


# ───────────────────────────── YAML wiring ─────────────────────────────────


def _yaml_text(task_name):
    return (TASKS_DIR / f"{task_name}.yaml").read_text()


def test_subset_yamls_point_at_the_parent_corpus():
    utils = load_task_utils()
    for task, parent in SUBSET_TASKS.items():
        text = _yaml_text(task)
        parent_text = _yaml_text(parent)

        assert f"task: {task}" in text
        assert f"process_docs: !function utils.subset_{task.split('_', 1)[1]}" in text
        assert "output_type: loglikelihood_rolling" in text

        def dataset_pin(source):
            path = re.search(r"^dataset_path: (.+)$", source, re.M).group(1)
            shard = re.search(r"^\s+validation: (.+\.json\.gz)$", source, re.M)
            return (path, shard.group(1) if shard else None)

        assert dataset_pin(text) == dataset_pin(parent_text), (
            f"{task} does not load the exact corpus its indices were pinned "
            f"against"
        )

        pinned_docs = int(re.search(r"pinned_corpus_docs: (\d+)", text).group(1))
        assert pinned_docs == utils._SUBSET_SPECS[task][0], (
            f"{task}: YAML metadata disagrees with the utils.py pin"
        )


def test_subset_preset_only_differs_in_tasks():
    presets = yaml.safe_load((SCRIPTS / "eval_presets.yaml").read_text())["presets"]
    full = dict(presets["perplexity"])
    subset = dict(presets["perplexity_subset"])

    assert subset.pop("tasks") == [
        "wikitext",
        "zip2zip_pile_sub1k",
        "zip2zip_mc4_sub1k",
        "zip2zip_dc4_sub1k",
    ]
    full.pop("tasks")
    subset.pop("description")
    full.pop("description")
    assert subset == full, (
        "perplexity_subset must score in exactly the perplexity regime "
        "(window, seed, dtype, ...) or its numbers mean something else"
    )


# ─────────────────────────── shell plumbing ────────────────────────────────


def test_eval_ckpt_namespaces_subset_runs():
    source = (SCRIPTS / "eval_ckpt_rcp.sh").read_text()
    assert 'if [ "$PRESET" = "perplexity_subset" ]; then' in source
    assert "WANDB_PREFIX=${WANDB_PREFIX:-subset_ppl}" in source
    # Subset numbers are refused a spot in the full-run section.
    refusal_at = source.index('if [ "$WANDB_PREFIX" = "final" ]; then')
    assert source.index("WANDB_PREFIX=${WANDB_PREFIX:-subset_ppl}") < refusal_at
    # A LIMIT run scores only the first N of the pinned 1000 docs, so it must
    # not be able to log into the comparable subset_ppl/ series.
    assert (
        '[ -n "$LIMIT" ] && [ -n "$RESUME_WANDB_ID" ]' in source
        and '[ "$WANDB_PREFIX" = "subset_ppl" ]' in source
    )
    # Only the FULL perplexity preset may resolve the pipeline's [pending]
    # follow-up block; a quick subset score must leave it pending.
    assert 'if [ "$PRESET" = "perplexity" ]; then' in source
