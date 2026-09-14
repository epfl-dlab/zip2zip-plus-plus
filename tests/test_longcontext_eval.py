"""Protocol checks for the RULER evaluation path."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import types

import pytest
import yaml


REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "scripts"


def _load_script(name: str):
    path = SCRIPTS / name
    sys.path.insert(0, str(SCRIPTS))
    spec = importlib.util.spec_from_file_location(f"_test_{path.stem}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_longcontext_ruler_preset():
    presets = yaml.safe_load((SCRIPTS / "eval_presets.yaml").read_text())["presets"]
    ruler = presets["longcontext_ruler"]
    assert ruler["tasks"] == ["ruler"]
    assert ruler["ruler_lengths"] == (
        "4096,8192,16384,32768,65536,131072"
    )
    assert ruler["rope_mode"] == "phi3-longrope"
    assert ruler["fail_on_truncation"] is True
    assert ruler["force_greedy"] is True
    assert ruler["max_length"] == 131072
    assert [
        name for name in presets if name.startswith("longcontext_")
    ] == ["longcontext_ruler"]


def test_ruler_hotpot_uses_pinned_hf_mirror():
    harness = _load_script("eval_harness.py")
    from pathlib import Path

    from lm_eval import utils as lm_eval_utils
    from lm_eval.tasks.ruler import qa_utils

    harness._use_ruler_hotpot_mirror()

    # Exercise the same fresh-module !function loading path used by TaskManager.
    import yaml

    node = yaml.ScalarNode(
        tag="tag:yaml.org,2002:str", value="qa_utils.get_hotpotqa"
    )
    loader = yaml.SafeLoader("")
    try:
        callback = lm_eval_utils.import_function(
            loader, node, Path(qa_utils.__file__).with_name("qa_hotpot.yaml")
        )
    finally:
        loader.dispose()
    namespace = callback.__globals__
    seen = []

    def download_json(url):
        seen.append(url)
        return []

    namespace["download_json"] = download_json
    assert namespace["read_hotpotqa"]() == ([], [])
    assert seen == [harness._RULER_HOTPOT_MIRROR]
    assert namespace["read_hotpotqa"]() == ([], [])
    assert seen == [harness._RULER_HOTPOT_MIRROR]



def test_ruler_lengths_parser_rejects_ambiguous_ordering():
    harness = _load_script("eval_harness.py")
    assert harness._parse_ruler_lengths(
        "4096,8192,16384,32768,65536,131072"
    ) == [
        4096,
        8192,
        16384,
        32768,
        65536,
        131072,
    ]
    with pytest.raises(ValueError, match="strictly increasing"):
        harness._parse_ruler_lengths("8192,4096")
    with pytest.raises(ValueError, match="strictly increasing"):
        harness._parse_ruler_lengths("4096,4096")


def test_aggregate_ruler_accepts_split_task_results_and_requires_all_tasks():
    aggregate = _load_script("aggregate_longcontext.py")
    names = list(aggregate.RULER_TASKS)
    left, right = names[:6], names[6:]

    def payload(task_names, value):
        return {
            "results": {
                task: {"32768,none": value}
                for task in task_names
            },
            "args": {"ruler_lengths": "32768"},
        }

    result = aggregate.aggregate_ruler(
        [payload(left, 0.25), payload(right, 0.75)]
    )
    expected = (len(left) * 0.25 + len(right) * 0.75) / len(names)
    assert result["num_tasks"] == 13
    assert result["curve"]["32768"] == pytest.approx(expected)

    with pytest.raises(ValueError, match="missing"):
        aggregate.aggregate_ruler([payload(names[:-1], 0.5)])


def test_aggregate_ruler_merges_jobs_split_by_length():
    aggregate = _load_script("aggregate_longcontext.py")
    names = list(aggregate.RULER_TASKS)

    def payload(length, value):
        return {
            "results": {
                task: {f"{length},none": value}
                for task in names
            },
            "args": {"ruler_lengths": str(length)},
        }

    result = aggregate.aggregate_ruler(
        [payload(65536, 0.6), payload(131072, 0.2)]
    )
    assert result["curve"] == {"65536": 0.6, "131072": 0.2}

    with pytest.raises(ValueError, match="duplicate RULER result"):
        aggregate.aggregate_ruler([payload(65536, 0.6), payload(65536, 0.2)])


def test_fail_on_truncation_stops_before_base_model_forward():
    from zip2zip_core.lm_eval_adapter import Zip2ZipLM

    lm = object.__new__(Zip2ZipLM)
    lm._max_length = 8
    lm.fail_on_truncation = True
    lm.compression_stats = {
        "max_prompt_base_tokens": 0,
        "truncated_requests": 0,
    }
    lm.model = types.SimpleNamespace()

    with pytest.raises(ValueError, match="would be left-truncated"):
        lm._generate_base(
            list(range(8)), [], 2, False, 0.0, 1.0, 0
        )
    assert lm.compression_stats["truncated_requests"] == 1


def test_compressed_truncation_uses_base_position_space(monkeypatch):
    import zip2zip_core.lm_eval_adapter as adapter

    class FakeCompressor:
        def __init__(self, **kwargs):
            pass

        def encode(self, ids, **kwargs):
            return [10, 11], None, None

    monkeypatch.setattr(adapter, "LZWCompressor", FakeCompressor)

    lm = object.__new__(adapter.Zip2ZipLM)
    lm.cfg = types.SimpleNamespace(vocab_size=100, base_token_positions=True)
    lm._compressor_kwargs = {}
    lm._max_length = 8
    lm.fail_on_truncation = True
    lm.compression_stats = {"truncated_requests": 0}

    with pytest.raises(ValueError, match="8 base-space tokens"):
        lm._generate_compressed(
            list(range(8)), [], 2, False, 0.0, 1.0, 0
        )
    assert lm.compression_stats["truncated_requests"] == 1
