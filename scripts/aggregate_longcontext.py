"""Aggregate the 13 RULER tasks into a context-length curve."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


RULER_TASKS = (
    "niah_single_1",
    "niah_single_2",
    "niah_single_3",
    "niah_multikey_1",
    "niah_multikey_2",
    "niah_multikey_3",
    "niah_multiquery",
    "niah_multivalue",
    "ruler_vt",
    "ruler_cwe",
    "ruler_fwe",
    "ruler_qa_squad",
    "ruler_qa_hotpot",
)


def _load(path: str) -> dict:
    return json.loads(Path(path).read_text())


def _metric(row: dict, name: str) -> float:
    for key, value in row.items():
        if key.split(",", 1)[0] == name and isinstance(value, (int, float)):
            return float(value)
    raise KeyError(f"metric {name!r} is missing from row keys {sorted(row)}")


def _evaluated_lengths(row: dict) -> list[int]:
    lengths = []
    for key, value in row.items():
        name = key.split(",", 1)[0]
        if name.isdigit() and isinstance(value, (int, float)) and value >= 0:
            lengths.append(int(name))
    return sorted(set(lengths))


def aggregate_ruler(payloads: list[dict]) -> dict:
    task_metrics: dict[str, dict[int, float]] = {}
    declared_lengths: set[int] = set()
    for payload in payloads:
        lengths = (payload.get("args") or {}).get("ruler_lengths")
        if isinstance(lengths, str):
            lengths = [int(item) for item in lengths.split(",") if item]
        if lengths:
            if lengths != sorted(set(lengths)):
                raise ValueError(
                    f"RULER lengths must be unique and increasing: {lengths}"
                )
            declared_lengths.update(lengths)

        for name, row in (payload.get("results") or {}).items():
            if name not in RULER_TASKS:
                continue
            row_lengths = lengths or _evaluated_lengths(row)
            if not row_lengths:
                raise ValueError(f"no evaluated RULER lengths for task {name}")
            declared_lengths.update(row_lengths)
            metrics = task_metrics.setdefault(name, {})
            for length in row_lengths:
                if length in metrics:
                    raise ValueError(f"duplicate RULER result: {name}@{length}")
                value = _metric(row, str(length))
                if value < 0:
                    raise ValueError(f"unevaluated RULER result: {name}@{length}")
                metrics[length] = value

    missing = sorted(set(RULER_TASKS) - set(task_metrics))
    if missing:
        raise ValueError(f"RULER task mismatch: missing={missing}")

    lengths = sorted(declared_lengths) or [
        4096, 8192, 16384, 32768, 65536, 131072
    ]
    curve = {}
    per_task = {name: {} for name in RULER_TASKS}
    for length in lengths:
        values = []
        for name in RULER_TASKS:
            if length not in task_metrics[name]:
                raise ValueError(f"missing RULER result: {name}@{length}")
            value = task_metrics[name][length]
            per_task[name][str(length)] = value
            values.append(value)
        curve[str(length)] = sum(values) / len(values)
    return {
        "num_tasks": len(task_metrics),
        "curve": curve,
        "per_task": per_task,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ruler",
        nargs="+",
        required=True,
        help="RULER result JSONs, optionally split by task and/or length",
    )
    parser.add_argument("--output", help="Optional aggregate JSON path")
    args = parser.parse_args()

    aggregate = {
        "ruler": aggregate_ruler([_load(path) for path in args.ruler])
    }
    rendered = json.dumps(aggregate, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(rendered + "\n")


if __name__ == "__main__":
    main()
