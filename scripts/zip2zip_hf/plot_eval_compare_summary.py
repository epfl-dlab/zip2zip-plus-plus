"""Plot __all__ rows from eval_compare summary.csv as SVG bar charts.

Example:
  python scripts/zip2zip_hf/plot_eval_compare_summary.py \
      --summary outputs/eval_compare/summary.csv
"""

from __future__ import annotations

import argparse
import csv
import html
import math
from pathlib import Path

METRICS = [
    ("mean_actual_bytes_per_zip_token", "Actual bytes / zip token", "higher is better"),
    ("mean_theory_bytes_per_zip_token", "Theory bytes / zip token", "higher is better"),
    ("mean_compression_efficiency", "Compression efficiency", "closer to 1 is better"),
    ("mean_base_token_saving", "Base token saving", "higher is better"),
    ("mean_gpt2_mean_nll", "GPT-2 mean NLL", "lower is better"),
    ("mean_gpt2_bits_per_byte", "GPT-2 bits / byte", "lower is better"),
    ("mean_empty_or_too_short", "Empty / too short rate", "lower is better"),
    ("mean_repeat_4gram_rate", "Repeat 4-gram rate", "lower is better"),
    ("mean_max_4gram_count", "Max 4-gram count", "lower is better"),
    ("mean_degenerate_repetition", "Degenerate repetition rate", "lower is better"),
]


def parse_value(row: dict[str, str], metric: str) -> float:
    try:
        return float(row[metric])
    except Exception:
        return float("nan")


def format_value(value: float) -> str:
    if math.isnan(value):
        return "nan"
    if abs(value) >= 100:
        return f"{value:.0f}"
    if abs(value) >= 10:
        return f"{value:.2f}"
    if abs(value) >= 1:
        return f"{value:.3f}"
    return f"{value:.3g}"


def sorted_all_rows(summary_path: Path) -> list[dict[str, str]]:
    with summary_path.open(encoding="utf-8") as file:
        rows = [row for row in csv.DictReader(file) if row.get("category") == "__all__"]
    ms_order = {"MS2": 0, "MS3": 1, "MS4": 2}
    group_order = {"ft": 0, "scratch": 1, "unknown": 2}
    rows.sort(
        key=lambda row: (
            group_order.get(row.get("model_group", "unknown"), 99),
            ms_order.get(row.get("ms", ""), 99),
            row.get("model_slug", ""),
        )
    )
    return rows


def bar_chart_svg(rows: list[dict[str, str]], metric: str, title: str, note: str, width: int, height: int) -> str:
    labels = [f"{row.get('ms')} {row.get('model_group')}" for row in rows]
    colors = ["#4C78A8" if row.get("model_group") == "ft" else "#F58518" for row in rows]
    values = [parse_value(row, metric) for row in rows]
    finite = [value for value in values if not math.isnan(value)] or [0.0]
    ymin = min(0.0, min(finite))
    ymax = max(finite)
    if ymax == ymin:
        ymax = ymin + 1.0
    pad = (ymax - ymin) * 0.12
    ymax += pad
    if ymin < 0:
        ymin -= pad

    left, right, top, bottom = 78, 24, 74, 96
    plot_width = width - left - right
    plot_height = height - top - bottom
    bar_gap = 18
    bar_width = (plot_width - bar_gap * (len(values) + 1)) / max(1, len(values))

    def y_pos(value: float) -> float:
        return top + (ymax - value) / (ymax - ymin) * plot_height

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width / 2}" y="28" text-anchor="middle" font-family="Arial" font-size="18" font-weight="700">{html.escape(title)}</text>',
        f'<text x="{width / 2}" y="50" text-anchor="middle" font-family="Arial" font-size="12" fill="#555">{html.escape(note)}</text>',
    ]

    for index in range(5):
        tick = ymin + (ymax - ymin) * index / 4
        y = y_pos(tick)
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width - right}" y2="{y:.1f}" stroke="#ddd"/>')
        parts.append(f'<text x="{left - 8}" y="{y + 4:.1f}" text-anchor="end" font-family="Arial" font-size="11" fill="#555">{format_value(tick)}</text>')

    zero_y = y_pos(0.0)
    parts.append(f'<line x1="{left}" y1="{zero_y:.1f}" x2="{width - right}" y2="{zero_y:.1f}" stroke="#888" stroke-width="1.2"/>')
    parts.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" stroke="#333"/>')

    for index, (value, label, color) in enumerate(zip(values, labels, colors)):
        if math.isnan(value):
            continue
        x = left + bar_gap + index * (bar_width + bar_gap)
        y = min(y_pos(value), zero_y)
        bar_height = abs(zero_y - y_pos(value))
        parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width:.1f}" height="{bar_height:.1f}" fill="{color}" stroke="#222" stroke-width="0.6"/>')
        parts.append(f'<text x="{x + bar_width / 2:.1f}" y="{y - 7:.1f}" text-anchor="middle" font-family="Arial" font-size="11">{format_value(value)}</text>')
        ms, group = label.split()
        parts.append(f'<text x="{x + bar_width / 2:.1f}" y="{height - 54}" text-anchor="middle" font-family="Arial" font-size="12">{html.escape(ms)}</text>')
        parts.append(f'<text x="{x + bar_width / 2:.1f}" y="{height - 37}" text-anchor="middle" font-family="Arial" font-size="11" fill="#555">{html.escape(group)}</text>')

    parts.append('<rect x="700" y="18" width="14" height="14" fill="#4C78A8"/><text x="720" y="30" font-family="Arial" font-size="12">FT</text>')
    parts.append('<rect x="760" y="18" width="14" height="14" fill="#F58518"/><text x="780" y="30" font-family="Arial" font-size="12">Scratch</text>')
    parts.append("</svg>")
    return "\n".join(parts)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    rows = sorted_all_rows(args.summary)
    if not rows:
        raise ValueError(f"No category == __all__ rows found in {args.summary}")

    out_dir = args.out_dir or args.summary.parent / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    for metric, title, note in METRICS:
        path = out_dir / f"{metric}.svg"
        path.write_text(bar_chart_svg(rows, metric, title, note, 920, 520), encoding="utf-8")
        print(f"[saved] {path}")

    cell_width, cell_height = 760, 430
    num_rows = math.ceil(len(METRICS) / 2)
    combined_width, combined_height = cell_width * 2, cell_height * num_rows + 60
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{combined_width}" height="{combined_height}" viewBox="0 0 {combined_width} {combined_height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{combined_width / 2}" y="34" text-anchor="middle" font-family="Arial" font-size="24" font-weight="700">Eval Compare: __all__ Metrics Across 6 Models</text>',
    ]
    for index, (metric, title, note) in enumerate(METRICS):
        x = (index % 2) * cell_width
        y = 60 + (index // 2) * cell_height
        svg = bar_chart_svg(rows, metric, title, note, cell_width, cell_height)
        inner = svg.split(">", 1)[1].rsplit("</svg>", 1)[0]
        parts.append(f'<g transform="translate({x},{y})">{inner}</g>')
    parts.append("</svg>")
    combined_path = out_dir / "all_metrics_bar.svg"
    combined_path.write_text("\n".join(parts), encoding="utf-8")
    print(f"[saved] {combined_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
