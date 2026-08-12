"""Publication figure for hierarchical merge-size transfer.

The audited values come from the Repeat-4/Repeat-8 evaluation JSONs under
``/dlabscratch1/xinma/logs/eval``.  M3 is vx0.6.4.5 (trained with maximum
merge size 3); M4 is vx0.6.4.3 (trained with maximum merge size 4).

Usage:
    python plot/plot_merge_size_transfer.py
    python plot/plot_merge_size_transfer.py --output-dir /path/to/figures
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# Repeat-8 extrapolation of M3, evaluated with the current multi-view harness.
K_EVAL = np.array([3, 4, 5, 6, 7, 8])
STRICT_BYTE_PPL = np.array([
    1.1056712618,
    1.1534654876,
    1.1686796093,
    1.1703499639,
    1.1711654079,
    1.1712367105,
])
MULTI_VIEW_BYTE_PPL = np.array([
    1.0959358916,
    1.1026679268,
    1.1040350346,
    1.1043257040,
    1.1044348092,
    1.1044937425,
])
COMPRESSION = np.array([
    2.0284793001,
    2.1373441533,
    2.1555608939,
    2.1594070675,
    2.1632294844,
    2.1641014035,
])

# Native M4 at the matched K_eval=4 Repeat-8 segmentation.
M4_STRICT_REPEAT8_BYTE_PPL = 1.1121679450
M4_MULTI_REPEAT8_BYTE_PPL = 1.0989894566

# Matched-segmentation relative transfer penalties:
# 100 * (PPL(M3, K_eval=4) / PPL(M4, K_eval=4) - 1).
STRICT_TRANSFER_PCT = np.array([1.0675983239, 3.7132469835])
MULTI_TRANSFER_PCT = np.array([0.2524785799, 0.3347138756])
LENGTH4_COVERAGE = np.array([6.54, 29.90])


INK = "#171717"
INK2 = "#575757"
GRID = "#dedede"
SURFACE = "#ffffff"
STRICT = "#0072B2"       # Okabe-Ito blue
MULTI = "#009E73"        # Okabe-Ito green
COMP = "#6B7280"
UNSEEN = "#F3F4F6"


def style_axis(ax: plt.Axes) -> None:
    ax.set_facecolor(SURFACE)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color=GRID, linewidth=0.75)
    ax.xaxis.grid(False)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(axis="both", colors=INK2, labelsize=8.5, length=0, pad=4)


def make_figure() -> plt.Figure:
    matplotlib.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 10,
        "legend.fontsize": 8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(10.6, 3.35),
        gridspec_kw={"width_ratios": [1.35, 1.0, 0.9], "wspace": 0.40},
        facecolor=SURFACE,
    )
    ax_loss, ax_comp, ax_penalty = axes
    for ax in axes:
        style_axis(ax)

    # (a) Strict and multi-view loss under extrapolation.
    ax_loss.axvspan(3.02, 8.25, color=UNSEEN, zorder=0)
    ax_loss.axvline(3, color=INK2, linewidth=1.0, linestyle=(0, (3, 3)), zorder=1)
    ax_loss.plot(
        K_EVAL,
        STRICT_BYTE_PPL,
        color=STRICT,
        linewidth=2.1,
        marker="o",
        markersize=5.2,
        markerfacecolor=STRICT,
        markeredgecolor=SURFACE,
        markeredgewidth=0.9,
        label="M3 strict",
        zorder=4,
    )
    ax_loss.plot(
        K_EVAL,
        MULTI_VIEW_BYTE_PPL,
        color=MULTI,
        linewidth=1.8,
        linestyle=(0, (4, 2)),
        marker="o",
        markersize=4.7,
        markerfacecolor=MULTI,
        markeredgecolor=SURFACE,
        markeredgewidth=0.9,
        label="M3 multi-view",
        zorder=4,
    )
    ax_loss.scatter(
        [4], [M4_STRICT_REPEAT8_BYTE_PPL], color=STRICT,
        marker="*", s=105, edgecolor=SURFACE, linewidth=0.8,
        label="M4 strict", zorder=5,
    )
    ax_loss.scatter(
        [4], [M4_MULTI_REPEAT8_BYTE_PPL], color=MULTI,
        marker="*", s=105, edgecolor=SURFACE, linewidth=0.8,
        label="M4 multi-view", zorder=5,
    )
    ax_loss.annotate(
        "training limit", xy=(3, 1.1796), xytext=(3.13, 1.1796),
        ha="left", va="top", fontsize=7.5, color=INK2,
    )
    ax_loss.text(
        5.72, 1.1796, "unseen merge sizes",
        ha="center", va="top", fontsize=7.5, color=INK2,
    )
    ax_loss.annotate(
        "M4 strict",
        xy=(4, M4_STRICT_REPEAT8_BYTE_PPL),
        xytext=(4.28, 1.1162),
        arrowprops={"arrowstyle": "-", "color": STRICT, "lw": 0.8},
        color=STRICT,
        fontsize=7.5,
        ha="left",
        va="bottom",
    )
    ax_loss.set_xlim(2.75, 8.25)
    ax_loss.set_ylim(1.09, 1.18)
    ax_loss.set_xticks(K_EVAL)
    ax_loss.set_xlabel(r"Evaluation max merge size $K_{\mathrm{eval}}$")
    ax_loss.set_ylabel("Byte perplexity (lower is better)")
    ax_loss.set_title("(a) Repeat-8 extrapolation", loc="left", fontweight="bold")
    ax_loss.legend(
        loc="upper right",
        bbox_to_anchor=(1.01, 0.86),
        frameon=False,
        handlelength=2.3,
        borderaxespad=0,
    )

    # (b) Compression saturates at the same time as the loss curve.
    ax_comp.axvspan(3.02, 8.25, color=UNSEEN, zorder=0)
    ax_comp.axvline(3, color=INK2, linewidth=1.0, linestyle=(0, (3, 3)), zorder=1)
    ax_comp.plot(
        K_EVAL,
        COMPRESSION,
        color=COMP,
        linewidth=2.1,
        marker="o",
        markersize=5.2,
        markeredgecolor=SURFACE,
        markeredgewidth=0.9,
        zorder=4,
    )
    ax_comp.annotate(
        f"{COMPRESSION[0]:.3f}",
        (K_EVAL[0], COMPRESSION[0]),
        xytext=(7, -2),
        textcoords="offset points",
        fontsize=7.5,
        color=INK2,
        ha="left",
        va="top",
    )
    ax_comp.annotate(
        f"{COMPRESSION[-1]:.3f}",
        (K_EVAL[-1], COMPRESSION[-1]),
        xytext=(-2, 8),
        textcoords="offset points",
        fontsize=7.5,
        color=INK2,
        ha="right",
        va="bottom",
    )
    ax_comp.set_xlim(2.75, 8.25)
    ax_comp.set_ylim(2.01, 2.18)
    ax_comp.set_xticks(K_EVAL)
    ax_comp.set_xlabel(r"Evaluation max merge size $K_{\mathrm{eval}}$")
    ax_comp.set_ylabel("Base tokens / compressed token")
    ax_comp.set_title("(b) Repeat-8 compression", loc="left", fontweight="bold")

    # (c) Clean M3-transfer vs M4-native comparison at identical K_eval=4.
    x = np.arange(2)
    width = 0.32
    strict_bars = ax_penalty.bar(
        x - width / 2, STRICT_TRANSFER_PCT, width,
        color=STRICT, label="Strict", zorder=3,
    )
    multi_bars = ax_penalty.bar(
        x + width / 2, MULTI_TRANSFER_PCT, width,
        color=MULTI, label="Multi-view", zorder=3,
    )
    for bars in (strict_bars, multi_bars):
        for bar in bars:
            ax_penalty.annotate(
                f"{bar.get_height():.2f}%",
                (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=7.4,
                color=INK,
            )
    coverage_positions = [(-0.10, 1.43), (1.17, 4.18)]
    coverage_alignments = ["center", "right"]
    for (x_text, y_text), alignment, coverage in zip(
        coverage_positions, coverage_alignments, LENGTH4_COVERAGE
    ):
        ax_penalty.text(
            x_text, y_text, f"{coverage:.1f}% length-4\ncoverage",
            ha=alignment, va="bottom", fontsize=7.2, color=INK2,
        )
    ax_penalty.set_xticks(x)
    ax_penalty.set_xticklabels(["Repeat-4", "Repeat-8"])
    ax_penalty.set_ylim(0, 4.85)
    ax_penalty.set_ylabel("Transfer penalty in byte PPL (%)")
    ax_penalty.set_title("(c) Matched segmentation", loc="left", fontweight="bold")
    ax_penalty.legend(
        loc="upper left", frameon=False, ncol=1, labelspacing=0.35,
        handlelength=1.4, bbox_to_anchor=(0.01, 0.99),
    )
    ax_penalty.text(
        0.5,
        -0.22,
        r"$100\!\times\![\mathrm{PPL}(M3,K=4)/\mathrm{PPL}(M4,K=4)-1]$",
        transform=ax_penalty.transAxes,
        ha="center",
        va="top",
        fontsize=7.1,
        color=INK2,
    )

    fig.subplots_adjust(left=0.072, right=0.992, top=0.90, bottom=0.24)
    return fig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("plot"))
    parser.add_argument("--stem", default="merge_size_transfer_perplexity")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    fig = make_figure()
    for suffix, kwargs in (
        ("pdf", {}),
        ("svg", {}),
        ("png", {"dpi": 300}),
    ):
        path = args.output_dir / f"{args.stem}.{suffix}"
        fig.savefig(path, bbox_inches="tight", facecolor=SURFACE, **kwargs)
        print(f"Saved {path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
