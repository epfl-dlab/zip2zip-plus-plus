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
REPEAT4_COMPRESSION = np.array([1.6986079718, 1.7148797874, 1.7172041887, 1.7176766414, 1.7177561662, 1.7177743443])
WIKITEXT_COMPRESSION = np.array([1.1664272766, 1.1689825309, 1.1695630726, 1.1696598856, 1.1696598856, 1.1696598856])

# Native M4 at the matched K_eval=4 Repeat-8 segmentation.
M4_STRICT_REPEAT8_BYTE_PPL = 1.1121679450
M4_MULTI_REPEAT8_BYTE_PPL = 1.0989894566

# Matched-segmentation relative transfer penalties:
# 100 * (PPL(M3, K_eval=4) / PPL(M4, K_eval=4) - 1).
STRICT_TRANSFER_PCT = np.array([0.0827229508, 1.0675983239, 3.7132469835])
MULTI_TRANSFER_PCT = np.array([-0.0233673253, 0.2524785799, 0.3347138756])
# Fraction of scored compressed targets whose expansion length is exactly 4.
LENGTH4_TARGET_SHARE = np.array([0.3974632174, 2.8073336482, 15.9927013087])


INK = "#171717"
INK2 = "#575757"
GRID = "#dedede"
SURFACE = "#ffffff"
STRICT = "#0072B2"       # Okabe-Ito blue
MULTI = "#009E73"        # Okabe-Ito green
COMP = "#6B7280"
REPEAT4 = "#E69F00"      # Okabe-Ito orange
WIKITEXT = "#CC79A7"     # Okabe-Ito reddish purple
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
        figsize=(11.2, 3.35),
        gridspec_kw={"width_ratios": [1.0, 1.15, 1.35], "wspace": 0.38},
        facecolor=SURFACE,
    )
    ax_comp, ax_penalty, ax_loss = axes
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
    ax_loss.text(
        5.72, 1.29, "unseen merge sizes",
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
    ax_loss.annotate(
        "M4 multi-view",
        xy=(4, M4_MULTI_REPEAT8_BYTE_PPL),
        xytext=(4.28, 1.091),
        arrowprops={"arrowstyle": "-", "color": MULTI, "lw": 0.8},
        color=MULTI,
        fontsize=7.5,
        ha="left",
        va="top",
    )
    ax_loss.set_ylim(0.95, 1.30)
    loss_ticks = np.arange(0.95, 1.301, 0.05)
    ax_loss.set_yticks(loss_ticks)
    ax_loss.set_yticklabels([f"{tick:.3f}" for tick in loss_ticks])
    ax_loss.set_xticks(K_EVAL)
    ax_loss.set_xlabel(r"Evaluation max merge size $K_{\mathrm{eval}}$")
    ax_loss.set_ylabel("Byte perplexity (lower is better)")
    ax_loss.set_title("(c) Repeat-8 extrapolation", loc="left", fontweight="bold")
    ax_loss.legend(
        loc="lower right",
        bbox_to_anchor=(1.01, 0.03),
        frameon=False,
        handlelength=2.3,
        borderaxespad=0,
    )

    # (b) Compression saturates at the same time as the loss curve.
    ax_comp.plot(
        K_EVAL,
        COMPRESSION,
        color=COMP,
        linewidth=2.1,
        marker="o",
        markersize=5.2,
        markeredgecolor=SURFACE,
        markeredgewidth=0.9,
        label="Repeat-8",
        zorder=4,
    )
    ax_comp.plot(
        K_EVAL,
        REPEAT4_COMPRESSION,
        color=REPEAT4,
        linewidth=1.9,
        linestyle=(0, (4, 2)),
        marker="o",
        markersize=4.8,
        markeredgecolor=SURFACE,
        markeredgewidth=0.9,
        label="Repeat-4",
        zorder=4,
    )
    ax_comp.plot(
        K_EVAL,
        WIKITEXT_COMPRESSION,
        color=WIKITEXT,
        linewidth=1.9,
        linestyle=(0, (1.5, 1.5)),
        marker="o",
        markersize=4.8,
        markeredgecolor=SURFACE,
        markeredgewidth=0.9,
        label="WikiText",
        zorder=4,
    )
    ax_comp.set_xlim(2.75, 8.25)
    # A ratio of 1 is the natural no-compression reference.
    ax_comp.set_ylim(1.00, 2.35)
    ax_comp.set_xticks(K_EVAL)
    ax_comp.set_xlabel(r"Evaluation max merge size $K_{\mathrm{eval}}$")
    ax_comp.set_ylabel("Base tokens / compressed token")
    ax_comp.set_title("(a) Compression by corpus", loc="left", fontweight="bold")
    ax_comp.legend(loc="center right", bbox_to_anchor=(1.01, 0.34), frameon=False)

    # Matched M3-transfer vs. M4-native comparison at identical K_eval=4.
    x = np.arange(3)
    width = 0.28
    strict_bars = ax_penalty.bar(
        x - width / 2, STRICT_TRANSFER_PCT, width,
        color=STRICT, label="Strict", zorder=3,
    )
    multi_bars = ax_penalty.bar(
        x + width / 2, MULTI_TRANSFER_PCT, width,
        color=MULTI, label="Multi-view", zorder=3,
    )
    for bars in (strict_bars, multi_bars):
        x_offset = -4 if bars is strict_bars else 4
        for bar in bars:
            height = bar.get_height()
            label = "≈0%" if abs(height) < 0.05 else f"{height:.2f}%"
            ax_penalty.annotate(
                label,
                (bar.get_x() + bar.get_width() / 2, max(height, 0.0)),
                xytext=(x_offset, 8),
                textcoords="offset points",
                ha="center",
                va="bottom",
                bbox={"facecolor": SURFACE, "edgecolor": "none", "pad": 0.25},
                fontsize=7.4,
                color=INK,
            )
    ax_penalty.set_xticks(x)
    ax_penalty.set_xticklabels([
        f"WikiText\n({LENGTH4_TARGET_SHARE[0]:.1f}%)",
        f"Repeat-4\n({LENGTH4_TARGET_SHARE[1]:.1f}%)",
        f"Repeat-8\n({LENGTH4_TARGET_SHARE[2]:.1f}%)",
    ])
    ax_penalty.set_ylim(-0.12, 4.45)
    ax_penalty.set_ylabel("Relative byte-PPL difference: M3 vs. M4 (%)")
    ax_penalty.set_title("(b) Matched segmentation", loc="left", fontweight="bold")
    ax_penalty.legend(
        loc="upper left", frameon=False, ncol=1, labelspacing=0.35,
        handlelength=1.4, bbox_to_anchor=(0.01, 0.99),
    )
    ax_penalty.text(
        0.5,
        -0.20,
        "Parentheses: share of length-4 targets",
        transform=ax_penalty.transAxes,
        ha="center",
        va="top",
        fontsize=6.8,
        color=INK2,
    )
    ax_penalty.text(
        0.5,
        -0.32,
        r"$100\!\times\![\mathrm{PPL}(M3,K=4)/\mathrm{PPL}(M4,K=4)-1]$",
        transform=ax_penalty.transAxes,
        ha="center",
        va="top",
        fontsize=7.1,
        color=INK2,
    )

    fig.subplots_adjust(left=0.068, right=0.992, top=0.90, bottom=0.27)
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
