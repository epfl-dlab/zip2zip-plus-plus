"""Publication figure for hierarchical merge-size transfer.

The audited values come from the Repeat-4/Repeat-8 evaluation JSONs under
``/dlabscratch1/xinma/logs/eval``.  M3 is vx0.6.4.5 (trained with maximum
merge size 3); M4 is vx0.6.4.3 (trained with maximum merge size 4).

Every perplexity below was regenerated on 2026-09-03 with the fixed multi-view
harness, from these seven runs (``results_*<tag>*.json``):

    mvfix-m3-r8-k3, -k5, -k6, -k7, -k8   M3, WikiText x8 only, K_eval = 3,5..8
    mvfix-m3-all-k4                      M3, all three corpora, K_eval = 4
    mvfix-m4-all-k4                      M4, all three corpora, K_eval = 4

Strict perplexities reproduced the previous values exactly; only the
multi-view values moved, by 1e-6 to 2e-6.  The three compression arrays are
LZW properties of the corpus and ``K_eval`` alone -- independent of the
checkpoint and of the perplexity scorer -- so they are unchanged, and the
Repeat-8 column was re-verified against the five single-corpus runs above.

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
    1.1033719017,
    1.1489530517,
    1.1598237164,
    1.1542942817,
    1.1539484902,
    1.1538341796,
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
M4_MULTI_REPEAT8_BYTE_PPL = 1.1096440250

# Strict and exact multi-view byte PPL at matched K_eval=4 segmentation.
M3_TRANSFER_STRICT_BYTE_PPL = np.array([1.6599958493, 1.2049554316, 1.1534654876])
M3_TRANSFER_MULTI_BYTE_PPL = np.array([1.6301572635, 1.1975300953, 1.1489530517])
M4_NATIVE_STRICT_BYTE_PPL = np.array([1.6586237867, 1.1922272337, 1.1121679450])
M4_NATIVE_MULTI_BYTE_PPL = np.array([1.6301298759, 1.1863561672, 1.1096440250])


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
    ax_loss.axvspan(3.5, 8.25, color=UNSEEN, zorder=0)
    ax_loss.axvline(3.5, color=INK2, linewidth=1.0, linestyle=(0, (3, 3)), zorder=1)
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
        3.08, 1.29, "seen merge sizes",
        ha="center", va="top", fontsize=6.8, color=INK2,
    )
    ax_loss.text(
        5.85, 1.29, "unseen merge sizes",
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
    ax_loss.set_ylabel("Byte perplexity")
    ax_loss.set_title("(c) WikiText ×8 generalization", loc="left", fontweight="bold")
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
        label="WikiText ×8",
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
        label="WikiText ×4",
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
    width = 0.16
    m3_strict_bars = ax_penalty.bar(
        x - 1.75 * width, M3_TRANSFER_STRICT_BYTE_PPL, width,
        color=STRICT, edgecolor=STRICT, linewidth=1.4,
        label="M3 strict", zorder=3,
    )
    m4_strict_bars = ax_penalty.bar(
        x - 0.75 * width, M4_NATIVE_STRICT_BYTE_PPL, width,
        facecolor=SURFACE, edgecolor=STRICT, linewidth=1.4,
        hatch="///", label="M4 strict", zorder=3,
    )
    m3_multi_bars = ax_penalty.bar(
        x + 0.75 * width, M3_TRANSFER_MULTI_BYTE_PPL, width,
        color=MULTI, edgecolor=MULTI, linewidth=1.4,
        label="M3 multi-view", zorder=3,
    )
    m4_multi_bars = ax_penalty.bar(
        x + 1.75 * width, M4_NATIVE_MULTI_BYTE_PPL, width,
        facecolor=SURFACE, edgecolor=MULTI, linewidth=1.4,
        hatch="///", label="M4 multi-view", zorder=3,
    )
    for bars in (
        m3_strict_bars,
        m3_multi_bars,
        m4_strict_bars,
        m4_multi_bars,
    ):
        for bar in bars:
            height = bar.get_height()
            ax_penalty.annotate(
                f"{height:.4f}",
                (bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                rotation=90,
                fontsize=6.2,
                color=INK,
            )
    ax_penalty.set_xticks(x)
    ax_penalty.set_xticklabels([
        "WikiText",
        "WikiText ×4",
        "WikiText ×8",
    ])
    ax_penalty.set_ylim(1.0, 1.76)
    ax_penalty.set_ylabel("Byte perplexity")
    ax_penalty.set_title(
        "(b) M3: 3→4 vs. M4: 4→4*", loc="left", fontweight="bold"
    )
    ax_penalty.legend(
        loc="upper right", frameon=False, ncol=2, columnspacing=0.8,
        labelspacing=0.3, handlelength=1.3, fontsize=6.5,
        bbox_to_anchor=(1.01, 0.99),
    )
    ax_penalty.text(
        0.5,
        -0.13,
        r"* Train → evaluation max merge size; both use $K_{\mathrm{eval}}=4$",
        transform=ax_penalty.transAxes,
        ha="center",
        va="top",
        fontsize=7.1,
        color=INK2,
    )

    fig.subplots_adjust(left=0.068, right=0.992, top=0.90, bottom=0.17)
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
