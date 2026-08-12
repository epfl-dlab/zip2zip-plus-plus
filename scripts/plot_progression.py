"""Per-metric progression plots for the zip2zip-core recipe line.

One PNG per metric: x = recipe version (equally spaced), y = score, plus two
dashed reference lines (the uncompressed baseline and the released HF model as
evaluated by us). Every figure uses the same figure size AND the same fixed axes
margins, so the plots are exactly superimposable — flipping between them keeps
each version on the same pixel column.

    uv run --with matplotlib python scripts/plot_progression.py [outdir]

TO ADD A NEW VERSION: append its id to VERSIONS (and to LINEAGE unless it is an
abandoned branch), then add one block to SCORES. Nothing is positional, so a
missing metric just leaves a gap rather than shifting other points.

Where the numbers come from (RCP): the pipeline's final eval JSON for each run,
    $SCRATCH/logs/eval/results_<RUN_NAME>_step8000_default_<ts>.json     # MC + GSM8K
    $SCRATCH/logs/eval/results_<RUN_NAME>_step8000_wikitext_<ts>.json    # byte ppl
read with results[task][metric]; acc_norm where the paper's Table 3 uses it, acc
for WinoGrande, flexible-extract for GSM8K.
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# ── palette (dataviz reference instance, light mode) ──────────────────────────
SERIES = "#2a78d6"    # slot 1 — the recipe lineage
RELEASED = "#eb6834"  # slot 2 — the released model reference
DEAD = "#8f8e88"      # muted ink — abandoned branches
BASE_C = "#52514e"    # text-secondary — the baseline threshold
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"

# ── data ─────────────────────────────────────────────────────────────────────
VERSIONS = ["v0.2", "v0.3", "v0.4", "v0.5", "v0.6", "v0.6.1", "v0.6.2", "v0.6.3", "v0.6.4"]
# v0.6 (warm-start) and v0.6.1 (deeper encoder) both branched off v0.5 and were
# abandoned as negative results; the champion lineage skips them.
LINEAGE = ["v0.2", "v0.3", "v0.4", "v0.5", "v0.6.2", "v0.6.3", "v0.6.4"]
ABANDONED = ["v0.6", "v0.6.1"]

SCORES = {
    "v0.2":   dict(gsm8k=0.3556, arc_c=0.5410, arc_e=0.8089, hellaswag=0.7116,
                   openbookqa=0.4600, piqa=0.7992, winogrande=0.7466, wikitext=1.6951),
    "v0.3":   dict(gsm8k=0.3791, arc_c=0.4991, arc_e=0.7963, hellaswag=0.7023,
                   openbookqa=0.4600, piqa=0.7922, winogrande=0.7285, wikitext=1.7187),
    "v0.4":   dict(gsm8k=0.6096, arc_c=0.5401, arc_e=0.8068, hellaswag=0.7024,
                   openbookqa=0.4700, piqa=0.7894, winogrande=0.7127, wikitext=1.7105),
    "v0.5":   dict(gsm8k=0.6232, arc_c=0.5606, arc_e=0.8237, hellaswag=0.7250,
                   openbookqa=0.4740, piqa=0.7900, winogrande=0.7427, wikitext=1.6857),
    "v0.6":   dict(gsm8k=0.5565, arc_c=0.5529, arc_e=0.8270, hellaswag=0.7218,
                   openbookqa=0.4600, piqa=0.7965, winogrande=0.7443, wikitext=1.6927),
    "v0.6.1": dict(gsm8k=0.5974, arc_c=0.5486, arc_e=0.8178, hellaswag=0.7158,
                   openbookqa=0.4620, piqa=0.7971, winogrande=0.7490, wikitext=1.6858),
    "v0.6.2": dict(gsm8k=0.6505, arc_c=0.5452, arc_e=0.8190, hellaswag=0.7127,
                   openbookqa=0.4640, piqa=0.7976, winogrande=0.7388, wikitext=1.6866),
    "v0.6.3": dict(gsm8k=0.6520, arc_c=0.5512, arc_e=0.8224, hellaswag=0.7145,
                   openbookqa=0.4720, piqa=0.7927, winogrande=0.7451, wikitext=1.6918),
    "v0.6.4": dict(gsm8k=0.6770, arc_c=0.5700, arc_e=0.8304, hellaswag=0.7233,
                   openbookqa=0.4660, piqa=0.8003, winogrande=0.7443, wikitext=1.6574),
}

# Uncompressed continual-pretraining control, same data/recipe (MAX_CODEBOOK_SIZE=0).
# results_andrea-ctrl-phi35-4B-codebook0-1BData-v0.4-Zip2zipCore_step8000_default_20260716_161119
BASELINE = dict(gsm8k=0.7415, arc_c=0.5930, arc_e=0.8443, hellaswag=0.7372,
                openbookqa=0.4780, piqa=0.8161, winogrande=0.7514, wikitext=1.5777)

# The released model epfl-dlab/zip2zip-Phi-3.5-mini-instruct-v0.1, evaluated by us
# through the same harness: logs/eval/old/results_z2z_default_20260630_130627.json
# (+ results_zip2zip-Phi-3.5-mini-instruct-v0.1_perplexity_wikitext_20260713_145506
# for byte ppl). NB: a slide once transcribed WinoGrande as 0.74; the log says 0.7332.
RELEASED_V01 = dict(gsm8k=0.3791, arc_c=0.5640, arc_e=0.8224, hellaswag=0.7202,
                    openbookqa=0.4780, piqa=0.7976, winogrande=0.7332, wikitext=1.7367)

META = {
    "gsm8k":      ("GSM8K", "exact_match, flexible-extract", "Grade-school math, generative — higher is better", "pct"),
    "arc_c":      ("ARC-Challenge", "acc_norm", "Science QA, multiple choice — higher is better", "pct"),
    "arc_e":      ("ARC-Easy", "acc_norm", "Science QA, multiple choice — higher is better", "pct"),
    "hellaswag":  ("HellaSwag", "acc_norm", "Commonsense sentence completion — higher is better", "pct"),
    "openbookqa": ("OpenBookQA", "acc_norm", "Open-book science QA — higher is better", "pct"),
    "piqa":       ("PIQA", "acc_norm", "Physical commonsense — higher is better", "pct"),
    "winogrande": ("WinoGrande", "acc", "Coreference resolution — higher is better", "pct"),
    "wikitext":   ("WikiText", "byte perplexity", "Language modelling — LOWER is better", "ppl"),
}

# Fixed geometry — identical in every figure so the plots superimpose exactly.
FIGSIZE = (7.6, 4.8)
MARGINS = dict(left=0.105, right=0.975, top=0.762, bottom=0.126)

xi = {v: i for i, v in enumerate(VERSIONS)}


def tick_fmt(kind, span):
    """Enough precision that no two ticks render the same label."""
    if kind == "pct":
        dec = 1 if span * 100 < 6 else 0
        return FuncFormatter(lambda v, _: f"{v * 100:.{dec}f}%")
    return FuncFormatter(lambda v, _: f"{v:.3f}")


def plot_metric(key, outdir):
    name, unit, subtitle, kind = META[key]
    vals = {v: SCORES[v][key] for v in VERSIONS if key in SCORES[v]}
    base, rel = BASELINE[key], RELEASED_V01[key]
    lower_better = kind == "ppl"

    fig = plt.figure(figsize=FIGSIZE, facecolor=SURFACE)
    fig.subplots_adjust(**MARGINS)
    ax = fig.add_subplot(111)
    ax.set_facecolor(SURFACE)

    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.xaxis.grid(False)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
        ax.spines[s].set_linewidth(0.8)
    ax.tick_params(colors=INK2, labelsize=9.5, length=0, pad=6)

    # reference thresholds (dashed is correct: these ARE thresholds)
    ax.axhline(rel, color=RELEASED, linewidth=1.4, linestyle=(0, (6, 3, 1, 3)),
               zorder=2, label="released v0.1 (our eval)")
    ax.axhline(base, color=BASE_C, linewidth=1.4, linestyle=(0, (5, 4)),
               zorder=2, label="baseline (uncompressed)")

    # champion lineage
    lx = [xi[v] for v in LINEAGE if v in vals]
    ly = [vals[v] for v in LINEAGE if v in vals]
    ax.plot(lx, ly, color=SERIES, linewidth=2.0, zorder=4,
            solid_capstyle="round", label="recipe lineage")
    ax.plot(lx, ly, marker="o", markersize=8, linestyle="none", color=SERIES,
            markeredgecolor=SURFACE, markeredgewidth=2.0, zorder=5)

    # abandoned branches: dotted spurs from v0.5 make the branch explicit
    for v in ABANDONED:
        if v in vals:
            ax.plot([xi["v0.5"], xi[v]], [vals["v0.5"], vals[v]], color=DEAD,
                    linewidth=1.0, linestyle=(0, (1, 2.5)), zorder=2.5)
    axs = [xi[v] for v in ABANDONED if v in vals]
    ays = [vals[v] for v in ABANDONED if v in vals]
    ax.plot(axs, ays, marker="o", markersize=7.5, linestyle="none",
            markerfacecolor=SURFACE, markeredgecolor=DEAD, markeredgewidth=1.8,
            zorder=3, label="abandoned (branched off v0.5)")

    allv = list(vals.values()) + [base, rel]
    lo, hi = min(allv), max(allv)
    pad = (hi - lo) * 0.14 or 0.01
    ax.set_ylim(lo - pad, hi + pad * 1.25)
    ax.set_xlim(-0.45, len(VERSIONS) - 0.55)
    ax.set_xticks(range(len(VERSIONS)))
    ax.set_xticklabels(VERSIONS)
    ax.yaxis.set_major_formatter(tick_fmt(kind, hi - lo))

    # selective direct labels: the champion endpoint and the two references
    champ = LINEAGE[-1]
    cy = vals[champ]
    lab = f"{cy * 100:.1f}%" if kind == "pct" else f"{cy:.4f}"
    ax.annotate(lab, (xi[champ], cy), textcoords="offset points",
                xytext=(0, -20 if lower_better else 14), ha="center",
                fontsize=10.5, fontweight="bold", color=INK, zorder=6)
    # The two reference lines can coincide (e.g. OpenBookQA, both .478); drop the
    # released label below its line so the two never render on top of each other.
    span = ax.get_ylim()[1] - ax.get_ylim()[0]
    rel_dy = -13 if abs(base - rel) < 0.05 * span else 5
    for val, col, txt, dy in ((base, BASE_C, "baseline", 5),
                              (rel, RELEASED, "released v0.1", rel_dy)):
        s = f"{val * 100:.1f}%" if kind == "pct" else f"{val:.4f}"
        ax.annotate(f"{txt} {s}", (len(VERSIONS) - 0.62, val),
                    textcoords="offset points", xytext=(0, dy), ha="right",
                    fontsize=9, color=col, zorder=6)

    fig.text(MARGINS["left"], 0.955, name, fontsize=15, fontweight="bold",
             color=INK, ha="left", va="center")
    fig.text(MARGINS["left"], 0.902, subtitle, fontsize=9.5, color=INK2,
             ha="left", va="center")
    ax.set_ylabel(unit, fontsize=9.5, color=INK2, labelpad=8)

    leg = ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.015), ncol=2,
                    frameon=False, fontsize=8.8, handlelength=2.1,
                    columnspacing=1.8, borderaxespad=0.0)
    for t in leg.get_texts():
        t.set_color(INK2)

    path = os.path.join(outdir, f"{key}.png")
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    return path


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "plots"
    os.makedirs(outdir, exist_ok=True)
    for key in META:
        print("wrote", plot_metric(key, outdir))


if __name__ == "__main__":
    main()
