from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd


PLOT_DIR = Path(__file__).resolve().parent

DATASETS = {
    "loss": pd.read_csv(PLOT_DIR / "training_loss.csv"),
    "acc": pd.read_csv(PLOT_DIR / "training_acc.csv"),
}

RUNS = [
    ("With residual", "zeroinit", "#D0021B"),
    ("Without residual", "nores", "#4C8BF5"),
]

SMOOTH = 0.8  # Light EMA smoothing (effective span: roughly five points).


def ema(series, alpha=SMOOTH):
    return series.ewm(alpha=(1 - alpha), adjust=False).mean()


def metric_column(df, marker, metric):
    matches = [
        column
        for column in df.columns
        if marker in column and column.endswith(f" - {metric}")
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected one {marker!r} {metric} column, found {matches}"
        )
    return matches[0]


def plot_metric(ax, df, metric):
    steps = df["Step"]

    for label, marker, color in RUNS:
        values = df[metric_column(df, marker, metric)]
        smoothed = ema(values)
        final_value = smoothed.dropna().iloc[-1]
        metric_label = "loss" if metric == "loss" else "acc"

        ax.plot(
            steps,
            smoothed,
            color=color,
            linewidth=1.6,
            label=f"{label} ({metric_label}: {final_value:.3f})",
        )

    if metric == "loss":
        ax.set_title("Training Loss", fontsize=13)
        ax.set_ylabel("Loss", fontsize=12)
        ax.set_ylim(1.0, 8.0)
    else:
        ax.set_title("Training Accuracy", fontsize=13)
        ax.set_ylabel("Accuracy", fontsize=12)
        ax.set_ylim(0.0, 0.7)

    ax.set_xlabel("Step", fontsize=12)
    ax.tick_params(axis="both", labelsize=10)
    ax.legend(fontsize=9)
    ax.xaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, _: f"{int(x):,}")
    )
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(8.6, 3.8))

plot_metric(ax_loss, DATASETS["loss"], "loss")
plot_metric(ax_acc, DATASETS["acc"], "acc")

plt.tight_layout()

for extension in ("pdf", "png"):
    output = PLOT_DIR / f"residual_ablation.{extension}"
    plt.savefig(output, dpi=300, bbox_inches="tight")
    print(f"Saved -> {output}")

plt.show()
