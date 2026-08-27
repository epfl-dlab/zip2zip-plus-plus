from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd


PLOT_DIR = Path(__file__).resolve().parent

DATASETS = {
    "compression": {
        "loss": pd.read_csv(PLOT_DIR / "compression_loss.csv"),
        "acc": pd.read_csv(PLOT_DIR / "compression_acc.csv"),
    },
    "decompression": {
        "loss": pd.read_csv(PLOT_DIR / "decompression_loss.csv"),
        "acc": pd.read_csv(PLOT_DIR / "decompression_acc.csv"),
    },
}

# Keep the original colors for 20M--1B; purple is added for the new 7M model.
MODELS = [
    ("7M", "7M", "#9467BD"),
    ("20M", "20M", "#4C8BF5"),
    ("50M", "50M", "#F5A623"),
    ("150M", "150M", "#7ED321"),
    ("1B", "1B", "#D0021B"),
]

SMOOTH = 0.8  # Light EMA smoothing (effective span: roughly five points).
GROKKING_LOSS = {
    "compress": 4.0,
    "decompress": 4.4,
}


def ema(series, alpha=SMOOTH):
    return series.ewm(alpha=(1 - alpha), adjust=False).mean()


def metric_columns(df, model_size, direction, metric):
    """Find a W&B export column without depending on its run hash."""
    marker = f"Phi-{model_size}_{direction}_"
    suffix = f" - direction/{direction}_{metric}"
    matches = [
        column
        for column in df.columns
        if marker in column and column.endswith(suffix)
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected one {model_size} {direction}_{metric} column, found {matches}"
        )

    value = matches[0]
    return value, f"{value}__MIN", f"{value}__MAX"


def plot_metric(ax, df, direction, metric):
    steps = df["Step"]

    for model_size, label, color in MODELS:
        value_col, min_col, max_col = metric_columns(
            df, model_size, direction, metric
        )
        values = df[value_col]
        lower = df[min_col]
        upper = df[max_col]

        ax.plot(steps, values, color=color, linewidth=0.6, alpha=0.3)
        ax.plot(
            steps,
            ema(values),
            label=label,
            color=color,
            linewidth=1.4,
        )
        ax.fill_between(
            steps,
            ema(lower),
            ema(upper),
            color=color,
            alpha=0.15,
        )

    title_direction = {
        "compress": "Compression",
        "decompress": "Decompression",
    }[direction]
    if metric == "loss":
        grokking_loss = GROKKING_LOSS[direction]
        ax.axhline(
            y=grokking_loss,
            color="gray",
            linestyle="--",
            linewidth=1.0,
            label=f"Grokking (~{grokking_loss:g})",
        )
        ax.set_ylabel("Loss", fontsize=10)
        ax.set_title(f"{title_direction} Loss", fontsize=11)
        ax.set_ylim(top=15)
    else:
        ax.set_ylabel("Accuracy", fontsize=10)
        ax.set_title(f"{title_direction} Accuracy", fontsize=11)
        ax.set_ylim(-0.05, 1.05)

    ax.set_xlabel("Step", fontsize=10)
    legend_location = "upper right" if metric == "loss" else "upper left"
    ax.legend(
        title="Model",
        fontsize=8,
        title_fontsize=8,
        loc=legend_location,
    )
    ax.xaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, _: f"{int(x):,}")
    )
    ax.grid(True, linestyle="--", alpha=0.4)


fig, axes = plt.subplots(2, 2, figsize=(11, 7))

plot_metric(axes[0, 0], DATASETS["compression"]["loss"], "compress", "loss")
plot_metric(axes[0, 1], DATASETS["compression"]["acc"], "compress", "acc")
plot_metric(
    axes[1, 0], DATASETS["decompression"]["loss"], "decompress", "loss"
)
plot_metric(
    axes[1, 1], DATASETS["decompression"]["acc"], "decompress", "acc"
)

plt.tight_layout()

for extension in ("pdf", "png"):
    output = PLOT_DIR / f"phi_compression_decompression_grokking.{extension}"
    plt.savefig(output, dpi=300, bbox_inches="tight")
    print(f"Saved -> {output}")

plt.show()
