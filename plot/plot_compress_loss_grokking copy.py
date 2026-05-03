import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

df_loss = pd.read_csv("compress_loss_grokking.csv")
df_acc  = pd.read_csv("compress_acc_grokking.csv")

models = [
    ("zip2zip_20M_compress_ms4",  "20M",  "#4C8BF5"),
    ("zip2zip_50M_compress_ms4",  "50M",  "#F5A623"),
    ("zip2zip_150M_compress_ms4", "150M", "#7ED321"),
    ("zip2zip_1b_compress_ms4",   "1B",   "#D0021B"),
]

SMOOTH = 0.0  # EMA factor, higher = smoother

def ema(s, alpha=SMOOTH):
    return s.ewm(alpha=(1 - alpha), adjust=False).mean()

fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(11, 3.5))

# ── Loss ──────────────────────────────────────────────────────────────
for col_prefix, label, color in models:
    steps = df_loss["Step"]
    loss  = df_loss[f"{col_prefix} - loss"]
    lo    = df_loss[f"{col_prefix} - loss__MIN"]
    hi    = df_loss[f"{col_prefix} - loss__MAX"]
    ax_loss.plot(steps, loss, color=color, linewidth=0.6, alpha=0.3)
    ax_loss.plot(steps, ema(loss), label=label, color=color, linewidth=1.4)
    ax_loss.fill_between(steps, ema(lo), ema(hi), alpha=0.15, color=color)

ax_loss.axhline(y=4.8, color="gray", linestyle="--", linewidth=1.0, label="Grokking (4.8)")
ax_loss.set_xlabel("Step", fontsize=10)
ax_loss.set_ylabel("Loss", fontsize=10)
ax_loss.set_title("Compression Loss", fontsize=11)
ax_loss.set_ylim(top=20)
ax_loss.legend(title="Model", fontsize=8, title_fontsize=8)
ax_loss.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
ax_loss.grid(True, linestyle="--", alpha=0.4)

# ── Accuracy ──────────────────────────────────────────────────────────
for col_prefix, label, color in models:
    steps = df_acc["Step"]
    acc   = df_acc[f"{col_prefix} - acc"]
    lo    = df_acc[f"{col_prefix} - acc__MIN"]
    hi    = df_acc[f"{col_prefix} - acc__MAX"]
    ax_acc.plot(steps, acc, color=color, linewidth=0.6, alpha=0.3)
    ax_acc.plot(steps, ema(acc), label=label, color=color, linewidth=1.4)
    ax_acc.fill_between(steps, ema(lo), ema(hi), alpha=0.15, color=color)

ax_acc.set_xlabel("Step", fontsize=10)
ax_acc.set_ylabel("Accuracy", fontsize=10)
ax_acc.set_title("Compression Accuracy", fontsize=11)
ax_acc.legend(title="Model", fontsize=8, title_fontsize=8)
ax_acc.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
ax_acc.grid(True, linestyle="--", alpha=0.4)

plt.tight_layout()

out = "compress_grokking.pdf"
plt.savefig(out, dpi=300, bbox_inches="tight")
print(f"Saved → {out}")
plt.show()
