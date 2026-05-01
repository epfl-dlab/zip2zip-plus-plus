import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

df_loss = pd.read_csv("plot/curriculum_loss.csv")
df_acc  = pd.read_csv("plot/curriculum_acc.csv")

runs = [
    ("zip2zip_1b_ms8_no_curriculum_from_scratch",  "From scratch",        "#4C8BF5", "-",  1.4),
    ("zip2zip_1b_curriculum_2_phase_finetuning",    "Curriculum\n(2 phase)", "#F5A623", "-",  2.6),
    ("zip2zip_1b_curriculum_4_phases",              "Curriculum\n (4 phase)", "#7ED321", "-",  1.4),
    ("zip2zip_1b_curriculum_rand_control_s123",     "Random\n control",      "#D0021B", "--", 1.4),
]

SMOOTH = 0.0

def ema(s, alpha=SMOOTH):
    return s.ewm(alpha=(1 - alpha), adjust=False).mean()

fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(11, 3.5))

# ── Loss ──────────────────────────────────────────────────────────────
for col_prefix, label, color, ls, lw in runs:
    steps = df_loss["Step"]
    loss  = df_loss[f"{col_prefix} - loss"]
    lo    = df_loss[f"{col_prefix} - loss__MIN"]
    hi    = df_loss[f"{col_prefix} - loss__MAX"]
    ax_loss.plot(steps, loss, color=color, linewidth=0.6, alpha=0.3, linestyle=ls)
    ax_loss.plot(steps, ema(loss), label=label, color=color, linewidth=lw, linestyle=ls)
    ax_loss.fill_between(steps, ema(lo), ema(hi), alpha=0.15, color=color)

ax_loss.set_xlabel("Step", fontsize=10)
ax_loss.set_ylabel("Loss", fontsize=10)
ax_loss.set_title("Training Loss", fontsize=11)
ax_loss.set_ylim(0, 15)
ax_loss.legend(title="Strategy", fontsize=8, title_fontsize=8)
ax_loss.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
ax_loss.grid(True, linestyle="--", alpha=0.4)

# ── Accuracy ──────────────────────────────────────────────────────────
for col_prefix, label, color, ls, lw in runs:
    steps = df_acc["Step"]
    acc   = df_acc[f"{col_prefix} - acc"]
    lo    = df_acc[f"{col_prefix} - acc__MIN"]
    hi    = df_acc[f"{col_prefix} - acc__MAX"]
    ax_acc.plot(steps, acc, color=color, linewidth=0.6, alpha=0.3, linestyle=ls)
    ax_acc.plot(steps, ema(acc), label=label, color=color, linewidth=lw, linestyle=ls)
    ax_acc.fill_between(steps, ema(lo), ema(hi), alpha=0.15, color=color)

ax_acc.set_xlabel("Step", fontsize=10)
ax_acc.set_ylabel("Accuracy", fontsize=10)
ax_acc.set_title("Next-Token Accuracy", fontsize=11)
ax_acc.set_ylim(top=0.7)
ax_acc.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
ax_acc.grid(True, linestyle="--", alpha=0.4)

plt.tight_layout()

out = "plot/curriculum.pdf"
plt.savefig(out, dpi=300, bbox_inches="tight")
plt.savefig(out.replace(".pdf", ".png"), dpi=300, bbox_inches="tight")
print(f"Saved → {out}")
