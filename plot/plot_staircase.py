import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.rcParams.update({
    "font.family": "serif",
    "font.size": 16,
    "axes.labelsize": 18,
    "axes.titlesize": 19,
    "legend.fontsize": 14,
    "xtick.labelsize": 15,
    "ytick.labelsize": 15,
    "figure.dpi": 150,
})

# Flat (F) accuracies per ms setting
# Size 1 = Base token accuracy from the table
f_data = {
    3: {1: 39.0, 2: 32.5, 3: 58.6},
    4: {1: 37.8, 2: 31.5, 3: 41.0, 4: 67.5},
    5: {1: 38.5, 2: 32.2, 3: 41.4, 4: 50.0, 5: 73.9},
    6: {1: 38.8, 2: 32.4, 3: 42.0, 4: 51.5, 5: 55.9, 6: 80.8},
    7: {1: 38.1, 2: 31.0, 3: 39.9, 4: 49.4, 5: 54.4, 6: 57.9, 7: 82.9},
    8: {1: 38.1, 2: 31.7, 3: 40.6, 4: 49.9, 5: 54.5, 6: 59.1, 7: 62.0, 8: 86.9},
}

# Compression ratios from W&B at step 7270 (flat encoder, general text)
compression_general = {
    1: 1.00024,
    2: 1.25524,
    3: 1.26767,
    4: 1.27082,
    5: 1.27190,
    6: 1.27220,
    7: 1.27234,
    8: 1.27236,
}

# Simulated compression ratios for code data (fake, similar trend, peaks ~1.45)
compression_code = {
    1: 1.00030,
    2: 1.38200,
    3: 1.42100,
    4: 1.43800,
    5: 1.44500,
    6: 1.44800,
    7: 1.44950,
    8: 1.45000,
}

# Confusion matrix data from the image (ms=8, 7x7: t1-t7 vs p1-p7)
confusion_matrix = np.array([
    [0.937, 0.053, 0.007, 0.002, 0.001, 0.000, 0.000],
    [0.586, 0.382, 0.024, 0.005, 0.001, 0.001, 0.001],
    [0.351, 0.173, 0.437, 0.028, 0.006, 0.002, 0.002],
    [0.225, 0.096, 0.112, 0.522, 0.031, 0.007, 0.007],
    [0.155, 0.063, 0.050, 0.105, 0.570, 0.036, 0.021],
    [0.107, 0.045, 0.030, 0.037, 0.097, 0.615, 0.070],
    [0.042, 0.018, 0.012, 0.010, 0.013, 0.032, 0.872],
])

sizes = list(range(1, 9))

cmap = plt.cm.viridis
colors = [cmap(i / 6) for i in range(6)]

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 5.4),
                                     gridspec_kw={"width_ratios": [1, 1, 1.2], "wspace": 0.35})

# --- Panel (a): Accuracy staircase ---
for idx, (ms, accs) in enumerate(sorted(f_data.items())):
    xs = sorted(accs.keys())
    ys = [accs[s] for s in xs]
    color = colors[idx]

    stair_x = []
    stair_y = []
    for i, (x, y) in enumerate(zip(xs, ys)):
        if i == 0:
            stair_x.append(x - 0.4)
            stair_y.append(y)
        stair_x.append(x + 0.4)
        stair_y.append(y)
        if i < len(xs) - 1:
            stair_x.append(xs[i + 1] - 0.4)
            stair_y.append(ys[i + 1])

    ax1.plot(stair_x, stair_y, color=color, linewidth=2.0, alpha=0.85,
             label=f"ms={ms}")
    ax1.scatter(xs, ys, color=color, s=40, zorder=5, edgecolors="white", linewidths=0.5)

    max_s = max(xs)
    ax1.scatter([max_s], [accs[max_s]], color=color, s=100, zorder=6,
                edgecolors="black", linewidths=1.2, marker="D")

ax1.set_xlabel("Hyper Token Size")
ax1.set_ylabel("Accuracy (%)")
ax1.set_xticks(sizes)
ax1.set_xticklabels([str(s) for s in sizes])
ax1.set_title("(a) Accuracy by Hyper Token Size")
ax1.grid(axis="y", alpha=0.3)
ax1.legend(loc="upper left", framealpha=0.9)

# --- Panel (b): Compression ratio staircase ---
def draw_cr_staircase(ax, cr_dict, color, label, annotate_offset=10):
    ms_vals = sorted(cr_dict.keys())
    cr_vals = [cr_dict[ms] for ms in ms_vals]

    stair_x = []
    stair_y = []
    for i, (x, y) in enumerate(zip(ms_vals, cr_vals)):
        if i == 0:
            stair_x.append(x - 0.4)
            stair_y.append(y)
        stair_x.append(x + 0.4)
        stair_y.append(y)
        if i < len(ms_vals) - 1:
            stair_x.append(ms_vals[i + 1] - 0.4)
            stair_y.append(cr_vals[i + 1])

    ax.plot(stair_x, stair_y, color=color, linewidth=2.5, alpha=0.9, label=label)
    ax.scatter(ms_vals, cr_vals, color=color, s=60, zorder=5,
               edgecolors="white", linewidths=0.8)

    for ms, cr in zip(ms_vals, cr_vals):
        ax.annotate(f"{cr:.2f}", (ms, cr), textcoords="offset points",
                    xytext=(0, annotate_offset), ha="center", fontsize=14, color=color)

draw_cr_staircase(ax2, compression_general, "#2c7bb6", "general", annotate_offset=-15)
draw_cr_staircase(ax2, compression_code, "#d7191c", "code (fake)", annotate_offset=10)

ms_values = sorted(compression_general.keys())
ax2.set_xlabel("LZW Max Merge Size")
ax2.set_ylabel("Compression Ratio")
ax2.set_xticks(ms_values)
ax2.set_xticklabels([str(s) for s in ms_values])
ax2.set_title("(b) Compression Ratio by Max Merge Size")
ax2.set_ylim(0.95, 1.5)
ax2.grid(axis="y", alpha=0.3)
ax2.legend(loc="lower right", framealpha=0.9)

# --- Panel (c): Merge Size Confusion Matrix ---
labels = [f"p{i}" for i in range(1, 8)]
true_labels = [f"t{i}" for i in range(1, 8)]

im = ax3.imshow(confusion_matrix, cmap="YlOrRd", vmin=0, vmax=1, aspect="equal")

ax3.set_xticks(range(7))
ax3.set_xticklabels(labels, fontsize=15)
ax3.set_yticks(range(7))
ax3.set_yticklabels(true_labels, fontsize=15)
ax3.set_xlabel("Predicted merge size")
ax3.set_ylabel("True merge size")
ax3.set_title("(c) Merge Size Confusion Matrix")

for i in range(7):
    for j in range(7):
        val = confusion_matrix[i, j]
        text_color = "white" if val > 0.5 else "black"
        ax3.text(j, i, f"{val:.2f}", ha="center", va="center",
                 fontsize=13, color=text_color)

cbar = fig.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)
cbar.set_label("Row-normalized ratio", fontsize=15)
cbar.ax.tick_params(labelsize=13)

fig.tight_layout()
fig.savefig("plot/staircase_accuracy.png", bbox_inches="tight", dpi=150)
fig.savefig("plot/staircase_accuracy.pdf", bbox_inches="tight")
print("Saved plot/staircase_accuracy.png and plot/staircase_accuracy.pdf")
