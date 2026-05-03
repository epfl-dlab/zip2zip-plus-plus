"""Visualize hyper_causal_mask: training-time logit mask for codebook entries."""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap

T, K = 8, 8

full_mask   = np.ones((T, K))
causal_mask = np.tril(np.ones((T, K)))

cmap = ListedColormap(["#d9534f", "#5cb85c"])

fig, axes = plt.subplots(1, 2, figsize=(7.5, 4.4),
                         gridspec_kw={"wspace": 0.18})

titles = ["Without LZW causal mask", "With LZW causal mask"]
masks  = [full_mask, causal_mask]

for i, (ax, mask, title) in enumerate(zip(axes, masks, titles)):
    ax.imshow(mask, cmap=cmap, vmin=0, vmax=1, origin="upper", aspect="auto")
    ax.set_box_aspect(T / K)

    # minor grid lines
    ax.set_xticks(np.arange(-0.5, K, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, T, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.2)
    ax.tick_params(which="minor", length=0)

    ax.set_xticks(np.arange(K))
    ax.set_yticks(np.arange(T))
    ax.set_xticklabels([f"$k\\!=\\!{j}$" for j in range(K)], fontsize=12,
                       rotation=45, ha="right")
    ax.set_xlabel("Codebook entry index $k$", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)

    if i == 0:
        ax.set_yticklabels([f"$t\\!=\\!{j}$" for j in range(T)], fontsize=12)
        ax.set_ylabel("Sequence position $t$", fontsize=12)
    else:
        ax.set_yticklabels([f"$t\\!=\\!{j}$" for j in range(T)], fontsize=12)
        ax.set_ylabel("")

# Diagonal boundary on right panel
d = np.linspace(-0.5, K - 0.5, 300)
axes[1].plot(d, d, color="white", linewidth=2.0, linestyle="--", alpha=0.85)

# Legend
fig.legend(
    handles=[mpatches.Patch(color="#5cb85c", label="Visible (logit computed)"),
             mpatches.Patch(color="#d9534f", label="Masked ($-\\infty$)")],
    loc="lower center", ncol=2, fontsize=16, prop={"weight": "bold"},
    bbox_to_anchor=(0.5, -0.08), frameon=True, edgecolor="#cccccc",
)


plt.tight_layout(rect=[0, 0.08, 1, 1])

out = "/mnt/scratch/zip2zip-core/scripts/hyper_causal_mask.pdf"
plt.savefig(out, bbox_inches="tight", pad_inches=0.2, dpi=200)
plt.savefig(out.replace(".pdf", ".png"), bbox_inches="tight", pad_inches=0.2, dpi=600)
print(f"Saved {out}")
