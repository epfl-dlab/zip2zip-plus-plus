"""Visualize hyper_causal_mask: training-time logit mask for codebook entries."""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap

T, K = 8, 8

full_mask   = np.ones((T, K))
causal_mask = np.tril(np.ones((T, K)))

cmap = ListedColormap(["#d9534f", "#5cb85c"])

fig, axes = plt.subplots(1, 2, figsize=(7.5, 4.2),
                         gridspec_kw={"wspace": 0.18})

titles = ["Without hyper_causal_mask", "With hyper_causal_mask"]
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
    ax.set_xticklabels([f"$k\\!=\\!{j}$" for j in range(K)], fontsize=7,
                       rotation=45, ha="right")
    ax.set_xlabel("Codebook entry index $k$", fontsize=9)
    ax.set_title(title, fontsize=10, fontweight="bold", pad=8)

    if i == 0:
        ax.set_yticklabels([f"$t\\!=\\!{j}$" for j in range(T)], fontsize=7)
        ax.set_ylabel("Sequence position $t$", fontsize=9)
    else:
        ax.set_yticklabels([f"$t\\!=\\!{j}$" for j in range(T)], fontsize=7)
        ax.set_ylabel("")

# Diagonal boundary on right panel
d = np.linspace(-0.5, K - 0.5, 300)
axes[1].plot(d, d, color="white", linewidth=2.0, linestyle="--", alpha=0.85)

# Legend
fig.legend(
    handles=[mpatches.Patch(color="#5cb85c", label="Visible (logit computed)"),
             mpatches.Patch(color="#d9534f", label="Masked ($-\\infty$)")],
    loc="lower center", ncol=2, fontsize=9,
    bbox_to_anchor=(0.5, -0.02), frameon=True, edgecolor="#cccccc",
)

fig.suptitle("Hyper-token logit mask during training",
             fontsize=11, fontweight="bold", y=1.01)
plt.tight_layout(rect=[0, 0.08, 1, 1])

out = "/mnt/scratch/zip2zip-core/scripts/hyper_causal_mask.pdf"
plt.savefig(out, bbox_inches="tight", dpi=200)
plt.savefig(out.replace(".pdf", ".png"), bbox_inches="tight", dpi=200)
print(f"Saved {out}")
