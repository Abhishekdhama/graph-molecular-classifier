"""
Generate a publication-quality Jaccard stability bar chart.
Reads comparison_summary.json and outputs figures/jaccard_stability_chart.png
"""

import json
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['Segoe UI', 'Arial', 'DejaVu Sans']

# Load data
with open("results/comparison_summary.json") as f:
    data = json.load(f)

methods = list(data.keys())
scores  = [data[m]["mean_jaccard_across_mols"] for m in methods]

# Per-molecule std for error bars
per_mol_means = {m: [v["mean_jaccard"] for v in data[m]["per_mol"].values()] for m in methods}
stds = [np.std(per_mol_means[m]) for m in methods]

# Colors — warm palette matching the YlOrRd heatmaps
colors = ['#e8453c', '#f59e42', '#c0392b']  # red, amber, dark red
edge_colors = ['#b5342a', '#c07a2e', '#8e2a1f']

fig, ax = plt.subplots(figsize=(8, 5))
fig.patch.set_facecolor('#f8f8f8')
ax.set_facecolor('#f8f8f8')

bars = ax.bar(
    methods, scores,
    color=colors, edgecolor=edge_colors, linewidth=1.5,
    width=0.55, yerr=stds, capsize=6,
    error_kw=dict(lw=1.5, color='#333333', capthick=1.5),
    zorder=3
)

# Value labels on bars
for bar, score in zip(bars, scores):
    ax.text(
        bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.03,
        f"{score:.3f}",
        ha='center', va='bottom', fontsize=14, fontweight='bold', color='#1a1a1a'
    )

# Reference lines
ax.axhline(y=1.0, color='#2d7a3a', linestyle='--', alpha=0.4, lw=1.2, zorder=1)
ax.text(2.35, 1.01, "Perfect stability (1.0)", fontsize=8, color='#2d7a3a', alpha=0.7)

ax.axhline(y=0.0, color='#c0392b', linestyle='--', alpha=0.3, lw=1.0, zorder=1)

# Labels and title
ax.set_ylabel("Mean Jaccard Overlap (top-3 nodes)", fontsize=11, fontweight='bold', labelpad=10)
ax.set_title(
    "Explainability Stability Across 10 Random Seeds\n"
    "Higher = same explanation regardless of training seed",
    fontsize=13, fontweight='bold', pad=15, color='#1a1a1a'
)

ax.set_ylim(0, 0.72)
ax.set_yticks(np.arange(0, 0.8, 0.1))
ax.tick_params(axis='both', labelsize=10)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_color('#cccccc')
ax.spines['bottom'].set_color('#cccccc')
ax.yaxis.grid(True, alpha=0.3, linestyle='-', color='#cccccc', zorder=0)

# Winner annotation
best_idx = np.argmax(scores)
ax.annotate(
    "Most stable",
    xy=(best_idx, scores[best_idx] + stds[best_idx] + 0.01),
    xytext=(best_idx, scores[best_idx] + stds[best_idx] + 0.08),
    fontsize=9, fontweight='bold', color='#2d7a3a',
    ha='center',
    arrowprops=dict(arrowstyle='->', color='#2d7a3a', lw=1.5)
)

plt.tight_layout()
out_path = "figures/jaccard_stability_chart.png"
plt.savefig(out_path, dpi=200, bbox_inches='tight', facecolor='#f8f8f8')
plt.close()
print(f"Saved -> {out_path}")
