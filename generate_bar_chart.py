import json
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['Segoe UI', 'Arial', 'DejaVu Sans']

with open("results/comparison_summary.json") as f:
    data = json.load(f)

methods = list(data.keys())
scores = [data[m]["mean_jaccard_across_mols"] for m in methods]

per_mol_means = {m: [v["mean_jaccard"] for v in data[m]["per_mol"].values()] for m in methods}
stds = [np.std(per_mol_means[m]) for m in methods]

colors = ['#e8453c', '#f59e42', '#c0392b']
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

for i, (bar, score, std) in enumerate(zip(bars, scores, stds)):
    ax.text(
        bar.get_x() + bar.get_width() / 2, score + std + 0.025,
        f"{score:.3f}",
        ha='center', va='bottom', fontsize=13, fontweight='bold', color='#1a1a1a'
    )

ax.set_ylabel("Mean Jaccard Overlap (top-3 nodes)", fontsize=11, fontweight='bold', labelpad=10)
ax.set_title(
    "Explainability Stability Across 10 Random Seeds\nHigher = same explanation regardless of training seed",
    fontsize=13, fontweight='bold', pad=15, color='#1a1a1a'
)

ax.set_ylim(0, 0.82)
ax.set_yticks(np.arange(0, 0.9, 0.1))
ax.tick_params(axis='both', labelsize=10)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_color('#cccccc')
ax.spines['bottom'].set_color('#cccccc')
ax.yaxis.grid(True, alpha=0.3, linestyle='-', color='#cccccc', zorder=0)

best_idx = np.argmax(scores)
best_top = scores[best_idx] + stds[best_idx] + 0.075
ax.annotate(
    "Most stable",
    xy=(best_idx, best_top),
    xytext=(best_idx, best_top + 0.055),
    fontsize=9, fontweight='bold', color='#2d7a3a',
    ha='center',
    arrowprops=dict(arrowstyle='->', color='#2d7a3a', lw=1.5)
)

plt.tight_layout()
out_path = "figures/jaccard_stability_chart.png"
plt.savefig(out_path, dpi=200, bbox_inches='tight', facecolor='#f8f8f8')
plt.close()
print(f"Saved -> {out_path}")
