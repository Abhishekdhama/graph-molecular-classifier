import json
import numpy as np

PROBE_IDS   = [0, 1, 2, 3, 4]
RESULT_FILES = {
    "GNNExplainer":    "results/gnnexplainer_stability.json",
    "Gradient x Input": "results/grad_stability.json",
    "GNN-LRP":         "results/lrp_stability.json",
}


def load_results(path):
    with open(path) as f:
        raw = json.load(f)
    return {int(k): v for k, v in raw.items()}


def main():
    print(f"\n{'='*65}")
    print(f"  STABILITY COMPARISON — GNN Explainability Methods")
    print(f"  Metric: Mean Jaccard overlap of top-3 nodes across 10 seeds")
    print(f"{'='*65}")

    loaded = {}
    for method, path in RESULT_FILES.items():
        try:
            loaded[method] = load_results(path)
        except FileNotFoundError:
            print(f"  Missing: {path} — run the corresponding explain script first")

    if not loaded:
        print("  No results found. Run explain_gradient.py and explain_lrp.py first.")
        return

    methods = list(loaded.keys())
    col_w   = 20

    header = f"  {'Molecule':<20}" + "".join(f"{m:>{col_w}}" for m in methods)
    print(f"\n{header}")
    print(f"  {'─'*20}" + "─" * col_w * len(methods))

    per_method_scores = {m: [] for m in methods}

    for mol_id in PROBE_IDS:
        label_str = ""
        row = f"  {f'Mol {mol_id}':<20}"
        for method in methods:
            if method in loaded and mol_id in loaded[method]:
                stats     = loaded[method][mol_id]
                mean_j    = stats["mean_jaccard"]
                std_j     = stats["std_jaccard"]
                label_str = "mutagenic" if stats["true_label"] == 1 else "non-mutagenic"
                cell      = f"{mean_j:.3f} ± {std_j:.3f}"
                per_method_scores[method].append(mean_j)
            else:
                cell = "N/A"
            row += f"{cell:>{col_w}}"
        row += f"  ({label_str})"
        print(row)

    print(f"\n  {'─'*20}" + "─" * col_w * len(methods))

    avg_row = f"  {'Mean across mols':<20}"
    for method in methods:
        if per_method_scores[method]:
            avg = np.mean(per_method_scores[method])
            avg_row += f"{avg:.3f}{'':>{col_w - 5}}"
        else:
            avg_row += f"{'N/A':>{col_w}}"
    print(avg_row)

    print(f"\n{'─'*65}")
    print(f"  WINNER")
    print(f"{'─'*65}")

    ranked = sorted(
        [(m, np.mean(s)) for m, s in per_method_scores.items() if s],
        key=lambda x: x[1],
        reverse=True
    )
    for rank, (method, score) in enumerate(ranked, 1):
        bar = "█" * int(score * 30)
        print(f"  {rank}. {method:<22} {score:.3f}  {bar}")

    best_method, best_score = ranked[0]
    print(f"\n  Most stable: {best_method} (mean Jaccard = {best_score:.3f})")

    print(f"\n{'─'*65}")
    print(f"  INTERPRETATION")
    print(f"{'─'*65}")
    print(f"  Jaccard = 1.0 → identical top-3 nodes across all seed pairs")
    print(f"  Jaccard = 0.0 → completely different nodes every retrain")
    print(f"  Higher = more trustworthy for safety-critical applications")
    print(f"{'='*65}\n")

    summary = {
        method: {
            "mean_jaccard_across_mols": round(float(np.mean(scores)), 4),
            "per_mol": {
                mol_id: loaded[method][mol_id]
                for mol_id in PROBE_IDS
                if method in loaded and mol_id in loaded[method]
            }
        }
        for method, scores in per_method_scores.items()
        if scores
    }

    with open("results/comparison_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved → results/comparison_summary.json")


if __name__ == "__main__":
    main()