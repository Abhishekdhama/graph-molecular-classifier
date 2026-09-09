"""
explain_gradient.py — Gradient × Input Explainability

For each trained seed model, runs Gradient × Input attribution
on a fixed set of probe molecules. Saves node importance scores
per seed for stability comparison.

Why Gradient × Input?
─────────────────────
- Fully deterministic: same model → same explanation every time
- No perturbations, no sampling — just one backward pass
- Attribution score = gradient * input feature value
  (measures: "if this feature changed, how much would the prediction change?")

Usage:
    python explain_gradient.py

Output:
    results/grad_explanations.json   ← node scores per molecule per seed
    results/grad_stability.json      ← Jaccard overlap across seeds
"""

import os
import json
import torch
import numpy as np
from torch_geometric.datasets import TUDataset

from model import MoleculeGCN

# ── Config ────────────────────────────────────────────────────────────────────

SEEDS        = [42, 0, 1, 7, 13, 21, 99, 123, 256, 512]
HIDDEN_CH    = 64
TOP_K        = 3        # top-k nodes considered "important" for Jaccard
PROBE_IDS    = [0, 1, 2, 3, 4]   # which molecules to explain (fixed across all seeds)

os.makedirs("results", exist_ok=True)


# ── Core: Gradient × Input ────────────────────────────────────────────────────

def grad_x_input(model, data, device):
    """
    Computes Gradient × Input node importance scores.

    Steps:
        1. Enable grad on node features x
        2. Forward pass → get prediction
        3. Backward on predicted class score
        4. node_score = (grad * x).abs().sum(dim=-1)
           → scalar importance per node

    Returns:
        scores: list of floats, one per node (length = num_nodes in molecule)
        pred  : int, predicted class (0 or 1)
    """
    model.eval()
    data = data.to(device)

    # Clone x and require grad — original data.x stays clean
    x = data.x.float().clone().requires_grad_(True)

    out = model(x, data.edge_index, data.batch)
    pred = out.argmax(dim=1).item()

    # Backward on the predicted class score only
    out[0, pred].backward()

    # Gradient × Input: element-wise multiply, sum across feature dim
    node_scores = (x.grad * x).abs().sum(dim=-1)
    return node_scores.detach().cpu().tolist(), pred


# ── Jaccard Stability ─────────────────────────────────────────────────────────

def jaccard(set1, set2):
    """Overlap between two sets of top-k important nodes."""
    s1, s2 = set(set1), set(set2)
    if not s1 and not s2:
        return 1.0
    return len(s1 & s2) / len(s1 | s2)


def top_k_nodes(scores, k):
    """Returns indices of top-k highest scoring nodes."""
    arr = np.array(scores)
    return list(np.argsort(arr)[-k:])


def compute_stability(all_scores, mol_id):
    """
    For a given molecule, compute mean pairwise Jaccard
    across all seed pairs — this is the stability score.

    High Jaccard → method consistently identifies same nodes
    Low Jaccard → method is noisy/unstable
    """
    seed_topk = {}
    for seed, scores in all_scores[mol_id].items():
        seed_topk[seed] = top_k_nodes(scores, TOP_K)

    seeds = list(seed_topk.keys())
    pairwise = []
    for i in range(len(seeds)):
        for j in range(i + 1, len(seeds)):
            j_score = jaccard(seed_topk[seeds[i]], seed_topk[seeds[j]])
            pairwise.append(j_score)

    return round(np.mean(pairwise), 4), round(np.std(pairwise), 4)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'='*55}")
    print(f"  Gradient × Input Explainability")
    print(f"{'='*55}")
    print(f"  Seeds       : {len(SEEDS)} models")
    print(f"  Probe mols  : {PROBE_IDS}")
    print(f"  Top-k nodes : {TOP_K}")

    dataset = TUDataset(root="data/MUTAG", name="MUTAG")

    # Probe molecules — fixed, same across all seeds
    probe_mols = [dataset[i] for i in PROBE_IDS]

    # Structure: {mol_id: {seed: [scores]}}
    all_scores = {mol_id: {} for mol_id in PROBE_IDS}
    all_preds  = {mol_id: {} for mol_id in PROBE_IDS}

    for seed in SEEDS:
        model_path = f"models/seed_{seed}.pt"
        if not os.path.exists(model_path):
            print(f"  ⚠️  Missing models/seed_{seed}.pt — run train_multiseed.py first")
            continue

        model = MoleculeGCN(
            num_node_features=dataset.num_node_features,
            hidden_channels=HIDDEN_CH,
            num_classes=dataset.num_classes,
        ).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))

        print(f"\n  Seed {seed:>4} →", end=" ")
        for mol_id, mol in zip(PROBE_IDS, probe_mols):
            scores, pred = grad_x_input(model, mol, device)
            all_scores[mol_id][str(seed)] = scores
            all_preds[mol_id][str(seed)]  = pred
            print(f"mol{mol_id}(pred={pred})", end=" ")
        print()

    # ── Stability Report ──
    print(f"\n{'─'*55}")
    print(f"  STABILITY REPORT — Gradient × Input")
    print(f"  (Mean Jaccard overlap of top-{TOP_K} nodes across seed pairs)")
    print(f"{'─'*55}")

    stability_results = {}
    for mol_id in PROBE_IDS:
        mean_j, std_j = compute_stability(all_scores, mol_id)
        true_label = dataset[mol_id].y.item()
        label_str  = "mutagenic" if true_label == 1 else "non-mutagenic"
        print(f"  Mol {mol_id} ({label_str:>14}) │ "
              f"Jaccard {mean_j:.3f} ± {std_j:.3f}")
        stability_results[mol_id] = {
            "mean_jaccard": mean_j,
            "std_jaccard": std_j,
            "true_label": true_label,
        }

    # ── Save ──
    with open("results/grad_explanations.json", "w") as f:
        json.dump(all_scores, f, indent=2)
    with open("results/grad_stability.json", "w") as f:
        json.dump(stability_results, f, indent=2)

    print(f"\n  Saved → results/grad_explanations.json")
    print(f"  Saved → results/grad_stability.json")
    print(f"\n  Next step: run explain_lrp.py for GNN-LRP comparison")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()
