import os
import json
import torch
import numpy as np
from torch_geometric.datasets import TUDataset

from model import MoleculeGCN

SEEDS     = [42, 0, 1, 7, 13, 21, 99, 123, 256, 512]
HIDDEN_CH = 64
TOP_K     = 3
PROBE_IDS = [0, 1, 2, 3, 4]

os.makedirs("results", exist_ok=True)

try:
    from torch_geometric.explain import Explainer, GNNExplainer as GNNExplainerAlg
    NEW_API = True
except ImportError:
    from torch_geometric.nn import GNNExplainer
    NEW_API = False


def run_gnnexplainer(model, mol, device):
    mol = mol.to(device)

    if NEW_API:
        explainer = Explainer(
            model=model,
            algorithm=GNNExplainerAlg(epochs=200),
            explanation_type='model',
            node_mask_type='attributes',
            edge_mask_type='object',
            model_config=dict(
                mode='multiclass_classification',
                task_level='graph',
                return_type='raw',
            ),
        )
        explanation  = explainer(mol.x.float(), mol.edge_index, batch=mol.batch)
        node_scores  = explanation.node_mask.sum(dim=-1)
        with torch.no_grad():
            out  = model(mol.x.float(), mol.edge_index, mol.batch)
            pred = out.argmax(dim=1).item()
    else:
        explainer       = GNNExplainer(model, epochs=200, return_type='log_prob')
        node_mask, _    = explainer.explain_graph(mol.x.float(), mol.edge_index)
        node_scores     = node_mask.sum(dim=-1)
        with torch.no_grad():
            out  = model(mol.x.float(), mol.edge_index, mol.batch)
            pred = out.argmax(dim=1).item()

    return node_scores.detach().cpu().tolist(), pred


def jaccard(set1, set2):
    s1, s2 = set(set1), set(set2)
    if not s1 and not s2:
        return 1.0
    return len(s1 & s2) / len(s1 | s2)


def top_k_nodes(scores, k):
    return list(np.argsort(np.array(scores))[-k:])


def compute_stability(all_scores, mol_id):
    seed_topk = {seed: top_k_nodes(scores, TOP_K)
                 for seed, scores in all_scores[mol_id].items()}
    seeds    = list(seed_topk.keys())
    pairwise = [
        jaccard(seed_topk[seeds[i]], seed_topk[seeds[j]])
        for i in range(len(seeds))
        for j in range(i + 1, len(seeds))
    ]
    return round(np.mean(pairwise), 4), round(np.std(pairwise), 4)


def main():
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = TUDataset(
        root=r"D:\pyg-mini-project\mini_project\data\MUTAG",
        name="MUTAG",
        force_reload=False
    )

    print(f"\n{'='*55}")
    print(f"  GNNExplainer Explainability")
    print(f"  API: {'new (torch_geometric.explain)' if NEW_API else 'old (torch_geometric.nn)'}")
    print(f"{'='*55}")

    probe_mols = [dataset[i] for i in PROBE_IDS]
    all_scores = {mol_id: {} for mol_id in PROBE_IDS}

    for seed in SEEDS:
        model_path = f"models/seed_{seed}.pt"
        if not os.path.exists(model_path):
            print(f"  Missing models/seed_{seed}.pt — run train_multiseed.py first")
            continue

        model = MoleculeGCN(
            num_node_features=dataset.num_node_features,
            hidden_channels=HIDDEN_CH,
            num_classes=dataset.num_classes,
        ).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))

        print(f"  Seed {seed:>4} →", end=" ", flush=True)
        for mol_id, mol in zip(PROBE_IDS, probe_mols):
            scores, pred                  = run_gnnexplainer(model, mol, device)
            all_scores[mol_id][str(seed)] = scores
            print(f"mol{mol_id}(pred={pred})", end=" ", flush=True)
        print()

    print(f"\n{'─'*55}")
    print(f"  STABILITY — GNNExplainer  (top-{TOP_K} Jaccard across seeds)")
    print(f"{'─'*55}")

    stability_results = {}
    for mol_id in PROBE_IDS:
        mean_j, std_j = compute_stability(all_scores, mol_id)
        true_label    = dataset[mol_id].y.item()
        label_str     = "mutagenic" if true_label == 1 else "non-mutagenic"
        print(f"  Mol {mol_id} ({label_str:>14}) │ Jaccard {mean_j:.3f} ± {std_j:.3f}")
        stability_results[mol_id] = {
            "mean_jaccard": mean_j,
            "std_jaccard":  std_j,
            "true_label":   true_label,
        }

    with open("results/gnnexplainer_explanations.json", "w") as f:
        json.dump(all_scores, f, indent=2)
    with open("results/gnnexplainer_stability.json", "w") as f:
        json.dump(stability_results, f, indent=2)

    print(f"\n  Saved → results/gnnexplainer_explanations.json")
    print(f"  Saved → results/gnnexplainer_stability.json")
    print(f"\n  Next: python compare_stability.py")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()
