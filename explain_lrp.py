import os
import json
import torch
import numpy as np
from torch_geometric.datasets import TUDataset
from torch_geometric.nn import global_mean_pool

from model import MoleculeGCN

SEEDS     = [42, 0, 1, 7, 13, 21, 99, 123, 256, 512]
HIDDEN_CH = 64
TOP_K     = 3
PROBE_IDS = [0, 1, 2, 3, 4]
EPSILON   = 1e-6

os.makedirs("results", exist_ok=True)


def get_conv_weight(conv):
    if hasattr(conv, 'lin'):
        return conv.lin.weight.data
    elif hasattr(conv, 'weight'):
        return conv.weight.data
    else:
        raise AttributeError("Cannot find weight matrix in GCNConv layer")


def lrp_gcn(model, data, device):
    model.eval()
    data  = data.to(device)
    x     = data.x.float().clone()
    ei    = data.edge_index
    batch = data.batch

    with torch.no_grad():
        h0     = x
        h1     = torch.relu(model.conv1(h0, ei))
        h2     = torch.relu(model.conv2(h1, ei))
        h3     = torch.relu(model.conv3(h2, ei))
        pooled = global_mean_pool(h3, batch)
        a1     = torch.relu(model.lin1(pooled))
        out    = model.lin2(a1)

    pred = out.argmax(dim=1).item()

    # lin2 backward
    w2  = model.lin2.weight.data[pred]
    z2  = (a1[0] * w2).sum() + EPSILON
    R_a1 = a1[0] * w2 * (out[0, pred].item() / z2)

    # lin1 backward
    w1      = model.lin1.weight.data
    z1      = (pooled[0] @ w1.T) + EPSILON
    s1      = R_a1 / z1
    R_pool  = pooled[0] * (w1.T @ s1)

    # mean pooling backward — weighted by node activation magnitude
    h3_sum  = h3.abs().sum(dim=0, keepdim=True) + EPSILON
    R_h3    = (h3 / h3_sum) * R_pool.unsqueeze(0)

    # conv3 backward — propagate through weight matrix
    wc3     = get_conv_weight(model.conv3)
    z_c3    = (h2 @ wc3.T).abs() + EPSILON
    s_c3    = R_h3 / z_c3
    R_h2    = h2 * (s_c3 @ wc3)

    # conv2 backward
    wc2     = get_conv_weight(model.conv2)
    z_c2    = (h1 @ wc2.T).abs() + EPSILON
    s_c2    = R_h2 / z_c2
    R_h1    = h1 * (s_c2 @ wc2)

    # conv1 backward
    wc1     = get_conv_weight(model.conv1)
    z_c1    = (h0 @ wc1.T).abs() + EPSILON
    s_c1    = R_h1 / z_c1
    R_h0    = h0 * (s_c1 @ wc1)

    node_scores = R_h0.abs().sum(dim=1)
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
    print(f"  GNN-LRP Explainability")
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

        print(f"  Seed {seed:>4} →", end=" ")
        for mol_id, mol in zip(PROBE_IDS, probe_mols):
            scores, pred                  = lrp_gcn(model, mol, device)
            all_scores[mol_id][str(seed)] = scores
            print(f"mol{mol_id}(pred={pred})", end=" ")
        print()

    print(f"\n{'─'*55}")
    print(f"  STABILITY — GNN-LRP  (top-{TOP_K} Jaccard across seeds)")
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

    with open("results/lrp_explanations.json", "w") as f:
        json.dump(all_scores, f, indent=2)
    with open("results/lrp_stability.json", "w") as f:
        json.dump(stability_results, f, indent=2)

    print(f"\n  Saved → results/lrp_explanations.json")
    print(f"  Saved → results/lrp_stability.json")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()