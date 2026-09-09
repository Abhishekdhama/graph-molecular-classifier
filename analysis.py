import json
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
from torch_geometric.datasets import TUDataset

from model import MoleculeGCN

ATOM_TYPES = {0: 'C', 1: 'N', 2: 'O', 3: 'F', 4: 'I', 5: 'Cl', 6: 'Br'}
HIDDEN_CH  = 64
EPSILON    = 1e-6
TOP_K      = 3
PROBE_IDS  = [0, 1, 2, 3, 4]
SEEDS      = [42, 0, 1, 7, 13, 21, 99, 123, 256, 512]

os.makedirs("figures", exist_ok=True)


def load_json(path):
    with open(path) as f:
        raw = json.load(f)
    return {int(k): {s: v for s, v in sv.items()} for k, sv in raw.items()}


def build_graph(data):
    G  = nx.Graph()
    ei = data.edge_index.numpy()
    for i in range(data.x.size(0)):
        G.add_node(i, atom=ATOM_TYPES.get(data.x[i].argmax().item(), '?'))
    for k in range(ei.shape[1]):
        u, v = int(ei[0, k]), int(ei[1, k])
        if u < v:
            G.add_edge(u, v)
    return G


def find_groups(G):
    no2, coo = set(), set()
    for node, d in G.nodes(data=True):
        nbrs  = list(G.neighbors(node))
        o_nbrs = [n for n in nbrs if G.nodes[n]['atom'] == 'O']
        if d['atom'] == 'N' and len(o_nbrs) >= 2:
            no2.add(node)
            no2.update(o_nbrs)
        if d['atom'] == 'C' and len(o_nbrs) >= 2:
            coo.add(node)
            coo.update(o_nbrs)
    return no2, coo


def top_k_nodes(scores, k=TOP_K):
    return set(np.argsort(np.array(scores))[-k:])


def jaccard(s1, s2):
    s1, s2 = set(s1), set(s2)
    if not s1 and not s2:
        return 1.0
    return len(s1 & s2) / len(s1 | s2)


def no2_hit_rate(scores_by_seed, no2_nodes):
    if not no2_nodes:
        return None
    hits = 0
    for scores in scores_by_seed.values():
        topk = top_k_nodes(scores)
        if topk & no2_nodes:
            hits += 1
    return hits / len(scores_by_seed)


def find_worst_failure(explanations):
    worst_mol, worst_s1, worst_s2, worst_j = None, None, None, 1.0
    for mol_id, seed_scores in explanations.items():
        seeds = list(seed_scores.keys())
        for i in range(len(seeds)):
            for j in range(i + 1, len(seeds)):
                t1 = top_k_nodes(seed_scores[seeds[i]])
                t2 = top_k_nodes(seed_scores[seeds[j]])
                j_score = jaccard(t1, t2)
                if j_score < worst_j:
                    worst_j   = j_score
                    worst_mol = mol_id
                    worst_s1  = seeds[i]
                    worst_s2  = seeds[j]
    return worst_mol, worst_s1, worst_s2, worst_j


def print_no2_consistency(dataset, gnn_exp, grad_exp, lrp_exp):
    print(f"\n{'='*65}")
    print(f"  NO₂ CONSISTENCY CHECK — does each method find NO₂ group?")
    print(f"  (% of seeds where top-{TOP_K} nodes include a NO₂ atom)")
    print(f"{'='*65}")
    print(f"  {'Mol':<6} {'Label':<16} {'NO₂ nodes':<14} "
          f"{'GNNExplainer':>14} {'Grad×Input':>12} {'LRP':>8}")
    print(f"  {'─'*6} {'─'*16} {'─'*14} {'─'*14} {'─'*12} {'─'*8}")

    for mol_id in PROBE_IDS:
        mol        = dataset[mol_id]
        G          = build_graph(mol)
        no2, _     = find_groups(G)
        true_label = mol.y.item()
        label_str  = 'mutagenic' if true_label else 'non-mutagenic'

        gnn_rate  = no2_hit_rate(gnn_exp.get(mol_id,  {}), no2)
        grad_rate = no2_hit_rate(grad_exp.get(mol_id, {}), no2)
        lrp_rate  = no2_hit_rate(lrp_exp.get(mol_id,  {}), no2)

        def fmt(r):
            if r is None:
                return 'N/A (no NO₂)'
            return f"{r*100:.0f}%"

        print(f"  {mol_id:<6} {label_str:<16} {str(no2):<14} "
              f"{fmt(gnn_rate):>14} {fmt(grad_rate):>12} {fmt(lrp_rate):>8}")

    print()


def draw_failure_panel(ax, G, scores, pos, seed, no2_nodes, coo_nodes,
                       topk_nodes, true_label, pred):
    import matplotlib.colors as mcolors
    cmap = plt.cm.YlOrRd
    norm = mcolors.Normalize(vmin=min(scores), vmax=max(scores))

    n_colors = [cmap(norm(scores[n])) for n in G.nodes()]
    n_sizes  = [500 + 900 * norm(scores[n]) for n in G.nodes()]

    nx.draw_networkx_edges(G, pos, ax=ax, edge_color='#cccccc', width=1.5)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=n_colors,
                           node_size=n_sizes, alpha=0.95)

    labels = {n: G.nodes[n]['atom'] for n in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels=labels, ax=ax,
                            font_size=8, font_weight='bold', font_color='white')

    for node in topk_nodes:
        if node in pos:
            x, y = pos[node]
            ax.add_patch(plt.Circle((x, y), 0.08, fill=False,
                                    edgecolor='#00c8ff', linewidth=2.5,
                                    transform=ax.transData))

    for node in no2_nodes:
        if node in pos:
            ax.annotate('NO₂', xy=pos[node],
                        xytext=(pos[node][0] + 0.15, pos[node][1] + 0.15),
                        fontsize=7, color='#3b6fd4', fontweight='bold',
                        arrowprops=dict(arrowstyle='->', color='#3b6fd4', lw=1.2))

    for node in coo_nodes - no2_nodes:
        if node in pos:
            ax.annotate('COO', xy=pos[node],
                        xytext=(pos[node][0] - 0.2, pos[node][1] - 0.15),
                        fontsize=7, color='#e8453c', fontweight='bold',
                        arrowprops=dict(arrowstyle='->', color='#e8453c', lw=1.2))

    pred_str = 'Mutagenic' if pred == 1 else 'Non-mutagenic'
    correct  = pred == true_label
    color    = '#2d7a3a' if correct else '#c0392b'
    suffix   = '✓' if correct else '✗ WRONG'
    ax.set_title(f"GNNExplainer  |  Seed {seed}\nPred: {pred_str} {suffix}",
                 fontsize=11, fontweight='bold', color=color, pad=8)
    ax.axis('off')


def visualize_failure_case(dataset, gnn_exp, mol_id, seed1, seed2, jaccard_score, device):
    mol        = dataset[mol_id]
    true_label = mol.y.item()
    G          = build_graph(mol)
    pos        = nx.spring_layout(G, seed=42, k=1.2)
    no2, coo   = find_groups(G)

    scores1 = gnn_exp[mol_id][seed1]
    scores2 = gnn_exp[mol_id][seed2]
    topk1   = top_k_nodes(scores1)
    topk2   = top_k_nodes(scores2)

    def get_pred(seed):
        model = MoleculeGCN(
            num_node_features=dataset.num_node_features,
            hidden_channels=HIDDEN_CH,
            num_classes=dataset.num_classes,
        ).to(device)
        model.load_state_dict(torch.load(f"models/seed_{seed}.pt", map_location=device))
        model.eval()
        with torch.no_grad():
            data = mol.to(device)
            out  = model(data.x.float(), data.edge_index, data.batch)
            return out.argmax(dim=1).item()

    pred1 = get_pred(seed1)
    pred2 = get_pred(seed2)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.patch.set_facecolor('#f8f8f8')

    label_str = 'Mutagenic' if true_label == 1 else 'Non-mutagenic'
    shared    = topk1 & topk2
    diff1     = topk1 - topk2
    diff2     = topk2 - topk1

    fig.suptitle(
        f"Failure Case — Molecule {mol_id}  |  True: {label_str}  |  "
        f"GNNExplainer Jaccard = {jaccard_score:.3f}\n"
        f"⬤ Cyan circle = top-{TOP_K} nodes  |  "
        f"Shared top-3: {shared}  |  "
        f"Seed {seed1} only: {diff1}  |  Seed {seed2} only: {diff2}",
        fontsize=11, fontweight='bold', y=1.03
    )

    draw_failure_panel(axes[0], G, scores1, pos, seed1,
                       no2, coo, topk1, true_label, pred1)
    draw_failure_panel(axes[1], G, scores2, pos, seed2,
                       no2, coo, topk2, true_label, pred2)

    legend = [
        mpatches.Patch(facecolor='#ffe066', label='Low importance'),
        mpatches.Patch(facecolor='#e8453c', label='High importance'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='none',
                   markeredgecolor='#00c8ff', markersize=12,
                   markeredgewidth=2.5, label=f'Top-{TOP_K} nodes'),
    ]
    fig.legend(handles=legend, loc='lower center', ncol=3,
               fontsize=9, framealpha=0.9, bbox_to_anchor=(0.5, -0.07))

    plt.tight_layout()
    path = f"figures/failure_mol{mol_id}_seed{seed1}_vs_{seed2}.png"
    plt.savefig(path, dpi=200, bbox_inches='tight', facecolor='#f8f8f8')
    plt.close()
    print(f"\n  Failure case saved → {path}")


def print_top3_table(gnn_exp, dataset, mol_id, seed1, seed2):
    mol    = dataset[mol_id]
    G      = build_graph(mol)
    no2, _ = find_groups(G)
    atom   = lambda n: G.nodes[n]['atom']

    s1 = top_k_nodes(gnn_exp[mol_id][seed1])
    s2 = top_k_nodes(gnn_exp[mol_id][seed2])

    print(f"\n{'─'*55}")
    print(f"  TOP-3 NODE COMPARISON — Mol {mol_id} | GNNExplainer")
    print(f"{'─'*55}")
    print(f"  {'Node':<8} {'Atom':<8} {'Seed '+seed1:<14} {'Seed '+seed2:<14} {'In NO₂?'}")
    print(f"  {'─'*8} {'─'*8} {'─'*14} {'─'*14} {'─'*8}")

    all_nodes = sorted(s1 | s2)
    for n in all_nodes:
        in1 = '✓ top-3' if n in s1 else '—'
        in2 = '✓ top-3' if n in s2 else '—'
        in_no2 = '✓ YES' if n in no2 else 'no'
        print(f"  {n:<8} {atom(n):<8} {in1:<14} {in2:<14} {in_no2}")

    shared = s1 & s2
    print(f"\n  Shared top-3 : {shared} — atoms: {[atom(n) for n in shared]}")
    print(f"  Jaccard      : {jaccard(s1, s2):.3f}")
    print(f"  NO₂ nodes    : {no2}")


def main():
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = TUDataset(
        root=r"D:\pyg-mini-project\mini_project\data\MUTAG",
        name="MUTAG",
        force_reload=False
    )

    gnn_exp  = load_json("results/gnnexplainer_explanations.json")
    grad_exp = load_json("results/grad_explanations.json")
    lrp_exp  = load_json("results/lrp_explanations.json")

    print_no2_consistency(dataset, gnn_exp, grad_exp, lrp_exp)

    print("  Finding worst GNNExplainer failure case...")
    mol_id, s1, s2, j = find_worst_failure(gnn_exp)
    print(f"  Worst case → Mol {mol_id}, seed {s1} vs seed {s2}, Jaccard = {j:.3f}")

    print_top3_table(gnn_exp, dataset, mol_id, s1, s2)
    visualize_failure_case(dataset, gnn_exp, mol_id, s1, s2, j, device)

    print(f"\n{'='*65}")
    print(f"  Done. Check figures/ and terminal output above.")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    main()