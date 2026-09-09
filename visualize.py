
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import networkx as nx
from torch_geometric.datasets import TUDataset
from torch_geometric.nn import global_mean_pool

from model import MoleculeGCN

ATOM_TYPES = {0: 'C', 1: 'N', 2: 'O', 3: 'F', 4: 'I', 5: 'Cl', 6: 'Br'}
ATOM_COLORS = {'C': '#4a4a4a', 'N': '#3b6fd4', 'O': '#e8453c',
               'F':  '#33c26e', 'I': '#9b59b6', 'Cl': '#2ecc71', 'Br': '#e67e22'}

HIDDEN_CH = 64
EPSILON   = 1e-6
SEED      = 123

os.makedirs("figures", exist_ok=True)


def get_conv_weight(conv):
    if hasattr(conv, 'lin'):
        return conv.lin.weight.data
    return conv.weight.data


def load_model(dataset, seed, device):
    model = MoleculeGCN(
        num_node_features=dataset.num_node_features,
        hidden_channels=HIDDEN_CH,
        num_classes=dataset.num_classes,
    ).to(device)
    model.load_state_dict(torch.load(f"models/seed_{seed}.pt", map_location=device))
    model.eval()
    return model


def grad_x_input(model, data, device):
    data = data.to(device)
    x    = data.x.float().clone().requires_grad_(True)
    out  = model(x, data.edge_index, data.batch)
    pred = out.argmax(dim=1).item()
    out[0, pred].backward()
    scores = (x.grad * x).abs().sum(dim=-1)
    return scores.detach().cpu().numpy(), pred


def lrp(model, data, device):
    data = data.to(device)
    x, ei, batch = data.x.float(), data.edge_index, data.batch

    with torch.no_grad():
        h0     = x
        h1     = torch.relu(model.conv1(h0, ei))
        h2     = torch.relu(model.conv2(h1, ei))
        h3     = torch.relu(model.conv3(h2, ei))
        pooled = global_mean_pool(h3, batch)
        a1     = torch.relu(model.lin1(pooled))
        out    = model.lin2(a1)

    pred  = out.argmax(dim=1).item()
    w2    = model.lin2.weight.data[pred]
    z2    = (a1[0] * w2).sum() + EPSILON
    R_a1  = a1[0] * w2 * (out[0, pred].item() / z2)

    w1     = model.lin1.weight.data
    z1     = (pooled[0] @ w1.T) + EPSILON
    R_pool = pooled[0] * (w1.T @ (R_a1 / z1))

    h3_sum = h3.abs().sum(dim=0, keepdim=True) + EPSILON
    R_h3   = (h3 / h3_sum) * R_pool.unsqueeze(0)

    wc3  = get_conv_weight(model.conv3)
    R_h2 = h2 * ((R_h3 / ((h2 @ wc3.T).abs() + EPSILON)) @ wc3)

    wc2  = get_conv_weight(model.conv2)
    R_h1 = h1 * ((R_h2 / ((h1 @ wc2.T).abs() + EPSILON)) @ wc2)

    wc1  = get_conv_weight(model.conv1)
    R_h0 = h0 * ((R_h1 / ((h0 @ wc1.T).abs() + EPSILON)) @ wc1)

    scores = R_h0.abs().sum(dim=1)
    return scores.detach().cpu().numpy(), pred


def gnnexplainer_scores(model, data, device):
    data = data.to(device)
    try:
        from torch_geometric.explain import Explainer, GNNExplainer as Alg
        exp  = Explainer(
            model=model,
            algorithm=Alg(epochs=200),
            explanation_type='model',
            node_mask_type='attributes',
            edge_mask_type='object',
            model_config=dict(mode='multiclass_classification',
                              task_level='graph', return_type='raw'),
        )
        explanation = exp(data.x.float(), data.edge_index, batch=data.batch)
        scores      = explanation.node_mask.sum(dim=-1).detach().cpu().numpy()
    except ImportError:
        from torch_geometric.nn import GNNExplainer
        exp    = GNNExplainer(model, epochs=200, return_type='log_prob')
        mask, _ = exp.explain_graph(data.x.float(), data.edge_index)
        scores  = mask.sum(dim=-1).detach().cpu().numpy()

    with torch.no_grad():
        out  = model(data.x.float(), data.edge_index, data.batch)
        pred = out.argmax(dim=1).item()

    return scores, pred


def build_graph(data):
    G  = nx.Graph()
    n  = data.x.size(0)
    ei = data.edge_index.numpy()

    for i in range(n):
        atom_idx  = data.x[i].argmax().item()
        G.add_node(i, atom=ATOM_TYPES.get(atom_idx, '?'))

    for k in range(ei.shape[1]):
        u, v = int(ei[0, k]), int(ei[1, k])
        if u < v:
            G.add_edge(u, v)

    return G


def find_groups(G):
    no2, coo = set(), set()
    for node, data in G.nodes(data=True):
        neighbors = list(G.neighbors(node))
        o_nbrs    = [n for n in neighbors if G.nodes[n]['atom'] == 'O']
        if data['atom'] == 'N' and len(o_nbrs) >= 2:
            no2.add(node)
            no2.update(o_nbrs)
        if data['atom'] == 'C' and len(o_nbrs) >= 2:
            coo.add(node)
            coo.update(o_nbrs)
    return no2, coo


def draw_panel(ax, G, scores, pos, title, pred, true_label, no2_nodes, coo_nodes):
    norm    = mcolors.Normalize(vmin=scores.min(), vmax=scores.max())
    cmap    = plt.cm.YlOrRd
    n_colors = [cmap(norm(scores[n])) for n in G.nodes()]
    n_sizes  = [600 + 1000 * norm(scores[n]) for n in G.nodes()]
    labels   = {n: G.nodes[n]['atom'] for n in G.nodes()}

    nx.draw_networkx_edges(G, pos, ax=ax, edge_color='#cccccc', width=1.5, alpha=0.8)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=n_colors,
                           node_size=n_sizes, alpha=0.95)
    nx.draw_networkx_labels(G, pos, labels=labels, ax=ax,
                            font_size=8, font_weight='bold', font_color='white')

    for node in no2_nodes:
        if node in pos:
            ax.annotate('', xy=pos[node], xytext=(pos[node][0], pos[node][1] + 0.18),
                        arrowprops=dict(arrowstyle='->', color='#3b6fd4', lw=1.5))

    for node in coo_nodes:
        if node in pos:
            ax.annotate('', xy=pos[node], xytext=(pos[node][0], pos[node][1] - 0.18),
                        arrowprops=dict(arrowstyle='->', color='#e8453c', lw=1.5))

    pred_str  = 'Mutagenic' if pred == 1 else 'Non-mutagenic'
    true_str  = 'Mutagenic' if true_label == 1 else 'Non-mutagenic'
    correct   = pred == true_label
    color     = '#2d7a3a' if correct else '#c0392b'
    ax.set_title(f"{title}\nPred: {pred_str}", fontsize=11,
                 fontweight='bold', color=color, pad=10)
    ax.axis('off')


def visualize_molecule(mol_id, dataset, device):
    mol        = dataset[mol_id]
    true_label = mol.y.item()
    model      = load_model(dataset, SEED, device)

    G   = build_graph(mol)
    pos = nx.spring_layout(G, seed=42, k=1.2)

    no2_nodes, coo_nodes = find_groups(G)

    print(f"  Molecule {mol_id} — {'Mutagenic' if true_label else 'Non-mutagenic'}")
    print(f"  NO₂ group nodes: {no2_nodes}")
    print(f"  COO group nodes: {coo_nodes}")

    print("  Running GNNExplainer...", end=" ", flush=True)
    gnn_scores, gnn_pred = gnnexplainer_scores(model, mol, device)
    print("done")

    print("  Running Gradient × Input...", end=" ", flush=True)
    grad_scores, grad_pred = grad_x_input(model, mol, device)
    print("done")

    print("  Running GNN-LRP...", end=" ", flush=True)
    lrp_scores, lrp_pred = lrp(model, mol, device)
    print("done")

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
    fig.patch.set_facecolor('#f8f8f8')

    label_str = 'Mutagenic' if true_label == 1 else 'Non-mutagenic'
    fig.suptitle(
        f"Molecule {mol_id}  |  True Label: {label_str}  |  Seed {SEED}  |  "
        f"Node color = importance  (yellow → red = low → high)",
        fontsize=12, fontweight='bold', y=1.01
    )

    draw_panel(axes[0], G, gnn_scores,  pos, "GNNExplainer\n(Perturbation-based)",
               gnn_pred,  true_label, no2_nodes, coo_nodes)
    draw_panel(axes[1], G, grad_scores, pos, "Gradient × Input\n(Gradient-based)",
               grad_pred, true_label, no2_nodes, coo_nodes)
    draw_panel(axes[2], G, lrp_scores,  pos, "GNN-LRP\n(Parameter decomposition)",
               lrp_pred,  true_label, no2_nodes, coo_nodes)

    sm  = plt.cm.ScalarMappable(cmap=plt.cm.YlOrRd)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, orientation='vertical',
                        fraction=0.015, pad=0.02, shrink=0.85)
    cbar.set_label('Normalized Importance', fontsize=10)
    cbar.set_ticks([0, 0.5, 1])
    cbar.set_ticklabels(['Low', 'Mid', 'High'])

    legend_elements = [
        plt.Line2D([0], [0], color='#3b6fd4', lw=2, label='↑ NO₂ group'),
        plt.Line2D([0], [0], color='#e8453c', lw=2, label='↓ COO group'),
    ]
    fig.legend(handles=legend_elements, loc='lower center',
               ncol=2, fontsize=9, framealpha=0.9,
               bbox_to_anchor=(0.5, -0.06))

    plt.tight_layout()
    out_path = f"figures/mol{mol_id}_comparison.png"
    plt.savefig(out_path, dpi=200, bbox_inches='tight', facecolor='#f8f8f8')
    plt.close()
    print(f"\n  Saved → {out_path}")


def find_misclassified(dataset, device):
    model = load_model(dataset, SEED, device)
    misclassified = []
    with torch.no_grad():
        for i, mol in enumerate(dataset):
            mol  = mol.to(device)
            out  = model(mol.x.float(), mol.edge_index, mol.batch)
            pred = out.argmax(dim=1).item()
            if pred != mol.y.item():
                misclassified.append(i)
    return misclassified


def main():
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = TUDataset(
        root=r"D:\pyg-mini-project\mini_project\data\MUTAG",
        name="MUTAG",
        force_reload=False
    )

    print(f"\n{'='*55}")
    print(f"  COO / NO₂ Visualization")
    print(f"  Model: seed {SEED} (best, acc=89.47%)")
    print(f"{'='*55}\n")

    print("  Finding misclassified molecules...")
    misclassified = find_misclassified(dataset, device)
    print(f"  Misclassified by seed {SEED}: {misclassified}")

    # Visualize all misclassified molecules
    for mol_id in misclassified:
        print(f"\n  → Visualizing molecule {mol_id}")
        visualize_molecule(mol_id, dataset, device)

    # Also visualize probe molecules 0-4 for comparison
    print(f"\n  → Visualizing probe molecules 0-4")
    for mol_id in range(5):
        if mol_id not in misclassified:
            visualize_molecule(mol_id, dataset, device)

    print(f"\n{'='*55}")
    print(f"  All figures saved → figures/")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()
