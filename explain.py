"""
explain.py — GNNExplainer for MoleculeGCN

Yeh script trained model pe GNNExplainer chalata hai aur batata hai ki
model ne kaunse atoms/bonds ko important samjha mutagenic prediction ke liye.

What it does:
  1. Model train karta hai
  2. GNNExplainer run karta hai selected molecules pe
  3. Har molecule ke liye dikhata hai ki kaunse atoms important hain
"""

import torch
import torch.nn.functional as F
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.explain import Explainer, GNNExplainer
from model import MoleculeGCN

# Atom types in MUTAG (one-hot index → element

ATOM_NAMES = ['C', 'N', 'O', 'F', 'I', 'Cl', 'Br']


def get_atom_name(one_hot_vector):
    """One-hot feature vector se atom name nikalta hai."""
    idx = one_hot_vector.argmax().item()
    return ATOM_NAMES[idx] if idx < len(ATOM_NAMES) else f'?{idx}'


# 1. Data loading

print("=" * 60)
print(" GNNExplainer — Molecular Toxicity Explanation")
print("=" * 60)

print("\n📊 Step 1: Dataset load ho raha hai...")
dataset = TUDataset(root='../data', name='MUTAG')
torch.manual_seed(42)
dataset = dataset.shuffle()

split = int(len(dataset) * 0.8)
train_dataset = dataset[:split]
test_dataset = dataset[split:]

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
print(f"  {len(dataset)} molecules loaded (Train: {len(train_dataset)}, Test: {len(test_dataset)})")


# 2. model train

print("\n Step 2: Model train ho raha hai (100 epochs)...")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = MoleculeGCN(
    num_node_features=dataset.num_features,
    hidden_channels=64,
    num_classes=dataset.num_classes
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.CrossEntropyLoss()


def train_one_epoch():
    model.train()
    total_loss = 0
    for batch in train_loader:
        batch = batch.to(device)
        out = model(batch.x, batch.edge_index, batch.batch)
        loss = criterion(out, batch.y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch.num_graphs
    return total_loss / len(train_dataset)


@torch.no_grad()
def evaluate(loader):
    model.eval()
    correct = 0
    total = 0
    for batch in loader:
        batch = batch.to(device)
        out = model(batch.x, batch.edge_index, batch.batch)
        pred = out.argmax(dim=1)
        correct += (pred == batch.y).sum().item()
        total += batch.num_graphs
    return correct / total


# Train for 100 epochs

for epoch in range(1, 101):
    loss = train_one_epoch()
    if epoch % 25 == 0:
        test_acc = evaluate(test_loader)
        print(f"  Epoch {epoch:3d} │ Loss: {loss:.4f} │ Test Acc: {test_acc:.3f}")

train_acc = evaluate(train_loader)
test_acc = evaluate(test_loader)
print(f"\n  Training complete! Train Acc: {train_acc:.3f} │ Test Acc: {test_acc:.3f}")


# 3.  GNNExplainer setup

print("\n" + "=" * 60)
print(" Step 3: GNNExplainer setup ho raha hai...")
print("=" * 60)

print("""
    GNNExplainer kya karta hai?
   ─────────────────────────────────────────────────
   Yeh ek trained model ko "question" karta hai:
   "Bhai, tune yeh prediction KYUN di?"

   Kaise karta hai:
   1. Har node (atom) ko ek "importance mask" deta hai (0 to 1)
   2. Har edge (bond) ko bhi ek mask deta hai
   3. Yeh masks LEARN karta hai — optimize karta hai taaki
      sirf important atoms/bonds se bhi same prediction aaye
   4. Jo atoms ka mask value HIGH hai → woh important hain

   Socho aise:
   Agar model bole "yeh molecule mutagenic hai" →
   GNNExplainer batata hai "haan, KYUNKI yeh NO2 group hai"
   ─────────────────────────────────────────────────
""")

# Model eval mode
model.eval()

# Explainer setup
# 'phenomenon' = graph-level classification
# algorithm = GNNExplainer (learns masks via optimization)

explainer = Explainer(
    model=model,
    algorithm=GNNExplainer(epochs=200, lr=0.01),
    explanation_type='phenomenon',       # "model ne yeh prediction kyun di"
    node_mask_type='attributes',         
    edge_mask_type='object',             
        mode='multiclass_classification',
        task_level='graph',              # graph-level task hai
        return_type='raw',               # model raw logits return karta hai
    ),
)

print("Explainer ready!\n")


# 4. Molecules explaination

print("=" * 60)
print("  🧪 Step 4: Molecules explain ho rahe hain...")
print("=" * 60)

# Pick interesting molecules — some mutagenic, some non-mutagenic

molecules_to_explain = []
mutagenic_count = 0
non_mutagenic_count = 0

for i in range(len(test_dataset)):
    label = test_dataset[i].y.item()
    if label == 1 and mutagenic_count < 3:
        molecules_to_explain.append((i, test_dataset[i]))
        mutagenic_count += 1
    elif label == 0 and non_mutagenic_count < 2:
        molecules_to_explain.append((i, test_dataset[i]))
        non_mutagenic_count += 1
    if mutagenic_count >= 3 and non_mutagenic_count >= 2:
        break

print(f"\n  {len(molecules_to_explain)} molecules selected for explanation\n")

for mol_idx, (dataset_idx, data) in enumerate(molecules_to_explain):
    data = data.to(device)
    true_label = data.y.item()
    label_text = "MUTAGENIC " if true_label == 1 else "NON-MUTAGENIC "

    # check the model prediction

    with torch.no_grad():
        batch_vec = torch.zeros(data.num_nodes, dtype=torch.long, device=device)
        pred = model(data.x, data.edge_index, batch_vec)
        pred_class = pred.argmax(dim=1).item()
        pred_text = "mutagenic" if pred_class == 1 else "non-mutagenic"
        confidence = F.softmax(pred, dim=1)[0]

    print(f"{'─' * 60}")
    print(f"  Molecule #{mol_idx + 1} (test set index: {dataset_idx})")
    print(f"     Atoms: {data.num_nodes}  │  Bonds: {data.num_edges // 2}")
    print(f"     True Label:  {label_text}")
    print(f"     Prediction:  {pred_text} (confidence: {confidence[pred_class]:.1%})")
    correct = " CORRECT" if pred_class == true_label else "❌ WRONG"
    print(f"     Result:      {correct}")

    # GNNExplainer:
    # Single graph ke liye batch vector banana padta hai
    # target = model ki prediction 

    explanation = explainer(
        data.x,
        data.edge_index,
        target=torch.tensor([pred_class], device=device),
        batch=torch.zeros(data.num_nodes, dtype=torch.long, device=device),
    )

    #Node importance nikalo ──
    # node_mask shape: [num_nodes, num_features]
    # Har node ka overall importance = uske mask values ka mean

    node_mask = explanation.node_mask
    node_importance = node_mask.mean(dim=1)   # [num_nodes]

    # Normalize to 0-1

    if node_importance.max() > 0:
        node_importance = node_importance / node_importance.max()

    # Edge importance nikalo ──

    edge_mask = explanation.edge_mask   # [num_edges]
    if edge_mask is not None and edge_mask.max() > 0:
        edge_importance = edge_mask / edge_mask.max()
    else:
        edge_importance = edge_mask

    # ── Results
    print(f"\n    Atom Importance (GNNExplainer results):")
    print(f"     {'Atom':>6} │ {'Element':>7} │ {'Importance':>10} │ {'Bar':>20}")
    print(f"     {'─'*6}─┼─{'─'*7}─┼─{'─'*10}─┼─{'─'*20}")

    # Sort by importance (descending)

    sorted_indices = node_importance.argsort(descending=True)

    for rank, node_idx in enumerate(sorted_indices):
        node_idx = node_idx.item()
        imp = node_importance[node_idx].item()
        atom_name = get_atom_name(data.x[node_idx])
        bar = "█" * int(imp * 15) + "░" * (15 - int(imp * 15))

        #  highlight the Top-3 

        marker = " ⭐" if rank < 3 else ""
        print(f"     {node_idx:>6} │ {atom_name:>7} │ {imp:>10.4f} │ {bar}{marker}")

    # Top important edges 

    if edge_importance is not None:
        print(f"\n     🔗 Top Important Bonds:")
        edge_sorted = edge_importance.argsort(descending=True)

        shown_pairs = set()
        bond_count = 0
        for edge_idx in edge_sorted:
            edge_idx = edge_idx.item()
            src = data.edge_index[0, edge_idx].item()
            dst = data.edge_index[1, edge_idx].item()

            # skip the Duplicate bonds  (undirected graph hai)

            pair = (min(src, dst), max(src, dst))
            if pair in shown_pairs:
                continue
            shown_pairs.add(pair)

            imp = edge_importance[edge_idx].item()
            src_atom = get_atom_name(data.x[src])
            dst_atom = get_atom_name(data.x[dst])
            bar = "█" * int(imp * 15) + "░" * (15 - int(imp * 15))
            print(f"       {src_atom}({src}) ── {dst_atom}({dst}) │ {imp:.4f} │ {bar}")

            bond_count += 1
            if bond_count >= 5:
                break

    # Summary
    top3_atoms = [get_atom_name(data.x[sorted_indices[i].item()])
                  for i in range(min(3, len(sorted_indices)))]
    print(f"\n     📌 Summary: Model ne sabse zyada dhyan diya → {', '.join(top3_atoms)} atoms pe")
    print()


# ══════════════════════════════════════════════
# 5. OVERALL INTERPRETATION
# ══════════════════════════════════════════════
print("=" * 60)
print("  📖 Overall Interpretation")
print("=" * 60)
print("""
  🧬 MUTAG dataset mein, mutagenicity mainly inn chemical
     groups ki wajah se hoti hai:

     • NO₂ (nitro group)  — nitrogen + oxygen atoms
     • NH₂ (amino group)  — nitrogen + hydrogen
     • Aromatic rings     — carbon ring structuresS

     Agar GNNExplainer baar baar N (Nitrogen) aur O (Oxygen)
     atoms ko highlight kar raha hai mutagenic molecules mein,
     toh model ne SAHI pattern seekha hai!

     Note: GNNExplainer har run pe thoda different results
     de sakta hai kyunki yeh optimization-based method hai.
     Multiple runs ka average lena better hota hai.
""")
print("=" * 60)
print("  Explanation complete!")
print("=" * 60)
