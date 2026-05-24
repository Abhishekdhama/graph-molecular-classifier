"""
train.py — Training & evaluation pipeline for MoleculeGCN

Run:  python train.py

Dataset: MUTAG (188 chemical compounds)
  • Nodes = atoms (C, N, O, F, I, Cl, Br)
  • Edges = chemical bonds
  • Label = mutagenic (1) or non-mutagenic (0)

Pipeline: Load data → Create model → Train (100 epochs) → Evaluate
"""

import torch
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from model import MoleculeGCN

print("=" * 55)
print("  🧪 Mini Project: Molecular Toxicity Classification")
print("=" * 55)


# ══════════════════════════════════════════════
# 1. LOAD & PREPARE DATA
# ══════════════════════════════════════════════
print("\n📊 Step 1: Loading MUTAG Dataset")
print("-" * 40)

dataset = TUDataset(root='../data', name='MUTAG')

# Shuffle for random train/test split
torch.manual_seed(42)
dataset = dataset.shuffle()

# 80/20 split
split = int(len(dataset) * 0.8)
train_dataset = dataset[:split]
test_dataset = dataset[split:]

print(f"  Total molecules:  {len(dataset)}")
print(f"  Training:         {len(train_dataset)}")
print(f"  Testing:          {len(test_dataset)}")
print(f"  Num classes:      {dataset.num_classes} (0=non-mutagenic, 1=mutagenic)")
print(f"  Atom features:    {dataset.num_features} (one-hot atom type)")

# Show a sample molecule
sample = dataset[0]
print(f"\n  Sample molecule:")
print(f"    Atoms:  {sample.num_nodes}")
print(f"    Bonds:  {sample.num_edges}")
print(f"    Label:  {sample.y.item()} ({'mutagenic' if sample.y.item() == 1 else 'non-mutagenic'})")

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


# ══════════════════════════════════════════════
# 2. CREATE MODEL
# ══════════════════════════════════════════════
print("\n\n🧠 Step 2: Creating Model")
print("-" * 40)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = MoleculeGCN(
    num_node_features=dataset.num_features,    # 7
    hidden_channels=64,
    num_classes=dataset.num_classes            # 2
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.CrossEntropyLoss()

print(f"  Device: {device}")
print(f"  Model:\n{model}")
print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")


# ══════════════════════════════════════════════
# 3. TRAINING FUNCTION
# ══════════════════════════════════════════════

def train():
    """One epoch of training."""
    model.train()
    total_loss = 0

    for batch in train_loader:
        batch = batch.to(device)

        # Forward pass
        out = model(batch.x, batch.edge_index, batch.batch)
        loss = criterion(out, batch.y)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch.num_graphs

    return total_loss / len(train_dataset)


# ══════════════════════════════════════════════
# 4. EVALUATION FUNCTION
# ══════════════════════════════════════════════

@torch.no_grad()
def test(loader):
    """Compute accuracy on a DataLoader."""
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


# ══════════════════════════════════════════════
# 5. TRAIN!
# ══════════════════════════════════════════════
print("\n\n🏋️ Step 3: Training for 100 Epochs")
print("-" * 55)

best_test_acc = 0
best_epoch = 0

for epoch in range(1, 101):
    loss = train()
    train_acc = test(train_loader)
    test_acc = test(test_loader)

    if test_acc > best_test_acc:
        best_test_acc = test_acc
        best_epoch = epoch

    if epoch % 10 == 0:
        bar = "█" * int(train_acc * 20) + "░" * (20 - int(train_acc * 20))
        print(f"  Epoch {epoch:3d} │ Loss: {loss:.4f} │ "
              f"Train: {train_acc:.3f} │ Test: {test_acc:.3f} │ {bar}")


# ══════════════════════════════════════════════
# 6. FINAL RESULTS
# ══════════════════════════════════════════════
print("-" * 55)
print(f"""
╔═══════════════════════════════════════╗
║           FINAL RESULTS               ║
╠═══════════════════════════════════════╣
║  Best Test Accuracy:  {best_test_acc:.1%}          ║
║  At Epoch:            {best_epoch:3d}              ║
║                                       ║
║  {int(best_test_acc * len(test_dataset))}/{len(test_dataset)} molecules correctly classified  ║
╚═══════════════════════════════════════╝
""")

print("🎓 Core Concepts Used:")
print("  ┌─────────────────────────────────────────────────────┐")
print("  │ Tensors & Autograd   → loss.backward()             │")
print("  │ Training Loop        → forward → loss → back → step│")
print("  │ Graph Data Objects   → Data(x, edge_index, y)      │")
print("  │ GCNConv Layers       → 3-layer message passing     │")
print("  │ Global Pooling       → node features → graph vector│")
print("  └─────────────────────────────────────────────────────┘")

