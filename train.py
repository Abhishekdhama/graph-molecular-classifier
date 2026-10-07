import torch
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from model import MoleculeGCN

print("=" * 55)
print("  Mini Project: Molecular Toxicity Classification")
print("=" * 55)

dataset = TUDataset(root='../data', name='MUTAG')

torch.manual_seed(42)
dataset = dataset.shuffle()

split = int(len(dataset) * 0.8)
train_dataset = dataset[:split]
test_dataset = dataset[split:]

print(f"  Total molecules:  {len(dataset)}")
print(f"  Training:         {len(train_dataset)}")
print(f"  Testing:          {len(test_dataset)}")
print(f"  Num classes:      {dataset.num_classes} (0=non-mutagenic, 1=mutagenic)")
print(f"  Atom features:    {dataset.num_features} (one-hot atom type)")

sample = dataset[0]
print(f"\n  Sample molecule:")
print(f"    Atoms:  {sample.num_nodes}")
print(f"    Bonds:  {sample.num_edges}")
print(f"    Label:  {sample.y.item()} ({'mutagenic' if sample.y.item() == 1 else 'non-mutagenic'})")

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = MoleculeGCN(
    num_node_features=dataset.num_features,
    hidden_channels=64,
    num_classes=dataset.num_classes
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.CrossEntropyLoss()

print(f"  Device: {device}")
print(f"  Model:\n{model}")
print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")


def train():
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
def test(loader):
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


print("\n\n Training for 100 Epochs")
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
