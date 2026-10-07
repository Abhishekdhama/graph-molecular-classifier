import os
import json
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from sklearn.model_selection import StratifiedShuffleSplit

from model import MoleculeGCN

SEEDS = [42, 0, 1, 7, 13, 21, 99, 123, 256, 512]
HIDDEN_CH = 64
EPOCHS = 200
LR = 0.01
BATCH_SIZE = 32
DROPOUT = 0.5
TEST_SIZE = 0.2

os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_loaders(dataset, seed: int):
    labels = [data.y.item() for data in dataset]

    sss = StratifiedShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=seed)
    train_idx, test_idx = next(sss.split(range(len(dataset)), labels))

    train_data = [dataset[int(i)] for i in train_idx]
    test_data = [dataset[int(i)] for i in test_idx]

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, test_loader, test_data


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index, batch.batch)
        loss = F.cross_entropy(out, batch.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    for batch in loader:
        batch = batch.to(device)
        out = model(batch.x, batch.edge_index, batch.batch)
        pred = out.argmax(dim=1)
        correct += (pred == batch.y).sum().item()
        total += batch.num_graphs
    return correct / total


def train_one_seed(seed: int, dataset, device):
    print(f"\n{'─'*55}")
    print(f"  Seed {seed:>4}")
    print(f"{'─'*55}")

    set_seed(seed)

    train_loader, test_loader, _ = get_loaders(dataset, seed)

    model = MoleculeGCN(
        num_node_features=dataset.num_node_features,
        hidden_channels=HIDDEN_CH,
        num_classes=dataset.num_classes,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_test_acc = 0.0
    best_epoch = 0

    for epoch in range(1, EPOCHS + 1):
        loss = train_one_epoch(model, train_loader, optimizer, device)
        train_acc = evaluate(model, train_loader, device)
        test_acc = evaluate(model, test_loader, device)

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_epoch = epoch
            torch.save(model.state_dict(), f"models/seed_{seed}.pt")

        if epoch % 25 == 0:
            bar = "█" * int(test_acc * 20) + "░" * (20 - int(test_acc * 20))
            print(f"  Epoch {epoch:>3} │ Loss {loss:.4f} │ "
                  f"Train {train_acc:.3f} │ Test {test_acc:.3f} │ {bar}")

    print(f"\n  Best Test Acc: {best_test_acc:.3f} at epoch {best_epoch}")
    return best_test_acc, best_epoch


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*55}")
    print(f"  GNN Explainability — Multi-Seed Training")
    print(f"{'='*55}")
    print(f"  Device  : {device}")
    print(f"  Seeds   : {SEEDS}")
    print(f"  Epochs  : {EPOCHS}")

    dataset = TUDataset(root="data/MUTAG", name="MUTAG", force_reload=False)
    print(f"  Dataset : MUTAG ({len(dataset)} molecules, "
          f"{dataset.num_classes} classes)")

    summary = {}

    for seed in SEEDS:
        best_acc, best_epoch = train_one_seed(seed, dataset, device)
        summary[str(seed)] = {
            "best_test_acc": round(best_acc, 4),
            "best_epoch": best_epoch,
        }

    print(f"\n{'='*55}")
    print(f"  FINAL SUMMARY ACROSS ALL SEEDS")
    print(f"{'='*55}")
    accs = [v["best_test_acc"] for v in summary.values()]
    for seed, stats in summary.items():
        bar = "█" * int(stats["best_test_acc"] * 20)
        print(f"  Seed {seed:>4} │ {stats['best_test_acc']:.3f} │ {bar}")

    print(f"\n  Mean Acc : {np.mean(accs):.3f}")
    print(f"  Std  Acc : {np.std(accs):.3f}")
    print(f"  Min  Acc : {np.min(accs):.3f}")
    print(f"  Max  Acc : {np.max(accs):.3f}")

    with open("results/training_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n  Models saved → models/seed_*.pt")
    print(f"  Summary saved → results/training_summary.json")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()