<p align="center">
  <h1 align="center">🧬 MolTox-GCN</h1>
  <p align="center">
    <strong>Molecular Toxicity Classification using Graph Convolutional Networks</strong>
  </p>
  <p align="center">
    A PyTorch Geometric project that predicts whether chemical compounds are mutagenic<br>
    by learning molecular graph representations with GCNConv message passing.
  </p>
</p>

---

## Overview

MolTox-GCN is a graph-level binary classification model that takes molecular graphs (atoms as nodes, bonds as edges) and predicts mutagenicity. The model uses a 3-layer Graph Convolutional Network followed by global mean pooling and a fully connected classifier head.

| Property | Value |
|---|---|
| **Dataset** | MUTAG (188 molecules) |
| **Task** | Binary classification (mutagenic / non-mutagenic) |
| **Architecture** | 3× GCNConv → Global Mean Pool → 2× Linear |
| **Framework** | PyTorch + PyTorch Geometric |
| **Test Accuracy** | 73.7% (28/38 molecules) |

## Project Structure

```
pyg-mini-project/
├── README.md              # Project documentation
├── GUIDE.md               # Exhaustive code guide — every concept & line explained
├── mini_project/
│   ├── model.py           # MoleculeGCN — GCN architecture (imported by train.py)
│   └── train.py           # Entry point — data loading, training loop, evaluation
└── data/                  # Auto-generated on first run (MUTAG dataset cache)
```

> **Note**: Only `train.py` needs to be run. It imports `model.py` automatically. See [GUIDE.md](GUIDE.md) for a detailed walkthrough of every piece of code.

## Prerequisites

- Python 3.10+
- pip

## Installation

```bash
# 1. Create and activate a virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1        # Windows PowerShell

# 2. Install PyTorch (CPU)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# 3. Install PyTorch Geometric
pip install torch-geometric
```

> **GPU users**: Replace the PyTorch install command with the appropriate CUDA version from [pytorch.org](https://pytorch.org/get-started/locally/).

## Usage

```bash
cd mini_project
python train.py
```

### Output

```
=======================================================
  🧪 Mini Project: Molecular Toxicity Classification
=======================================================

📊 Step 1: Loading MUTAG Dataset
  Total molecules:  188
  Training:         150
  Testing:          38
  Num classes:      2 (0=non-mutagenic, 1=mutagenic)
  Atom features:    7 (one-hot atom type)

  Sample molecule:
    Atoms:  21
    Bonds:  44
    Label:  1 (mutagenic)

🧠 Step 2: Creating Model
  Device: cpu
  Parameters: 13,122

🏋️ Step 3: Training for 100 Epochs
  Epoch  10 │ Loss: 0.5557 │ Train: 0.753 │ Test: 0.658 │ ███████████████░░░░░
  Epoch  20 │ Loss: 0.5127 │ Train: 0.773 │ Test: 0.684 │ ███████████████░░░░░
  Epoch  30 │ Loss: 0.4886 │ Train: 0.773 │ Test: 0.658 │ ███████████████░░░░░
  Epoch  40 │ Loss: 0.4949 │ Train: 0.773 │ Test: 0.711 │ ███████████████░░░░░
  Epoch  50 │ Loss: 0.4650 │ Train: 0.767 │ Test: 0.711 │ ███████████████░░░░░
  Epoch  60 │ Loss: 0.4873 │ Train: 0.787 │ Test: 0.632 │ ███████████████░░░░░
  Epoch  70 │ Loss: 0.4527 │ Train: 0.793 │ Test: 0.632 │ ███████████████░░░░░
  Epoch  80 │ Loss: 0.4538 │ Train: 0.800 │ Test: 0.711 │ ████████████████░░░░
  Epoch  90 │ Loss: 0.4211 │ Train: 0.800 │ Test: 0.684 │ ████████████████░░░░
  Epoch 100 │ Loss: 0.4309 │ Train: 0.787 │ Test: 0.632 │ ███████████████░░░░░

╔═══════════════════════════════════════╗
║           FINAL RESULTS               ║
╠═══════════════════════════════════════╣
║  Best Test Accuracy:  73.7%          ║
║  At Epoch:             66              ║
║                                       ║
║  28/38 molecules correctly classified  ║
╚═══════════════════════════════════════╝
```

## Architecture

```
Input Molecule (Graph)
       │
       ▼
┌─────────────┐
│  GCNConv 1  │   7 → 64 channels  (atom features → hidden)
│  + ReLU     │
├─────────────┤
│  GCNConv 2  │  64 → 64 channels  (1-hop → 2-hop neighborhoods)
│  + ReLU     │
├─────────────┤
│  GCNConv 3  │  64 → 64 channels  (2-hop → 3-hop neighborhoods)
│  + ReLU     │
├─────────────┤
│ Global Mean │   Aggregate all node embeddings into a single
│   Pooling   │   graph-level representation vector
├─────────────┤
│  Linear 1   │  64 → 64 + ReLU + Dropout(0.5)
├─────────────┤
│  Linear 2   │  64 → 2  (class logits)
└─────────────┘
       │
       ▼
  Prediction: Mutagenic / Non-mutagenic
```

## Hyperparameters

| Parameter | Value |
|---|---|
| Hidden channels | 64 |
| GCN layers | 3 |
| Dropout | 0.5 |
| Optimizer | Adam |
| Learning rate | 0.01 |
| Batch size | 32 |
| Epochs | 100 |
| Train/Test split | 80/20 |
| Random seed | 42 |

## License

This project is for educational purposes.
