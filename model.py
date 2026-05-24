"""
model.py — MoleculeGCN: Graph-level classification with GCN

Classifies molecular graphs as mutagenic or non-mutagenic.

Architecture:
    3× GCNConv layers (message passing)
    → Global Mean Pooling (node features → graph feature)
    → 2× Linear layers (classifier head)
"""

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool


class MoleculeGCN(torch.nn.Module):
    """
    GCN for Graph Classification.
    
    How it works:
    ─────────────────────────────────────────────────────
    INPUT: A molecule (graph)
    
      Atoms (nodes) with features
          ↓
      GCNConv Layer 1  ← each atom learns from its bonded neighbors
          ↓
      GCNConv Layer 2  ← now atoms know about 2-hop neighbors
          ↓
      GCNConv Layer 3  ← 3-hop neighborhood information
          ↓
      Global Mean Pool ← AVERAGE all atom features → one vector for the molecule
          ↓
      Linear + ReLU    ← classify the molecule
          ↓
      Linear           ← output: [score_class0, score_class1]
    
    OUTPUT: Prediction — mutagenic or non-mutagenic
    ─────────────────────────────────────────────────────
    """

    def __init__(self, num_node_features, hidden_channels, num_classes):
        super().__init__()

        # ── GCN Layers (message passing) ──
        # Each layer: aggregate neighbor features → transform → activate
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)

        # ── Classifier Head ──
        self.lin1 = torch.nn.Linear(hidden_channels, hidden_channels)
        self.lin2 = torch.nn.Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index, batch):
        """
        Forward pass.

        Args:
            x:          Node features       [total_nodes_in_batch, num_features]
            edge_index: Edge connectivity    [2, total_edges_in_batch]
            batch:      Graph membership     [total_nodes_in_batch]
                        e.g. [0,0,0,1,1,1,1,2,2,...] 
                        means first 3 nodes → graph 0, next 4 → graph 1, etc.
        
        Returns:
            Class scores for each graph    [num_graphs_in_batch, num_classes]
        """
        # ── Message Passing ──
        # Each GCNConv aggregates neighbor features and transforms them
        x = self.conv1(x, edge_index)
        x = F.relu(x)

        x = self.conv2(x, edge_index)
        x = F.relu(x)

        x = self.conv3(x, edge_index)
        x = F.relu(x)

        # ── Global Pooling ──
        # We need ONE vector per graph (not per node!)
        # global_mean_pool averages all node features within each graph
        # Uses 'batch' to know which nodes belong to which graph
        x = global_mean_pool(x, batch)
        # Shape: [total_nodes, hidden] → [num_graphs, hidden]

        # ── Classification ──
        x = self.lin1(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)

        return x    # [num_graphs, num_classes]
