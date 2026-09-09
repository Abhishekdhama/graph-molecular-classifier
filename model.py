"""
========================================
 Molecular Toxicity Classification
========================================

  model.py — GCN model for graph-level classification

  This model classifies entire MOLECULES (graphs) as
  mutagenic or non-mutagenic.

  Architecture:
    3x GCNConv layers (message passing)
    → Global Mean Pooling (node features → graph feature)
    → 2x Linear layers (classifier head)
========================================
"""

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool


class MoleculeGCN(torch.nn.Module):
   

    def __init__(self, num_node_features, hidden_channels, num_classes):
        super().__init__()

        
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)

        
        self.lin1 = torch.nn.Linear(hidden_channels, hidden_channels)
        self.lin2 = torch.nn.Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index, batch):
        
        
        x = self.conv1(x, edge_index)
        x = F.relu(x)

        x = self.conv2(x, edge_index)
        x = F.relu(x)

        x = self.conv3(x, edge_index)
        x = F.relu(x)

        
        x = global_mean_pool(x, batch)
        

        # Classification
        x = self.lin1(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)

        return x    # [num_graphs, num_classes]
