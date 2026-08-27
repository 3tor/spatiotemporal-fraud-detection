import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class BaselineMLP(nn.Module):
    """
    Tier 1 Baseline: Tabular Deep Learning
    Evaluates raw node features without any graph structure.
    """
    def __init__(self, in_channels=166, hidden_channels=64, out_channels=1):
        super(BaselineMLP, self).__init__()
        self.lin1 = nn.Linear(in_channels, hidden_channels)
        self.lin2 = nn.Linear(hidden_channels, hidden_channels // 2)
        self.lin3 = nn.Linear(hidden_channels // 2, out_channels)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x, edge_index=None):
        # edge_index is ignored; MLP only uses node features (x)
        x = F.relu(self.lin1(x))
        x = self.dropout(x)
        x = F.relu(self.lin2(x))
        x = self.dropout(x)
        x = self.lin3(x) 
        return x  # Return raw logits (BCEWithLogitsLoss will apply the Sigmoid)

class BaselineGCN(nn.Module):
    """
    Tier 2 Baseline: Static Graph Neural Network
    Evaluates spatial message passing without temporal sequence tracking.
    """
    def __init__(self, in_channels=166, hidden_channels=64, out_channels=1):
        super(BaselineGCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels // 2)
        self.lin = nn.Linear(hidden_channels // 2, out_channels)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x, edge_index):
        # Pass features through the graph topology
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.lin(x)
        return x