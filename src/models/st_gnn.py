import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class SpatialEncoder(nn.Module):
    """
    Stage 1: Spatial representation learning per temporal snapshot.
    """
    def __init__(self, in_channels=166, hidden_channels=64, out_channels=32, dropout=0.2):
        super(SpatialEncoder, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index):
        # 1st Hop Message Passing
        h = self.conv1(x, edge_index)
        h = F.relu(h)
        h = self.dropout(h)
        
        # 2nd Hop Message Passing
        h = self.conv2(h, edge_index)
        h = F.relu(h)
        return h  # Shape: (N, 32)

class SpatioTemporalGNN(nn.Module):
    """
    Stage 2 & 3: Decoupled ST-GNN (Spatial GCN + Temporal GRU + MLP Head).
    """
    def __init__(self, in_channels=166, spatial_dim=32, rnn_hidden=32, dropout=0.3):
        super(SpatioTemporalGNN, self).__init__()
        # Shared spatial encoder across time snapshots
        self.spatial_encoder = SpatialEncoder(
            in_channels=in_channels, 
            hidden_channels=64, 
            out_channels=spatial_dim,
            dropout=0.2
        )
        
        # Temporal sequence processor
        self.temporal_gru = nn.GRU(
            input_size=spatial_dim,
            hidden_size=rnn_hidden,
            num_layers=1,
            batch_first=False
        )
        
        # Classification MLP head
        self.mlp = nn.Sequential(
            nn.Linear(rnn_hidden, 16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(16, 1)  # Raw logits for BCEWithLogitsLoss
        )

    def forward(self, x_seq, edge_index_seq):
        """
        Args:
            x_seq: List of node feature tensors [X_(t-2), X_(t-1), X_t]
            edge_index_seq: List of edge index tensors [E_(t-2), E_(t-1), E_t]
        """
        spatial_embeddings = []
        
        # 1. Extract spatial graph representations for each snapshot in the window
        for x_t, edge_idx_t in zip(x_seq, edge_index_seq):
            h_t = self.spatial_encoder(x_t, edge_idx_t)
            spatial_embeddings.append(h_t)
            
        # Stack into sequence tensor of shape: (T, Num_Nodes, Spatial_Dim)
        seq_tensor = torch.stack(spatial_embeddings, dim=0)
        
        # 2. Process temporal evolution through GRU
        gru_out, h_n = self.temporal_gru(seq_tensor)
        
        # Extract the final hidden state corresponding to the target time step t
        final_embedding = h_n.squeeze(0)  # Shape: (Num_Nodes, RNN_Hidden)
        
        # 3. Project through MLP classifier
        logits = self.mlp(final_embedding)
        return logits