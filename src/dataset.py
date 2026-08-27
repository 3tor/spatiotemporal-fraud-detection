import os
import torch
import pandas as pd
from torch_geometric.datasets import EllipticBitcoinDataset

# Determine project root directory dynamically (one level up from src/)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
DEFAULT_DATA_DIR = os.path.join(PROJECT_ROOT, "data")

def get_elliptic_dataset(root_dir: str = DEFAULT_DATA_DIR):
    """
    Downloads the Elliptic dataset via PyG and constructs strict 
    chronological masks for Train (1-30), Val (31-34), and Test (35-49).
    """
    print(f"Downloading/Loading Elliptic Dataset to {root_dir}...")
    dataset = EllipticBitcoinDataset(root=root_dir)
    data = dataset[0]  # The dataset is a single massive graph

    # Read the raw features CSV to map the time_step column
    raw_features_path = os.path.join(root_dir, 'raw', 'elliptic_txs_features.csv')
    
    print("Mapping temporal steps from raw features...")
    feat_df = pd.read_csv(raw_features_path, header=None)
    time_steps = torch.from_numpy(feat_df.iloc[:, 1].values).to(torch.long)
    
    # Attach time_step to the PyG Data object
    data.time_step = time_steps

    # y labels: 0 = Licit, 1 = Illicit, 2 = Unknown
    is_labeled = (data.y != 2)

    # ---------------------------------------------------------
    # Create the 3-Way Chronological Split Masks
    # ---------------------------------------------------------
    # Train: Time steps 1 to 30
    data.custom_train_mask = is_labeled & (data.time_step <= 30)
    
    # Val: Time steps 31 to 34 (Used for Early Stopping & Threshold Tuning)
    data.custom_val_mask = is_labeled & (data.time_step >= 31) & (data.time_step <= 34)
    
    # Test: Time steps 35 to 49 (Out-of-Sample Performance)
    data.custom_test_mask = is_labeled & (data.time_step >= 35)

    print("\n--- Dataset Split Statistics ---")
    print(f"Total Nodes (Transactions): {data.num_nodes}")
    print(f"Total Edges (Flows): {data.num_edges}")
    print(f"Total Labeled Nodes: {is_labeled.sum().item()}")
    print(f"  - Train Labeled Nodes: {data.custom_train_mask.sum().item()}")
    print(f"  - Val Labeled Nodes:   {data.custom_val_mask.sum().item()}")
    print(f"  - Test Labeled Nodes:  {data.custom_test_mask.sum().item()}")
    print(f"Total Unknown Nodes (y=2): {(data.y == 2).sum().item()} (Used for structure only)")

    return data

if __name__ == "__main__":
    data = get_elliptic_dataset()
    print("\nData Object successfully constructed!")
    print(data)

def build_temporal_snapshot_cache(data):
    """
    Precomputes per-timestep node features and edge indices for fast windowing.
    """
    snapshots = {}
    unique_times = torch.unique(data.time_step).tolist()
    
    for t in sorted(unique_times):
        # Nodes active at time step t
        node_mask_t = (data.time_step == t)
        
        # Filter edge_index to only include edges where source and target are at time t
        edge_mask_t = node_mask_t[data.edge_index[0]] & node_mask_t[data.edge_index[1]]
        edge_index_t = data.edge_index[:, edge_mask_t]
        
        snapshots[t] = {
            'x': data.x,  # Full feature matrix
            'edge_index': edge_index_t,
            'node_mask': node_mask_t
        }
    return snapshots