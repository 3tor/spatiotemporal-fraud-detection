import os
import torch
from dataset import get_elliptic_dataset, build_temporal_snapshot_cache
from models.st_gnn import SpatioTemporalGNN
from evaluate import evaluate_stgnn

def evaluate_saved_model():
    # 1. Define paths
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    CHECKPOINT_PATH = os.path.abspath(os.path.join(CURRENT_DIR, "..", "checkpoints", "SpatioTemporalGNN_best.pt"))
    
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"Error: Checkpoint not found at {CHECKPOINT_PATH}")
        print("Please run train.py first to generate the saved weights.")
        return

    # 2. Load the dataset
    print("Loading Elliptic Dataset...")
    data = get_elliptic_dataset()
    snapshots = build_temporal_snapshot_cache(data)
    
    # 3. Initialize the EMPTY model architecture (must match the trained model's dimensions)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = SpatioTemporalGNN(in_channels=data.x.size(1), spatial_dim=32, rnn_hidden=32)
    
    # 4. LOAD THE SAVED WEIGHTS
    print(f"Loading weights from {CHECKPOINT_PATH}...")
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device, weights_only=True))
    model.to(device)
    
    # 5. Evaluate instantly
    print("\n--- Running Instant Test Set Evaluation (Steps 35-49) ---")
    optimal_tau = 0.85 
    
    test_metrics = evaluate_stgnn(model, data, snapshots, data.custom_test_mask, lookback=3, threshold=optimal_tau)
    
    print(f"Test F1 (Illicit): {test_metrics['F1 (Illicit)']:.4f}")
    print(f"Test Precision:    {test_metrics['Precision']:.4f}")
    print(f"Test Recall:       {test_metrics['Recall']:.4f}")
    print(f"Test PR-AUC:       {test_metrics['PR-AUC']:.4f}")
    print("==========================================================")

if __name__ == "__main__":
    evaluate_saved_model()