import os
import torch
import numpy as np
import random
import matplotlib.pyplot as plt
from dataset import get_elliptic_dataset, build_temporal_snapshot_cache
from models.st_gnn import SpatioTemporalGNN
from evaluate import evaluate_stgnn

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def plot_concept_drift():
    set_seed(42)
    
    print("Loading Dataset...")
    data = get_elliptic_dataset()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    data = data.to(device)

    snapshots = build_temporal_snapshot_cache(data)
    
    # Initialize empty model
    model = SpatioTemporalGNN(in_channels=data.x.size(1), spatial_dim=32, rnn_hidden=32).to(device)
    
    # Load the trained weights
    checkpoint_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "checkpoints", "SpatioTemporalGNN_best.pt"))
    if not os.path.exists(checkpoint_path):
        print(f"Error: Could not find checkpoint at {checkpoint_path}. Run train.py first.")
        return
        
    print("Loading pre-trained ST-GNN weights...")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    model.eval()

    # We will evaluate chronologically on the Test Set (Steps 35-49)
    test_steps = list(range(35, 50))
    f1_scores = []
    
    optimal_tau = 0.85 # Using the threshold from training
    lookback = 3
    
    print("Evaluating per timestep...")
    for t in test_steps:
        # Create a temporary mask that ONLY isolates the current timestep
        step_mask = (data.time_step == t) & data.custom_test_mask
        
        if step_mask.sum() == 0:
            f1_scores.append(0)
            continue
            
        metrics = evaluate_stgnn(model, data, snapshots, step_mask, lookback, threshold=optimal_tau)
        f1_scores.append(metrics['F1 (Illicit)'])
        print(f"Step {t} | F1: {metrics['F1 (Illicit)']:.4f}")

    # --- PLOT CONCEPT DRIFT ---
    print("\nGenerating Concept Drift Chart...")
    plt.figure(figsize=(10, 5))
    plt.plot(test_steps, f1_scores, marker='o', linestyle='-', color='#d62728', linewidth=2, label="ST-GNN (T=3)")
    
    # Highlight the darknet shutdown event (Concept Drift)
    plt.axvline(x=43, color='black', linestyle='--', alpha=0.6)
    plt.text(42.8, max(f1_scores)*0.85, 'Darknet Market\nShutdown (Step 43)', 
             horizontalalignment='right', fontsize=10, bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

    plt.title('Model Resilience Under Concept Drift (Test Set)')
    plt.xlabel('Chronological Time Step (approx. 2 weeks each)')
    plt.ylabel('Illicit F1-Score')
    plt.xticks(test_steps)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    
    # Save figure
    fig_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "figures"))
    os.makedirs(fig_dir, exist_ok=True)
    fig_path = os.path.join(fig_dir, "temporal_drift.png")
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"Drift chart saved to: {fig_path}")

if __name__ == "__main__":
    plot_concept_drift()