import os
import torch
import matplotlib.pyplot as plt
import seaborn as sns

from dataset import get_elliptic_dataset
from models.st_gnn import SpatioTemporalGNN
from train import train_stgnn_model

# --- Dynamically resolve absolute paths ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
FIGURES_DIR = os.path.join(PROJECT_ROOT, "figures")

def plot_temporal_drift(metrics_dict, save_path=os.path.join(FIGURES_DIR, "temporal_drift.png")):
    """
    Generates a line plot showing the F1 score at each test time step.
    """
    # Ensure the figures directory actually exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 5))
    
    # Extract time steps and F1 scores
    steps = list(metrics_dict.keys())
    f1_scores = list(metrics_dict.values())
    
    # Plot the ST-GNN trajectory
    plt.plot(steps, f1_scores, marker='o', linestyle='-', linewidth=2, color='#2c3e50', label="ST-GNN (T=3)")
    
    # Highlight the "Darknet Shutdown" structural shift (~Step 43)
    plt.axvline(x=43, color='#e74c3c', linestyle='--', alpha=0.7, label="Major Market Shutdown (Step 43)")
    
    plt.title("Model Resilience Under Temporal Concept Drift (Steps 35-49)", fontsize=14, pad=15)
    plt.xlabel("Temporal Snapshot (t)", fontsize=12)
    plt.ylabel("Minority F1-Score (Illicit)", fontsize=12)
    plt.ylim(0, 1.0)
    plt.xticks(range(35, 50))
    plt.legend(loc="lower left")
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"\n✅ Plot saved successfully to: {save_path}")

if __name__ == "__main__":
    # 1. Load data
    data = get_elliptic_dataset()
    
    # 2. Re-initialize and train the model quickly
    st_model = SpatioTemporalGNN(in_channels=data.x.size(1), spatial_dim=32, rnn_hidden=32)
    test_metrics = train_stgnn_model(st_model, data, epochs=30, lookback=3)
    
    # 3. Generate the drift plot
    if "Per_Step_F1" in test_metrics:
        plot_temporal_drift(test_metrics["Per_Step_F1"])
    else:
        print("Error: 'Per_Step_F1' not found in test metrics.")