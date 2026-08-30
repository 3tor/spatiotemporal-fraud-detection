import os
import random
import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt

# Custom imports
from dataset import get_elliptic_dataset, build_temporal_snapshot_cache
from models.st_gnn import SpatioTemporalGNN
from loss import get_weighted_bce_loss
from evaluate import evaluate_stgnn, tune_optimal_threshold

def set_seed(seed=42):
    """Locks all random number generators for complete reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def run_ablation():
    print("Loading Dataset for Ablation Study...")
    data = get_elliptic_dataset()
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    data = data.to(device)
    
    snapshots = build_temporal_snapshot_cache(data)
    time_steps = sorted(snapshots.keys())
    
    criterion = get_weighted_bce_loss(device)
    labels = data.y.float().unsqueeze(1)
    
    lookbacks = [1, 3, 5]
    results = {}
    
    checkpoint_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "checkpoints"))
    os.makedirs(checkpoint_dir, exist_ok=True)

    for T in lookbacks:
        print(f"\n{'='*50}\nRUNNING ABLATION: Lookback Window T = {T}\n{'='*50}")
        
        # FIX Point 2: Lock seed BEFORE initializing model
        set_seed(42) 
        
        model = SpatioTemporalGNN(in_channels=data.x.size(1), spatial_dim=32, rnn_hidden=32).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
        
        best_val_f1 = 0.0
        optimal_tau = 0.50
        checkpoint_path = os.path.join(checkpoint_dir, f"STGNN_Ablation_T{T}.pt")
        
        # FIX Point 1: Train for 30 epochs (matches train.py)
        for epoch in range(1, 31):
            model.train()
            for t in time_steps:
                if t < T or t > 30:
                    continue
                
                node_mask_t = snapshots[t]['node_mask']
                train_mask_t = data.custom_train_mask & node_mask_t
                if train_mask_t.sum() == 0: continue
                
                x_seq = [snapshots[step]['x'] for step in range(t - T + 1, t + 1)]
                edge_idx_seq = [snapshots[step]['edge_index'] for step in range(t - T + 1, t + 1)]
                
                optimizer.zero_grad()
                logits = model(x_seq, edge_idx_seq)
                loss = criterion(logits[train_mask_t], labels[train_mask_t].to(device))
                loss.backward()
                optimizer.step()
                
            # Tune every 5 epochs
            if epoch % 5 == 0:
                val_metrics = evaluate_stgnn(model, data, snapshots, data.custom_val_mask, T, threshold=0.5)
                tau_star, tuned_f1 = tune_optimal_threshold(val_metrics["Raw_Probs"], val_metrics["Raw_Labels"])
                
                # FIX Point 3 (Part A): Save the best checkpoint
                if tuned_f1 > best_val_f1:
                    best_val_f1 = tuned_f1
                    optimal_tau = tau_star
                    torch.save(model.state_dict(), checkpoint_path)
                    print(f"Epoch {epoch:03d} | Tuned Val F1: {tuned_f1:.4f} (at τ={tau_star:.2f}) -> [!] Checkpoint Saved")
                else:
                    print(f"Epoch {epoch:03d} | Tuned Val F1: {tuned_f1:.4f} (at τ={tau_star:.2f})")

        # FIX Point 3 (Part B): Reload the best checkpoint before final evaluation
        if os.path.exists(checkpoint_path):
            model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
            print(f"--- Loaded Best Checkpoint (τ* = {optimal_tau:.2f}) ---")
            
        print(f"--- Final Test Set Evaluation (Steps 35-49) for T={T} ---")
        test_metrics = evaluate_stgnn(model, data, snapshots, data.custom_test_mask, T, threshold=optimal_tau)
        
        print(f"Test F1 (Illicit): {test_metrics['F1 (Illicit)']:.4f}")
        print(f"Test Precision:    {test_metrics['Precision']:.4f}")
        print(f"Test Recall:       {test_metrics['Recall']:.4f}")
        print(f"Test PR-AUC:       {test_metrics['PR-AUC']:.4f}")
        results[T] = test_metrics
        
    # --- PLOT ABLATION RESULTS ---
    print("\nGenerating Ablation Chart...")
    labels = ['T=1 (No Memory)', 'T=3 (Short)', 'T=5 (Long)']
    f1_scores = [results[1]['F1 (Illicit)'], results[3]['F1 (Illicit)'], results[5]['F1 (Illicit)']]
    precisions = [results[1]['Precision'], results[3]['Precision'], results[5]['Precision']]
    recalls = [results[1]['Recall'], results[3]['Recall'], results[5]['Recall']]
    pr_aucs = [results[1]['PR-AUC'], results[3]['PR-AUC'], results[5]['PR-AUC']]

    x = np.arange(len(labels))
    width = 0.2

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - 1.5 * width, f1_scores, width, label='F1-Score (Illicit)', color='#2ca02c')
    ax.bar(x - 0.5 * width, precisions, width, label='Precision', color='#1f77b4')
    ax.bar(x + 0.5 * width, recalls, width, label='Recall', color='#ff7f0e')
    ax.bar(x + 1.5 * width, pr_aucs, width, label='PR-AUC', color='#d62728')

    ax.set_ylabel('Score')
    ax.set_title('Ablation Study: Impact of Temporal Lookback Window (T)')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    fig_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "figures"))
    os.makedirs(fig_dir, exist_ok=True)
    fig_path = os.path.join(fig_dir, "ablation_lookback.png")
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"Ablation chart saved to: {fig_path}")

if __name__ == "__main__":
    run_ablation()