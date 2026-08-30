import os
import random
import numpy as np
import torch
import torch.optim as optim

# Import custom modules
from dataset import get_elliptic_dataset, build_temporal_snapshot_cache
from models.baselines import BaselineMLP, BaselineGCN
from models.st_gnn import SpatioTemporalGNN
from loss import get_weighted_bce_loss
from evaluate import evaluate_model, evaluate_stgnn, tune_optimal_threshold

def set_seed(seed=42):
    """Locks all random number generators for complete reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def train_baseline_model(model, data, epochs=100, lr=0.01):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = model.to(device)
    data = data.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = get_weighted_bce_loss(device)
    labels = data.y.float().unsqueeze(1)
    
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        
        logits = model(data.x, data.edge_index)
        train_logits = logits[data.custom_train_mask]
        train_labels = labels[data.custom_train_mask]
        
        loss = criterion(train_logits, train_labels)
        loss.backward()
        optimizer.step()
        
    # Final Out-of-Sample Test Evaluation
    print(f"\n--- Final Test Set Evaluation ({model.__class__.__name__}) ---")
    test_metrics = evaluate_model(model, data, data.custom_test_mask)
    for k, v in test_metrics.items():
        print(f"Test {k}: {v:.4f}")
    return test_metrics

def train_stgnn_model(model, data, epochs=30, lr=0.001, lookback=3):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = model.to(device)
    data = data.to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = get_weighted_bce_loss(device)
    labels = data.y.float().unsqueeze(1)
    
    snapshots = build_temporal_snapshot_cache(data)
    time_steps = sorted(snapshots.keys())
    
    best_val_f1 = 0.0
    optimal_tau = 0.50
    
    for epoch in range(1, epochs + 1):
        model.train()
        
        for t in time_steps:
            if t < lookback or t > 30: # Only train on steps <= 30
                continue
                
            node_mask_t = snapshots[t]['node_mask']
            train_mask_t = data.custom_train_mask & node_mask_t
            
            if train_mask_t.sum() == 0:
                continue
                
            x_seq = [snapshots[step]['x'] for step in range(t - lookback + 1, t + 1)]
            edge_idx_seq = [snapshots[step]['edge_index'] for step in range(t - lookback + 1, t + 1)]
            
            optimizer.zero_grad()
            logits = model(x_seq, edge_idx_seq)
            
            loss = criterion(logits[train_mask_t], labels[train_mask_t])
            loss.backward()
            optimizer.step()
            
        # Evaluate every 5 epochs and tune threshold
        if epoch % 5 == 0:
            val_metrics = evaluate_stgnn(model, data, snapshots, data.custom_val_mask, lookback, threshold=0.5)
            tau_star, tuned_f1 = tune_optimal_threshold(val_metrics["Raw_Probs"], val_metrics["Raw_Labels"])
            
            # --- CHECKPOINT SAVING LOGIC ---
            if tuned_f1 > best_val_f1:
                best_val_f1 = tuned_f1
                optimal_tau = tau_star
                
                checkpoint_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "checkpoints"))
                os.makedirs(checkpoint_dir, exist_ok=True)
                checkpoint_path = os.path.join(checkpoint_dir, f"SpatioTemporalGNN_T{lookback}_best.pt")
                torch.save(model.state_dict(), checkpoint_path)
                print(f"Epoch {epoch:03d} | Tuned Val F1: {tuned_f1:.4f} (at τ={tau_star:.2f}) -> [!] Checkpoint Saved")

    # --- Out-of-Sample Final Evaluation ---
    print(f"\n--- Final Test Set Evaluation (ST-GNN T={lookback}) ---")
    print(f"Applying optimal decision boundary: τ* = {optimal_tau:.2f}")
    
    # Load the best weights before final test
    checkpoint_path = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "checkpoints")), f"{model.__class__.__name__}_best.pt")
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, weights_only=True))
        
    test_metrics = evaluate_stgnn(model, data, snapshots, data.custom_test_mask, lookback, threshold=optimal_tau)
    
    print(f"Test F1 (Illicit): {test_metrics['F1 (Illicit)']:.4f}")
    print(f"Test Precision:    {test_metrics['Precision']:.4f}")
    print(f"Test Recall:       {test_metrics['Recall']:.4f}")
    print(f"Test PR-AUC:       {test_metrics['PR-AUC']:.4f}")
    
    return test_metrics

if __name__ == "__main__":
    print("Loading Dataset...")
    data = get_elliptic_dataset()
    
    # 2. Train Tabular MLP Baseline
    print("\n" + "="*50 + "\nBASELINE 1: Multi-Layer Perceptron (Tabular)\n" + "="*50)
    set_seed(42) # Reset seed for MLP
    mlp_model = BaselineMLP(in_channels=data.x.size(1))
    train_baseline_model(mlp_model, data, epochs=100)
    
    # 3. Train Static GCN Baseline
    print("\n" + "="*50 + "\nBASELINE 2: Static Graph Convolutional Network (GCN)\n" + "="*50)
    set_seed(42) # Reset seed for GCN
    gcn_model = BaselineGCN(in_channels=data.x.size(1))
    train_baseline_model(gcn_model, data, epochs=100)
    
    # 4. Train ST-GNN
    print("\n" + "="*50 + "\nPROPOSED MODEL: Spatio-Temporal GNN (T=3)\n" + "="*50)
    set_seed(42) # <-- FIX: Reset seed right before ST-GNN!
    st_model = SpatioTemporalGNN(in_channels=data.x.size(1), spatial_dim=32, rnn_hidden=32)
    train_stgnn_model(st_model, data, epochs=30, lookback=3)