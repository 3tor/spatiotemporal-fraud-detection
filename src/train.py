import torch
import torch.optim as optim

# Import custom modules
from dataset import get_elliptic_dataset, build_temporal_snapshot_cache
from models.st_gnn import SpatioTemporalGNN
from loss import get_weighted_bce_loss
from evaluate import evaluate_stgnn, tune_optimal_threshold

def train_stgnn_model(model, data, epochs=50, lr=0.001, lookback=3):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"--- Training ST-GNN on Device: {device} ---")
    
    model = model.to(device)
    data = data.to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = get_weighted_bce_loss(device)
    labels = data.y.float().unsqueeze(1)
    
    print("Building temporal snapshot cache...")
    snapshots = build_temporal_snapshot_cache(data)
    time_steps = sorted(snapshots.keys())
    
    best_val_f1 = 0.0
    optimal_tau = 0.50
    
    print(f"Starting Training: {model.__class__.__name__} with Lookback T={lookback}")
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        batches = 0
        
        # Mini-batch by sliding temporal window
        for t in time_steps:
            if t < lookback or t > 30: # Only train on steps <= 30
                continue
                
            node_mask_t = snapshots[t]['node_mask']
            train_mask_t = data.custom_train_mask & node_mask_t
            
            if train_mask_t.sum() == 0:
                continue
                
            # 1. Build chronological sequence [G_{t-2}, G_{t-1}, G_t]
            x_seq = [snapshots[step]['x'] for step in range(t - lookback + 1, t + 1)]
            edge_idx_seq = [snapshots[step]['edge_index'] for step in range(t - lookback + 1, t + 1)]
            
            # 2. Forward pass
            optimizer.zero_grad()
            logits = model(x_seq, edge_idx_seq)
            
            # 3. Loss & Backprop ONLY on known nodes at time step t
            loss = criterion(logits[train_mask_t], labels[train_mask_t])
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            batches += 1
            
        avg_loss = epoch_loss / max(1, batches)
        
        # Evaluate every 5 epochs
        if epoch % 5 == 0:
            # First, evaluate on validation set with default threshold to get raw probs
            val_metrics = evaluate_stgnn(model, data, snapshots, data.custom_val_mask, lookback, threshold=0.5)
            
            # Tune threshold tau on Validation probabilities
            tau_star, tuned_f1 = tune_optimal_threshold(val_metrics["Raw_Probs"], val_metrics["Raw_Labels"])
            
            print(f"Epoch {epoch:03d} | Avg Loss: {avg_loss:.4f} | Val PR-AUC: {val_metrics['PR-AUC']:.4f} | Tuned Val F1: {tuned_f1:.4f} (at τ={tau_star:.2f})")
            
            if tuned_f1 > best_val_f1:
                best_val_f1 = tuned_f1
                optimal_tau = tau_star

    # --- Out-of-Sample Final Evaluation ---
    print(f"\n--- Final Test Set Evaluation (Steps 35-49) ---")
    print(f"Applying optimal decision boundary: τ* = {optimal_tau:.2f}")
    
    test_metrics = evaluate_stgnn(model, data, snapshots, data.custom_test_mask, lookback, threshold=optimal_tau)
    
    print(f"Test F1 (Illicit): {test_metrics['F1 (Illicit)']:.4f}")
    print(f"Test Precision:    {test_metrics['Precision']:.4f}")
    print(f"Test Recall:       {test_metrics['Recall']:.4f}")
    print(f"Test PR-AUC:       {test_metrics['PR-AUC']:.4f}")
    print("-" * 50)
    
    return test_metrics

if __name__ == "__main__":
    data = get_elliptic_dataset()
    
    print("\n" + "="*50)
    print("PROPOSED MODEL: Spatio-Temporal GNN (T=3)")
    print("="*50)
    
    # Initialize the decoupled ST-GNN
    st_model = SpatioTemporalGNN(in_channels=data.x.size(1), spatial_dim=32, rnn_hidden=32)
    
    # Train and evaluate
    train_stgnn_model(st_model, data, epochs=30, lookback=3)