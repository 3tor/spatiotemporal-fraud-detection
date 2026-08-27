import torch
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, average_precision_score

def evaluate_model(model, data, mask, threshold=0.5):
    """
    Evaluates the model specifically on the illicit class (y = 1).
    """
    model.eval()
    with torch.no_grad():
        # Forward pass
        logits = model(data.x, data.edge_index)
        probs = torch.sigmoid(logits).squeeze()
        
        # Filter predictions and labels using the split mask
        masked_probs = probs[mask].cpu().numpy()
        masked_labels = data.y[mask].cpu().numpy()
        
        # Binarize predictions based on the threshold
        preds = (masked_probs >= threshold).astype(int)
        
        # Calculate strict minority-class metrics (pos_label=1)
        f1 = f1_score(masked_labels, preds, pos_label=1, zero_division=0)
        precision = precision_score(masked_labels, preds, pos_label=1, zero_division=0)
        recall = recall_score(masked_labels, preds, pos_label=1, zero_division=0)
        
        # Calculate PR-AUC (Average Precision) across all thresholds
        pr_auc = average_precision_score(masked_labels, masked_probs, pos_label=1)
        
    return {
        "F1 (Illicit)": f1,
        "Precision": precision,
        "Recall": recall,
        "PR-AUC": pr_auc
    }

def tune_optimal_threshold(model, data, val_mask, step=0.05):
    """
    Sweeps thresholds τ ∈ [0.10, 0.90] on validation data to maximize Illicit F1.
    """
    model.eval()
    with torch.no_grad():
        logits = model(data.x, data.edge_index) if not isinstance(model, SpatioTemporalGNN_Wrapper) else model.predict(data)
        probs = torch.sigmoid(logits).squeeze()
        
        masked_probs = probs[val_mask].cpu().numpy()
        masked_labels = data.y[val_mask].cpu().numpy()
        
        best_threshold = 0.50
        best_val_f1 = 0.0
        
        for tau in np.arange(0.10, 0.90, step):
            preds = (masked_probs >= tau).astype(int)
            f1 = f1_score(masked_labels, preds, pos_label=1, zero_division=0)
            if f1 > best_val_f1:
                best_val_f1 = f1
                best_threshold = tau
                
    return best_threshold, best_val_f1

def tune_optimal_threshold(probs, labels, step=0.05):
    """
    Sweeps thresholds τ ∈ [0.10, 0.90] to maximize Illicit F1.
    """
    best_threshold = 0.50
    best_f1 = 0.0
    
    for tau in np.arange(0.10, 0.90, step):
        preds = (probs >= tau).astype(int)
        f1 = f1_score(labels, preds, pos_label=1, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = tau
            
    return best_threshold, best_f1

def evaluate_stgnn(model, data, snapshots, mask, lookback=3, threshold=0.5):
    """
    Evaluates the ST-GNN by rolling a temporal window forward.
    """
    model.eval()
    all_probs = []
    all_labels = []
    
    time_steps = sorted(snapshots.keys())
    
    with torch.no_grad():
        for t in time_steps:
            if t < lookback:
                continue
                
            # Filter mask for this specific time step
            node_mask_t = snapshots[t]['node_mask']
            eval_mask_t = mask & node_mask_t
            
            if eval_mask_t.sum() == 0:
                continue
                
            # Build sequence
            x_seq = [snapshots[step]['x'] for step in range(t - lookback + 1, t + 1)]
            edge_idx_seq = [snapshots[step]['edge_index'] for step in range(t - lookback + 1, t + 1)]
            
            # Predict
            logits = model(x_seq, edge_idx_seq)
            probs = torch.sigmoid(logits).squeeze()
            
            all_probs.append(probs[eval_mask_t].cpu())
            all_labels.append(data.y[eval_mask_t].cpu())
            
    if len(all_probs) == 0:
        return {"F1 (Illicit)": 0, "PR-AUC": 0}
        
    final_probs = torch.cat(all_probs).numpy()
    final_labels = torch.cat(all_labels).numpy()
    
    preds = (final_probs >= threshold).astype(int)
    
    return {
        "F1 (Illicit)": f1_score(final_labels, preds, pos_label=1, zero_division=0),
        "Precision": precision_score(final_labels, preds, pos_label=1, zero_division=0),
        "Recall": recall_score(final_labels, preds, pos_label=1, zero_division=0),
        "PR-AUC": average_precision_score(final_labels, final_probs, pos_label=1),
        "Raw_Probs": final_probs,
        "Raw_Labels": final_labels
    }