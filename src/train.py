import torch
import torch.optim as optim
import matplotlib.pyplot as plt

# Import custom modules
from dataset import get_elliptic_dataset
from models.baselines import BaselineMLP, BaselineGCN
from loss import get_weighted_bce_loss
from evaluate import evaluate_model

def train_model(model, data, epochs=100, lr=0.01, weight_decay=1e-5):
    # Setup M4 Mac hardware acceleration (MPS) or fallback to CPU
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"--- Training on Device: {device} ---")
    
    model = model.to(device)
    data = data.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = get_weighted_bce_loss(device)
    
    # Ensure y is a float tensor of shape (N, 1) for BCEWithLogitsLoss
    labels = data.y.float().unsqueeze(1)
    
    best_val_f1 = 0.0
    
    print(f"Starting Training: {model.__class__.__name__}")
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        
        # Forward pass (Static GCN needs edge_index, MLP ignores it)
        logits = model(data.x, data.edge_index)
        
        # Apply the semi-supervised train mask
        train_logits = logits[data.custom_train_mask]
        train_labels = labels[data.custom_train_mask]
        
        # Compute loss ONLY on known training nodes
        loss = criterion(train_logits, train_labels)
        
        # Backpropagation
        loss.backward()
        optimizer.step()
        
        # Evaluate on Validation Set every 10 epochs
        if epoch % 10 == 0:
            val_metrics = evaluate_model(model, data, data.custom_val_mask)
            val_f1 = val_metrics['F1 (Illicit)']
            
            print(f"Epoch {epoch:03d} | Loss: {loss.item():.4f} | Val F1 (Illicit): {val_f1:.4f} | Val PR-AUC: {val_metrics['PR-AUC']:.4f}")
            
            # Save the best model
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                # Optional: torch.save(model.state_dict(), f"../data/{model.__class__.__name__}_best.pt")

    # Final Out-of-Sample Test Evaluation
    print(f"\n--- Final Test Set Evaluation ({model.__class__.__name__}) ---")
    test_metrics = evaluate_model(model, data, data.custom_test_mask)
    for k, v in test_metrics.items():
        print(f"Test {k}: {v:.4f}")
    print("-" * 50)
    
    return test_metrics

if __name__ == "__main__":
    # 1. Load the data using our pipeline from Phase 1
    data = get_elliptic_dataset()
    
    # 2. Train the Tier 1 Baseline (MLP - No Graph Features)
    print("\n" + "="*50)
    print("BASELINE 1: Multi-Layer Perceptron (Tabular)")
    print("="*50)
    mlp_model = BaselineMLP(in_channels=data.x.size(1))
    train_model(mlp_model, data, epochs=100)
    
    # 3. Train the Tier 2 Baseline (Static GCN - Graph Topology added)
    print("\n" + "="*50)
    print("BASELINE 2: Static Graph Convolutional Network (GCN)")
    print("="*50)
    gcn_model = BaselineGCN(in_channels=data.x.size(1))
    train_model(gcn_model, data, epochs=100)