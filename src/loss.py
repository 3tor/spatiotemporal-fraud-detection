import torch
import torch.nn as nn

def get_weighted_bce_loss(device):
    """
    Returns a BCE loss function weighted for the Elliptic dataset's class imbalance.
    Licit nodes (Class 0): ~42,019
    Illicit nodes (Class 1): ~4,545
    Ratio (Negative / Positive) ≈ 9.24
    """
    # Weight the positive (illicit) class higher to penalize missing fraud
    imbalance_ratio = torch.tensor([9.24]).to(device)
    
    # BCEWithLogitsLoss is numerically more stable than Sigmoid + BCELoss
    criterion = nn.BCEWithLogitsLoss(pos_weight=imbalance_ratio)
    return criterion