# Spatio-Temporal Graph Neural Networks for Financial Anomaly and Fraud Detection

This repository contains the official PyTorch implementation of the research project: **Spatio-Temporal Graph Neural Networks for Financial Anomaly and Fraud Detection**. 

This project investigates the application of a decoupled ST-GNN (Spatial GCN + Temporal GRU) on the Elliptic Bitcoin dataset to detect illicit transactions (money laundering) while remaining resilient to rapid topological shifts (concept drift) caused by darknet market shutdowns.

---

## 🚀 Quick Evaluation for Reviewers

A pre-trained ST-GNN model checkpoint is included in the `checkpoints/` directory. To evaluate the model's out-of-sample performance instantly without retraining, follow the setup instructions below and run the evaluation script:

```bash
# 1. Clone the repository and navigate to the project root
git clone [https://github.com/3tor/spatiotemporal-fraud-detection.git](https://github.com/3tor/spatiotemporal-fraud-detection.git)
cd spatiotemporal-fraud-detection

# 2. Run the instant evaluation script
cd src
python eval_checkpoint.py

How to run from scratch
# 3. Test Data Pipeline
cd src
python dataset.py

# 4. Train Baselines and ST-GNN
cd src
python train.py

# 5. Generate Concept Drift Plot
cd src
python plot_drift.py

# 6. Run Ablation Study
cd src
python ablation.py

spatiotemporal-fraud-detection/
├── checkpoints/                # Saved model weights (.pt files)
│   └── SpatioTemporalGNN_best.pt 
├── data/                       # Dataset directory (auto-downloads Elliptic data)
├── figures/                    # Generated charts for the academic report
│   ├── ablation_lookback.png
│   └── temporal_drift.png                
├── src/                        # Core source code
│   ├── models/
│   │   ├── baselines.py        # Tabular MLP and Static GCN architectures
│   │   └── st_gnn.py           # Proposed decoupled ST-GNN architecture
│   ├── ablation.py             # Script to run T=1, T=3, T=5 lookback experiments
│   ├── dataset.py              # Data ingestion and chronological masking
│   ├── eval_checkpoint.py      # Quick-start script for grading team
│   ├── evaluate.py             # Validation threshold tuning and metric calculation
│   ├── loss.py                 # Weighted BCE loss for 98% class imbalance
│   ├── plot_drift.py           # Generates per-timestep trajectory charts
│   └── train.py                # Master training loop
└── README.md