import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from dataset import get_elliptic_dataset
from models.st_gnn import SpatioTemporalGNN
from train import train_stgnn_model

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
FIGURES_DIR = os.path.join(PROJECT_ROOT, "figures")

def plot_ablation_results(results_df, save_path=os.path.join(FIGURES_DIR, "ablation_lookback.png")):
    """
    Plots a grouped bar chart comparing metrics across different lookback windows.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Set up the matplotlib figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Metrics to plot
    metrics = ['Test F1 (Illicit)', 'Test Precision', 'Test Recall', 'Test PR-AUC']
    x = np.arange(len(metrics))
    width = 0.25  # width of the bars
    
    # Extract data
    t1_scores = results_df.loc[results_df['Lookback (T)'] == 1, metrics].values[0]
    t3_scores = results_df.loc[results_df['Lookback (T)'] == 3, metrics].values[0]
    t5_scores = results_df.loc[results_df['Lookback (T)'] == 5, metrics].values[0]
    
    # Plot bars
    rects1 = ax.bar(x - width, t1_scores, width, label='T=1 (No Memory)', color='#bdc3c7')
    rects2 = ax.bar(x,         t3_scores, width, label='T=3 (Short Memory)', color='#3498db')
    rects3 = ax.bar(x + width, t5_scores, width, label='T=5 (Long Memory)', color='#2c3e50')
    
    # Formatting
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Ablation Study: Impact of Temporal Lookback Window (T)', fontsize=14, pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(['F1-Score', 'Precision', 'Recall', 'PR-AUC'], fontsize=11)
    ax.set_ylim(0, 1.0)
    ax.legend()
    
    # Add numerical labels on top of bars
    for rects in [rects1, rects2, rects3]:
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"\n✅ Ablation chart saved to: {save_path}")

if __name__ == "__main__":
    data = get_elliptic_dataset()
    lookback_windows = [1, 3, 5]
    results = []

    for t in lookback_windows:
        print("\n" + "="*50)
        print(f"RUNNING ABLATION: Lookback Window T = {t}")
        print("="*50)
        
        # Initialize fresh model for each run
        model = SpatioTemporalGNN(in_channels=data.x.size(1), spatial_dim=32, rnn_hidden=32)
        
        # Train (Using fewer epochs here just to get the ablation done quickly, 
        # for your final paper you might want to push this back to 30+)
        metrics = train_stgnn_model(model, data, epochs=25, lookback=t)
        
        results.append({
            'Lookback (T)': t,
            'Test F1 (Illicit)': metrics['F1 (Illicit)'],
            'Test Precision': metrics['Precision'],
            'Test Recall': metrics['Recall'],
            'Test PR-AUC': metrics['PR-AUC']
        })

    # Save and Plot
    results_df = pd.DataFrame(results)
    print("\n--- Final Ablation Results ---")
    print(results_df.to_string(index=False))
    
    plot_ablation_results(results_df)