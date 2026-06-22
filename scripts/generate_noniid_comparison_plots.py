"""
Non-IID Comparison Plots Generator
==================================
Generates publication-quality matplotlib figures comparing the results of
IID vs Non-IID (Dirichlet alpha=0.5) data partitioning across various 
network impairment experiments.

Reads CSV metrics from experiments/results/ and outputs generated plots
to experiments/results/comparison/.

Author: Frank Meyo
Project: FL Network Simulator — UNZA CS Final Year Project 2026
"""

import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def load_data(filepath: Path) -> pd.DataFrame:
    """Load CSV data if it exists, otherwise return an empty DataFrame."""
    if filepath.exists():
        return pd.read_csv(filepath)
    else:
        print(f"Warning: Data file not found at {filepath}")
        return pd.DataFrame()


def generate_plots() -> None:
    # Set up directories
    results_dir = Path("experiments/results")
    out_dir = results_dir / "comparison"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Apply professional seaborn styling
    sns.set_theme(style="whitegrid", palette="deep", font_scale=1.2)

    # -----------------------------------------------------------------------
    # Load Data
    # -----------------------------------------------------------------------
    file_map = {
        "baseline_fedavg": ("baseline_fedavg_metrics.csv", "baseline_fedavg_noniid_metrics.csv"),
        "severe_fedavg": ("severe_disruption_fedavg_metrics.csv", "severe_disruption_fedavg_noniid_metrics.csv"),
        "severe_fedprox": ("severe_disruption_fedprox_metrics.csv", "severe_disruption_fedprox_noniid_metrics.csv"),
        "rural_fedprox": ("rural_zambia_fedprox_metrics.csv", "rural_zambia_fedprox_noniid_metrics.csv"),
    }

    data = {}
    for key, (iid_file, noniid_file) in file_map.items():
        data[key] = {
            "iid": load_data(results_dir / iid_file),
            "noniid": load_data(results_dir / noniid_file)
        }

    # -----------------------------------------------------------------------
    # Figure 1: IID vs Non-IID Accuracy Curves (2x2 Grid)
    # -----------------------------------------------------------------------
    fig1, axs = plt.subplots(2, 2, figsize=(14, 8), dpi=150)
    fig1.suptitle("IID vs Non-IID Data Distribution Impact on Convergence", fontweight="bold")
    
    subplots_config = [
        ("baseline_fedavg", axs[0, 0], "Baseline FedAvg"),
        ("severe_fedavg", axs[0, 1], "Severe FedAvg"),
        ("severe_fedprox", axs[1, 0], "Severe FedProx"),
        ("rural_fedprox", axs[1, 1], "Rural FedProx")
    ]
    
    for key, ax, title in subplots_config:
        df_iid = data[key]["iid"]
        df_noniid = data[key]["noniid"]
        
        if not df_iid.empty:
            ax.plot(df_iid["round_num"], df_iid["global_accuracy"], 
                    linestyle='-', color='blue', linewidth=2, label='IID')
        if not df_noniid.empty:
            ax.plot(df_noniid["round_num"], df_noniid["global_accuracy"], 
                    linestyle='--', color='orange', linewidth=2, label='Non-IID')
            
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Communication Round")
        ax.set_ylabel("Global Accuracy")
        ax.set_ylim(0, 1.05)
        ax.legend()
        
    fig1.tight_layout()
    fig1.savefig(out_dir / "iid_vs_noniid_accuracy.png")
    plt.close(fig1)
    print("Saved Figure 1: iid_vs_noniid_accuracy.png")

    # -----------------------------------------------------------------------
    # Figure 2: FedAvg vs FedProx (IID vs Non-IID Under Severe Disruption)
    # -----------------------------------------------------------------------
    fig2, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(12, 6), dpi=150)
    fig2.suptitle("FedAvg vs FedProx: IID vs Non-IID Under Severe Disruption", fontweight="bold")
    
    # Left: IID
    df_severe_fedavg_iid = data["severe_fedavg"]["iid"]
    df_severe_fedprox_iid = data["severe_fedprox"]["iid"]
    
    if not df_severe_fedavg_iid.empty:
        ax_left.plot(df_severe_fedavg_iid["round_num"], df_severe_fedavg_iid["global_accuracy"], 
                     marker='o', color='red', label='FedAvg')
    if not df_severe_fedprox_iid.empty:
        ax_left.plot(df_severe_fedprox_iid["round_num"], df_severe_fedprox_iid["global_accuracy"], 
                     marker='^', color='purple', label='FedProx')
    
    ax_left.set_title("IID Data Partitioning")
    ax_left.set_xlabel("Communication Round")
    ax_left.set_ylabel("Global Accuracy")
    ax_left.set_ylim(0, 1.05)
    ax_left.legend()

    # Right: Non-IID
    df_severe_fedavg_noniid = data["severe_fedavg"]["noniid"]
    df_severe_fedprox_noniid = data["severe_fedprox"]["noniid"]
    
    if not df_severe_fedavg_noniid.empty:
        ax_right.plot(df_severe_fedavg_noniid["round_num"], df_severe_fedavg_noniid["global_accuracy"], 
                      marker='o', color='red', label='FedAvg')
    if not df_severe_fedprox_noniid.empty:
        ax_right.plot(df_severe_fedprox_noniid["round_num"], df_severe_fedprox_noniid["global_accuracy"], 
                      marker='^', color='purple', label='FedProx')
                      
    ax_right.set_title("Non-IID Data Partitioning")
    ax_right.set_xlabel("Communication Round")
    ax_right.set_ylabel("Global Accuracy")
    ax_right.set_ylim(0, 1.05)
    ax_right.legend()

    fig2.tight_layout()
    fig2.savefig(out_dir / "noniid_fedavg_vs_fedprox_severe.png")
    plt.close(fig2)
    print("Saved Figure 2: noniid_fedavg_vs_fedprox_severe.png")

    # -----------------------------------------------------------------------
    # Figure 3: Final Accuracy: IID vs Non-IID Comparison Bar Chart
    # -----------------------------------------------------------------------
    fig3, ax3 = plt.subplots(figsize=(12, 6), dpi=150)
    
    labels = ["Baseline", "Severe FedAvg", "Severe FedProx", "Rural FedProx"]
    keys = ["baseline_fedavg", "severe_fedavg", "severe_fedprox", "rural_fedprox"]
    
    iid_accs = []
    noniid_accs = []
    
    for key in keys:
        df_iid = data[key]["iid"]
        df_noniid = data[key]["noniid"]
        
        iid_acc = df_iid["global_accuracy"].iloc[-1] if not df_iid.empty else 0
        noniid_acc = df_noniid["global_accuracy"].iloc[-1] if not df_noniid.empty else 0
        
        iid_accs.append(iid_acc)
        noniid_accs.append(noniid_acc)

    x = range(len(labels))
    width = 0.35
    
    bar1 = ax3.bar([i - width/2 for i in x], iid_accs, width, label='IID', color='blue', alpha=0.8)
    bar2 = ax3.bar([i + width/2 for i in x], noniid_accs, width, label='Non-IID', color='orange', alpha=0.8)
    
    ax3.set_title("Final Accuracy: IID vs Non-IID Comparison", pad=15, fontweight="bold")
    ax3.set_xticks(x)
    ax3.set_xticklabels(labels)
    ax3.set_ylabel("Final Global Accuracy")
    ax3.set_ylim(0.95, 1.0)
    ax3.legend(loc='lower left')
    
    # Add value labels
    for bar in bar1 + bar2:
        height = bar.get_height()
        if height > 0:
            ax3.annotate(f"{height:.4f}",
                         xy=(bar.get_x() + bar.get_width() / 2, height),
                         xytext=(0, 3), 
                         textcoords="offset points",
                         ha='center', va='bottom', fontweight="bold", fontsize=9)

    fig3.tight_layout()
    fig3.savefig(out_dir / "accuracy_degradation_bar.png")
    plt.close(fig3)
    print("Saved Figure 3: accuracy_degradation_bar.png")


if __name__ == "__main__":
    print("Generating non-IID comparison plots...")
    generate_plots()
    print("Done!")
