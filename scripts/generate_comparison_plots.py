"""
Comparison Plots Generator
===========================
Generates publication-quality matplotlib figures comparing the results of
various Federated Learning network impairment experiments.

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
    files = {
        "baseline_fedavg": results_dir / "baseline_fedavg_metrics.csv",
        "urban_fedavg": results_dir / "urban_zambia_fedavg_metrics.csv",
        "rural_fedavg": results_dir / "rural_zambia_fedavg_metrics.csv",
        "severe_fedavg": results_dir / "severe_disruption_fedavg_metrics.csv",
        "rural_fedprox": results_dir / "rural_zambia_fedprox_metrics.csv",
        "severe_fedprox": results_dir / "severe_disruption_fedprox_metrics.csv",
    }

    data = {name: load_data(path) for name, path in files.items()}

    # Check if we have data to plot (in case the script is run before experiments)
    if all(df.empty for df in data.values()):
        print("No CSV files found. Please run the experiments first.")
        return

    # -----------------------------------------------------------------------
    # Figure 1: Network Comparison (Accuracy)
    # -----------------------------------------------------------------------
    fig1, ax1 = plt.subplots(figsize=(12, 6), dpi=150)
    
    colors = {
        "baseline_fedavg": "blue",
        "urban_fedavg": "green",
        "rural_fedavg": "orange",
        "severe_fedavg": "red"
    }
    
    labels = {
        "baseline_fedavg": "Baseline (Perfect Network)",
        "urban_fedavg": "Urban Zambia",
        "rural_fedavg": "Rural Zambia",
        "severe_fedavg": "Severe Disruption"
    }

    for key in ["baseline_fedavg", "urban_fedavg", "rural_fedavg", "severe_fedavg"]:
        df = data[key]
        if not df.empty:
            ax1.plot(df["round_num"], df["global_accuracy"], marker='o', 
                     linewidth=2, color=colors[key], label=labels[key])

    ax1.set_title("FedAvg Accuracy Under SSA Network Conditions", pad=15, fontweight="bold")
    ax1.set_xlabel("Communication Round")
    ax1.set_ylabel("Global Accuracy")
    ax1.set_ylim(0, 1.05)
    ax1.legend()
    fig1.tight_layout()
    fig1.savefig(out_dir / "network_comparison_accuracy.png")
    plt.close(fig1)
    print("Saved Figure 1: network_comparison_accuracy.png")

    # -----------------------------------------------------------------------
    # Figure 2: FedAvg vs FedProx
    # -----------------------------------------------------------------------
    fig2, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(12, 6), dpi=150)
    fig2.suptitle("FedAvg vs FedProx Under Network Impairment", fontweight="bold")

    # Left: Rural
    if not data["rural_fedavg"].empty:
        ax_left.plot(data["rural_fedavg"]["round_num"], data["rural_fedavg"]["global_accuracy"], 
                     marker='o', label="FedAvg", color="orange")
    if not data["rural_fedprox"].empty:
        ax_left.plot(data["rural_fedprox"]["round_num"], data["rural_fedprox"]["global_accuracy"], 
                     marker='^', label="FedProx (μ=0.01)", color="purple")
    ax_left.set_title("Rural Zambia Conditions")
    ax_left.set_xlabel("Communication Round")
    ax_left.set_ylabel("Global Accuracy")
    ax_left.set_ylim(0, 1.05)
    ax_left.legend()

    # Right: Severe
    if not data["severe_fedavg"].empty:
        ax_right.plot(data["severe_fedavg"]["round_num"], data["severe_fedavg"]["global_accuracy"], 
                      marker='o', label="FedAvg", color="red")
    if not data["severe_fedprox"].empty:
        ax_right.plot(data["severe_fedprox"]["round_num"], data["severe_fedprox"]["global_accuracy"], 
                      marker='^', label="FedProx (μ=0.01)", color="purple")
    ax_right.set_title("Severe Disruption Conditions")
    ax_right.set_xlabel("Communication Round")
    ax_right.set_ylabel("Global Accuracy")
    ax_right.set_ylim(0, 1.05)
    ax_right.legend()

    fig2.tight_layout()
    fig2.savefig(out_dir / "fedavg_vs_fedprox.png")
    plt.close(fig2)
    print("Saved Figure 2: fedavg_vs_fedprox.png")

    # -----------------------------------------------------------------------
    # Figure 3: Dropout Impact (Dual Y-Axis)
    # -----------------------------------------------------------------------
    profiles = ["baseline_fedavg", "urban_fedavg", "rural_fedavg", "severe_fedavg"]
    plot_labels = ["Baseline", "Urban", "Rural", "Severe"]
    
    drop_rates = []
    final_accs = []
    
    for key in profiles:
        df = data[key]
        if not df.empty:
            # Calculate overall drop rate
            total_dropped = df["dropped_updates"].sum()
            total_participating = df["participating_clients"].sum()
            total_attempts = total_dropped + total_participating
            
            rate = (total_dropped / total_attempts * 100) if total_attempts > 0 else 0
            acc = df["global_accuracy"].iloc[-1] * 100
            
            drop_rates.append(rate)
            final_accs.append(acc)
        else:
            drop_rates.append(0)
            final_accs.append(0)

    fig3, ax_acc = plt.subplots(figsize=(12, 6), dpi=150)
    ax_drop = ax_acc.twinx()

    x = range(len(plot_labels))
    width = 0.35

    # Plot bars
    bar1 = ax_drop.bar([i - width/2 for i in x], drop_rates, width, label='Drop Rate (%)', color="#F44336", alpha=0.8)
    bar2 = ax_acc.bar([i + width/2 for i in x], final_accs, width, label='Final Accuracy (%)', color="#2196F3", alpha=0.8)

    ax_acc.set_title("Impact of Client Dropout on Final Accuracy", pad=15, fontweight="bold")
    ax_acc.set_xticks(x)
    ax_acc.set_xticklabels(plot_labels)
    ax_drop.set_ylabel("Drop Rate (%)", color="#F44336", fontweight="bold")
    ax_acc.set_ylabel("Final Accuracy (%)", color="#2196F3", fontweight="bold")

    # Add legends
    lines, labels_list = ax_drop.get_legend_handles_labels()
    lines2, labels_list2 = ax_acc.get_legend_handles_labels()
    ax_acc.legend(lines + lines2, labels_list + labels_list2, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2)

    fig3.tight_layout()
    fig3.savefig(out_dir / "dropout_impact.png")
    plt.close(fig3)
    print("Saved Figure 3: dropout_impact.png")

    # -----------------------------------------------------------------------
    # Figure 4: Communication Cost
    # -----------------------------------------------------------------------
    bytes_mb = []
    for key in profiles:
        df = data[key]
        if not df.empty:
            total_bytes = df["total_bytes_transmitted"].sum()
            bytes_mb.append(total_bytes / (1024 * 1024))
        else:
            bytes_mb.append(0)

    fig4, ax4 = plt.subplots(figsize=(12, 6), dpi=150)
    bars = ax4.bar(plot_labels, bytes_mb, color="#4CAF50", alpha=0.8)
    
    ax4.set_title("Total Communication Cost (FedAvg)", pad=15, fontweight="bold")
    ax4.set_ylabel("Total Data Transmitted (MB)")
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            ax4.annotate(f"{height:.1f}",
                         xy=(bar.get_x() + bar.get_width() / 2, height),
                         xytext=(0, 3),  # 3 points vertical offset
                         textcoords="offset points",
                         ha='center', va='bottom', fontweight="bold")

    fig4.tight_layout()
    fig4.savefig(out_dir / "communication_cost.png")
    plt.close(fig4)
    print("Saved Figure 4: communication_cost.png")


if __name__ == "__main__":
    print("Generating comparison plots...")
    generate_plots()
    print("Done!")
