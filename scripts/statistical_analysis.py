"""
Statistical Analysis of Multi-Seed Experiments
==============================================
Aggregates metrics across 5 seeds for 6 experiments and performs
Wilcoxon signed-rank tests with Cohen's d effect size for 
FedAvg vs FedProx comparisons.

Author: Frank Meyo, FL Network Simulator UNZA 2026
"""

import os
import numpy as np
import pandas as pd
from scipy import stats

RESULTS_DIR = "experiments/results/multiseed"
OUTPUT_FILE = "experiments/results/statistical_summary.txt"

EXPERIMENTS = [
    "baseline_fedavg",
    "urban_zambia_fedavg",
    "rural_zambia_fedavg",
    "severe_disruption_fedavg",
    "rural_zambia_fedprox",
    "severe_disruption_fedprox"
]

def cohen_d(x, y):
    """Calculate Cohen's d for paired samples (x - y)."""
    diff = np.array(x) - np.array(y)
    # If standard deviation is 0, effect size is either 0 or undefined.
    std_diff = np.std(diff, ddof=1)
    if std_diff == 0:
        return 0.0
    d = np.mean(diff) / std_diff
    return d

def analyze_experiment(exp_name):
    final_accs = []
    final_losses = []
    drop_rates = []
    total_bytes = []
    
    for seed in range(1, 6):
        csv_path = os.path.join(RESULTS_DIR, f"{exp_name}_seed{seed}_metrics.csv")
        if not os.path.exists(csv_path):
            continue
            
        df = pd.read_csv(csv_path)
        if len(df) == 0:
            continue
            
        # Final round metrics
        final_row = df.iloc[-1]
        final_accs.append(final_row["global_accuracy"])
        final_losses.append(final_row["global_loss"])
        total_bytes.append(df["total_bytes_transmitted"].sum())
        
        # Calculate overall drop rate for this seed
        # Assuming dropped_updates and participating_clients represent the counts per round
        total_participating = df["participating_clients"].sum()
        total_dropped = df["dropped_updates"].sum()
        total_attempted = total_participating + total_dropped
        
        drop_rate = total_dropped / total_attempted if total_attempted > 0 else 0.0
        drop_rates.append(drop_rate)
        
    return {
        "final_accs": final_accs,
        "mean_acc": np.mean(final_accs) if final_accs else 0.0,
        "std_acc": np.std(final_accs, ddof=1) if len(final_accs) > 1 else 0.0,
        "mean_loss": np.mean(final_losses) if final_losses else 0.0,
        "std_loss": np.std(final_losses, ddof=1) if len(final_losses) > 1 else 0.0,
        "mean_drop": np.mean(drop_rates) if drop_rates else 0.0,
        "std_drop": np.std(drop_rates, ddof=1) if len(drop_rates) > 1 else 0.0,
        "mean_bytes": np.mean(total_bytes) if total_bytes else 0.0,
        "std_bytes": np.std(total_bytes, ddof=1) if len(total_bytes) > 1 else 0.0
    }

def main():
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    
    results = {}
    for exp in EXPERIMENTS:
        results[exp] = analyze_experiment(exp)
        
    output = []
    output.append("=========================================================================================")
    output.append("FL NETWORK SIMULATOR — MULTI-SEED STATISTICAL SUMMARY")
    output.append("=========================================================================================\n")
    
    output.append(f"{'Experiment':<32} | {'Final Acc (%)':<15} | {'Final Loss':<15} | {'Drop Rate (%)':<15} | {'Total Bytes':<15}")
    output.append("-" * 105)
    
    for exp in EXPERIMENTS:
        r = results[exp]
        if not r["final_accs"]:
            output.append(f"{exp:<32} | {'NO DATA':<15} | {'NO DATA':<15} | {'NO DATA':<15} | {'NO DATA':<15}")
            continue
            
        acc_str = f"{r['mean_acc']*100:.2f} ± {r['std_acc']*100:.2f}"
        loss_str = f"{r['mean_loss']:.4f} ± {r['std_loss']:.4f}"
        drop_str = f"{r['mean_drop']*100:.2f} ± {r['std_drop']*100:.2f}"
        
        # Format bytes as MB for readability, or raw bytes
        # User requested mean + std, I'll provide raw bytes or MB
        mean_mb = r['mean_bytes'] / (1024 * 1024)
        std_mb = r['std_bytes'] / (1024 * 1024)
        bytes_str = f"{mean_mb:.1f}MB ± {std_mb:.1f}MB"
        
        output.append(f"{exp:<32} | {acc_str:<15} | {loss_str:<15} | {drop_str:<15} | {bytes_str:<15}")
        
    output.append("\n=========================================================================================")
    output.append("STATISTICAL SIGNIFICANCE TESTING (Wilcoxon Signed-Rank Test)")
    output.append("=========================================================================================\n")
    
    # Comparisons
    comparisons = [
        ("rural_zambia_fedavg", "rural_zambia_fedprox", "Rural Zambia"),
        ("severe_disruption_fedavg", "severe_disruption_fedprox", "Severe Disruption")
    ]
    
    for exp1, exp2, label in comparisons:
        acc1 = results[exp1]["final_accs"]
        acc2 = results[exp2]["final_accs"]
        
        output.append(f"Comparison: {label} (FedAvg vs FedProx)")
        if len(acc1) == 5 and len(acc2) == 5:
            try:
                # Wilcoxon signed-rank test
                # zero_method='zsplit' handles cases where differences are exactly 0
                w, p = stats.wilcoxon(acc1, acc2, zero_method='zsplit')
                d = cohen_d(acc2, acc1) # Positive d means FedProx > FedAvg
                
                output.append(f"  - FedAvg Mean Acc:  {np.mean(acc1)*100:.2f}%")
                output.append(f"  - FedProx Mean Acc: {np.mean(acc2)*100:.2f}%")
                output.append(f"  - Wilcoxon W-stat:  {w:.1f}")
                output.append(f"  - p-value:          {p:.4f}")
                output.append(f"  - Cohen's d:        {d:.2f}")
                if p < 0.05:
                    output.append("  -> Result: SIGNIFICANT difference (p < 0.05)")
                else:
                    output.append("  -> Result: NO significant difference (p >= 0.05)")
            except ValueError as e:
                output.append(f"  -> Could not compute test: {e}")
        else:
            output.append("  -> Could not compute test: Missing samples (need exactly 5 pairs of results).")
            output.append(f"     Found {len(acc1)} for FedAvg, {len(acc2)} for FedProx.")
        output.append("")
        
    summary_text = "\n".join(output)
    print(summary_text)
    
    with open(OUTPUT_FILE, "w") as f:
        f.write(summary_text + "\n")
    print(f"\n✅ Summary saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
