"""
Statistical Analysis of Multi-Seed Experiments (n=10 Cohort)
=============================================================
Aggregates metrics across 10 seeds for the full IID and non-IID cohorts (76 experiments).
Performs:
  - Non-parametric Wilcoxon signed-rank tests (paired, zero_method='zsplit', two-sided,
    mode='approx').  The mode is set explicitly to the normal approximation so the
    p-value is reproducible across scipy versions; scipy <= 1.8 defaulted to 'approx',
    scipy >= 1.9 changed the default to 'auto' which may silently switch to the exact
    method for small n, producing a different p-value from the same data.
  - Paired Cohen's d (d_z = mean(diff) / std(diff, ddof=1)) with 95% bootstrap confidence intervals.
  - Statistical power calculation and Minimum Detectable Effect (MDE at alpha=0.05, power=0.80)
    via the non-central t-distribution (scipy.stats.nct).
  - Sample standard deviation (ddof=1) across all metrics.
  - Proper binary units (MiB, i.e. bytes / 1024^2) and accuracy differences reported in
    percentage points (pp).
  - Discriminating convergence threshold analysis (rounds to 90% and 95% accuracy).

Addresses Supervisor & Peer Review Findings: #1, #2, #3, #14, #15, #20.

Author: Frank Meyo, FL Network Simulator UNZA 2026
"""

import os
import numpy as np
import pandas as pd
import scipy
from scipy import stats

N_SEEDS = 10

RESULTS_DIR = "experiments/results/multiseed"
NONIID_DIR = os.path.join(RESULTS_DIR, "noniid")
OUTPUT_FILE = "experiments/results/statistical_summary.txt"

# IID experiments
IID_EXPERIMENTS = [
    "baseline_fedavg",
    "urban_zambia_fedavg",
    "rural_zambia_fedavg",
    "severe_disruption_fedavg",
    "rural_zambia_fedprox",
    "severe_disruption_fedprox",
]

# Non-IID experiments
NONIID_EXPERIMENTS = [
    "rural_zambia_fedavg_noniid",
    "rural_zambia_fedprox_noniid",
    "severe_disruption_fedavg_noniid",
    "severe_disruption_fedprox_noniid",
]

# -- Bytes-to-MiB divisor: 1024^2 = 1,048,576 (genuine binary mebibyte) --
BYTES_PER_MIB = 1024 * 1024  # 1,048,576


def cohen_d_paired(x, y):
    """
    Calculate Cohen's d for paired samples (dz):
        dz = mean(diff) / std(diff, ddof=1)
    where diff = x - y.
    """
    diff = np.array(x) - np.array(y)
    std_diff = np.std(diff, ddof=1)
    if std_diff == 0:
        return 0.0
    return float(np.mean(diff) / std_diff)


def cohen_d_paired_ci(x, y, alpha=0.05, n_boot=2000, seed=42):
    """
    Compute 95% bootstrap confidence interval for paired Cohen's d.
    """
    rng = np.random.RandomState(seed)
    diff = np.array(x) - np.array(y)
    n = len(diff)
    if n < 3:
        return (None, None)

    boot_ds = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, size=n)
        sample = diff[idx]
        s = np.std(sample, ddof=1)
        if s > 0:
            boot_ds.append(np.mean(sample) / s)
        else:
            boot_ds.append(0.0)

    ci_lower = np.percentile(boot_ds, 100 * (alpha / 2))
    ci_upper = np.percentile(boot_ds, 100 * (1 - alpha / 2))
    return float(ci_lower), float(ci_upper)


def paired_power(d, n, alpha=0.05):
    """
    Statistical power for paired comparison using the non-central t-distribution
    (scipy.stats.nct), matching paired t-test approximation for Wilcoxon signed-rank test.
    (Wilcoxon asymptotic relative efficiency is ~0.955 for normal differences).
    """
    if n < 2 or abs(d) < 1e-6:
        return alpha
    df = n - 1
    nc = abs(d) * np.sqrt(n)
    t_crit = stats.t.ppf(1 - alpha / 2, df)
    power = (1 - stats.nct.cdf(t_crit, df, nc)) + stats.nct.cdf(-t_crit, df, nc)
    return float(power)


def minimum_detectable_effect(n, alpha=0.05, target_power=0.80):
    """
    Compute the Minimum Detectable Effect size (|d|) at target_power=0.80 and alpha=0.05.
    """
    if n < 2:
        return None
    from scipy.optimize import brentq
    def objective(d_val):
        return paired_power(d_val, n, alpha) - target_power
    try:
        mde = brentq(objective, 0.01, 5.0)
        return float(mde)
    except Exception:
        return None


def analyze_experiment(exp_name, results_dir, n_seeds=N_SEEDS):
    final_accs, final_losses, drop_rates, total_bytes = [], [], [], []
    r90_rounds, r95_rounds = [], []
    missing_seeds = []

    for seed in range(1, n_seeds + 1):
        csv_path = os.path.join(results_dir, f"{exp_name}_seed{seed}_metrics.csv")
        if not os.path.exists(csv_path):
            missing_seeds.append(seed)
            continue

        df = pd.read_csv(csv_path)
        if len(df) == 0:
            missing_seeds.append(seed)
            continue

        final_row = df.iloc[-1]
        final_accs.append(final_row["global_accuracy"])
        final_losses.append(final_row["global_loss"])
        total_bytes.append(df["total_bytes_transmitted"].sum())

        total_participating = df["participating_clients"].sum()
        total_dropped = df["dropped_updates"].sum()
        total_attempted = total_participating + total_dropped
        drop_rates.append(total_dropped / total_attempted if total_attempted > 0 else 0.0)

        # Discriminating convergence speed
        # Rounds to 90%
        hit_90 = df[df["global_accuracy"] >= 0.90]
        r90_rounds.append(int(hit_90.iloc[0]["round_num"]) if len(hit_90) > 0 else -1)

        # Rounds to 95%
        hit_95 = df[df["global_accuracy"] >= 0.95]
        r95_rounds.append(int(hit_95.iloc[0]["round_num"]) if len(hit_95) > 0 else -1)

    n_found = len(final_accs)
    return {
        "final_accs": final_accs,
        "n_found": n_found,
        "missing_seeds": missing_seeds,
        "mean_acc": np.mean(final_accs) if n_found else 0.0,
        "std_acc": np.std(final_accs, ddof=1) if n_found > 1 else 0.0,
        "min_acc": np.min(final_accs) if n_found else 0.0,
        "max_acc": np.max(final_accs) if n_found else 0.0,
        "mean_loss": np.mean(final_losses) if n_found else 0.0,
        "std_loss": np.std(final_losses, ddof=1) if n_found > 1 else 0.0,
        "mean_drop": np.mean(drop_rates) if n_found else 0.0,
        "std_drop": np.std(drop_rates, ddof=1) if n_found > 1 else 0.0,
        "mean_bytes": np.mean(total_bytes) if n_found else 0.0,
        "std_bytes": np.std(total_bytes, ddof=1) if n_found > 1 else 0.0,
        "r90_mean": np.mean([r for r in r90_rounds if r > 0]) if any(r > 0 for r in r90_rounds) else None,
        "r95_mean": np.mean([r for r in r95_rounds if r > 0]) if any(r > 0 for r in r95_rounds) else None,
    }


def run_comparison(output, results, exp1, exp2, label, n_seeds=N_SEEDS):
    acc1 = results[exp1]["final_accs"]
    acc2 = results[exp2]["final_accs"]

    output.append(f"Comparison: {label} (FedAvg vs FedProx)")

    if results[exp1]["missing_seeds"] or results[exp2]["missing_seeds"]:
        output.append(f"  ⚠️  Incomplete data: {exp1} missing seeds {results[exp1]['missing_seeds']}, "
                       f"{exp2} missing seeds {results[exp2]['missing_seeds']}")

    n_pairs = len(acc1)
    if n_pairs >= 5 and len(acc2) == n_pairs:
        try:
            # Two-sided Wilcoxon signed-rank test.
            # mode='approx' is set EXPLICITLY to force the normal approximation
            # regardless of scipy version. This is a deliberate decision to avoid
            # the version-sensitivity discovered during earlier analysis: scipy < 1.9
            # defaulted to 'approx', scipy >= 1.9 changed the default to 'auto'
            # which can silently switch to the exact method for small n, producing
            # a different p-value from the same data. Pinning mode='approx' ensures
            # the reported "Asymptotic p" is genuinely computed via the asymptotic
            # normal approximation in every environment.
            w, p = stats.wilcoxon(acc2, acc1,
                                  zero_method='zsplit',
                                  alternative='two-sided',
                                  mode='approx')
            d = cohen_d_paired(acc2, acc1)  # positive d => FedProx > FedAvg
            ci_lower, ci_upper = cohen_d_paired_ci(acc2, acc1)
            power = paired_power(d, n_pairs)
            mde = minimum_detectable_effect(n_pairs)

            mean1 = np.mean(acc1) * 100
            mean2 = np.mean(acc2) * 100
            diff_pp = mean2 - mean1

            output.append(f"  - Paired Cohort:    n = {n_pairs} paired seeds")
            output.append(f"  - FedAvg Mean Acc:  {mean1:.2f}% (SD={np.std(acc1, ddof=1)*100:.2f} pp)")
            output.append(f"  - FedProx Mean Acc: {mean2:.2f}% (SD={np.std(acc2, ddof=1)*100:.2f} pp)")
            output.append(f"  - Mean Difference:  {diff_pp:+.2f} percentage points (pp)")
            output.append(f"  - Wilcoxon W-stat:  W = {w:.1f} (zero_method='zsplit', two-sided, mode='approx')")
            output.append(f"  - Asymptotic p:     p = {p:.4f}")
            if ci_lower is not None:
                output.append(f"  - Paired Cohen's d: d_z = {d:.2f} [95% CI: {ci_lower:.2f}, {ci_upper:.2f}]")
            else:
                output.append(f"  - Paired Cohen's d: d_z = {d:.2f}")
            output.append(f"  - Statistical Power: {power*100:.1f}% (at observed effect size |d|={abs(d):.2f})")
            if mde is not None:
                output.append(f"  - Min. Detectable Effect (MDE): d = {mde:.2f} (at alpha=0.05, power=80%)")

            if p < 0.05:
                output.append("  -> Significance:    STATISTICALLY SIGNIFICANT difference (p < 0.05)")
            else:
                output.append("  -> Significance:    NO statistically significant difference (p >= 0.05)")
        except ValueError as e:
            output.append(f"  -> Could not compute test: {e}")
    else:
        output.append("  -> Could not compute test: Insufficient samples.")
        output.append(f"     Found {len(acc1)} for FedAvg, {len(acc2)} for FedProx.")
    output.append("")


def print_summary_table(output, results, experiments):
    header = (f"{'Experiment':<36} | {'n':<3} | {'Final Acc (%)':<16} | {'Acc Range [Min, Max]':<22} | "
              f"{'Final Loss':<15} | {'Drop Rate (%)':<15} | {'Delivered (MiB)':<18} | {'R90/R95':<10}")
    output.append(header)
    output.append("-" * len(header))
    for exp in experiments:
        r = results[exp]
        if not r["final_accs"]:
            output.append(f"{exp:<36} | {'0':<3} | {'NO DATA':<16} | {'NO DATA':<22} | {'NO DATA':<15} | {'NO DATA':<15} | {'NO DATA':<18} | {'N/A':<10}")
            continue
        acc_str = f"{r['mean_acc']*100:.2f} ± {r['std_acc']*100:.2f}"
        range_str = f"[{r['min_acc']*100:.2f}%, {r['max_acc']*100:.2f}%]"
        loss_str = f"{r['mean_loss']:.4f} ± {r['std_loss']:.4f}"
        drop_str = f"{r['mean_drop']*100:.2f} ± {r['std_drop']*100:.2f}"
        # Genuine MiB: bytes / 1024^2 = bytes / 1,048,576
        mean_mib = r['mean_bytes'] / BYTES_PER_MIB
        std_mib = r['std_bytes'] / BYTES_PER_MIB
        bytes_str = f"{mean_mib:.1f} ± {std_mib:.1f} MiB"

        r90_s = f"{r['r90_mean']:.1f}" if r['r90_mean'] is not None else ">20"
        r95_s = f"{r['r95_mean']:.1f}" if r['r95_mean'] is not None else ">20"
        conv_s = f"{r90_s} / {r95_s}"

        flag = "" if r["n_found"] == N_SEEDS else "  ⚠️ incomplete"
        output.append(f"{exp:<36} | {r['n_found']:<3} | {acc_str:<16} | {range_str:<22} | "
                      f"{loss_str:<15} | {drop_str:<15} | {bytes_str:<18} | {conv_s:<10}{flag}")


def main():
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    iid_results = {exp: analyze_experiment(exp, RESULTS_DIR) for exp in IID_EXPERIMENTS}
    noniid_results = {exp: analyze_experiment(exp, NONIID_DIR) for exp in NONIID_EXPERIMENTS}

    output = []
    output.append("=" * 145)
    output.append("FL NETWORK SIMULATOR — MULTI-SEED STATISTICAL SUMMARY (n=10 COHORT)")
    output.append(f"SciPy Version: {scipy.__version__} (Pinned) | NumPy Version: {np.__version__} | Cohort Size: n={N_SEEDS}")
    output.append("Methodology: Sample SD (ddof=1), Wilcoxon Signed-Rank (zero_method='zsplit', two-sided, mode='approx'), paired Cohen's d (dz)")
    output.append(f"MiB Conversion: bytes / {BYTES_PER_MIB} (1024², genuine binary mebibyte)")
    output.append("=" * 145 + "\n")

    output.append("--- 1. IID COHORT (Full n=10 Seeds, Post-Seeding Fix) ---\n")
    print_summary_table(output, iid_results, IID_EXPERIMENTS)

    output.append("\n--- 2. NON-IID COHORT (Full n=10 Seeds, Dirichlet Partitioning) ---\n")
    print_summary_table(output, noniid_results, NONIID_EXPERIMENTS)

    output.append("\n" + "=" * 145)
    output.append("--- 3. PAIRED STATISTICAL SIGNIFICANCE TESTS (FedAvg vs FedProx) ---")
    output.append("=" * 145 + "\n")

    output.append("[A] IID Network Scenarios:\n")
    iid_comparisons = [
        ("rural_zambia_fedavg", "rural_zambia_fedprox", "Rural Zambia (IID)"),
        ("severe_disruption_fedavg", "severe_disruption_fedprox", "Severe Disruption (IID)"),
    ]
    for exp1, exp2, label in iid_comparisons:
        run_comparison(output, iid_results, exp1, exp2, label)

    output.append("[B] Non-IID Network Scenarios:\n")
    noniid_comparisons = [
        ("rural_zambia_fedavg_noniid", "rural_zambia_fedprox_noniid", "Rural Zambia (Non-IID)"),
        ("severe_disruption_fedavg_noniid", "severe_disruption_fedprox_noniid", "Severe Disruption (Non-IID)"),
    ]
    for exp1, exp2, label in noniid_comparisons:
        run_comparison(output, noniid_results, exp1, exp2, label)

    summary_text = "\n".join(output)
    print(summary_text)

    with open(OUTPUT_FILE, "w") as f:
        f.write(summary_text + "\n")
    print(f"\n✅ Summary successfully saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
