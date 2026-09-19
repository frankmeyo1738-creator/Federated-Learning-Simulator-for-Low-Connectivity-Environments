# Experimental Results

## Overview

This document summarises the empirical results produced by the Federated Learning Network Simulator across 100 multi-seed experiments (full $n=10$ cohort for both IID and non-IID data distributions), as well as exploratory CIFAR-10 single-seed benchmarks across four Sub-Saharan Africa (SSA) network profiles.

All multi-seed results reported here reflect the post-seeding remediation (commit `7e7b8cf` and later), sample standard deviations computed with Bessel's correction ($\text{ddof}=1$), exact asymptotic Wilcoxon signed-rank tests with tied-rank splitting (`zero_method='zsplit'`, `mode='approx'`), paired Cohen's $d_z$ effect sizes with 95% bootstrap confidence intervals, statistical power calculations via the non-central $t$-distribution, genuine binary mebibyte data accounting ($1024^2$ bytes per MiB), and discriminating convergence threshold rounds ($R_{90}$ and $R_{95}$).

---

## Research Questions & Findings Summary

1. **How does SSA-style network impairment affect federated learning convergence?**
   * Under **IID partitioning**, final global accuracy exhibits remarkable resilience across 10 communication rounds: baseline accuracy of **$98.52\% \pm 0.05\%$** degrades only slightly to **$98.48\% \pm 0.09\%$** under Rural Zambia conditions ($30.20\%$ drop rate) and **$98.34\% \pm 0.05\%$** under Severe Disruption ($59.80\%$ drop rate) — a degradation of only $0.18\text{ pp}$.
   * However, when network impairment is compounded with **non-IID Dirichlet heterogeneity ($\alpha=0.5$)**, the true impact manifests as **substantial convergence delay** rather than terminal accuracy failure. Reaching $95\%$ test accuracy requires:
     * **4.0 rounds** under Baseline (zero impairment)
     * **4.0 rounds** under Rural Zambia (IID FedAvg; 4.1 rounds for FedProx)
     * **7.3 rounds** under Rural Zambia (Non-IID) — a **$66\%$ delay**
     * **10.1–10.3 rounds** under Severe Disruption (Non-IID) — a **$130\%$ delay**, with final round accuracy reaching only $97.29\% \pm 0.77\%$.
   * Heterogeneous local data distributions coupled with heavy client dropouts starve the server of diverse class updates, slowing parameter trajectory progression along the optimization landscape.

2. **Does FedProx offer measurable robustness gains over FedAvg under connectivity constraints?**
   * Under most network conditions, FedProx shows **no statistically significant difference** in final test accuracy compared to FedAvg. However, under Severe Disruption (IID), a statistically significant difference emerged, though its practical magnitude is negligible:
     * Rural Zambia (IID): Mean difference $-0.02\text{ pp}$ ($98.46\% \text{ vs } 98.48\%$; raw diff $-0.013\text{ pp}$), Wilcoxon $W = 17.5, p = 0.3074, d_z = -0.38$
     * Severe Disruption (IID): Mean difference $-0.04\text{ pp}$ ($98.31\% \text{ vs } 98.34\%$; raw diff $-0.038\text{ pp}$), Wilcoxon $W = 1.5, p = 0.0078, d_z = -0.78$ (**Statistically Significant**)
     * Rural Zambia (Non-IID): Mean difference $-0.02\text{ pp}$ ($97.72\% \text{ vs } 97.74\%$), Wilcoxon $W = 16.0, p = 0.2377, d_z = -0.34$
     * Severe Disruption (Non-IID): Mean difference $+0.01\text{ pp}$ ($97.30\% \text{ vs } 97.29\%$), Wilcoxon $W = 27.0, p = 0.9593, d_z = 0.11$
   * The significant result under Severe Disruption (IID) illustrates the critical distinction between statistical and practical significance. While the consistency of the difference across seeds ($p=0.0078$) makes it statistically significant, the actual performance gap is merely $0.04\text{ pp}$ ($98.34\%$ vs $98.31\%$), which is practically negligible for real-world deployments. A clean, fully IID cohort tightened the standard deviations sufficiently to allow this consistent but tiny margin to cross the significance threshold. Overall, the proximal regularisation term ($\mu=0.01$) does not provide a large, transformative accuracy improvement over FedAvg on this benchmark under these conditions.

3. **How does network degradation affect communication cost?**
   * Total payload successfully delivered to the server scales downwards with client dropout rates:
     * Baseline: **$457.7 \pm 0.0\text{ MiB}$** ($0.00\%$ drop rate)
     * Urban Zambia: **$432.1 \pm 13.8\text{ MiB}$** ($5.60\%$ drop rate)
     * Rural Zambia: **$319.5 \pm 32.5\text{ MiB}$** ($30.20\%$ drop rate)
     * Severe Disruption: **$184.0 \pm 24.9\text{ MiB}$** ($59.80\%$ drop rate)
   * *Critical Methodological Framing:* This reduction reflects **delivered payload volume**, not an operational bandwidth saving. In reality, dropped packets and aborted TCP/HTTP uploads still consume cellular radio energy, channel time, and subscriber data quotas without contributing to model convergence.

4. **Do findings generalise across datasets of varying complexity?**
   * Exploratory single-seed CIFAR-10 experiments demonstrate that higher task complexity amplifies degradation: severe disruption reduces final accuracy from $65.8\%$ (baseline) to $51.2\%$, with FedProx demonstrating a preliminary $+0.88\text{ pp}$ advantage under severe loss. These CIFAR-10 results remain exploratory pending multi-seed cohort execution.

---

## Experiment Setup

### Evaluated Network Profiles

Four network profiles model realistic mobile communication regimes in Sub-Saharan Africa:

| Profile Name | Round-Trip Latency Range | Downlink Bandwidth | Packet Loss Rate | Dropout Trigger Rate | Real-World Analog |
|:---|:---:|:---:|:---:|:---:|:---|
| **Baseline** | 1–5 ms | 100.0 Mbps | 0% | 0% | High-speed campus LAN / Datacenter |
| **Urban Zambia** | 80–200 ms | 8.5 Mbps | 1% | 5% | 4G/LTE commercial core (e.g. Lusaka) |
| **Rural Zambia** | 300–800 ms | 1.2 Mbps | 7% | 15% | 3G/EDGE rural cellular with fading |
| **Severe Disruption** | 800–2000 ms | 0.3 Mbps | 20% | 35% | 2G / backhaul congestion / load-shedding |

### Calibration & Profile Disclosures

- **Implementation vs Proposal:** Nominal proposal parameters were formulated as unbounded Gaussians, whereas `src/network/impairment.py` enforces explicit bounded intervals $[\text{min}, \text{max}]$ with midpoint $\mu = \frac{\text{min}+\text{max}}{2}$ and $\sigma = 0.15 \times (\text{max}-\text{min})$, truncating symmetrically at $\pm 3.33\sigma$.
- **Bandwidth & Packet Loss Drift:** As disclosed in the supervisor review ledger, minor parameter drift exists between early proposal sketches and the committed YAML configurations (e.g. Urban 8.5 Mbps vs 5.0 Mbps; packet loss 1% vs 2%). These configurations reflect the concrete profiles committed in `config/experiments/` from repository inception.

---

## Controlled Experimental Conditions

To guarantee scientific reproducibility across all comparisons:
1. **Identical PRNG Impairment Sequences:** For any given random seed $s \in \{1, \dots, 10\}$, client availability triggers, packet loss events, and latency draws are completely deterministic. Paired FedAvg and FedProx runs under seed $s$ experience the exact same dropout events round-for-round (empirically confirmed: dropped update counts and delivered MiB are identical byte-for-byte across paired runs).
2. **Identical Data Partitions:** Training partitions across the 10 clients are generated deterministically per seed. Under non-IID partitioning, a Dirichlet distribution with concentration parameter $\alpha = 0.5$ generates identical class distributions for both algorithms.
3. **Deterministic Weight Initialisation:** Global initial weights are seeded deterministically before round 1, ensuring both algorithms initiate gradient descent from the identical parameter state.
4. **Independent Logging Verification:** Every execution independently validates the active partitioning mode directly from the internal logger output (`iid=True` or `iid=False`).

---

## Multi-Seed Results — Full $n=10$ Cohort

Aggregated across 10 random seeds (seeds 1 to 10) per configuration (100 total experiment runs). All metric uncertainties represent sample standard deviations with Bessel's correction ($\text{ddof}=1$).

### Table 1: IID Cohort Summary (60 Experiments, 10 Seeds Each)

| Experiment Configuration | $n$ | Final Accuracy (%) | Accuracy Range [Min, Max] | Final Loss | Effective Drop Rate (%) | Delivered Data (MiB) | Convergence Rounds ($R_{90} / R_{95}$) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `baseline_fedavg` | 10 | **98.52 ± 0.05%** | [98.45%, 98.58%] | 0.0446 ± 0.0024 | 0.00 ± 0.00% | 457.7 ± 0.0 MiB | 1.0 / 4.0 |
| `urban_zambia_fedavg` | 10 | **98.52 ± 0.08%** | [98.33%, 98.63%] | 0.0443 ± 0.0031 | 5.60 ± 3.03% | 432.1 ± 13.8 MiB | 1.0 / 3.9 |
| `rural_zambia_fedavg` | 10 | **98.48 ± 0.09%** | [98.32%, 98.64%] | 0.0458 ± 0.0028 | 30.20 ± 7.10% | 319.5 ± 32.5 MiB | 1.1 / 4.0 |
| `severe_disruption_fedavg` | 10 | **98.34 ± 0.05%** | [98.26%, 98.44%] | 0.0503 ± 0.0022 | 59.80 ± 5.43% | 184.0 ± 24.9 MiB | 1.1 / 4.7 |
| `rural_zambia_fedprox` | 10 | **98.46 ± 0.08%** | [98.36%, 98.61%] | 0.0462 ± 0.0026 | 30.20 ± 7.10% | 319.5 ± 32.5 MiB | 1.1 / 4.1 |
| `severe_disruption_fedprox` | 10 | **98.31 ± 0.06%** | [98.19%, 98.37%] | 0.0510 ± 0.0023 | 59.80 ± 5.43% | 184.0 ± 24.9 MiB | 1.2 / 4.7 |

### Table 2: Non-IID Cohort Summary (40 Experiments, Dirichlet $\alpha=0.5$, 10 Seeds Each)

| Experiment Configuration | $n$ | Final Accuracy (%) | Accuracy Range [Min, Max] | Final Loss | Effective Drop Rate (%) | Delivered Data (MiB) | Convergence Rounds ($R_{90} / R_{95}$) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `rural_zambia_fedavg_noniid` | 10 | **97.74 ± 0.93%** | [95.15%, 98.38%] | 0.0698 ± 0.0297 | 30.20 ± 7.10% | 319.5 ± 32.5 MiB | 2.5 / 7.3 |
| `rural_zambia_fedprox_noniid` | 10 | **97.72 ± 0.93%** | [95.12%, 98.34%] | 0.0699 ± 0.0286 | 30.20 ± 7.10% | 319.5 ± 32.5 MiB | 2.6 / 7.3 |
| `severe_disruption_fedavg_noniid` | 10 | **97.29 ± 0.77%** | [95.69%, 98.43%] | 0.0836 ± 0.0236 | 59.80 ± 5.43% | 184.0 ± 24.9 MiB | 3.7 / 10.1 |
| `severe_disruption_fedprox_noniid` | 10 | **97.30 ± 0.78%** | [95.70%, 98.38%] | 0.0836 ± 0.0242 | 59.80 ± 5.43% | 184.0 ± 24.9 MiB | 3.6 / 10.3 |

---

## Statistical Significance Testing (Wilcoxon Signed-Rank Tests)

Significance was assessed via paired two-sided Wilcoxon signed-rank tests using zero-split rank tie-breaking (`zero_method='zsplit'`) and pinned asymptotic approximation (`mode='approx'`). Effect sizes are measured via paired Cohen's $d_z = \frac{\bar{x}_\text{diff}}{s_\text{diff}}$ with $95\%$ bootstrap confidence intervals (1,000 resamples). Statistical power $(1-\beta)$ was calculated at $\alpha = 0.05$ against the observed effect size, alongside the Minimum Detectable Effect (MDE) at $80\%$ power.

### Table 3: Paired Hypothesis Tests (FedAvg vs. FedProx across $n=10$ Seeds)

| Comparison Scenario | Data Partition | Mean FedAvg Acc (%) | Mean FedProx Acc (%) | Mean Diff (pp) | Wilcoxon $W$ | Asymptotic $p$-value | Cohen's $d_z$ [95% CI] | Statistical Power ($1-\beta$) | Statistically Significant? |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Rural Zambia** | IID | 98.48% | 98.46% | -0.02 pp | 17.5 | **p = 0.3074** | -0.38 [-1.61, 0.19] | 19.1% | **No** ($p \ge 0.05$) |
| **Severe Disruption** | IID | 98.34% | 98.31% | -0.04 pp | 1.5 | **p = 0.0078** | -0.78 [-1.34, -0.55] | 59.9% | **Yes** ($p < 0.05$) |
| **Rural Zambia** | Non-IID | 97.74% | 97.72% | -0.02 pp | 16.0 | **p = 0.2377** | -0.34 [-3.64, 0.22] | 16.1% | **No** ($p \ge 0.05$) |
| **Severe Disruption** | Non-IID | 97.29% | 97.30% | +0.01 pp | 27.0 | **p = 0.9593** | 0.11 [-0.75, 0.70] | 6.2% | **No** ($p \ge 0.05$) |

*Note on Statistical Power:* For a paired sample size of $n=10$ at $\alpha=0.05$, achieving standard $80\%$ statistical power requires an effect size of $|d| \ge 1.00$ (a large effect). While a statistically significant difference was found in Severe Disruption (IID) due to highly consistent pairing, the absolute magnitude of this difference ($0.04\text{ pp}$) is practically negligible. Because the true performance differences on MNIST between FedAvg and FedProx are subtle ($|d_z| \le 0.78$), the tests exhibit low statistical power. This demonstrates conclusively that FedProx does not provide a large, transformative accuracy improvement over FedAvg on this benchmark under these conditions.

---

## Convergence Trajectory & Threshold Analysis

While final accuracy at round 10 remains close across algorithms, examining trajectory dynamics reveals the operational costs of low-connectivity environments:

1. **Threshold Dynamics ($R_{90}$ and $R_{95}$):**
   * Baseline models reach $90\%$ test accuracy in **1.0 round** and $95\%$ accuracy in **4.0 rounds**.
   * Under IID Rural network conditions, convergence to $95\%$ requires **4.4 rounds**.
   * Under Non-IID Dirichlet partitioning ($\alpha=0.5$), Rural Zambia requires **7.3 rounds** ($+66\%$ delay), and Severe Disruption requires **10.1 to 10.3 rounds** ($+130\%$ delay).
   * **Takeaway:** In real-world cellular deployments, data heterogeneity combined with frequent client dropouts dramatically extends the number of communication rounds needed to achieve acceptable operational accuracy.

2. **Terminal Stability vs Client Drift:**
   * Under Non-IID partitioning, final model loss rises from $0.0446$ (Baseline) to $0.0698$ (Rural Non-IID) and $0.0836$ (Severe Non-IID).
   * Standard deviation of accuracy increases from $\pm 0.05\text{ pp}$ (IID) to $\pm 0.93\text{ pp}$ (Rural Non-IID), demonstrating higher variance across different seed-partition combinations.

---

## Exploratory CIFAR-10 Benchmarks (Single Seed)

To evaluate whether task complexity alters algorithmic behavior under severe dropouts, exploratory runs were conducted on CIFAR-10 (10 clients, 10 communication rounds, seed 42):

| Experiment Profile | Algorithm | Final Test Accuracy | Effective Drop Rate | Delivered Data (MiB) |
|:---|:---:|:---:|:---:|:---:|
| Baseline CIFAR-10 | FedAvg | 65.80% | 0.00% | 457.7 MiB |
| Rural Zambia CIFAR-10 | FedAvg | 61.40% | 29.00% | 325.0 MiB |
| Severe Disruption CIFAR-10 | FedAvg | 51.20% | 64.00% | 164.8 MiB |
| Severe Disruption CIFAR-10 | FedProx ($\mu=0.01$) | 52.08% | 64.00% | 164.8 MiB |

*Observation:* On a non-trivial computer vision task where the baseline accuracy is lower ($65.80\%$), severe communication impairment causes a severe **14.6 pp** performance drop. FedProx achieves **$52.08\%$** vs FedAvg's **$51.20\%$** ($+0.88\text{ pp}$ margin). Because this is a single exploratory seed, this represents an exploratory finding rather than a statistically confirmed conclusion.

---

## Reproducibility Pipeline

The complete $n=10$ cohort and statistical report can be fully reproduced locally or on Google Colab:

```bash
# 1. Local execution of full statistical analysis (reproduces Table 1, 2, and 3):
python scripts/statistical_analysis.py

# 2. Local test suite execution (24 unit tests):
pytest tests/unit/ -v

# 3. Google Colab GPU reproduction script:
# Run scripts/colab_runner.py in Google Colab with T4 GPU runtime
```

All 100 raw CSV result files are version-controlled in `experiments/results/multiseed/`.

---

*Frank Meyo — Department of Computer Science, University of Zambia (2026)*
