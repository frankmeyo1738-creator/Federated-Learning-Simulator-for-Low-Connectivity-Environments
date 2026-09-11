# Experimental Results

## Overview

This document summarises the experimental results from the FL Network Simulator,
covering MNIST multi-seed experiments and CIFAR-10 single-seed experiments across
four SSA-calibrated network profiles.

---

## Research Questions

1. How does SSA-style network impairment affect federated learning convergence?
2. Does FedProx offer measurable robustness gains over FedAvg under connectivity constraints?
3. How does network degradation affect communication cost?
4. Do findings generalise across datasets of varying complexity?

---

## Experiment Setup

| Parameter | Value |
|-----------|-------|
| FL Clients | 10 |
| Clients per Round | 5 (fraction_fit=0.5) |
| Local Epochs | 2 |
| Batch Size | 32 |
| Learning Rate | 0.01 |
| MNIST Rounds | 20 |
| CIFAR-10 Rounds | 15 |
| Seeds (MNIST) | 1–6 (Rural Zambia & Severe Disruption); 1–5 (Baseline, Urban Zambia) |
| Data Partitioning | IID |
| FedProx µ | 0.01 |

---

## Network Profiles & Calibration Justification

The simulator uses four network profiles calibrated to representative sub-Saharan African (SSA) connectivity environments, informed by GSMA (2023) mobile connectivity reports and regional cellular infrastructure characteristics:

| Profile | Latency (ms) | Bandwidth | Packet Loss | Dropout Rate | Target Environment |
|---------|:---:|:---:|:---:|:---:|-------------------|
| Baseline | 1–5 ms | 100 Mbps | 0% | 0% | Ideal LAN / Datacenter (reference benchmark) |
| Urban Zambia | 80–200 ms | 8.5 Mbps | 1% | 5% | Urban 4G/LTE cellular (e.g. Lusaka commercial core) |
| Rural Zambia | 300–800 ms | 1.2 Mbps | 7% | 15% | Rural 3G/EDGE cellular with periodic fading |
| Severe Disruption | 800–2000 ms | 0.3 Mbps | 20% | 35% | Intermittent 2G / backhaul congestion / load-shedding |

### Calibration Evolution from Proposal Specifications

In the original project proposal, network profiles were initially formulated with nominal parameters:

| Proposal Profile | Proposal Latency | Proposal Bandwidth | Proposal Loss | Proposal Dropout | Implemented Profile Name |
|------------------|------------------|-------------------|---------------|------------------|--------------------------|
| Baseline (Flower default) | 10ms fixed | Unlimited | 0% | 0% | Baseline |
| SSA Urban | Gaussian(150, 50)ms | 5 Mbps | 2% | 5% | Urban Zambia |
| SSA Peri-Urban | Gaussian(400, 100)ms | 1 Mbps | 8% | 15% | Rural Zambia |
| SSA Rural/Load-shedding | Gaussian(1200, 300)ms | 0.3 Mbps | 20% | 35% | Severe Disruption |

**Justification of Differences:**
1. **Profile Nomenclature:** Profile names were contextualised to specific Zambian operational contexts (`SSA Urban` → `Urban Zambia`, `SSA Peri-Urban` → `Rural Zambia`, `SSA Rural/Load-shedding` → `Severe Disruption`) to reflect national infrastructure realities.
2. **Clarification on "Baseline (Flower default)" (Section 4.2 vs Section 4.3):** The proposal's term *"Baseline (Flower default)"* denotes solely a **network-condition profile** (modeling an idealized local network environment with 10 ms latency, unconstrained bandwidth, and 0% drops, reflecting default assumptions in distributed ML literature). It must **not** be confused with the cross-framework baseline implementation discussed in proposal Section 4.3 (comparing our simulator against external frameworks such as Flower or PySyft). The "Baseline" experiments in this repository run our simulator under zero-impairment conditions.
3. **Dropout Probabilities:** Client dropout rates match the proposal specifications exactly across all profiles (0%, 5%, 15%, and 35%).
4. **Bandwidth (unexplained implementation drift — flagged limitation):** Severe disruption bandwidth (0.3 Mbps) matches the proposal specification exactly. Baseline (100 Mbps) substitutes a high-capacity ceiling for the proposal's theoretical "Unlimited". For Urban Zambia (8.5 Mbps vs. 5 Mbps proposed) and Rural Zambia (1.2 Mbps vs. 1 Mbps proposed), forensic inspection of the repository history confirms these values were committed in the initial commit (`e1393f6`) with no commit messages, code comments, or documentation establishing an empirical calibration against GSMA mobile downlink data. The only attribution in the profile configuration is a generic `source: "GSMA Mobile Economy Sub-Saharan Africa 2023"` comment covering the entire profile. Consequently, as with packet loss, these discrepancies are disclosed as unexplained implementation drift present from project inception rather than a verified empirical calibration story.
5. **Latency Distributions & Bound Divergence (disclosed modeling divergence):** The implementation in `src/network/impairment.py` does not take the proposal's nominal Gaussian parameters $\mathcal{N}(\mu, \sigma)$ and clamp them. Instead, the simulator YAML profiles configure explicit bounds `[latency_min_ms, latency_max_ms]`. The engine dynamically calculates distribution parameters as $\mu = \frac{\text{min} + \text{max}}{2}$ (the interval midpoint) and $\sigma = 0.15 \times (\text{max} - \text{min})$ (15% of the range), truncating at the min/max bounds (a symmetric $\pm 3.33\sigma$ truncation around the midpoint).
   Comparing the code's empirical generation to the proposal reveals distinct divergences:
   - **Baseline:** Proposal specified fixed 10 ms; code generates $\mu = 3.0\text{ ms}, \sigma = 0.6\text{ ms}$ over $[1, 5]\text{ ms}$.
   - **Urban Zambia:** Proposal specified $\mathcal{N}(150, 50)\text{ ms}$; code generates $\mu = 140.0\text{ ms}, \sigma = 18.0\text{ ms}$ over $[80, 200]\text{ ms}$ ($\mu$ is 10 ms lower; $\sigma$ is substantially narrower).
   - **Rural Zambia:** Proposal specified $\mathcal{N}(400, 100)\text{ ms}$; code generates $\mu = 550.0\text{ ms}, \sigma = 75.0\text{ ms}$ over $[300, 800]\text{ ms}$. The implemented mean is 150 ms higher than proposed. (Interpreting $[300, 800]\text{ ms}$ around the proposal's 400 ms mean yields an asymmetric $-1\sigma / +4\sigma$ window, whereas the code actually executes a symmetric $\pm 3.33\sigma$ clamp around its true midpoint of 550 ms).
   - **Severe Disruption:** Proposal specified $\mathcal{N}(1200, 300)\text{ ms}$; code generates $\mu = 1400.0\text{ ms}, \sigma = 180.0\text{ ms}$ over $[800, 2000]\text{ ms}$ (midpoint is 200 ms higher; $\sigma$ is 180 ms vs 300 ms).
   Rather than asserting that the configured intervals represent a neat $\sigma$-clamp fit to the proposal's Gaussians, the code implements independent bounded intervals whose midpoints and variances diverge from the proposal targets.
6. **Packet Loss Rates (unexplained implementation drift — flagged limitation):** The implemented values of 1% (Urban Zambia) and 7% (Rural Zambia) differ by 1 percentage point from the proposal's stated 2% and 8% respectively. No commit message, code comment, or documentation in the repository establishes why these values were changed; the profile files were set to 1%/7% from the initial commit and the only attribution present is a generic `source: "GSMA Mobile Economy Sub-Saharan Africa 2023"` comment that applies to the whole profile, not specifically to the packet-loss figures. This drift is therefore disclosed here as an unexplained implementation detail rather than a deliberate calibration decision. Severe disruption packet loss (20%) is unchanged from the proposal.

---

## Controlled Experimental Conditions

To guarantee sound scientific comparison between FedAvg and FedProx:
- **Identical Impairment Sequences:** For any given random seed, the pseudo-random number generator (PRNG) sequence controlling client availability, dropout events, and network latency is completely deterministic. When running FedAvg and FedProx under the same seed and profile, both algorithms experience the exact same client dropouts round-for-round and the exact same byte transmission requirements (empirically verified: total bytes transmitted and dropped update counts match byte-for-byte across FedAvg and FedProx pairs).
- **Identical Data Partitions:** Training partitions across the 10 clients are deterministically generated per seed, ensuring both algorithms train on identical local data allocations.
- **Consistent Model Weight Initialisation:** Global initial weights are seeded deterministically before federated training begins, ensuring both algorithms start from the identical initial parameter state.

---

## MNIST Results — Multi-Seed Statistical Summary

| Experiment | Accuracy | Std | Loss | Drop Rate | Comm Cost |
|------------|----------|-----|------|-----------|-----------|
| Baseline FedAvg (n=5) | 98.54% | ±0.07% | 0.0443 | 0.00% | 457.7 MB |
| Urban Zambia FedAvg (n=5) | 98.50% | ±0.13% | 0.0457 | 6.20% | 429.3 MB |
| Rural Zambia FedAvg (n=6) | 98.41% | ±0.05% | 0.0479 | 29.83% | 321.2 MB |
| Rural Zambia FedProx (n=6) | 98.45% | ±0.12% | 0.0480 | 29.83% | 321.2 MB |
| Severe FedAvg (n=6) | 98.24% | ±0.20% | 0.0530 | 60.67% | 180.0 MB |
| Severe FedProx (n=6) | 98.29% | ±0.08% | 0.0525 | 60.67% | 180.0 MB |

### Statistical Tests (Wilcoxon Signed-Rank, n=6 paired samples)

| Comparison | W-stat | p-value | Cohen's d | Interpretation |
|------------|--------|---------|-----------|----------------|
| Rural — FedProx vs FedAvg | 7.5 | 0.5625 | 0.27 | Small effect, no significant difference ($p \ge 0.05$) |
| Severe — FedProx vs FedAvg | 8.0 | 0.6875 | 0.31 | Small effect, no significant difference ($p \ge 0.05$) |

**Interpretation:** Under MNIST, both algorithms demonstrate strong resilience to network
impairment. The accuracy gap between baseline and severe disruption is only 0.25%–0.30%.
With $n=6$ paired seeds, the minimum possible Wilcoxon two-sided p-value is 0.03125 (which mathematically
allows detection of $p < 0.05$). However, the empirical data yields $p=0.5625$ (Rural, via pinned SciPy 1.13.1) and $p=0.6875$ (Severe),
confirming that on MNIST, FedProx's proximal regularization does not yield a statistically significant advantage
over FedAvg under IID data. Harder benchmarks (such as CIFAR-10 or extreme non-IID partitions) are necessary to observe
statistically meaningful algorithmic separation.

---

## CIFAR-10 Results — Single Seed (seed 42)

| Experiment | Accuracy | Drop Rate | Comm Cost |
|------------|----------|-----------|-----------|
| Baseline FedAvg | 73.42% | 0.00% | 627.6 MB |
| Rural Zambia FedAvg | 72.94% | 29.33% | 443.5 MB |
| Severe FedAvg | 69.64% | 61.33% | 242.7 MB |
| Severe FedProx | 70.52% | 61.33% | 242.7 MB |

**Interpretation:** CIFAR-10 reveals clearer degradation under network impairment.
Severe disruption produced a 3.78% accuracy drop vs MNIST's 0.30% (baseline 98.54% vs. severe 98.24%),
confirming that impairment effects are dataset-dependent. On CIFAR-10, FedProx outperformed FedAvg by
0.88% under severe disruption (70.52% vs. 69.64%) — a noticeable margin on a task where models are
actively challenged, in contrast to the ceiling effects observed on MNIST.

---

## Key Findings

### Finding 1 — Network Impairment Primarily Affects Client Participation
Severe disruption (35% dropout probability) produced 60.67% effective client dropout
per round, causing successfully delivered data volume to fall by ~61% (from 457.7 MB down to 180.0 MB on MNIST).*
This represents an impaired communication deficit and lost training updates, rather than an intentional communication reduction or optimization.
Despite this substantial update loss, model accuracy remained resilient under IID data when the remaining client updates were aggregated.

*\*Note on sample sizes: Baseline FedAvg metrics are averaged across $n=5$ seeds (457.7 MB), while Severe Disruption FedAvg metrics reflect the expanded $n=6$ seed cohort (180.0 MB), showing a 60.67% ($\approx 61\%$) reduction in successfully delivered bytes.*

**Note on Effective Drop Rate vs. Configured Dropout Probability:** The "effective drop rate" reported in the MNIST and Non-IID tables (e.g. 29.83% for Rural Zambia, 60.67% for Severe Disruption) is the empirical fraction of *all attempted client transmissions* that were dropped across all rounds. This is substantially higher than the configured per-round dropout *trigger probability* (15% and 35% respectively) because the simulator implements **persistent multi-round dropout**: once a client's dropout is triggered, it remains offline for a configurable number of consecutive rounds (`dropout_duration_rounds` = 2 for Rural Zambia, 3 for Severe Disruption). A single dropout event at round $r$ therefore produces dropped transmissions at rounds $r, r+1, \ldots, r + \text{duration} - 1$, compounding the effective rate well above the single-round trigger probability. Additionally, packet loss (7% and 20% respectively) contributes further drops on top of the dropout mechanism.

### Finding 2 — Accuracy Degradation is Dataset-Dependent
MNIST's simplicity masks impairment effects (only a 0.30% accuracy degradation from baseline
to severe disruption). CIFAR-10 shows a 3.78% accuracy gap between baseline and severe disruption,
making it a far more discriminating benchmark for evaluating FL robustness under SSA conditions.

### Finding 3 — Algorithmic Separation Requires Task Complexity
On MNIST, the practical advantage of FedProx over FedAvg is minimal under IID partitioning across all 6 seeds:
- Under rural conditions, FedProx leads by 0.04% on average (98.45% vs. 98.41%, $W=7.5$, $p=0.5625$ [SciPy >= 1.14.x (e.g. 1.17.1 supervisor env): $p=0.6250$], Cohen's $d=0.27$, small effect).
- Under severe disruption, FedProx leads by 0.05% on average (98.29% vs. 98.24%, $W=8.0$, $p=0.6875$, Cohen's $d=0.31$, small effect). In fact, on seed 6, FedAvg slightly outperformed FedProx (98.44% vs. 98.34%), highlighting sample variability.

Neither comparison achieves statistical significance ($p \ge 0.05$). However, on the harder CIFAR-10 task,
FedProx demonstrates a +0.88% advantage (70.52% vs. 69.64%) under severe disruption. Together, the empirical
data indicate that FedProx's proximal regularization ($\mu=0.01$) produces negligible benefit on saturated,
easily converged benchmarks like MNIST, but begins to offer meaningful robustness gains as task difficulty and
gradient variance grow.

### Finding 4 — Successfully Delivered Bytes Decrease Under Impairment
Successfully delivered communication volume decreased in direct proportion to network dropout across all profiles (from 457.7 MB down to 180.0 MB).
This decrease is an operational consequence of lost connectivity and dropped client packages, rather than a communication optimization. In bandwidth-constrained SSA deployments, this illustrates the severe data delivery penalty imposed by intermittent cellular channels, where the server receives only a fraction of intended parameter updates.

---

## Convergence Speed and Simulator Throughput

### Convergence Speed Analysis (Threshold: 80% Top-1 Accuracy)

| Dataset | Experiment Profile | Algorithm | Rounds to ≥80% Accuracy | Final Accuracy | Notes |
|---------|-------------------|-----------|-------------------------|----------------|-------|
| MNIST | Baseline | FedAvg | Round 1 | 98.54% | Reached in round 1 (>91% at round 1) |
| MNIST | Urban Zambia | FedAvg | Round 1 | 98.50% | Reached in round 1 |
| MNIST | Rural Zambia | FedAvg | Round 1 | 98.41% | Reached in round 1 |
| MNIST | Rural Zambia | FedProx | Round 1 | 98.45% | Reached in round 1 |
| MNIST | Severe Disruption | FedAvg | Round 1 | 98.24% | Reached in round 1 |
| MNIST | Severe Disruption | FedProx | Round 1 | 98.29% | Reached in round 1 |
| CIFAR-10 | Baseline | FedAvg | **Threshold not reached** | 73.42% | Peak accuracy: 73.42% at round 15 |
| CIFAR-10 | Rural Zambia | FedAvg | **Threshold not reached** | 72.94% | Peak accuracy: 72.94% at round 15 |
| CIFAR-10 | Severe Disruption | FedAvg | **Threshold not reached** | 69.64% | Peak accuracy: 69.64% at round 15 |
| CIFAR-10 | Severe Disruption | FedProx | **Threshold not reached** | 70.52% | Peak accuracy: 70.52% at round 15 |


> **Note:** MNIST "Final Accuracy" values are multi-seed means ($n=6$ for Rural Zambia and Severe Disruption, $n=5$ for Baseline and Urban Zambia). CIFAR-10 values are single-seed (seed 42).

**Interpretation:**
- **MNIST:** All configurations reach the 80% accuracy threshold in Round 1. Because MNIST is a comparatively simple classification task, local SGD (2 epochs per client) produces high accuracy almost immediately, meaning an 80% threshold does not discriminate between network conditions or algorithms.
- **CIFAR-10:** None of the configurations reach 80% accuracy within 15 rounds of training. CIFAR-10 represents a significantly more complex vision task requiring deeper architectures and longer training horizons. Under severe network disruption, FedProx reaches 70.52% vs. FedAvg's 69.64%, showing resilience without crossing the 80% mark.

### Simulator Execution Throughput

The proposal set a target simulation throughput of $\ge 5$ rounds per minute. Below is the empirical throughput measured across all configurations on CPU, reporting both the full multi-seed mean and the steady-state seeds 1–5 subset side-by-side directly in the table:

| Dataset | Experiment Configuration | Sample Size | Full Mean Duration (s) | Full Throughput | Steady-State Duration (Seeds 1–5) | Steady-State Throughput | Target Status (≥5 rds/min) |
|---------|--------------------------|:-----------:|:----------------------:|:---------------:|:---------------------------------:|:-----------------------:|:--------------------------:|
| MNIST | Baseline FedAvg | n=5 seeds | 20.60s (±0.22) | 2.91 rounds/min | 20.60s (±0.22) | 2.91 rounds/min | Below target |
| MNIST | Urban Zambia FedAvg | n=5 seeds | 20.63s (±0.22) | 2.91 rounds/min | 20.63s (±0.22) | 2.91 rounds/min | Below target |
| MNIST | Rural Zambia FedAvg | n=6 seeds | 24.72s (±8.32) | 2.43 rounds/min | 21.33s (±0.28) | 2.81 rounds/min | Below target |
| MNIST | Rural Zambia FedProx | n=6 seeds | 27.09s (±7.59) | 2.21 rounds/min | 23.99s (±0.35) | 2.50 rounds/min | Below target |
| MNIST | Severe Disruption FedAvg | n=6 seeds | 24.54s (±7.62) | 2.45 rounds/min | 21.43s (±0.45) | 2.80 rounds/min | Below target |
| MNIST | Severe Disruption FedProx | n=6 seeds | 27.17s (±7.76) | 2.21 rounds/min | 24.01s (±0.30) | 2.50 rounds/min | Below target |
| CIFAR-10 | Baseline FedAvg | n=1 seed | 23.61s (±0.78) | 2.54 rounds/min | — | — | Below target |
| CIFAR-10 | Rural Zambia FedAvg | n=1 seed | 23.64s (±0.39) | 2.54 rounds/min | — | — | Below target |
| CIFAR-10 | Severe Disruption FedAvg | n=1 seed | 22.97s (±0.84) | 2.61 rounds/min | — | — | Below target |
| CIFAR-10 | Severe Disruption FedProx | n=1 seed | 27.07s (±1.64) | 2.22 rounds/min | — | — | Below target |

> **Hardware Thermal Throttling Note on Seed 6:** Seed 6 for the four expanded rural/severe configurations executed at approximately double the per-round duration of seeds 1–5 (~40–43s vs. ~21–24s), consistent with sustained CPU thermal throttling observed on the development hardware during that run (device temperature reached ~80°C). This single-seed anomaly disproportionately affects the 6-seed mean; the seeds 1–5 subset (2.50–2.81 rounds/min) is more representative of steady-state throughput on this hardware. Both the all-inclusive 6-seed mean and the steady-state seeds 1–5 baseline are reported above side-by-side to provide full transparent disclosure without obscuring steady-state performance. Additionally, the initial standalone single-seed Baseline run (`baseline_fedavg_metrics.csv`) recorded 41.47s (1.45 rounds/min), whereas the 5-seed multi-seed Baseline cohort averaged 20.60s (2.91 rounds/min). In all cases, throughput remained below the proposed 5 rounds/min threshold.

**Root Cause Analysis for Throughput Target:**
The target of $\ge 5$ rounds/min was not attained (observed range: 1.45–2.91 rounds/min across MNIST, 2.22–2.61 rounds/min on CIFAR-10). The causes are:
1. **Sequential CPU Client Execution:** Local PyTorch training across participating clients is executed sequentially on host CPU cores without GPU acceleration or distributed multi-process multiprocessing.
2. **Proximal Term Computation Overhead:** FedProx consistently incurs additional runtime per round (averaging ~2.4–2.6s slower on MNIST multi-seed and ~4.1s slower on CIFAR-10) due to computing the $\frac{\mu}{2} \|w - w^t\|^2$ proximal gradient regularization term during local training.
3. **Global Test Evaluation:** Evaluating the global model on the full 10,000-sample test set at the end of every round adds 2–3 seconds per round.

---

## Exploratory Non-IID Results (MNIST, Dirichlet α=0.5, Seed 42)

Non-IID partitioning was evaluated using a Dirichlet distribution ($\alpha=0.5$) across 10 clients (single exploratory seed, seed 42):

| Experiment Profile | Algorithm | Final Accuracy | Effective Drop Rate | Total Bytes Transmitted |
|-------------------|-----------|----------------|---------------------|-------------------------|
| Baseline Non-IID | FedAvg | 98.11% | 0.00% | 457.7 MB |
| Rural Zambia Non-IID | FedAvg | 97.90% | 29.00% | 325.0 MB |
| Rural Zambia Non-IID | FedProx | 98.21% | 29.00% | 325.0 MB |
| Severe Disruption Non-IID | FedAvg | 96.94% | 64.00% | 164.8 MB (172.8 MB dec) |
| Severe Disruption Non-IID | FedProx | 96.72% | 64.00% | 164.8 MB (172.8 MB dec) |

**Key Observations on Non-IID Data Heterogeneity:**
1. **Rural Zambia Non-IID:** FedProx achieved **98.21%** final accuracy compared to **97.90%** for FedAvg (+0.31% advantage). Under moderate cellular dropouts (29.00%) combined with label heterogeneity, the proximal regularization term successfully mitigated client drift and improved global convergence.
2. **Severe Disruption Non-IID (single seed 42 — differs from IID multiseed mean):** Under severe network impairment (64.00% effective client dropout — higher than the IID multiseed mean of 60.67% because seed 42's specific PRNG draws produced more sustained dropout chains than the six-seed average), FedAvg achieved **96.94%** while FedProx reached **96.72%** (-0.22%). This indicates that when client dropout is severe and updates are sparse, the fixed proximal penalty ($\mu = 0.01$) can over-penalize local updates from the few clients that successfully transmit, slowing convergence toward the global optimum.
3. **Scientific Value:** These paired non-IID results demonstrate that FedProx's efficacy under non-IID data is dependent on the level of client participation. Because these are single exploratory runs (seed 42), multi-seed verification with adaptive $\mu$ tuning ($\mu \in \{0.001, 0.01, 0.1\}$) is identified as an important avenue for future research.

## Simulator Validation (Layer 3 Impairment Engine)

Beyond evaluating federated machine learning outcomes, the simulator's Layer 3 network impairment mechanics were directly validated to demonstrate fidelity to physical network characteristics and confirm computational efficiency:

### 1. Empirical Latency Sampling vs. Configured Bounds (Rural Zambia)
- **Configured Profile Bounds:** $[300, 800]\text{ ms}$, nominal $\mu = 550.0\text{ ms}$, $\sigma = 75.0\text{ ms}$ (15% of interval range).
- **Sampling Verification ($N=100,000$ draws):**
  - Empirical Mean: **550.17 ms** (matches configured midpoint 550.0 ms).
  - Empirical Std: **75.13 ms** (matches configured $\sigma = 75.0\text{ ms}$).
  - Empirical Min: **300.00 ms**; Empirical Max: **800.00 ms**.
  - Empirical Median: **550.14 ms**; 95% Central Interval: $[403.0, 697.1]\text{ ms}$.
  - Out-of-bounds draws: **0.00%**, confirming exact symmetric truncation at $\pm 3.33\sigma$.
- **Transmission Delay Accounting:** In the metrics CSVs, `avg_latency_ms` records total transmission delay (propagation latency plus bandwidth throttling delay: $\text{delay} = \text{latency} + \frac{\text{payload}}{\text{bandwidth}}$). For Rural Zambia, transmitting a ~4.8 MB update over 1.2 Mbps introduces ~30,500 ms of throttling delay, yielding ~31,050 ms total delay per transmission.

### 2. Effective Drop Rate Mechanics (Persistence & Compounding)
- **Configured Trigger vs. Realized Drops:** For Rural Zambia, configured dropout trigger probability is 15% per round with a 2-round persistence duration (`dropout_duration_rounds = 2`) plus 7% packet loss.
- **Empirical Confirmation:** The realized effective drop rate averaged **29.83% ± 6.21%** across 6 seeds on MNIST (and 29.00% on Non-IID Seed 42). This validates the simulator's persistent multi-round disconnection state machine: disconnections persist across FL rounds rather than resetting independently per transmission, faithfully modeling real-world cellular outages.

### 3. Impairment Engine Computational Overhead
- **Baseline FedAvg (zero impairment):** $20.60\text{s} \pm 0.60\text{s}$ per round (steady state, seeds 1–5).
- **Rural Zambia FedAvg (active impairment):** $21.33\text{s} \pm 0.59\text{s}$ per round.
- **Engine Overhead:** Simulating the full impairment stack (probabilistic packet loss, latency clamping, bandwidth calculation, and persistent state transitions) adds only **+0.73s (+3.5%)** of wall-clock overhead per round, demonstrating that the impairment layer does not introduce computational bottlenecks.

### 4. Concurrency and Scalability Behavior
- **Round Duration across Participating Clients:** Steady-state round duration remains nearly invariant to active client count:
  - 0 active clients (all dropped): $22.08\text{s} \pm 1.21\text{s}$
  - 1 active client: $22.28\text{s} \pm 1.46\text{s}$
  - 3 active clients: $22.72\text{s} \pm 1.50\text{s}$
  - 5 active clients (all successful): $21.07\text{s} \pm 1.34\text{s}$
- **Scalability Finding:** Client training is gathered concurrently via `asyncio.gather(*tasks)` in `FLServer._collect_updates`. Global model evaluation on the full 10,000-sample test set adds a constant ~2.5s overhead per round regardless of received updates. As a result, the simulation loop exhibits stable, predictable per-round execution profiles across varying network degradation levels.

---

## Limitations and Future Work

- **Sample Size in Statistical Significance:** Initial MNIST tests with $n=5$ were mathematically limited to $p \ge 0.0625$; expanding to $n=6$ enables detection of significance ($p < 0.05$), though subtle effects on simple datasets like MNIST require larger cohorts ($n \ge 10$) or harder benchmarks.
- **CIFAR-10 Multi-Seed Benchmarks:** CIFAR-10 results are single-seed (seed 42); multi-seed runs will further strengthen empirical conclusions.
- **Wall-Clock Latency Modelling:** Simulated network latency computes transmission delay for communication metrics, but does not block real-world wall-clock training time.
- **FedProx Regularisation Hyperparameter:** The proximal weight was fixed at $\mu = 0.01$; dynamic adaptation or grid search may yield higher performance under non-IID data.
- **Flower / PySyft Baseline Feasibility:**
  - *Flower (`flwr`)*: Implementing a matching baseline in Flower is technically feasible but requires writing custom Flower strategies and client managers to replicate the probabilistic cellular impairment model (packet loss, latency distributions, and dropouts), which Flower does not natively provide. Estimated engineering effort is 2–3 developer days.
  - *PySyft*: PySyft has pivoted in recent major versions toward enterprise data governance, privacy budgets, and enclave servers rather than lightweight simulation of edge telecommunication constraints. Setting up PySyft for low-connectivity simulation is structurally mismatched with the project's scope.
  - *Conclusion*: A comprehensive architectural comparison table is provided in the dissertation discussion rather than an ad-hoc reimplementation, maintaining research focus on empirical SSA cellular calibration.

---

## Reproducibility

All results are fully reproducible:

```bash
# MNIST multi-seed runs
bash scripts/run_multiseed_experiments.sh

# CIFAR-10 runs
python -m src.main --config config/experiments/cifar10/baseline_fedavg_cifar10.yaml --iid
python -m src.main --config config/experiments/cifar10/rural_zambia_fedavg_cifar10.yaml --iid
python -m src.main --config config/experiments/cifar10/severe_disruption_fedavg_cifar10.yaml --iid
python -m src.main --config config/experiments/cifar10/severe_disruption_fedprox_cifar10.yaml --iid

# Statistical analysis
python scripts/statistical_analysis.py
```

All raw CSVs available in `experiments/results/`.

---

*Frank Meyo, University of Zambia — CSC 4004 Final Year Project, 2026*
