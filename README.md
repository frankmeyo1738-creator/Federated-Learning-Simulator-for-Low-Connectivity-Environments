# Federated Learning Network Simulator for Low-Connectivity Environments

> A reproducible federated learning simulator designed for Sub-Saharan African (SSA) network conditions, evaluating privacy-preserving distributed AI under bandwidth constraints, latency, packet loss, and client dropout.

**Author:** Frank Meyo — University of Zambia, Department of Computing and Informatics  
**Supervisor:** Mr. Mofya Phiri  
**Project:** CSC 4004 Final Year Project, 2026  
**License:** MIT

---

## Motivation

Federated learning (FL) enables distributed model training without sharing raw data — a critical property for privacy-sensitive applications in healthcare, agriculture, and finance. However, most FL research assumes reliable, high-bandwidth network conditions that do not reflect the infrastructure realities of Sub-Saharan Africa, where connectivity is characterised by high latency, frequent dropouts, and limited bandwidth.

This simulator addresses that gap by providing a configurable, reproducible environment for evaluating FL algorithms under SSA-inspired network profiles, enabling researchers to assess algorithm behaviour before real-world deployment.

---

## Architecture

The simulator is built across four layers:
- **Layer 1 — Configuration:** YAML-driven experiment configuration and validation (`src/config.py`)
- **Layer 2 — FL Core:** FedAvg and FedProx clients and server orchestration (`src/core/`)
- **Layer 3 — Network Impairment:** Latency sampling, packet loss, bandwidth throttling, and persistent client dropout (`src/network/`)
- **Layer 4 — Analytics:** Per-round metrics, CSV export, matplotlib visualisations, and statistical tracking (`src/analytics/`)

---

## Network Profiles

Four SSA-inspired network profiles reflecting typical mobile communication regimes in Sub-Saharan Africa (inspired by GSMA Mobile Economy SSA data):

| Profile | Latency (ms) | Bandwidth | Packet Loss | Dropout Prob | Dropout Duration |
|---------|-------------|-----------|-------------|--------------|------------------|
| Baseline | 1–5 | 100 Mbps | 0% | 0% | 0 rounds |
| Urban Zambia | 80–200 | 8.5 Mbps | 1% | 5% | 1 round |
| Rural Zambia | 300–800 | 1.2 Mbps | 7% | 15% | 2 rounds |
| Severe Disruption | 800–2000 | 0.3 Mbps | 20% | 35% | 3 rounds |

*Note on Profile Parameters:* Latency parameters represent operational $[\text{min}, \text{max}]$ clamping bounds with midpoint $\mu$ and $\sigma = 0.15 \times \text{range}$. Implementation parameters are embedded directly in experiment YAMLs; minor variances from early proposal sketches are disclosed in `docs/RESULTS.md` §7.2 as implementation drift.

---

## Quickstart

```bash
# Clone and install
git clone https://github.com/frankmeyo1738-creator/Federated-Learning-Simulator-for-Low-Connectivity-Environments.git
cd Federated-Learning-Simulator-for-Low-Connectivity-Environments
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Run full test suite (37 unit & integration tests)
python3 -m pytest -v tests/

# Run a smoke test
python3 -m src.main --config config/experiments/smoke_test.yaml --iid

# Run multi-seed analysis
python3 scripts/statistical_analysis.py
```

---

## Experiment Results — Multi-Seed $n=10$ Cohort

Comprehensive evaluation across **76 total experiment runs** ($n=10$ paired random seeds 1–10 across both IID and non-IID Dirichlet $\alpha=0.5$ partitions). All uncertainties represent sample standard deviations ($\text{ddof}=1$), data volumes are reported in binary mebibytes ($1024^2$ B), and accuracy differences in percentage points (pp).

### 1. IID Network Scenarios (MNIST, 10 rounds)

### 1. IID Network Scenarios (MNIST, 10 rounds)

| Experiment | Final Accuracy (%) | Final Loss | Drop Rate (%) | Delivered (MiB) |
|------------|-------------------|------------|---------------|-----------------|
| Baseline FedAvg | $98.52 \pm 0.05$ | $0.0446 \pm 0.0024$ | $0.00 \pm 0.00$ | $457.7 \pm 0.0$ |
| Urban Zambia FedAvg | $98.52 \pm 0.08$ | $0.0443 \pm 0.0031$ | $5.60 \pm 3.03$ | $432.1 \pm 13.8$ |
| Rural Zambia FedAvg | $98.48 \pm 0.09$ | $0.0458 \pm 0.0028$ | $30.20 \pm 7.10$ | $319.5 \pm 32.5$ |
| Rural Zambia FedProx | $98.46 \pm 0.08$ | $0.0462 \pm 0.0026$ | $30.20 \pm 7.10$ | $319.5 \pm 32.5$ |
| Severe Disruption FedAvg | $98.34 \pm 0.05$ | $0.0503 \pm 0.0022$ | $59.80 \pm 5.43$ | $184.0 \pm 24.9$ |
| Severe Disruption FedProx | $98.31 \pm 0.06$ | $0.0510 \pm 0.0023$ | $59.80 \pm 5.43$ | $184.0 \pm 24.9$ |

### 2. Non-IID Dirichlet ($\alpha=0.5$) Scenarios (MNIST, 10 rounds)

| Experiment | Final Accuracy (%) | Final Loss | Drop Rate (%) | Delivered (MiB) |
|------------|-------------------|------------|---------------|-----------------|
| Rural Zambia FedAvg | $97.74 \pm 0.93$ | $0.0698 \pm 0.0297$ | $30.20 \pm 7.10$ | $319.5 \pm 32.5$ |
| Rural Zambia FedProx | $97.72 \pm 0.93$ | $0.0699 \pm 0.0286$ | $30.20 \pm 7.10$ | $319.5 \pm 32.5$ |
| Severe Disruption FedAvg | $97.29 \pm 0.77$ | $0.0836 \pm 0.0236$ | $59.80 \pm 5.43$ | $184.0 \pm 24.9$ |
| Severe Disruption FedProx | $97.30 \pm 0.78$ | $0.0836 \pm 0.0242$ | $59.80 \pm 5.43$ | $184.0 \pm 24.9$ |

### Paired Statistical Comparisons (Wilcoxon Signed-Rank, $n=10$, SciPy 1.16.3 pinned)

- **Rural Zambia (IID):** FedProx vs FedAvg — diff $= -0.02\text{ pp}$ ($98.46\% \text{ vs } 98.48\%$), $W = 17.5, p = 0.3074, d_z = -0.38$ (not significant)
- **Severe Disruption (IID):** FedProx vs FedAvg — diff $= -0.04\text{ pp}$ ($98.31\% \text{ vs } 98.34\%$), $W = 1.5, p = 0.0078, d_z = -0.78$ (**statistically significant, but practically negligible**)
- **Rural Zambia (Non-IID):** FedProx vs FedAvg — diff $= -0.02\text{ pp}$ ($97.72\% \text{ vs } 97.74\%$), $W = 16.0, p = 0.2377, d_z = -0.34$ (not significant)
- **Severe Disruption (Non-IID):** FedProx vs FedAvg — diff $= +0.01\text{ pp}$ ($97.30\% \text{ vs } 97.29\%$), $W = 27.0, p = 0.9593, d_z = 0.11$ (not significant)

---

## Key Findings

1. **Network impairment impacts delivered volume and convergence latency, not final IID accuracy.** On MNIST IID, severe disruption (60% effective drop rate) reduces final accuracy by only $0.18\text{ pp}$ ($98.52\% \rightarrow 98.34\%$).
2. **Non-IID data compounded with dropouts delays convergence significantly.** Under Non-IID Dirichlet ($\alpha=0.5$), reaching $95\%$ accuracy requires $4.0$ rounds under Baseline, increasing to $7.3$ rounds under Rural Zambia ($+82.5\%$) and $10.1$ rounds under Severe Disruption ($+152.5\%$).
3. **Statistical vs. Practical Significance in FedAvg vs. FedProx.** Across all evaluated conditions, mean performance differences remain within $\pm 0.04\text{ pp}$ (between $-0.04\text{ pp}$ and $+0.01\text{ pp}$). While three comparisons showed no statistically significant difference ($p \ge 0.2377$), Severe Disruption (IID) reached statistical significance ($p = 0.0078$, FedAvg $+0.04\text{ pp}$ over FedProx) due to consistent pairing ($8/10$ seed wins, $2$ ties, $0$ losses). However, this $0.04\text{ pp}$ margin ($98.34\%$ vs $98.31\%$) is practically negligible for real-world deployments.
4. **Delivered communication volume scales inversely with dropout.** Delivered payload drops from $457.7\text{ MiB}$ (Baseline) to $184.0\text{ MiB}$ (Severe Disruption). This reflects **lost updates / channel waste**, not an efficiency saving.

---

## Reproducibility & Testing

- Python 3.9+
- Full test suite: **37 passing automated tests**
  - `tests/unit/` (25 unit tests): aggregation math, configuration validation, network impairment distributions, data partitioning.
  - `tests/integration/` (12 integration tests): PRNG determinism, multi-round dropout state persistence, binary MiB conversion accounting, and FedProx $\mu=0$ mathematical equivalence to FedAvg.
- Run complete test suite:
  ```bash
  python3 -m pytest -v tests/
  ```

---

## Project Structure
```
src/
├── config.py          # Layer 1 — Config loading and validation
├── core/              # Layer 2 — FL server and client orchestration
├── network/           # Layer 3 — Network impairment engine
├── analytics/         # Layer 4 — Metrics tracking, CSV export, plotting
├── algorithms/        # FedAvg and FedProx aggregation implementations
├── data/              # Dataset download and IID/Non-IID partitioning
└── models/            # CNN architectures (MNISTNet, CIFAR10Net)

config/experiments/    # Self-contained YAML experiment configurations
experiments/results/   # Output CSVs, plots, and statistical summary
scripts/               # Multi-seed runner and statistical analysis scripts
tests/
├── unit/              # 25 unit tests across 4 modules
└── integration/       # 12 integration tests across 4 target modules
```

---

## Citation

If you use this simulator in your research, please cite:
Meyo, F. (2026). Federated Learning Network Simulator for Low-Connectivity Environments.
University of Zambia, Department of Computing and Informatics.
GitHub: https://github.com/frankmeyo1738-creator/Federated-Learning-Simulator-for-Low-Connectivity-Environments
