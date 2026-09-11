# Federated Learning Network Simulator for Low-Connectivity Environments

> A reproducible federated learning simulator calibrated to Sub-Saharan African (SSA) network conditions, evaluating privacy-preserving distributed AI under bandwidth constraints, latency, packet loss, and client dropout.

**Author:** Frank Meyo — University of Zambia, Department of Computing and Informatics  
**Supervisor:** Mr. Mofya Phiri  
**Project:** CSC 4004 Final Year Project, 2026  
**License:** MIT

---

## Motivation

Federated learning (FL) enables distributed model training without sharing raw data — a critical property for privacy-sensitive applications in healthcare, agriculture, and finance. However, most FL research assumes reliable, high-bandwidth network conditions that do not reflect the infrastructure realities of Sub-Saharan Africa, where connectivity is characterised by high latency, frequent dropouts, and limited bandwidth.

This simulator addresses that gap by providing a configurable, reproducible environment for evaluating FL algorithms under SSA-calibrated network profiles, enabling researchers to assess algorithm behaviour before real-world deployment.

---

## Architecture

The simulator is built across four layers:
Layer 1 — Configuration      YAML-driven experiment and network profile loading
Layer 2 — FL Core            FedAvg and FedProx clients and server orchestration
Layer 3 — Network Impairment Latency, packet loss, bandwidth throttling, client dropout
Layer 4 — Analytics          Per-round metrics, CSV export, visualisation

---

## Network Profiles

Four SSA-calibrated profiles, derived from GSMA 2023 connectivity data:

| Profile | Latency (ms) | Bandwidth | Packet Loss | Dropout Prob |
|---------|-------------|-----------|-------------|--------------|
| Baseline | 1–5 | 100 Mbps | 0% | 0% |
| Urban Zambia | 80–200 | 8.5 Mbps | 1% | 5% |
| Rural Zambia | 300–800 | 1.2 Mbps | 7% | 15% |
| Severe Disruption | 800–2000 | 0.3 Mbps | 20% | 35% |

---

## Quickstart

```bash
# Clone and install
git clone https://github.com/frankmeyo1738-creator/Federated-Learning-Simulator-for-Low-Connectivity-Environments.git
cd Federated-Learning-Simulator-for-Low-Connectivity-Environments
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Run a smoke test
python -m src.main --config config/experiments/smoke_test.yaml --iid

# Run standard benchmark suite (6 core experiments)
bash scripts/run_all_experiments.sh

# Run multi-seed statistical suite (34 runs)
bash scripts/run_multiseed_experiments.sh

# Run non-IID Dirichlet suite (5 experiments)
bash scripts/run_noniid_experiments.sh
```

---

## Experiment Results

### MNIST — Multi-Seed (baseline/urban: n=5 seeds; rural/severe: n=6 seeds — 34 runs total)

| Experiment | Final Accuracy | Std | Drop Rate |
|------------|---------------|-----|-----------|
| Baseline FedAvg | 98.54% | ±0.07% | 0.00% |
| Urban Zambia FedAvg | 98.50% | ±0.13% | 6.20% |
| Rural Zambia FedAvg | 98.41% | ±0.05% | 29.83% |
| Rural Zambia FedProx | 98.45% | ±0.12% | 29.83% |
| Severe Disruption FedAvg | 98.24% | ±0.20% | 60.67% |
| Severe Disruption FedProx | 98.29% | ±0.08% | 60.67% |

**Statistical tests (Wilcoxon signed-rank, n=6):**
- Rural Zambia: FedProx vs FedAvg — p=0.5625, Cohen's d=0.27 (small effect, not significant)
- Severe Disruption: FedProx vs FedAvg — p=0.6875, Cohen's d=0.31 (small effect, not significant)

### CIFAR-10 — Single Seed (seed 42)

| Experiment | Final Accuracy | Drop Rate | Bytes Transmitted |
|------------|---------------|-----------|-------------------|
| Baseline FedAvg | 73.42% | 0.00% | 627.6 MB |
| Rural Zambia FedAvg | 72.94% | 29.33% | 443.5 MB |
| Severe Disruption FedAvg | 69.64% | 61.33% | 242.7 MB |
| Severe Disruption FedProx | 70.52% | 61.33% | 242.7 MB |

---

## Key Findings

1. **Network impairment primarily affects client participation and delivered data volume, not accuracy.** Under MNIST, severe disruption (60.67% effective dropout) reduced accuracy by only 0.30% (98.54% → 98.24%) while causing successfully delivered communication volume to fall by ~61% (from 457.7 MB down to 180.0 MB).* This reduction reflects dropped parameter updates and lost connectivity rather than an intentional optimization.

2. **CIFAR-10 reveals clearer degradation.** On a harder dataset, severe disruption produced a 3.78% accuracy drop, demonstrating that impairment effects are dataset-dependent.

3. **FedProx shows modest robustness gains under severe disruption.** FedProx outperformed FedAvg by 0.88% on CIFAR-10 under severe disruption. On MNIST, FedProx leads by 0.05% on average (98.29% vs 98.24%, p=0.6875, Cohen's d=0.31, small effect, not significant). Notably, on seed 6, FedAvg slightly outperformed FedProx (98.44% vs 98.34%), illustrating sample variability and the absence of a reliable algorithmic advantage on this task.

4. **Successfully delivered bytes decrease in direct proportion to dropout rate.** Due to severe network drops, ~61% fewer bytes successfully reached the aggregation server under severe disruption compared to baseline (180.0 MB vs. 457.7 MB), illustrating the communication deficit in intermittent deployments.

*\*Note on sample sizes: Baseline FedAvg metrics are averaged across $n=5$ seeds, while Severe Disruption FedAvg metrics are averaged across the expanded $n=6$ seed cohort.*

---

## Limitations

- MNIST multi-seed statistical tests show no significant FedAvg vs FedProx difference; CIFAR-10 results are single-seed only
- Simulated latency affects transmission delay metrics but not wall-clock training time
- Flower/PySyft framework comparison deferred to future work
- µ hyperparameter tuning for FedProx not explored

---

## Reproducibility

- Python 3.9+
- All dependencies: `pip install -r requirements.txt`
- All experiment configs: `config/experiments/`
- All results: `experiments/results/`
- Fixed seeds (1–5) for Baseline and Urban Zambia runs; seeds 1–6 for Rural Zambia and Severe Disruption runs
- Run `pytest tests/unit/ -v` to verify 24/24 tests pass

---

## Project Structure
```
src/
├── config.py          # Layer 1 — Config loading and validation
├── core/              # Layer 2 — FL server and client
├── network/           # Layer 3 — Network impairment engine
├── analytics/         # Layer 4 — Metrics tracking and export
├── algorithms/        # FedAvg and FedProx implementations
└── data/              # Dataset loading and partitioning

config/experiments/    # YAML experiment configurations
experiments/results/   # Output CSVs and plots
scripts/               # Analysis and experiment runner scripts
tests/unit/            # 24 unit tests
```

---

## Citation

If you use this simulator in your research, please cite:
Meyo, F. (2026). Federated Learning Network Simulator for Low-Connectivity Environments.
University of Zambia, Department of Computing and Informatics.
GitHub: https://github.com/frankmeyo1738-creator/Federated-Learning-Simulator-for-Low-Connectivity-Environments
