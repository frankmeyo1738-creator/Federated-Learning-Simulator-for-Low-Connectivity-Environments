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

# Run full experiment suite
bash scripts/run_all_experiments.sh
```

---

## Experiment Results

### MNIST — Multi-Seed (5 seeds × 6 experiments = 30 runs)

| Experiment | Final Accuracy | Std | Drop Rate |
|------------|---------------|-----|-----------|
| Baseline FedAvg | 98.54% | ±0.07% | 0.00% |
| Urban Zambia FedAvg | 98.50% | ±0.13% | 6.20% |
| Rural Zambia FedAvg | 98.41% | ±0.05% | 27.60% |
| Rural Zambia FedProx | 98.45% | ±0.14% | 27.60% |
| Severe Disruption FedAvg | 98.20% | ±0.20% | 60.40% |
| Severe Disruption FedProx | 98.28% | ±0.08% | 60.40% |

**Statistical tests (Wilcoxon signed-rank, n=5):**
- Rural Zambia: FedProx vs FedAvg — p=0.8125, Cohen's d=0.24 (no significant difference)
- Severe Disruption: FedProx vs FedAvg — p=0.4375, Cohen's d=0.50 (medium effect, not significant)

### CIFAR-10 — Single Seed (seed 42)

| Experiment | Final Accuracy | Drop Rate | Bytes Transmitted |
|------------|---------------|-----------|-------------------|
| Baseline FedAvg | 73.42% | 0.00% | 627.6 MB |
| Rural Zambia FedAvg | 72.94% | 29.33% | 443.5 MB |
| Severe Disruption FedAvg | 69.64% | 61.33% | 242.7 MB |
| Severe Disruption FedProx | 70.52% | 61.33% | 242.7 MB |

---

## Key Findings

1. **Network impairment primarily affects participation and communication cost, not accuracy.** Under MNIST, severe disruption (60% dropout) reduced accuracy by only 0.34% while cutting communication volume by 64%.

2. **CIFAR-10 reveals clearer degradation.** On a harder dataset, severe disruption produced a 3.78% accuracy drop, demonstrating that impairment effects are dataset-dependent.

3. **FedProx shows modest robustness gains under severe disruption.** FedProx outperformed FedAvg by 0.88% on CIFAR-10 and 0.08% on MNIST under severe conditions, though MNIST differences were not statistically significant (p=0.44).

4. **Communication cost scales directly with dropout rate.** Severe disruption transmitted 64% less data than baseline, with important implications for bandwidth-constrained deployments.

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
- Fixed seeds (1–5) for all multi-seed runs
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
