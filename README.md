# Federated Learning Network Simulator for Low-Connectivity Environments

> A reproducible federated learning simulator designed for Sub-Saharan African (SSA) network conditions, evaluating privacy-preserving distributed AI under bandwidth constraints, latency, packet loss, and client dropout.

**Author:** Frank Meyo — University of Zambia, Department of Computing and Informatics  
**Supervisor:** Mr. Mofya Phiri  
**Project:** CSC 4004 Final Year Project, 2026  
**License:** MIT

**Evidence update, 16 October 2026:** the current cohort is **110 runs** (60 IID, 40 impaired non-IID, ten matched non-IID Baseline). Tasks 1–6 are uncommitted corrections based on `b7164d6`; the existing `v1.0.8` tag does not contain them. See [runner coverage and complete 110-run commands](docs/REPRODUCIBILITY.md), [AI disclosures](docs/AI_USAGE_LOG.md), [proposed Appendices A–C](docs/APPENDIX_MATERIAL.md), and [verification record](docs/DOCUMENTATION_VERIFICATION.md).

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

*Profile qualification:* latency uses Gaussian midpoint mu and sigma = 0.15 × range, clamped to configured bounds. Parameters are modelling assumptions, not demonstrated cellular calibration. Urban/Rural bandwidth and packet-loss differences from the proposal are disclosed as implementation drift in [results](docs/RESULTS.md). Production selects CUDA when available and CPU otherwise; it does not select Apple MPS. Bandwidth delay is analytical only; propagation sleeps occur during production execution.

---

## Quickstart

Use a separate copy of the corrected working tree and a compatible Python 3.11+ interpreter. The unpublished corrections are not obtained merely by cloning the existing remote release. Installation below requires package access; it was not executed during Task 6.

```bash
# Install in the prepared copy
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt

# Collect tests without executing fixtures (requires full dependencies)
python3 -m pytest --collect-only -q tests/

# Run multi-seed analysis
python3 scripts/statistical_analysis.py

# Generate genuine diagnostics and separately derived historical delay corrections
python3 scripts/verify_layer3_fidelity.py
python3 scripts/audit_historical_delays.py
```

---

## Experiment Results — Multi-Seed $n=10$ Cohort

Evaluation covers **110 runs** ($n=10$ seeds 1–10 per configuration), including the matched non-IID Baseline. All uncertainties below are sample SDs (ddof=1); volumes are binary MiB and differences are percentage points. The matrix is asymmetric: FedProx exists only in Rural/Severe. Analysis commands write generated summaries but do not train models or download datasets. Training and full test execution are separate operations; see the reproduction guide.

### 1. IID Network Scenarios (MNIST, 20 rounds)

| Experiment | Final Accuracy (%) | Final Loss | Drop Rate (%) | Delivered (MiB) |
|------------|-------------------|------------|---------------|-----------------|
| Baseline FedAvg | $98.52 \pm 0.05$ | $0.0446 \pm 0.0024$ | $0.00 \pm 0.00$ | $457.7 \pm 0.0$ |
| Urban Zambia FedAvg | $98.52 \pm 0.08$ | $0.0443 \pm 0.0031$ | $5.60 \pm 3.03$ | $432.1 \pm 13.8$ |
| Rural Zambia FedAvg | $98.48 \pm 0.09$ | $0.0458 \pm 0.0028$ | $30.20 \pm 7.10$ | $319.5 \pm 32.5$ |
| Rural Zambia FedProx | $98.46 \pm 0.08$ | $0.0462 \pm 0.0026$ | $30.20 \pm 7.10$ | $319.5 \pm 32.5$ |
| Severe Disruption FedAvg | $98.34 \pm 0.05$ | $0.0503 \pm 0.0022$ | $59.80 \pm 5.43$ | $184.0 \pm 24.9$ |
| Severe Disruption FedProx | $98.31 \pm 0.06$ | $0.0510 \pm 0.0023$ | $59.80 \pm 5.43$ | $184.0 \pm 24.9$ |

### 2. Non-IID Dirichlet ($\alpha=0.5$) Scenarios (MNIST, 20 rounds)

| Experiment | Final Accuracy (%) | Final Loss | Drop Rate (%) | Delivered (MiB) |
|------------|-------------------|------------|---------------|-----------------|
| Baseline FedAvg | $98.20 \pm 0.13$ | $0.0557 \pm 0.0050$ | $0.00 \pm 0.00$ | $457.7 \pm 0.0$ |
| Rural Zambia FedAvg | $97.74 \pm 0.93$ | $0.0698 \pm 0.0297$ | $30.20 \pm 7.10$ | $319.5 \pm 32.5$ |
| Rural Zambia FedProx | $97.72 \pm 0.93$ | $0.0699 \pm 0.0286$ | $30.20 \pm 7.10$ | $319.5 \pm 32.5$ |
| Severe Disruption FedAvg | $97.29 \pm 0.77$ | $0.0836 \pm 0.0236$ | $59.80 \pm 5.43$ | $184.0 \pm 24.9$ |
| Severe Disruption FedProx | $97.30 \pm 0.78$ | $0.0836 \pm 0.0242$ | $59.80 \pm 5.43$ | $184.0 \pm 24.9$ |

### Paired Statistical Comparisons (Wilcoxon Signed-Rank, $n=10$, SciPy 1.16.3 pinned)

- **Rural Zambia (IID):** FedProx vs FedAvg — unrounded mean diff = -0.013 pp, $W = 17.5, p = 0.3074, d_z = -0.38$ (not significant)
- **Severe Disruption (IID):** FedProx vs FedAvg — diff $= -0.04\text{ pp}$ ($98.31\% \text{ vs } 98.34\%$), $W = 1.5, p = 0.0078, d_z = -0.78$ (**statistically significant, but practically negligible**)
- **Rural Zambia (Non-IID):** FedProx vs FedAvg — diff $= -0.02\text{ pp}$ ($97.72\% \text{ vs } 97.74\%$), $W = 16.0, p = 0.2377, d_z = -0.34$ (not significant)
- **Severe Disruption (Non-IID):** FedProx vs FedAvg — diff $= +0.01\text{ pp}$ ($97.30\% \text{ vs } 97.29\%$), $W = 27.0, p = 0.9593, d_z = 0.11$ (not significant)

The procedure is asymptotic, two-sided Wilcoxon with `zero_method='zsplit'`, `correction=False`; bootstrap intervals use 2,000 paired resamples, seed 42. Severe IID survives Bonferroni adjustment across four comparisons (p_adj = 0.0312). Observed-effect power is withdrawn; MDE uses a paired-t approximation. Nonsignificance is not equivalence.

---

## Key Findings

1. **Small observed IID accuracy differences:** Baseline and Severe FedAvg differ by 0.18 pp in mean terminal MNIST accuracy (98.52% versus 98.34%), alongside substantial differences in delivered volume.
2. **Matched convergence contrasts:** IID Baseline R95 = 4.0; non-IID Baseline = 6.0. Heterogeneity adds a descriptive +2.0 rounds; Rural/Severe non-IID then add +1.3/+4.1 rounds (+21.67%/+68.33% relative to matched non-IID Baseline). The +82.5%/+152.5% contrasts against IID Baseline combine both changes. No significance test for this decomposition is claimed.
3. **Statistical versus absolute magnitude:** the four mean algorithm differences range from -0.038 to +0.008 pp. Severe IID remains significant after adjustment, but no equivalence or deployment-performance conclusion follows from these small observed differences.
4. **Delivered payload:** the mean falls from 457.7 MiB (Baseline) to 184.0 MiB (Severe). This reflects lost updates, not a measured efficiency saving, energy cost or subscriber-data cost.

---

## Reproducibility & Testing

- Current requirements target compatible Python 3.11+ environments; package pins do not prove historical installed versions or hardware.
- Current inventory: **148 cases, 135 unit + 13 integration**: the previous 140 plus eight release-evidence guard tests. Historical v1.0.2 had 37; v1.0.8 had 38.
- The historical Task 6 checks recorded 102 analysis/diagnostic passes and dependency-limited collection. Task 9A subsequently passed all original 140 locally but found three clean-checkout failures. Task 9A.1 repairs and verifies serialization and provenance; its current results are recorded separately in [release verification](docs/RELEASE_VERIFICATION.md). Eight partition tests can download MNIST and some integration tests train synthetic models.
- [Test allocations, safe commands and runner limitations](docs/REPRODUCIBILITY.md). No single current Colab mode covers 110: default `noniid_baseline` covers ten; `all`/`force_all` cover the original 100.

## Corrected diagnostic and timing evidence

The [genuine engine observations](experiments/results/diagnostics/README.md) contain 10,000 actual method returns with fixed seeds and zero observed inclusive-bounds violations. The old shifted NumPy diagnostic remains preserved under `superseded-evidence/`; it is invalid as raw-engine evidence.

[Selection-aware theory](experiments/results/diagnostics/DROPOUT_THEORY.md) predicts 0%, 5.95%, 26.208599%, 60.732510%; observed rates are 0%, 5.60%, 30.20%, 59.80%. Rural exceeds its pointwise 95% predictive interval; universal theoretical agreement is not claimed.

[Historical delay audit](experiments/results/diagnostics/historical-delay-audit/AUDIT.md): 100 original runs use the binary divisor; ten matched Baseline non-IID runs use decimal Mbps. Separately derived corrections alter bandwidth only, not entire mixed-delay metrics or original CSVs.

Read-only timing audit: pooled recorded throughput is 2.610 rounds/minute across 110 runs, below the 5-round/minute target. Per-run historical hardware is incompletely documented. The timers include propagation waits; they do not isolate overhead or establish thermal throttling. See [results and qualifications](docs/RESULTS.md).

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
├── unit/              # 127 cases across 8 modules
└── integration/       # 13 cases across 4 modules
```

---

## Citation

If you use this simulator in your research, please cite:
Meyo, F. (2026). Federated Learning Network Simulator for Low-Connectivity Environments.
University of Zambia, Department of Computing and Informatics.
GitHub: https://github.com/frankmeyo1738-creator/Federated-Learning-Simulator-for-Low-Connectivity-Environments

## Release evidence verification

The proposed corrected version is `v1.1.0`; no tag, commit or publication is asserted. Evidence CSVs have exact path-specific byte-preservation attributes. Original observations are unchanged; a frozen pre-repair record retains earlier working-tree hashes and source versions. From a full checkout, run the three documented analysis scripts followed by `python3 scripts/verify_release_evidence.py --strict-checkout`. See [the policy and verification procedure](docs/RELEASE_VERIFICATION.md). Thesis and audit deliverables under `output/` are excluded from the source release.
