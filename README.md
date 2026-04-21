# Federated Learning Network Simulator for Low-Connectivity Environments

A Python-based FL network simulation framework for testing federated 
learning algorithms under realistic Sub-Saharan African network conditions.

## Project Status
🚧 Under active development — Final Year Project, UNZA CS 2026

## Supervisor
Mr. Mofya Phiri, Department of Computing and Informatics, UNZA

## Tech Stack
Python 3.11+ · PyTorch · asyncio · PyYAML · pytest

## Architecture

The simulator is built with a clean 4-layer architecture:

1. **Configuration Layer** — YAML-driven experiment config with validation
2. **Simulation Engine** — FL training loop orchestration (FedAvg / FedProx)
3. **Network Impairment Engine** — Realistic network condition simulation
4. **Analytics Layer** — Metrics tracking, CSV export, and visualization

## Network Profiles

| Profile | Latency (ms) | Bandwidth | Packet Loss | Dropout |
|---------|-------------|-----------|-------------|---------|
| Baseline | 1–5 | 100 Mbps | 0% | 0% |
| Urban Zambia | 80–200 | 8.5 Mbps | 1% | 5% |
| Rural Zambia | 300–800 | 1.2 Mbps | 7% | 15% |
| Severe Disruption | 800–2000 | 0.3 Mbps | 20% | 35% |

## Milestones
- [x] Proposal submitted
- [ ] Core simulator (Week 5-8)
- [ ] Network impairment engine (Week 9-10)
- [ ] Experiments (Week 12-13)
- [ ] Final submission: September 29, 2026

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run an experiment
python -m src.main --config config/experiments/example_experiment.yaml
```

## License
Academic use — UNZA Department of Computing and Informatics
