# Non-IID Seed-42 Archive

## Provenance

These 4 CSV files are **archived copies** of the original single-seed (seed=42)
non-IID experiment results. They **predate commit `7e7b8cf`** and the subsequent
decision to expand to n=10 multi-seed non-IID runs on Google Colab.

Each file corresponds to a single run of a non-IID (Dirichlet α=0.5) experiment
using `seed: 42`, executed without the `--iid` flag (i.e. Dirichlet partitioning
active). These were the runs reported in the initial supervisor forensic review
response.

## Contents

| File | Config | Seed | Partitioning |
|------|--------|------|--------------|
| `rural_zambia_fedavg_noniid_metrics.csv` | `rural_zambia_fedavg_noniid` | 42 | Dirichlet α=0.5 |
| `rural_zambia_fedprox_noniid_metrics.csv` | `rural_zambia_fedprox_noniid` | 42 | Dirichlet α=0.5 |
| `severe_disruption_fedavg_noniid_metrics.csv` | `severe_disruption_fedavg_noniid` | 42 | Dirichlet α=0.5 |
| `severe_disruption_fedprox_noniid_metrics.csv` | `severe_disruption_fedprox_noniid` | 42 | Dirichlet α=0.5 |

The originals remain in place at `experiments/results/`.

## Why Archived

Retained for audit trail. The n=10 multiseed non-IID re-run on Colab (seeds 1–10,
configs in `config/experiments/multiseed/noniid/`) will supersede these as the
canonical statistical dataset.

## CLI Invocation Note

Non-IID runs use **no `--iid` flag** (the default). Example:
```
python -m src.main --config config/experiments/multiseed/noniid/rural_zambia_fedavg_noniid_seed1.yaml
```
IID runs require the explicit flag:
```
python -m src.main --config config/experiments/multiseed/baseline_fedavg_seed1.yaml --iid
```
