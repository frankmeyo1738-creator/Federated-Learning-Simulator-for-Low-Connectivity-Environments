# Pre-Seeding-Fix Archive — IID Multiseed CSVs (Seeds 1–5)

## Provenance

These 30 CSV files are **archived copies** of the original seed 1–5 metric results
for all six IID experiment configurations. They **predate commit `7e7b8cf`**, which
introduced a critical seeding fix: prior to that commit, `random`, `numpy`, and
`torch` RNGs were not all seeded before data partitioning, meaning results were
not fully reproducible.

## What's Here

30 files — 6 IID configs × 5 seeds (1–5):
- `baseline_fedavg_seed{1..5}_metrics.csv`
- `rural_zambia_fedavg_seed{1..5}_metrics.csv`
- `rural_zambia_fedprox_seed{1..5}_metrics.csv`
- `severe_disruption_fedavg_seed{1..5}_metrics.csv`
- `severe_disruption_fedprox_seed{1..5}_metrics.csv`
- `urban_zambia_fedavg_seed{1..5}_metrics.csv`

The originals remain in place at `experiments/results/multiseed/`.

## Why Archived, Not Deleted

Retained for audit trail and comparison purposes (supervisor review, DAAD
documentation). The n=10 re-run on Colab (seeds 1–10) will supersede these as
the canonical statistical dataset, but these files document what was submitted
pre-fix.

## Key Commit Reference

| Commit | Description |
|--------|-------------|
| `7e7b8cf` | Fix seeding bug, expand to n=6 stats, add non-IID FedAvg comparison |

These CSVs were produced **before** `7e7b8cf` was merged.
