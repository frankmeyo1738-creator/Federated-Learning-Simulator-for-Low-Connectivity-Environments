#!/bin/bash
# run_multiseed_experiments.sh
# Runs the full 34-experiment multi-seed suite sequentially on local hardware.
#
# Experiment breakdown:
#   Baseline FedAvg      — seeds 1–5 (n=5)   = 5 runs
#   Urban Zambia FedAvg  — seeds 1–5 (n=5)   = 5 runs
#   Rural Zambia FedAvg  — seeds 1–6 (n=6)   = 6 runs
#   Rural Zambia FedProx — seeds 1–6 (n=6)   = 6 runs
#   Severe Disruption FedAvg  — seeds 1–6 (n=6) = 6 runs
#   Severe Disruption FedProx — seeds 1–6 (n=6) = 6 runs
#   Total: 34 runs
#
# Measured throughput on M2 Air CPU: ~14–24 min per run.
# Estimated total: 9–13 hours on CPU. Use a GPU (e.g. NVIDIA T4) to accelerate.
#
# Usage: bash scripts/run_multiseed_experiments.sh

set -e
source .venv/bin/activate

echo "🌱 Starting Multi-Seed Experiment Suite (34 runs)"
echo "Estimated time: ~9–13 hours on M2 Air CPU (14–24 min/run)"
echo "Started at: $(date)"
echo "Results → experiments/results/multiseed/"
echo "=================================="

RUN=0
TOTAL=34

run_experiment() {
    local config="$1"
    local label="$2"
    RUN=$((RUN + 1))
    echo ""
    echo "▶️  [$RUN/$TOTAL] $label"
    python -m src.main --config "$config" --iid
    if [ $? -ne 0 ]; then
        echo "❌ [$RUN/$TOTAL] FAILED: $label — aborting."
        exit 1
    fi
    echo "✅ [$RUN/$TOTAL] Done — $label"
    echo "----------------------------------"
}

# ── Baseline FedAvg (seeds 1–5) ──────────────────────────────────────────────
for seed in 1 2 3 4 5; do
    run_experiment \
        "config/experiments/multiseed/baseline_fedavg_seed${seed}.yaml" \
        "baseline_fedavg_seed${seed}"
done

# ── Urban Zambia FedAvg (seeds 1–5) ──────────────────────────────────────────
for seed in 1 2 3 4 5; do
    run_experiment \
        "config/experiments/multiseed/urban_zambia_fedavg_seed${seed}.yaml" \
        "urban_zambia_fedavg_seed${seed}"
done

# ── Rural Zambia FedAvg (seeds 1–6) ──────────────────────────────────────────
for seed in 1 2 3 4 5 6; do
    run_experiment \
        "config/experiments/multiseed/rural_zambia_fedavg_seed${seed}.yaml" \
        "rural_zambia_fedavg_seed${seed}"
done

# ── Rural Zambia FedProx (seeds 1–6) ─────────────────────────────────────────
for seed in 1 2 3 4 5 6; do
    run_experiment \
        "config/experiments/multiseed/rural_zambia_fedprox_seed${seed}.yaml" \
        "rural_zambia_fedprox_seed${seed}"
done

# ── Severe Disruption FedAvg (seeds 1–6) ─────────────────────────────────────
for seed in 1 2 3 4 5 6; do
    run_experiment \
        "config/experiments/multiseed/severe_disruption_fedavg_seed${seed}.yaml" \
        "severe_disruption_fedavg_seed${seed}"
done

# ── Severe Disruption FedProx (seeds 1–6) ────────────────────────────────────
for seed in 1 2 3 4 5 6; do
    run_experiment \
        "config/experiments/multiseed/severe_disruption_fedprox_seed${seed}.yaml" \
        "severe_disruption_fedprox_seed${seed}"
done

echo ""
echo "🎉 All 34 runs complete!"
echo "Finished at: $(date)"
echo "Run 'python scripts/statistical_analysis.py' to regenerate the statistical summary."
