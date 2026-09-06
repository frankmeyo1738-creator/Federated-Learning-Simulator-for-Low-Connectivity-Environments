#!/bin/bash
# run_all_experiments.sh
# Automates running the 6 standard benchmark simulation experiments sequentially
# (MNIST, IID data distribution, default seed 42).
#
# Suite breakdown:
#   1. Baseline FedAvg               (config/experiments/baseline_fedavg.yaml)
#   2. Urban Zambia FedAvg           (config/experiments/urban_zambia_fedavg.yaml)
#   3. Rural Zambia FedAvg           (config/experiments/rural_zambia_fedavg.yaml)
#   4. Severe Disruption FedAvg      (config/experiments/severe_disruption_fedavg.yaml)
#   5. Rural Zambia FedProx          (config/experiments/rural_zambia_fedprox.yaml)
#   6. Severe Disruption FedProx     (config/experiments/severe_disruption_fedprox.yaml)
#
# For multi-seed statistical evaluation (34 runs), see: scripts/run_multiseed_experiments.sh
# For non-IID Dirichlet partition evaluation (5 runs), see: scripts/run_noniid_experiments.sh
#
# Usage: bash scripts/run_all_experiments.sh (from repository root)

set -e
source .venv/bin/activate

# List of experiment configs to run
EXPERIMENTS=(
    "baseline_fedavg.yaml"
    "urban_zambia_fedavg.yaml"
    "rural_zambia_fedavg.yaml"
    "severe_disruption_fedavg.yaml"
    "rural_zambia_fedprox.yaml"
    "severe_disruption_fedprox.yaml"
)

echo "🚀 Starting 6-Experiment Standard Benchmark Suite (IID, seed 42)"
echo "================================================================"

for config in "${EXPERIMENTS[@]}"; do
    echo ""
    echo "▶️  Running $config..."
    python -m src.main --config "config/experiments/$config" --iid
    
    if [ $? -ne 0 ]; then
        echo "❌ Experiment $config failed! Aborting."
        exit 1
    fi
    echo "✅ Finished $config."
    echo "----------------------------------"
done

echo ""
echo "🎉 All experiments completed successfully!"
echo "Check the experiments/results/ directory for the CSVs and plots."
