#!/bin/bash
# run_noniid_experiments.sh
# Automates running the 4 non-IID simulation experiments sequentially.

# Activate the virtual environment
source .venv/bin/activate

# List of experiment configs to run
EXPERIMENTS=(
    "baseline_fedavg_noniid.yaml"
    "severe_disruption_fedavg_noniid.yaml"
    "severe_disruption_fedprox_noniid.yaml"
    "rural_zambia_fedprox_noniid.yaml"
)

echo "🚀 Starting Non-IID Experiment Suite"
echo "=================================="

for config in "${EXPERIMENTS[@]}"; do
    echo ""
    echo "▶️  Running $config..."
    python -m src.main --config "config/experiments/$config"
    
    if [ $? -ne 0 ]; then
        echo "❌ Experiment $config failed! Aborting."
        exit 1
    fi
    echo "✅ Finished $config."
    echo "----------------------------------"
done

echo ""
echo "🎉 All non-IID experiments completed successfully!"
echo "Check the experiments/results/ directory for the CSVs and plots."
