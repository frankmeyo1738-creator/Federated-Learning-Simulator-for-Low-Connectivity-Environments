#!/bin/bash
# run_all_experiments.sh
# Automates running all 6 simulation experiments sequentially.

# Activate the virtual environment
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

echo "🚀 Starting Full Experiment Suite"
echo "=================================="

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
