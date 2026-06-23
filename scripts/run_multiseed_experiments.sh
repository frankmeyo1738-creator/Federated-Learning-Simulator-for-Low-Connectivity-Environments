#!/bin/bash
echo "🌱 Starting Multi-Seed Experiment Suite (30 runs)"
echo "Estimated time: ~90 mins on Colab T4 GPU"
echo "Started at: $(date)"
echo "Results → experiments/results/multiseed/"
echo "=================================="
echo "▶️  [1/30] baseline_fedavg_seed1"
python -m src.main --config config/experiments/multiseed/baseline_fedavg_seed1.yaml --iid && \
echo "✅ [1/30] Done" && \
echo "▶️  [2/30] baseline_fedavg_seed2"
python -m src.main --config config/experiments/multiseed/baseline_fedavg_seed2.yaml --iid && \
echo "✅ [2/30] Done" && \
echo "▶️  [3/30] baseline_fedavg_seed3"
python -m src.main --config config/experiments/multiseed/baseline_fedavg_seed3.yaml --iid && \
echo "✅ [3/30] Done" && \
echo "▶️  [4/30] baseline_fedavg_seed4"
python -m src.main --config config/experiments/multiseed/baseline_fedavg_seed4.yaml --iid && \
echo "✅ [4/30] Done" && \
echo "▶️  [5/30] baseline_fedavg_seed5"
python -m src.main --config config/experiments/multiseed/baseline_fedavg_seed5.yaml --iid && \
echo "✅ [5/30] Done" && \
echo "▶️  [6/30] urban_zambia_fedavg_seed1"
python -m src.main --config config/experiments/multiseed/urban_zambia_fedavg_seed1.yaml --iid && \
echo "✅ [6/30] Done" && \
echo "▶️  [7/30] urban_zambia_fedavg_seed2"
python -m src.main --config config/experiments/multiseed/urban_zambia_fedavg_seed2.yaml --iid && \
echo "✅ [7/30] Done" && \
echo "▶️  [8/30] urban_zambia_fedavg_seed3"
python -m src.main --config config/experiments/multiseed/urban_zambia_fedavg_seed3.yaml --iid && \
echo "✅ [8/30] Done" && \
echo "▶️  [9/30] urban_zambia_fedavg_seed4"
python -m src.main --config config/experiments/multiseed/urban_zambia_fedavg_seed4.yaml --iid && \
echo "✅ [9/30] Done" && \
echo "▶️  [10/30] urban_zambia_fedavg_seed5"
python -m src.main --config config/experiments/multiseed/urban_zambia_fedavg_seed5.yaml --iid && \
echo "✅ [10/30] Done" && \
echo "▶️  [11/30] rural_zambia_fedavg_seed1"
python -m src.main --config config/experiments/multiseed/rural_zambia_fedavg_seed1.yaml --iid && \
echo "✅ [11/30] Done" && \
echo "▶️  [12/30] rural_zambia_fedavg_seed2"
python -m src.main --config config/experiments/multiseed/rural_zambia_fedavg_seed2.yaml --iid && \
echo "✅ [12/30] Done" && \
echo "▶️  [13/30] rural_zambia_fedavg_seed3"
python -m src.main --config config/experiments/multiseed/rural_zambia_fedavg_seed3.yaml --iid && \
echo "✅ [13/30] Done" && \
echo "▶️  [14/30] rural_zambia_fedavg_seed4"
python -m src.main --config config/experiments/multiseed/rural_zambia_fedavg_seed4.yaml --iid && \
echo "✅ [14/30] Done" && \
echo "▶️  [15/30] rural_zambia_fedavg_seed5"
python -m src.main --config config/experiments/multiseed/rural_zambia_fedavg_seed5.yaml --iid && \
echo "✅ [15/30] Done" && \
echo "▶️  [16/30] severe_disruption_fedavg_seed1"
python -m src.main --config config/experiments/multiseed/severe_disruption_fedavg_seed1.yaml --iid && \
echo "✅ [16/30] Done" && \
echo "▶️  [17/30] severe_disruption_fedavg_seed2"
python -m src.main --config config/experiments/multiseed/severe_disruption_fedavg_seed2.yaml --iid && \
echo "✅ [17/30] Done" && \
echo "▶️  [18/30] severe_disruption_fedavg_seed3"
python -m src.main --config config/experiments/multiseed/severe_disruption_fedavg_seed3.yaml --iid && \
echo "✅ [18/30] Done" && \
echo "▶️  [19/30] severe_disruption_fedavg_seed4"
python -m src.main --config config/experiments/multiseed/severe_disruption_fedavg_seed4.yaml --iid && \
echo "✅ [19/30] Done" && \
echo "▶️  [20/30] severe_disruption_fedavg_seed5"
python -m src.main --config config/experiments/multiseed/severe_disruption_fedavg_seed5.yaml --iid && \
echo "✅ [20/30] Done" && \
echo "▶️  [21/30] rural_zambia_fedprox_seed1"
python -m src.main --config config/experiments/multiseed/rural_zambia_fedprox_seed1.yaml --iid && \
echo "✅ [21/30] Done" && \
echo "▶️  [22/30] rural_zambia_fedprox_seed2"
python -m src.main --config config/experiments/multiseed/rural_zambia_fedprox_seed2.yaml --iid && \
echo "✅ [22/30] Done" && \
echo "▶️  [23/30] rural_zambia_fedprox_seed3"
python -m src.main --config config/experiments/multiseed/rural_zambia_fedprox_seed3.yaml --iid && \
echo "✅ [23/30] Done" && \
echo "▶️  [24/30] rural_zambia_fedprox_seed4"
python -m src.main --config config/experiments/multiseed/rural_zambia_fedprox_seed4.yaml --iid && \
echo "✅ [24/30] Done" && \
echo "▶️  [25/30] rural_zambia_fedprox_seed5"
python -m src.main --config config/experiments/multiseed/rural_zambia_fedprox_seed5.yaml --iid && \
echo "✅ [25/30] Done" && \
echo "▶️  [26/30] severe_disruption_fedprox_seed1"
python -m src.main --config config/experiments/multiseed/severe_disruption_fedprox_seed1.yaml --iid && \
echo "✅ [26/30] Done" && \
echo "▶️  [27/30] severe_disruption_fedprox_seed2"
python -m src.main --config config/experiments/multiseed/severe_disruption_fedprox_seed2.yaml --iid && \
echo "✅ [27/30] Done" && \
echo "▶️  [28/30] severe_disruption_fedprox_seed3"
python -m src.main --config config/experiments/multiseed/severe_disruption_fedprox_seed3.yaml --iid && \
echo "✅ [28/30] Done" && \
echo "▶️  [29/30] severe_disruption_fedprox_seed4"
python -m src.main --config config/experiments/multiseed/severe_disruption_fedprox_seed4.yaml --iid && \
echo "✅ [29/30] Done" && \
echo "▶️  [30/30] severe_disruption_fedprox_seed5"
python -m src.main --config config/experiments/multiseed/severe_disruption_fedprox_seed5.yaml --iid && \
echo "✅ [30/30] Done" 
echo "🎉 All 30 runs complete!"
echo "Finished at: $(date)"
