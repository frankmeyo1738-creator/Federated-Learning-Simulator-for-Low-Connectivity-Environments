#!/bin/bash
# run_noniid_baseline.sh
# Runs the 10-seed non-IID Baseline FedAvg matched comparator experiment.
#
# Purpose:
#   Provides a matched non-IID Baseline (ideal network, Dirichlet alpha=0.5 partitioning)
#   to isolate the network impairment component of the +82.5% / +152.5% convergence delay
#   relative to the IID Baseline (Supervisor Feedback Point #12).
#
# Design:
#   - Algorithm: FedAvg only (matched to non-IID Rural/Severe cohort)
#   - Network: Baseline profile (0% dropout, 0% packet loss, 1-5 ms latency)
#   - Data: Non-IID Dirichlet alpha=0.5 (same as all other non-IID runs)
#   - Seeds: 1-10 (n=10 paired seeds)
#   - Rounds: 20 (matching all other runs)
#   - Output: experiments/results/multiseed/noniid/
#
# Expected runtime: ~22 s/round x 20 rounds x 10 seeds = ~75-80 min on M2 Air
# Usage: bash scripts/run_noniid_baseline.sh

set -e
echo "Non-IID Baseline FedAvg Matched Comparator (10 seeds)"
echo "Network: Ideal Baseline (0% dropout, 0% loss) | Data: Dirichlet alpha=0.5"
echo "Started: $(date)"
echo "=================================================="

TOTAL=10; RUN=0
for seed in 1 2 3 4 5 6 7 8 9 10; do
    RUN=$((RUN + 1))
    CONFIG="config/experiments/multiseed/noniid/baseline_fedavg_noniid_seed${seed}.yaml"
    echo ""
    echo "[${RUN}/${TOTAL}] baseline_fedavg_noniid_seed${seed}"
    python3 -m src.main --config "$CONFIG"
    if [ $? -ne 0 ]; then echo "FAILED seed${seed} — aborting."; exit 1; fi
    echo "Done [${RUN}/${TOTAL}]"
done

echo ""
echo "All 10 seeds complete. Results: experiments/results/multiseed/noniid/"
echo "Next: python3 scripts/statistical_analysis.py"
echo "Finished: $(date)"
