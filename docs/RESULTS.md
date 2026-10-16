# Experimental results and evidence boundaries

Updated 16 October 2026 from preserved CSVs and Tasks 1–5. No ML runs were regenerated. Tasks 1–6 are included in the verified local v1.1.0 release; the existing v1.0.8 tag predates them. GitHub publication has not occurred. See [release verification](RELEASE_VERIFICATION.md) for the dated checks and remaining publication gate.

## Cohort and statistical procedure

There are **110 MNIST runs**: 60 IID, 40 impaired non-IID and ten matched non-IID Baseline runs. Eleven configurations each have seeds 1–10 and 20 rounds. Ten clients exist and five are selected per round. Non-IID partitioning is Dirichlet alpha 0.5. The matrix is asymmetric: there is no Baseline/Urban FedProx or Urban non-IID cohort.

The [generated statistical summary](../experiments/results/statistical_summary.txt) uses sample SD (ddof=1), paired two-sided asymptotic Wilcoxon with zero_method='zsplit', mode='approx', correction=False, and paired Cohen's d_z with 95% percentile-bootstrap intervals: **2,000 paired resamples, RandomState seed 42**. Observed-effect power is withdrawn. MDE ≈1.00 at 80% power is a paired-t approximation, not exact Wilcoxon power.

| Configuration | Final accuracy (%), mean ± sample SD | R95 mean first-hit round |
|---|---:|---:|
| Baseline FedAvg IID | 98.52 ± 0.05 | 4.0 |
| Urban FedAvg IID | 98.52 ± 0.08 | 3.9 |
| Rural FedAvg IID | 98.48 ± 0.09 | 4.0 |
| Rural FedProx IID | 98.46 ± 0.08 | 4.1 |
| Severe FedAvg IID | 98.34 ± 0.05 | 4.7 |
| Severe FedProx IID | 98.31 ± 0.06 | 4.7 |
| Baseline FedAvg non-IID | 98.20 ± 0.13 | 6.0 |
| Rural FedAvg non-IID | 97.74 ± 0.93 | 7.3 |
| Rural FedProx non-IID | 97.72 ± 0.93 | 7.3 |
| Severe FedAvg non-IID | 97.29 ± 0.77 | 10.1 |
| Severe FedProx non-IID | 97.30 ± 0.78 | 10.3 |

R95 is a first-hit criterion, not sustained convergence. Threshold means are withheld if any seed fails to reach the threshold.

| Paired comparison | FedProx − FedAvg (pp), unrounded mean | W | Asymptotic p | Bonferroni p_adj | d_z [95% bootstrap CI] |
|---|---:|---:|---:|---:|---|
| Rural IID | -0.013 | 17.5 | 0.3074 | 1.0000 | -0.38 [-1.61, 0.19] |
| Severe IID | -0.038 | 1.5 | 0.0078 | 0.0312 | -0.78 [-1.34, -0.55] |
| Rural non-IID | -0.021 | 16.0 | 0.2377 | 0.9510 | -0.34 [-3.64, 0.22] |
| Severe non-IID | +0.008 | 27.0 | 0.9593 | 1.0000 | 0.11 [-0.75, 0.70] |

Severe IID survives Bonferroni adjustment over four comparisons. No material FedProx advantage was observed at mu = 0.01; this is not an equivalence test or proof about other datasets/hyperparameters. Differences are calculated before rounding cohort means.

## Matched convergence contrasts

FedAvg IID Baseline R95 = 4.0; non-IID Baseline = 6.0. Their descriptive contrast is **+2.0 rounds (+50%)**. Relative to matched non-IID Baseline:

- Rural non-IID: 7.3 rounds, **+1.3 (+21.67%)**.
- Severe non-IID: 10.1 rounds, **+4.1 (+68.33%)**.

Total contrasts against IID Baseline are +3.3 rounds (+82.50%) and +6.1 (+152.50%). They combine partition and network changes. Percentages with different denominators must not be added. No confidence interval or significance test for this decomposition is claimed.

Current code seeds before model initialisation; the earlier bug left initial weights unseeded, while partitioning already used a seeded NumPy generator. CSVs support seed labels and recorded outcomes, not historical device manifests or cross-device bit identity.

## Genuine Layer 3 evidence

Sources: [diagnostic report](../experiments/results/diagnostics/layer3_validation_report.txt), [method explanation](../experiments/results/diagnostics/README.md), [provenance](../experiments/results/diagnostics/layer3_provenance.json).

| Profile | Seed | Engine calls | Mean propagation (ms) | Sample SD (ms) |
|---|---:|---:|---:|---:|
| Baseline | 100 | 2,500 | 2.979889 | 0.600208 |
| Urban | 101 | 2,500 | 140.285986 | 18.130461 |
| Rural | 102 | 2,500 | 549.977460 | 75.820811 |
| Severe | 103 | 2,500 | 1401.944613 | 179.009523 |

Only sleep is suppressed during these actual production-method calls. Returned values are unadjusted. All 10,000 observations fall within inclusive bounds. This checks sampled bounds, not field calibration or empirical normality. The Gaussian is clamped, not rejection-sampled.

The October 10 diagnostic generated NumPy values and shifted them toward preset means. Its generator and outputs remain [preserved with hashes](../experiments/results/diagnostics/superseded-evidence/pre-correction-16-v1.0.8/README.md), but cannot support raw-engine claims. The student confirms Google Antigravity assistance from direct recollection; the underlying model is unspecified. Genuine samples above are fresh diagnostics, not recovered historical ML samples.

## Selection-aware dropout validation

The [derivation](../experiments/results/diagnostics/DROPOUT_THEORY.md) includes 10 clients, selection of 5 without replacement, triggers on selected eligible transmissions, persistent disconnections, packet loss after disconnection checks, initially online clients and 20 rounds. Expected means use rational arithmetic; predictive distributions use numerical event enumeration, not Monte Carlo.

| Profile | Predicted 20-round drop (%) | Observed mean (%) | Observed sample SD (pp) |
|---|---:|---:|---:|
| Baseline | 0.000000 | 0.00 | 0.00 |
| Urban | 5.950000 | 5.60 | 3.03 |
| Rural | 26.208599 | 30.20 | 7.10 |
| Severe | 60.732510 | 59.80 | 5.43 |

Observations are from canonical ten-seed FedAvg IID strata. Matching seeded outcomes across groups are not independent extra replications. Rural's 30.20% is outside the model's pointwise 95% predictive interval [23.1%, 29.4%], although inside the exploratory four-profile Bonferroni band [22.2%, 30.3%]. These are predictive intervals under model assumptions, not confidence intervals for an estimated observed mean. The discrepancy remains; blanket “matches theory” wording is withdrawn.

## Historical delay units and delivered payload

The [historical audit](../experiments/results/diagnostics/historical-delay-audit/AUDIT.md) supports binary-divisor classification for 100 original runs and decimal-Mbps classification for the ten matched non-IID Baseline runs. Git/source evidence and numerical compatibility agree, but are not an attested execution manifest.

Saved avg_latency_ms is a **cumulative successful-upload mean of propagation plus analytical bandwidth delay**. For cumulative delivered bytes C, successful uploads S, fixed bandwidth w and historical divisor d, derive the decimal mean by adding 8000*C/(w*S) × (1/1000000 − 1/d) to the saved mean. Never scale the whole mixed metric. Variable payloads are handled linearly; current-round means use cumulative weighted differences. Sixty no-success rounds have missing current-round delay means, not zero observations.

All 2,200 row derivatives are separate from originals, with provenance. Individual historical latency draws and serialized wire payloads are unavailable. Production bandwidth conversion is now decimal; bandwidth delay is analytical, not slept and does not schedule arrivals or enforce deadlines.

Delivered parameter payload is unchanged: Baseline 457.7 ± 0.0 MiB, Urban 432.1 ± 13.8, Rural 319.5 ± 32.5, Severe 184.0 ± 24.9 (ten-seed means ± sample SD). Reduced delivered bytes mean lost updates, not bandwidth efficiency. Energy, subscriber quota and protocol overhead are not measured.

## Timing, hardware and overhead — read-only Task 5

Pooled throughput = 60 × total recorded rounds / sum(round_duration_seconds). Across 110 runs: 2,200 rounds, 50,567.148399 recorded seconds, **2.610390 rounds/minute**; mean round 22.985067 s; mean summed duration/run 459.701349 s; range 391.320742–859.863056 s. No run averages 5 rounds/minute. Original 100-run pooled throughput is 2.590970; matched non-IID Baseline is 2.821903.

The round timer includes selection, local training/transfers, propagation waits, impairment checks, aggregation and evaluation. It excludes initial setup, metric recording after the timer, and final CSV/plot export. The tracker sums these durations as total experiment time; that is not complete process runtime. The Colab wrapper separately times the child process. Sleeps may overlap other work; neither summing propagation delays nor comparing profile means isolates overhead.

**Target status: NOT MET in recorded results.** The proposal's M2/8 GB hardware condition is not verified for each run. Documentation reports local M2 development and Colab T4 execution; October 13 supervisor feedback specifically corroborates T4 non-IID Baseline seed 1. No complete per-run hardware/software map, retained run logs or notebook outputs were found. Current analysis provenance cannot establish historical training hardware.

The old 398–860 s / mean 467 s summary is not the current 110-run aggregate. Thermal throttling, CPU-only timing, +0.73 s/round isolated overhead, and “no measurable overhead above a 1.28 s compute noise floor” are unsupported. Baseline round SD 1.283907 s is descriptive variability, not an overhead bound. The +0.73 s withdrawal remains; no new benchmark is claimed.

## Exploratory CIFAR-10: current files versus earlier prose

Read-only extraction from four current experiments/results/cifar10/*_metrics.csv files gives 15 rounds/run:

| Config | Final accuracy (%) |
|---|---:|
| baseline_fedavg_cifar10 | 73.42 |
| rural_zambia_fedavg_cifar10 | 72.94 |
| severe_disruption_fedavg_cifar10 | 69.64 |
| severe_disruption_fedprox_cifar10 | 70.52 |

Previous documentation figures 65.80/61.40/51.20/52.08% do not match these final rows and lack a run/version attribution here. They are superseded as current-CSV summaries, not silently treated as validated earlier observations. The Severe difference is +0.88 pp, a single-seed hypothesis for future work, not general superiority. Task 7 reconciled these source files with the thesis, and the Task 8 template report retains the traceable 69.64%/70.52% Severe values while explicitly withdrawing the untraceable 61.57%/62.45% pair. The four current files trace to `d678bef097b9be4b0480b2470c48b85441e23043`, use the seed-42 configurations and predate the model-initialisation seeding fix. No missing historical run or hardware attribution has been invented. No Baseline FedProx output exists in this set.

## Profile and reproduction qualifications

Profiles are SSA-inspired assumptions, not demonstrated field calibration. Urban bandwidth/loss 8.5 Mbps/1% differs from proposal 5 Mbps/2%; Rural 1.2 Mbps/7% differs from 1 Mbps/8%. No preserved evidence justifies those differences as empirical calibration. Disconnection state probability is not the same quantity as selected-update drop rate.

[Reproduction guide](REPRODUCIBILITY.md): exact runner coverage and complete 110-run commands. [AI disclosure](AI_USAGE_LOG.md): historical attributions and Codex/student responsibilities. Current [test inventory](REPRODUCIBILITY.md#test-inventory--appendix-b-source-material): 148 cases (the original 140 plus eight release guards). Historical Task 6 recorded 102 targeted passes; Task 9A recorded 140 local passes and three publication-checkout failures. Current Task 9A.1 results are separately recorded in [release verification](RELEASE_VERIFICATION.md).
