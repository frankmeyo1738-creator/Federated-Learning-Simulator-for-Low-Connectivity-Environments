# Reproduction guide and Appendix C source material

Updated 16 October 2026. This describes the corrected working tree; Tasks 1–6 are not committed or tagged. Existing tag `v1.0.8` at `b7164d6` predates those tasks. Checking out that tag alone does not reproduce the corrected diagnostics or documentation.

## Evidence review versus experiment regeneration

Analysis reads the preserved observations; regeneration executes ML and produces new observations. No training, dataset download, benchmark, or production runner was executed for this documentation task. Commands below labelled **training** are future reproduction instructions, not commands executed in this audit. Use a separate copy of the corrected working tree for regeneration: the CLI writes the configured result filenames and does not preserve an existing CSV automatically.

Hardware and package pins are not historical execution manifests. Production selects CUDA if available, otherwise CPU; it does not select Apple MPS. Same-device deterministic checks do not establish CPU/GPU bit identity, and wall-clock durations are not deterministic outputs.

## Environment

Run from the repository root. For a new reproduction environment, use Python 3.11 or later compatible with the pinned wheels:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

Installation requires package access; it was not executed during Task 6. `requirements.txt` pins Torch 2.8.0, torchvision 0.23.0, SciPy 1.16.3, NumPy 2.0.2, pandas 2.3.3, PyYAML 6.0.3, matplotlib, seaborn, pytest and pytest-asyncio. Actual CUDA support depends on the installed build and available device.

`requirements_colab.txt` is a different historical file: SciPy 1.13.1, with Torch/torchvision omitted. It is not an equivalent lockfile for the current environment. The notebook installs unpinned dependencies and has no retained outputs. `colab_runner.py` checks Torch/CUDA but does not install requirements. Record the actual environment for any future execution; neither requirements file establishes the versions used for each saved historical run.

## Actual runner coverage

| Runner | Actual coverage | Partition/output behaviour |
|---|---|---|
| `scripts/run_multiseed_experiments.sh` | 34 IID runs: Baseline/Urban FedAvg seeds 1–5 (10 runs); Rural/Severe × FedAvg/FedProx seeds 1–6 (24 runs) | Always passes `--iid`; activates `.venv`; writes multiseed CSVs. Does not cover all 60 IID runs. |
| `scripts/run_noniid_experiments.sh` | Five seed-42 configs: Baseline FedAvg, Rural FedAvg/FedProx, Severe FedAvg/FedProx | No `--iid`; activates `.venv`; writes root `experiments/results/`, not the ten-seed cohort. Header says four, executable list contains five. |
| `scripts/run_noniid_baseline.sh` | Matched Baseline non-IID FedAvg seeds 1–10 | No `--iid`; uses current `python3`, does not activate a venv; writes `multiseed/noniid/`. Ten runs, not the full matrix. |
| `scripts/colab_runner.py` | Mode-dependent; see below | Uses child `python`; hard-coded Colab paths, Drive persistence and Git operations. Existing files are generally skipped. |
| `scripts/run_all_experiments.sh` | Six seed-42 IID configs | Legacy exploratory runner; not the 110-run cohort. |

These descriptions come from executable loops, flags, YAML seeds and output paths, not comments or runtime estimates.

### Colab modes

The current source assignment is **`MODE = "noniid_baseline"`**. These are source settings, not CLI flags or environment variables; `--mode all` is not implemented. Production runner behaviour was not changed.

| Mode | Coverage/behaviour |
|---|---|
| `noniid_baseline` (default) | Ten matched non-IID Baseline runs; skips existing CSVs. On a complete checkout it may execute no training. |
| `all` | Original 100 runs: 60 IID + 40 impaired non-IID, seeds 1–10; skips existing CSVs. Does not include the additional Baseline non-IID cohort. |
| `force_all` | Deletes the original 100 output CSVs in the runner's checkout and regenerates that 100-run matrix. Does not include the additional ten. |
| `stats_only` | No training; regenerates the statistical summary. It still passes through the runner's earlier setup/Drive/Git configuration. Use the statistical script directly for analysis. |
| `missing_20` | Historical repair of Rural/Severe IID, both algorithms, seeds 1–5; removes those CSVs. Not a full-cohort command. |
| `missing_24` | Historical repair of Rural/Severe IID, both algorithms, seeds 1–6 (24 runs); removes those CSVs. It is included in the execution-loop condition. Not a full-cohort command. |

No existing single mode executes all 110 runs. Conceptually `force_all` plus `noniid_baseline` covers 100 + 10, but the latter will skip already existing comparator CSVs. To actually regenerate all 110 without changing runner behaviour, use the explicit CLI loops below.

The Colab runner defaults to Drive mounting and GitHub-push setup. On an existing checkout it invokes Git reset/pull; execution can remove outputs, create commits and push if enabled. It is not a read-only verification command, and neither importing it nor passing `--help` is a safe audit method. Its hard-coded repository location is under `/content/`.

## Complete 110-run regeneration — TRAINING, not executed in Task 6

In a separate working-tree copy, after environment setup, these commands execute exactly the 11 released MNIST configurations × seeds 1–10. They work locally or in a prepared Colab shell; CUDA selection is automatic. Initial runs may download MNIST. They overwrite results in that copy, so retain the original evidence separately.

```bash
set -e
for experiment in baseline_fedavg urban_zambia_fedavg rural_zambia_fedavg severe_disruption_fedavg rural_zambia_fedprox severe_disruption_fedprox; do
    for seed in 1 2 3 4 5 6 7 8 9 10; do
        python3 -m src.main --config "config/experiments/multiseed/${experiment}_seed${seed}.yaml" --iid
    done
done

for experiment in baseline_fedavg_noniid rural_zambia_fedavg_noniid severe_disruption_fedavg_noniid rural_zambia_fedprox_noniid severe_disruption_fedprox_noniid; do
    for seed in 1 2 3 4 5 6 7 8 9 10; do
        python3 -m src.main --config "config/experiments/multiseed/noniid/${experiment}_seed${seed}.yaml"
    done
done
```

Coverage: 60 IID + 50 non-IID = 110 runs, 20 rounds each, 10 total clients, fraction_fit 0.5. Non-IID defaults to Dirichlet alpha 0.5. There is no Baseline/Urban FedProx cohort and no Urban non-IID cohort. This is not a complete two-algorithm × four-profile × two-partition factorial matrix. The five-run legacy non-IID shell script cannot supply the 50-run non-IID cohort.

Regenerated decimal-Mbps latency metrics will differ from the historical 100 binary-divisor CSVs. Treat new runs as new evidence, not a byte-identical replacement of historical timing or total-delay observations. Current cohort statistics can be checked against frozen files without retraining.

## Analysis and diagnostic commands — no ML training

These commands write only their documented summaries/diagnostic outputs. They were previously executed in Tasks 1–4; in Task 6 their algorithms and existing outputs were preserved.

```bash
python3 scripts/statistical_analysis.py
python3 scripts/verify_layer3_fidelity.py
python3 scripts/audit_historical_delays.py
```

- Statistical analysis: 110-run input manifest, 2,000 paired bootstrap resamples (seed 42), asymptotic two-sided Wilcoxon with `zsplit` and `correction=False`, Bonferroni reporting, matched Baseline and descriptive R95 contrasts. Writes `experiments/results/statistical_summary.txt`.
- Layer 3: 2,500 real engine latency calls/profile, seeds 100–103; temporary sleep suppression only; raw returned floats, observed cohort drops and selection-aware finite-horizon theory. Writes the active diagnostic CSVs/report/provenance under `experiments/results/diagnostics/`. Does not execute the superseded generator.
- Historical delay audit: source/Git evidence plus cumulative accounting, Decimal arithmetic and separate derived outputs under `diagnostics/historical-delay-audit/`. Requires the existing Git history; a source copy without Git objects cannot supply its historical-provenance check.

Fresh diagnostic samples are not recovered historical training draws. Derived decimal delays are not raw observations. Original cohort and archived evidence must remain separate. See [results](RESULTS.md) and [diagnostic explanation](../experiments/results/diagnostics/README.md).

## Test inventory — Appendix B source material

After Task 9A.1 the suite contains **148 cases: 135 unit + 13 integration**. This includes all original 140 plus eight release-evidence guard tests. The historical 102 analysis/diagnostic cases are expanded pytest cases, not merely counts of test functions.

| File | Cases | Scope |
|---|---:|---|
| `tests/unit/test_aggregation.py` | 5 | Aggregation |
| `tests/unit/test_config.py` | 5 | Configuration |
| `tests/unit/test_network.py` | 7 | Network methods |
| `tests/unit/test_partitioning.py` | 8 | Partitioning; fixture can download MNIST |
| `tests/unit/test_statistical_analysis.py` | 33 | Analysis only |
| `tests/unit/test_layer3_fidelity.py` | 22 | Engine diagnostics; no training |
| `tests/unit/test_dropout_theory.py` | 30 | Probability model/lifecycle; no training |
| `tests/unit/test_historical_delay_audit.py` | 17 | Historical accounting and portable Git-byte validation; no training |
| `tests/unit/test_release_evidence.py` | 8 | Exact evidence identity, mutation rejection and bounded precision checks; no training |
| `tests/integration/test_fedprox_mu_zero.py` | 4 | Includes synthetic-model training |
| `tests/integration/test_bandwidth_units.py` | 4 | Payload units/analytical delay |
| `tests/integration/test_dropout_persistence.py` | 3 | Dropout lifecycle |
| `tests/integration/test_deterministic_seed.py` | 2 | Includes synthetic-model training |

Historical releases: `v1.0.2` had 37 (25 + 12); October 10 `v1.0.8` had 38 (25 + 13). The earlier 140 inventory was **not a claim of 140 passing executions in Task 6**. In Task 6, full collection obtained 122 cases and four import errors: three integration modules need matplotlib and the partitioning module needs torchvision. Seaborn and pytest-asyncio are also absent from the limited audit environment. The remaining 18 cases are directly counted from those unparameterized modules. No package installation or dataset download was attempted to remove these limitations.

The following four-file command exercises the 102 analysis/diagnostic cases without training or dataset downloads; it writes fixtures and regenerated test outputs only in temporary directories:

```bash
python3 -m pytest -q -p no:cacheprovider tests/unit/test_statistical_analysis.py tests/unit/test_layer3_fidelity.py tests/unit/test_dropout_theory.py tests/unit/test_historical_delay_audit.py
```

Full collection (imports tests, does not execute fixtures) and full execution are distinct:

```bash
python3 -m pytest --collect-only -q -p no:cacheprovider tests
```

Future full-suite execution, **not performed in Task 6**, is `python3 -m pytest -v tests/`. It needs all requirements; it executes small synthetic training tests and the eight partition tests can download MNIST. Do not describe it as a no-training/no-download command.

See [verification record](DOCUMENTATION_VERIFICATION.md) for commands actually checked and [appendix material](APPENDIX_MATERIAL.md) for proposed report text.


## Release serialization and provenance checks

Task 9A.1 adds a fourth analysis-only command after the existing three:

```bash
python3 scripts/statistical_analysis.py
python3 scripts/verify_layer3_fidelity.py
python3 scripts/audit_historical_delays.py
python3 scripts/verify_release_evidence.py --strict-checkout
```

Use a full Git checkout with history; no ML dataset is required for these four commands. They read only released files and Git objects and write generated analysis/verification outputs. Do not run the Colab wrapper for this check. The exact eight evidence CSV attributes preserve their recorded bytes through staging and checkout. The frozen `pre-9a1-working-tree/` snapshot retains older provenance, source versions and fourteen original CRLF input copies; the actual 110 cohort files are not rewritten. New generated metadata records the executing environment and its input serialization. The verifier rejects changed observations, schemas, experiment membership, source/output hashes and archive bytes. It records bounded numerical precision differences without adjusting outputs.

`--strict-checkout` requires the original cohort files to match their historical Git blobs exactly. The original development working tree contains fourteen recorded CRLF variants. For a deliberate check there, omit that flag: only those exact pre-recorded bytes are accepted, with a second independent comparison to historical Git bytes allowing CRLF/LF only. The record explicitly lists every such variant. Regenerate all three analyses in the environment being checked before running the verifier; old executable paths are historical metadata, not dependencies.

The current status and actual environment are in [release verification](RELEASE_VERIFICATION.md). Existing test dependencies, cached MNIST needs and synthetic training behavior remain as described above. Task 9A.1 does not install dependencies or verify a new installation of the Python 3.11+/SciPy 1.16.3 pins.
