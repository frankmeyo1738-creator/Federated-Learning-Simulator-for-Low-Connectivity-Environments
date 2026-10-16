# Proposed material for report Appendices A, B and C

Prepared by OpenAI Codex on 16 October 2026 at Frank Meyo's request. This is repository draft material for student review, not an edited thesis or a claim that the student personally executed Codex's checks. Historical attributions and the student's October 16 confirmations are preserved in [AI_USAGE_LOG.md](AI_USAGE_LOG.md).

## Appendix A — disclosure wording

> I used Claude/Antigravity, GitHub Copilot and ChatGPT during the historical periods recorded in my AI usage log. I confirm those original attributions and periods. For the October 10 latency diagnostic, I recall using Google Antigravity with AI assistance, but cannot identify the underlying model as Gemini or Claude. I do not claim that I personally wrote or independently validated that diagnostic.
>
> Supervisor review identified that it generated NumPy observations and shifted them toward preset report means, rather than observing the simulator. Its classification as raw engine evidence is withdrawn. The original generator and outputs remain preserved with hashes as superseded evidence.
>
> During the October 15–16 correction sessions, I directed the work and reviewed outputs with additional ChatGPT assistance. OpenAI Codex implemented Tasks 1–4 and their automated checks: reproducible statistical summaries, genuine engine observations, selection-aware dropout theory and separate historical bandwidth-delay derivatives. Task 5 was a read-only timing and hardware audit. Codex also drafted the Task 6 repository documentation and this proposed appendix material. I retain responsibility for reviewing it and explaining the implementation in my oral defence. I do not present those automated checks as my own independent execution or validation.

The log separates original September statistical verification from current results: Rural IID W=26.0, p=0.8782 appears in the older summary at 42e71e2; regenerated current data give W=17.5, p=0.3074. The former “1,000 bootstrap resamples” description was a documentation error: that commit already specified 2,000. The current procedure uses 2,000 paired resamples and no Wilcoxon continuity correction.

Do not repeat the former blanket “no AI-generated data” guarantee without the diagnostic disclosure. No original ML cohort observations were altered in the Codex corrections. Synthetic fixtures, theoretical probabilities, fresh engine samples and retrospective derivatives have different evidential status.

The historical seeding correction concerns model weights drawn before Torch was seeded; partitioning already used a seeded NumPy generator. Do not claim loader.py was rewritten to fix that issue. The historical log's local “38 tests” verification event remains undated; 38 is the October suite allocation, not proof of an early-September run.

## Appendix B — test allocation and verification wording

Current inventory: **140 cases**, comprising **127 unit** and **13 integration**.

| Module | Cases |
|---|---:|
| Unit: aggregation | 5 |
| Unit: config | 5 |
| Unit: network | 7 |
| Unit: partitioning | 8 |
| Unit: statistical analysis | 33 |
| Unit: genuine Layer 3 diagnostic | 22 |
| Unit: dropout theory | 30 |
| Unit: historical delay audit | 17 |
| Integration: FedProx | 4 |
| Integration: bandwidth units | 4 |
| Integration: dropout persistence | 3 |
| Integration: deterministic seeding | 2 |

> The earlier v1.0.2 suite had 37 tests (25 unit, 12 integration). The October 10 v1.0.8 suite had 38 (25 unit, 13 integration), adding a bandwidth regression test. Tasks 1–4 added 102 analysis/diagnostic cases, all passing in the recorded Codex checks. The current inventory is 140; full current-suite execution is not claimed. Task 6 full collection obtained 122 cases with four import errors in the limited audit environment; the other 18 are counted directly from the unparameterized source modules. No package or dataset download was performed to resolve that limitation.

The eight partition tests use a fixture that can download MNIST; some integration tests train tiny synthetic models. Collection, analysis-only checks, historic supervisor pass counts and full execution are distinct claims. See [the full inventory and commands](REPRODUCIBILITY.md#test-inventory--appendix-b-source-material) and [verification record](DOCUMENTATION_VERIFICATION.md).

## Appendix C — scripts, data and reproduction wording

> The frozen MNIST evidence consists of 110 runs: six IID and five non-IID configurations, each at seeds 1–10 and 20 rounds. The matrix is asymmetric. Complete regeneration requires the explicit 60-IID/50-non-IID CLI procedure documented in docs/REPRODUCIBILITY.md, or an equivalent controlled manifest in a separate working-tree copy.
>
> The actual scripts are run_multiseed_experiments.sh (34 IID runs), run_noniid_experiments.sh (five legacy seed-42 runs), run_noniid_baseline.sh (ten matched Baseline runs), and colab_runner.py (mode-dependent). There is no scripts/run_multiseed.py. Colab currently defaults to noniid_baseline; all/force_all cover the original 100, excluding the added ten. Existing-file skipping is not regeneration, and no single current mode covers all 110.
>
> Analysis uses scripts/statistical_analysis.py, scripts/verify_layer3_fidelity.py and scripts/audit_historical_delays.py. These respectively generate the statistical summary, genuine engine diagnostics plus finite-horizon theory comparison, and separate historical delay derivatives. They do not retrain the cohort. Historical CSVs and superseded evidence remain preserved.

The exact future training commands and environment setup are in [REPRODUCIBILITY.md](REPRODUCIBILITY.md#complete-110-run-regeneration--training-not-executed-in-task-6). Do not describe them as executed during this documentation audit.

The corrected working tree has **no new release tag or commit**. Existing v1.0.8/b7164d6 does not contain Tasks 1–6. Replace release wording only when an actual authorized release exists; do not invent a tag.

| Evidence type | Location |
|---|---|
| Original observations | experiments/results/multiseed/ and its noniid/ subdirectory |
| Generated statistics | experiments/results/statistical_summary.txt |
| Fresh engine samples | experiments/results/diagnostics/engine_propagation_latency_draws.csv |
| Empirical drop audit and theory | diagnostics/cohort_dropout_audit.csv, dropout_theory_by_round.csv, dropout_theory_distribution.csv, DROPOUT_THEORY.md |
| Historical bandwidth derivatives | diagnostics/historical-delay-audit/ |
| Invalid diagnostic preserved | diagnostics/superseded-evidence/pre-correction-16-v1.0.8/ |

Paths beginning diagnostics/ in this table are relative to experiments/results/. Provenance JSONs record source/input/output hashes and the analysis environment; they do not establish historical training hardware.

## Items to settle before thesis incorporation

- Exact date/environment of the historical local test execution, if it is to be dated; otherwise retain the explicitly undated account.
- CIFAR figures: current saved final accuracies are 73.42%, 72.94%, 69.64% and 70.52%. Earlier repository prose and the thesis give different values. Confirm which run/version those earlier figures describe; do not substitute an invented history. The current single-seed +0.88 pp Severe difference remains exploratory.
- Historical hardware mapping is incomplete. Qualify attribution rather than inventing a machine/thermal explanation. The accepted +0.73 s overhead withdrawal stays in force.
- The student must review/adapt proposed thesis wording before use. No student independent rerun of Tasks 1–5 or personal review of this newly drafted Task 6 material is claimed here.


## Dated release-integrity addendum — Task 9A.1

The preceding Task 6 appendix draft is historical preparation material. Task 8's official template report organises reproduction instructions in Appendix A, tests in B, code/configuration in C and AI disclosure in D. The source inventory is now 148 tests (135 unit + 13 integration), including the eight new release guards. See RELEASE_VERIFICATION.md for current execution results and the preserved distinction from the earlier 102 passes. Task 9A.1 does not edit the thesis or claim a published release.
