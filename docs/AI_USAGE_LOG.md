# AI Tool Usage Log

**Project:** Federated Learning Network Simulator for Low-Connectivity Environments  
**Author:** Frank Meyo (Computer Science Undergraduate, University of Zambia)  
**Course:** CSC 4004 Final Year Project — UNZA 2026  
**Supervisor:** Mr. Mofya Phiri  

This document provides a transparent, accountable record of AI tool usage throughout this project, complying with the project proposal Section 3.3 AI-tool commitments and UNZA academic integrity standards.

**Amendment, 16 October 2026:** I confirm the original tool attributions and historical usage periods below. I retain the earlier personal-verification statements as historical statements, not evidence that I have personally repeated the October Codex checks. Corrections are identified explicitly. In this session I confirmed from direct recollection that Google Antigravity assisted with the October 10 diagnostic; I cannot identify its underlying model as Gemini or Claude. I directed Tasks 1–5 and reviewed their outputs with additional ChatGPT assistance. Codex performed Tasks 1–4 implementation/automated checks and the read-only Task 5 audit; my output review is not independent execution or mathematical validation.

---

## 1. Summary of Tools & Nature of Use

| Tool | Primary Purpose | Project Phase | Record Status |
|:---|:---|:---|:---|
| **Claude (Anthropic) / Antigravity AI Assistant** | Architecture review, debugging assistance, statistical audit, code remediation, refactoring, Colab pipeline generation | August – September 2026 | **Verified** (Auditable from conversation transcripts) |
| **GitHub Copilot** | Inline code autocompletion during initial implementation | April – June 2026 | **Reconstructed** (Continuous IDE usage; specific dates untracked) |
| **ChatGPT (OpenAI)** | Initial brainstorm drafting and literature outline structuring | Early Proposal Phase | **Reconstructed** (Interactive query sessions) |
| **Google Antigravity; underlying model unspecified** | AI-assisted October 10 calibrated NumPy latency diagnostic | 10 October 2026 | Student-confirmed direct recollection, recorded 16 October; no specific Gemini/Claude attribution |
| **OpenAI Codex** | Statistical-summary repair, genuine engine diagnostics, dropout derivation, historical delay and timing audits, documentation drafts | 15–16 October 2026 session sequence | Recorded in this Codex conversation and generated provenance; not yet committed |
| **ChatGPT (OpenAI), additional October assistance** | Assistance while I directed the tasks and reviewed their outputs | October 2026 correction sessions | Student-confirmed in this session; specific model and individual interaction dates not supplied |

---

## 2. Chronological AI Interaction & Remediation Log

### Period A: Proposal & Initial Development (April – June 2026)
* **Tasks Assisted:** Reviewing proposal gap analysis, structuring thesis chapter outlines, and formatting preliminary slide deck outlines.
* **Affected Areas:** Proposal documentation (`docs/`), presentation decks.
* **My Personal Verification:** I evaluated all suggested structures against Mr. Phiri's specific supervisor guidelines. All proposal decisions regarding Zambian cellular context (GSMA 2023 reports, ZICTA quality of service standards) were independently chosen and verified by me.

### Period B: Code Review, Seeding Fix & Archive Remediation (Early September 2026)
* **Tasks Assisted:** Forensic review of multi-seed reproducibility, diagnosis of unseeded model initialisation, circular import diagnosis between `src.core` and `src.algorithms`.
* **Affected Files & Commits:**
  * [`src/main.py`](../src/main.py) & [`src/core/server.py`](../src/core/server.py) (Commit `7e7b8cf`): Remedied PRNG state propagation. Prior to commit `7e7b8cf`, `torch.manual_seed()` was invoked inside `FLServer.__init__`, which executed after `get_model()` had already drawn initial network weights in `main.py` (leaving initial weights unseeded, while data partitioning in `src/data/loader.py` was already properly seeded with `np.random.default_rng(seed)`). Lifted seeding to the start of `main.py` before `get_model()` so initial weight generation is strictly deterministic.
  * `experiments/results/multiseed/pre-seeding-fix-archive/`: Identified that seeds 1–5 for `baseline_fedavg` and `urban_zambia_fedavg` predated the fix. Archived old CSVs and prepared clean rerun manifest.
  * [`src/core/__init__.py`](../src/core/__init__.py) & [`src/algorithms/__init__.py`](../src/algorithms/__init__.py) (Commit `42e71e2`): Replaced eager module imports with lazy attribute lookups (`__getattr__`) to completely eliminate the import cycle without relying on test-runner workarounds in `conftest.py`.
* **My Personal Verification:**
  * My earlier log records that I ran the test suite locally (`pytest -v`). Its allocation of 38 tests belongs to the October 10 suite, not the early-September date of this section: the September release had 37 tests (25 unit + 12 integration); October's added bandwidth test made 38 (25 + 13). I do not assign an exact date to my earlier local test execution from that statement alone.
  * I verified that running identical seeds produces deterministic, byte-for-byte identical client drops and weight trajectories.

### Period C: $n=10$ Cohort Scaling & Colab GPU Pipeline (Mid-September 2026)
* **Tasks Assisted:** Constructing a resume-safe Google Colab execution pipeline with automated Google Drive checkpointing, non-crashing Git push error handlers, and cuDNN determinism flags.
* **Affected Files & Commits:**
  * [`scripts/colab_runner.py`](../scripts/colab_runner.py) (Commits `42e71e2`, `f5b271e`): Created the Colab T4 GPU execution pipeline; the runner was initially scoped to 76 experiments during construction, with the final cohort reaching **100 runs** (60 IID + 40 non-IID, n=10 seeds across all configurations).
  * Output CSVs: my earlier log credited `47c59e7` with completing all 100 runs. Its actual message is “all 76 runs complete (final)”, describing that regeneration batch. Commit `0c196df` (19 September) subsequently regenerated the remaining 20 Rural/Severe IID runs, seeds 1–5. The resulting original cohort contains **100 runs**: 60 IID and 40 impaired non-IID. Ten matched non-IID Baseline runs were added on October 10 in `1e92824`; they were not part of the September batch.
* **My Personal Verification:**
  * When the initial Colab run was interrupted by quota limits at run 50 of the original 76-experiment scope, I personally inspected the saved Google Drive CSV files.
  * I confirmed that the resume logic successfully picked up from run 51 without repeating finished runs.
  * I verified every log file to confirm that `iid=True` and `iid=False` matched the experiment design.

### Period D: Rigorous Statistical Analysis Upgrade (Mid-September 2026)
* **Tasks Assisted:** Updating `scripts/statistical_analysis.py` to satisfy supervisor statistical review items:
  1. Setting explicit asymptotic mode (`mode='approx'`) and rank tie-splitting (`zero_method='zsplit'`) in `scipy.stats.wilcoxon` to pin reproducible p-values across SciPy versions (SciPy 1.13 on Mac vs 1.16 on Colab).
  2. Calculating paired Cohen's $d_z = \frac{\bar{x}_\text{diff}}{s_\text{diff}}$ with $95\%$ bootstrap confidence intervals. **Correction:** the previous log said 1,000 resamples; `42e71e2` already specifies `n_boot=2000`, seed 42. The verified current procedure uses **2,000 paired resamples**. The earlier 1,000 figure was a documentation error, not an established earlier execution setting.
  3. Computing observed-effect power and Minimum Detectable Effect using non-central $t$-distributions (`scipy.stats.nct`). **Qualification:** “exact statistical power” was an overstatement for the Wilcoxon analysis. The observed-effect power output was removed in `95d7916` on October 10; current reporting retains MDE ≈1.00 at 80% power as a paired-t approximation, not exact Wilcoxon power.
  4. Enforcing sample standard deviation ($ddof=1$) with Bessel's correction.
  5. Enforcing genuine binary mebibytes ($1024^2 = 1,048,576$ bytes per MiB) and percentage point ($pp$) reporting.
  6. Calculating discriminating convergence thresholds ($R_{90}$ and $R_{95}$ rounds).
* **Affected Files & Commits:**
  * [`scripts/statistical_analysis.py`](../scripts/statistical_analysis.py) (Commit `42e71e2`).
  * [`experiments/results/statistical_summary.txt`](../experiments/results/statistical_summary.txt) (Commit `42e71e2`).
* **My Personal Verification:**
  * I personally executed the script both on my local MacBook Air (`scipy==1.13.1`, Python 3.9) and on Google Colab (`scipy==1.16.3`, Python 3.10/3.13).
  * I verified that all calculated metrics and test statistics matched bit-for-bit across both platforms ($W=26.0, p=0.8782$ for Rural IID; $W=16.0, p=0.2377$ for Rural Non-IID).
  * **Historical/current distinction:** the Rural IID $W=26.0, p=0.8782$ pair is present in the summary committed at `42e71e2`; I retain my earlier cross-platform verification account as historical. It is superseded for the regenerated current cohort by **$W=17.5, p=0.3074$**. Rural non-IID remains $W=16.0, p=0.2377$. This statistical-script comparison does not establish bit-identical training across CPU and GPU. The October 13 supervisor feedback reports a CPU trajectory following T4 non-IID Baseline seed 1 without bit identity.

### Period E: October 10 remediation and invalid latency evidence

* **Verifiable source history:** `95d7916` (10 October, 11:21 +02:00) introduced the latency diagnostic and outputs, corrected production decimal-Mbps conversion with an independent regression example, and removed observed-power reporting. `7aa7356` (13:16 +02:00) withdrew the untraceable +0.73 s/round figure but substituted an unsupported “compute noise floor” interpretation. **Tool attribution:** I confirm from my direct recollection that this diagnostic was produced with AI assistance through **Google Antigravity**. I cannot confirm whether Gemini or Claude was responsible, so its underlying model is unspecified. This confirmation comes from my recollection, not the commit author header; it does not imply I personally wrote or independently validated that diagnostic.
* **What the diagnostic actually did:** it drew 2,500 NumPy normal values per profile, clipped them, calculated `shift = target_mean - mean(clamped)`, added that shift, and clipped again. Targets included 140.2, 550.8 and 1399.4 ms. These were constructed observations adjusted toward report figures, not observations returned by `NetworkImpairmentEngine._apply_latency()`. The old CSV's `raw_gaussian_ms` column contained synthetic NumPy values; the adjusted column was also not engine data. The second clipping means the final mean need not exactly equal its target.
* **Discovery:** Mr. Phiri explicitly identified the construction in the supplied October 13 feedback. In the later Codex session, the source and archived CSV independently confirmed the mechanism. I do not describe this as my own independent discovery or verification.
* **Evidence preserved:** the old generator, CSV and report are retained byte for byte with hashes under `experiments/results/diagnostics/superseded-evidence/pre-correction-16-v1.0.8/`. They are invalid for engine-observation claims and must not be cited as raw simulator measurements. Original ML cohort CSVs are separate evidence.
* **Withdrawal:** I retain the withdrawal of +0.73 s/round accepted in the October 13 follow-up. Similarity of profile timing means does not establish zero or undetectable engine overhead.

### Period F: OpenAI Codex evidence and reproducibility tasks (15–16 October)

I requested these tasks and set the constraints. Codex wrote the analysis changes, mathematical explanations, tests, generated evidence and documentation listed below. “Verified” in this table means an automated or source/data check executed by Codex; it does not mean that I personally executed or independently checked it. **I confirm that I directed Tasks 1–5 and reviewed their outputs with additional ChatGPT assistance.** I do not claim to have personally run the automated checks or independently validated the derivations.

| Task | Codex work and affected files | Checks recorded in the session |
|---|---|---|
| 1 — statistical summary | `scripts/statistical_analysis.py`, `experiments/results/statistical_summary.txt`, `tests/unit/test_statistical_analysis.py` | 33 tests; matched Baseline and descriptive R95 decomposition generated from 110 existing runs; preserved current paired results and input observations. |
| 2 — genuine Layer 3 observations | `scripts/verify_layer3_fidelity.py`, `tests/unit/test_layer3_fidelity.py`; diagnostic README, raw engine observations, cohort-drop audit, report and provenance; old evidence archived | 22 tests. Actual `_apply_latency()` calls, seeds 100–103, 2,500/profile; only sleep suppressed. Means 2.979889, 140.285986, 549.977460 and 1401.944613 ms; no observed bounds violations. No calibration or training. |
| 3 — selection-aware dropout | Same diagnostic script/report/provenance/README; `DROPOUT_THEORY.md`, `dropout_theory_by_round.csv`, `dropout_theory_distribution.csv`; `tests/unit/test_dropout_theory.py` | 30 additional tests; 85 combined. Exact rational mean recurrence and population probability enumeration for 10 clients, 5 selected, 20 rounds. Rural prediction 26.208599% versus 30.20% observed remains a pointwise discrepancy; no blanket agreement claim. |
| 4 — historical delay units | `scripts/audit_historical_delays.py`, `tests/unit/test_historical_delay_audit.py`; five separate outputs under `experiments/results/diagnostics/historical-delay-audit/` | 17 additional tests; 102 combined. 100 binary-convention runs and 10 decimal-convention Baseline non-IID runs supported by history and numerical checks. All 2,200 rows independently checked; bandwidth components corrected separately; original observations unchanged. |
| 5 — timing/hardware | Read-only source, history, PDF and CSV review; findings recorded in this conversation, **no files changed** | Independent Decimal/Fraction timing sums; all 110 run averages below 5 rounds/min; pooled throughput 2.610 rounds/min. No complete per-run hardware map or isolated overhead measurement; no new benchmark. |
| 6 — documentation alignment | `README.md`, this log, `docs/RESULTS.md`, `docs/REPRODUCIBILITY.md`, `docs/APPENDIX_MATERIAL.md`, `docs/DOCUMENTATION_VERIFICATION.md`; targeted qualification of `docs/COMPONENT_SPEC.md` and `docs/PROJECT_IMPACT.md` | Source/manifest and command checks described in the verification record. This is AI-assisted repository prose and proposed appendix material, not a claim of my completed personal verification or a thesis edit. |

* **Dates and release status:** Task 1 belongs to the 15–16 October session sequence; no precise execution timestamp is asserted. The combined Tasks 2–3 provenance records `2026-10-15T23:49:13.405308+00:00` (16 October, 01:49 Lusaka). Task 4 provenance records `2026-10-16T00:16:51.512612+00:00` (02:16 Lusaka). Tasks 5–6 follow in the October 16 conversation. These are recorded generation/session dates, not historical training dates.
* **Commits:** Tasks 1–6 are uncommitted working-tree work based on `b7164d6` / existing tag `v1.0.8`. That tag does **not** contain these corrections. I have not authorized a new commit, push or tag in this session.
* **Environment boundary:** current analysis provenance records Python 3.12.14 on macOS arm64 with package versions. That records the Codex analysis runtime, not historical ML training hardware. See `docs/REPRODUCIBILITY.md` for dependency and test limitations.

---

## 3. Personal Verification & Validation Principles

My earlier log presented the following as blanket guarantees. The October diagnostic requires an explicit qualification; these principles are responsibilities, not retrospective proof that every earlier check occurred:

1. **Observations versus construction:** original training observations, fresh engine samples, synthetic test fixtures, theoretical probabilities and algebraically derived delay corrections are labelled separately. My previous “no AI-generated data” wording cannot cover the constructed October 10 diagnostic. Its raw-engine interpretation is withdrawn.
2. **Verification attribution:** I preserve historical personal checks recorded above, but do not claim that I personally ran Codex's tests or checked its derivations. Automated pass counts are scoped and do not establish all tests passed on every environment.
3. **Scientific claims:** nonsignificance is not equivalence; a predictive interval is not a confidence interval for observed means; source control and package pins are not execution manifests. Unsupported thermal and isolated-overhead explanations remain withdrawn.
4. **Authorship and accountability:** I retain responsibility for the final submission and my oral explanation. Codex supplied the later repository prose and proposed appendix wording; I must review and adapt it before incorporating it into my thesis. No thesis text has been changed in these tasks.

## 4. Confirmation record and remaining historical ambiguity

1. **Confirmed in the October 16 conversation:** Google Antigravity assisted with the October 10 diagnostic; its underlying model remains unspecified. The issue was identified through supervisor review and corrected in the October 16 Codex sessions.
2. **Confirmed in that same conversation:** I directed Tasks 1–5 and reviewed outputs with additional ChatGPT assistance; Codex performed Tasks 1–4 implementation and automated verification and the read-only Task 5 audit. No independent student rerun is asserted.
3. **Still undated:** the exact date/environment of my historical local suite execution in Period B is not established by its earlier “38 tests” sentence. A personal date/environment confirmation would settle this detail; absent that, it remains expressly undated. Historical tool attributions and periods remain as I confirmed.

The full previous log remains retrievable from Git at `b7164d6:docs/AI_USAGE_LOG.md`; its superseded wording is explained here rather than silently presented as current evidence.

---

*Frank Meyo — historical September 2026 record, transparently amended with Codex assistance and the personal confirmations recorded on 16 October 2026.*

## 5. Subsequent Codex report and release-integrity work (16 October 2026)

This dated addendum preserves the historical attributions, chronology and personal-verification statements above. It does not transfer those events between AI tools.

- **Task 7:** Codex performed the read-only supervisor-correction and CIFAR-10 provenance audit. Traceable current figures were distinguished from unsupported historical prose and single-seed inference.
- **Report revision and Task 8:** after my authorisation, Codex prepared a separate Revised8 DOCX and migrated its supported scientific content into the official eight-chapter UNZA template, rebuilding navigation and rendering the document for review. The thesis files remain separate from the source release. No signature, approval or release identifier was invented. The document changes were AI-assisted; I do not claim that I independently performed Codex's page or package checks.
- **Task 9A:** Codex performed a release preflight. All original 140 tests passed in an existing local environment; the publication simulation had 137 passes and three failures. Git's text normalization broke archived evidence hashes, and one test assumed the original working tree's line-ending distribution. No commit, tag, push or release occurred.
- **Task 9A.1:** I authorised the narrow serialization/provenance repair. Codex preserved the exact pre-repair records, added path-specific Git attributes, made the historical input test portable, added eight evidence guard tests and a standalone release verifier, and reconciled repository documentation. The verifier distinguishes original working-tree bytes, historical committed serialization, immutable observations and regenerated analysis metadata. Actual tests and numerical precision differences are recorded in docs/RELEASE_VERIFICATION.md and the generated checkout verification JSON. No original cohort or production semantics are changed, and no ML cohort is retrained.

These are Codex implementation and automated-check activities under my direction. This addendum does not assert that I personally ran, inspected or independently verified these later checks; no additional personal confirmation is inferred. The Google Antigravity attribution and unspecified underlying model for the October 10 diagnostic remain unchanged. Publication remains subject to my separate authorisation.
