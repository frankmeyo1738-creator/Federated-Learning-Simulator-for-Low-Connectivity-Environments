# AI Tool Usage Log

**Project:** Federated Learning Network Simulator for Low-Connectivity Environments  
**Author:** Frank Meyo (Computer Science Undergraduate, University of Zambia)  
**Course:** CSC 4004 Final Year Project — UNZA 2026  
**Supervisor:** Mr. Mofya Phiri  

This document provides a transparent, accountable record of AI tool usage throughout this project, complying with the project proposal Section 3.3 AI-tool commitments and UNZA academic integrity standards.

---

## 1. Summary of Tools & Nature of Use

| Tool | Primary Purpose | Project Phase | Record Status |
|:---|:---|:---|:---|
| **Claude (Anthropic) / Antigravity AI Assistant** | Architecture review, debugging assistance, statistical audit, code remediation, refactoring, Colab pipeline generation | August – September 2026 | **Verified** (Auditable from conversation transcripts) |
| **GitHub Copilot** | Inline code autocompletion during initial implementation | July – August 2026 | **Reconstructed** (Continuous IDE usage; specific dates untracked) |
| **ChatGPT (OpenAI)** | Initial brainstorm drafting and literature outline structuring | Early Proposal Phase | **Reconstructed** (Interactive query sessions) |

---

## 2. Chronological AI Interaction & Remediation Log

### Period A: Proposal & Initial Development (Early August 2026)
* **Tasks Assisted:** Reviewing proposal gap analysis, structuring thesis chapter outlines, and formatting preliminary slide deck outlines.
* **Affected Areas:** Proposal documentation (`docs/`), presentation decks.
* **My Personal Verification:** I evaluated all suggested structures against Mr. Phiri's specific supervisor guidelines. All proposal decisions regarding Zambian cellular context (GSMA 2023 reports, ZICTA quality of service standards) were independently chosen and verified by me.

### Period B: Code Review, Seeding Fix & Archive Remediation (Early September 2026)
* **Tasks Assisted:** Forensic review of multi-seed reproducibility, diagnosis of PRNG re-initialization between seeds, circular import diagnosis between `src.core` and `src.algorithms`.
* **Affected Files & Commits:**
  * [`src/main.py`](file:///src/main.py) & [`src/core/server.py`](file:///src/core/server.py) (Commit `7e7b8cf`): Remedied PRNG state propagation so `torch.manual_seed()` and `random.seed()` consistently seed both client partition assignment and model weight initialization.
  * `experiments/results/multiseed/pre-seeding-fix-archive/`: Identified that seeds 1–5 for `baseline_fedavg` and `urban_zambia_fedavg` predated the fix. Archived old CSVs and prepared clean rerun manifest.
  * [`src/core/__init__.py`](file:///src/core/__init__.py) & [`src/algorithms/__init__.py`](file:///src/algorithms/__init__.py) (Commit `42e71e2`): Replaced eager module imports with lazy attribute lookups (`__getattr__`) to completely eliminate the import cycle without relying on test-runner workarounds in `conftest.py`.
* **My Personal Verification:**
  * I ran the full test suite locally (`pytest tests/unit/ -v`) confirming all 24 unit tests pass cleanly.
  * I verified that running identical seeds produces deterministic, byte-for-byte identical client drops and weight trajectories.

### Period C: $n=10$ Cohort Scaling & Colab GPU Pipeline (Mid-September 2026)
* **Tasks Assisted:** Constructing a resume-safe Google Colab execution pipeline with automated Google Drive checkpointing, non-crashing Git push error handlers, and cuDNN determinism flags.
* **Affected Files & Commits:**
  * [`scripts/colab_runner.py`](file:///scripts/colab_runner.py) (Commits `42e71e2`, `f5b271e`): Created full 76-experiment runner for Colab T4 GPU.
  * Output CSVs (Commit `47c59e7`): Completed all 76 runs (60 IID + 40 Non-IID Dirichlet $\alpha=0.5$).
* **My Personal Verification:**
  * When the initial Colab run was interrupted by quota limits at 50/76, I personally inspected the saved Google Drive CSV files.
  * I confirmed that the resume logic successfully picked up from run 51 without repeating finished runs.
  * I verified every log file to confirm that `iid=True` and `iid=False` matched the experiment design.

### Period D: Rigorous Statistical Analysis Upgrade (Mid-September 2026)
* **Tasks Assisted:** Updating `scripts/statistical_analysis.py` to satisfy supervisor statistical review items:
  1. Setting explicit asymptotic mode (`mode='approx'`) and rank tie-splitting (`zero_method='zsplit'`) in `scipy.stats.wilcoxon` to pin reproducible p-values across SciPy versions (SciPy 1.13 on Mac vs 1.16 on Colab).
  2. Calculating paired Cohen's $d_z = \frac{\bar{x}_\text{diff}}{s_\text{diff}}$ with $95\%$ bootstrap confidence intervals (1,000 resamples).
  3. Computing exact statistical power ($1-\beta$) and Minimum Detectable Effect (MDE at $\alpha=0.05, 80\%$ power) using non-central $t$-distributions (`scipy.stats.nct`).
  4. Enforcing sample standard deviation ($ddof=1$) with Bessel's correction.
  5. Enforcing genuine binary mebibytes ($1024^2 = 1,048,576$ bytes per MiB) and percentage point ($pp$) reporting.
  6. Calculating discriminating convergence thresholds ($R_{90}$ and $R_{95}$ rounds).
* **Affected Files & Commits:**
  * [`scripts/statistical_analysis.py`](file:///scripts/statistical_analysis.py) (Commit `42e71e2`).
  * [`experiments/results/statistical_summary.txt`](file:///experiments/results/statistical_summary.txt) (Commit `42e71e2`).
* **My Personal Verification:**
  * I personally executed the script both on my local MacBook Air (`scipy==1.13.1`, Python 3.9) and on Google Colab (`scipy==1.16.3`, Python 3.10/3.13).
  * I verified that all calculated metrics and test statistics matched bit-for-bit across both platforms ($W=26.0, p=0.8782$ for Rural IID; $W=16.0, p=0.2377$ for Rural Non-IID).

---

## 3. Personal Verification & Validation Principles

To ensure complete academic integrity, the following rules were strictly maintained:

1. **No Hallucinated or AI-Generated Data:** No experimental results, accuracy metrics, drop rates, or p-values were generated by AI. All empirical data originated from executed simulator runs recorded in CSV files and processed through verified Python scripts.
2. **Deterministic Code Auditing:** Every code suggestion was tested against the unit test suite before integration.
3. **Independent Statistical Cross-Checking:** All mathematical formulas implemented in Python (Bessel's correction, Cohen's $d_z$, bootstrap CI, power via non-central $t$) were cross-referenced against established statistical literature and verified with manual calculations.
4. **Dissertation Authorship:** All narrative arguments, interpretations of Zambian telecommunications constraints, and dissertation chapters represent my own synthesis and voice.

---

*Frank Meyo — September 2026*
