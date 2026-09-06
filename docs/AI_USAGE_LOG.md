# AI Tool Usage Log

**Project:** Federated Learning Network Simulator for Low-Connectivity Environments  
**Author:** Frank Meyo (2022067479)  
**Course:** CSC 4004 Final Year Project — UNZA 2026  
**Supervisor:** Mr. Mofya Phiri  

This log documents AI tool usage throughout the project, in compliance with the project proposal Section 3.3 AI-tool commitments and university academic integrity requirements.

---

## Usage Log

### 1. Claude (Anthropic) — Chat Interface

| Period | Task / Use | Project Area | Validation / Verification |
|--------|-----------|-------------|--------------------------|
| Early August 2026 | Proposal gap-analysis and revision guidance | Proposal development | Human review: all suggestions were evaluated against project scope and supervisor feedback before incorporation |
| August 2026 | PPTX slide deck creation assistance | Presentation materials | Human review: slide content was reviewed for accuracy against actual project deliverables |
| August 2026 | Validation documentation draft | Documentation (`docs/`) | Human review: documentation was verified against actual implementation and test results |
| Late August – September 2026 | Capstone milestone-plan structuring | Project planning | Human review: milestones were validated against actual progress and remaining work |
| September 2026 | Drafted supervisor correspondence | Communication | Human review: all correspondence was reviewed and edited before sending |
| September 2026 | Code audit: seeding bug identification, test/commit verification against supervisor's findings | Code quality (`src/main.py`, `src/core/server.py`) | Human review: all identified issues were independently verified in the codebase before any changes were made |
| September 2026 (ongoing) | Forensic code-review remediation: seeding fix, experiment re-runs, statistical analysis update, documentation improvements | Multiple: `src/`, `config/`, `docs/`, `scripts/` | Human review: every code change verified by running the test suite (24/24 pass); every experiment result verified from actual CSV output; no results fabricated |

### 2. GitHub Copilot

| Period | Task / Use | Project Area | Validation / Verification |
|--------|-----------|-------------|--------------------------|
| Throughout development | Code completion suggestions during implementation | Source code (`src/`) | Human review: all suggestions were accepted or rejected on a per-suggestion basis; no code was used without understanding and verifying its correctness |

### 3. ChatGPT (OpenAI)

| Period | Task / Use | Project Area | Validation / Verification |
|--------|-----------|-------------|--------------------------|
| Throughout development | Prose drafting alternatives and thesis section outlines | Writing and documentation | Human review: all AI-generated prose was reviewed, edited, and verified against actual project content before incorporation. Usage consistent with proposal Section 3.3 AI-tool commitment. |

---

## Verification Methodology

All AI-generated or AI-assisted outputs were subject to the following verification process:

1. **Code changes**: Verified by running the full unit test suite (`pytest tests/unit/ -v` — 24 tests) and, where applicable, reproducibility checks demonstrating deterministic behaviour under fixed seeds.

2. **Documentation**: Cross-referenced against actual implementation, configuration files, and experiment result CSVs. No metrics, p-values, or experiment outcomes were AI-generated — all were computed from actual data.

3. **Analysis**: Statistical results were computed using `scripts/statistical_analysis.py` from real experiment output. AI tools were used to review methodology, not to generate results.

4. **Research**: AI-provided explanations of FL concepts were validated against the cited academic literature.

---

## Distinction: Verified vs Reconstructed

- **Verified historical usage**: The Claude chat history from early August through September 2026 provides a verifiable record of the tasks listed above.
- **Reasonable reconstruction**: GitHub Copilot and ChatGPT usage was ongoing throughout development; specific dates are not recoverable, but the nature of usage is accurately described.
- **Ongoing entries**: This log should be updated as development and writing continue through final submission.

---

## Reminder

> **Continue adding entries to this log** as you use AI tools during the remaining development, writing, and submission preparation phases. Each entry should include the date/period, specific task, project area affected, and how the output was validated.

---

*Last updated: September 2026*
