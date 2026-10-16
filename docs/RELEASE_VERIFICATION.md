# Release serialization and verification

Task 9A.1, 16 October 2026. The corrected source is uncommitted and untagged; `v1.1.0` is a proposed version only. No publication is asserted.

## Evidence policy

Root `.gitattributes` retains ordinary `* text=auto` behavior. Only the eight exact CSV paths found in Task 9A have `-text` overrides. The additional frozen pre-repair snapshot directory is byte-preserved because it contains immutable historical records and original CRLF input copies. There is no global Git configuration change and no normalization or restaging of the original cohort CSVs. `/output/` is ignored so thesis documents and local audit files remain separate.

`experiments/results/release-verification/pre-9a1-working-tree/` preserves 28 exact files and a manifest identifying all 110 inputs. These include the older summaries/provenance, affected source versions and fourteen raw input copies whose CRLF bytes differ from historical Git serialization. The older hashes are not replaced. The original October 10 superseded archive and manifest remain unchanged. Snapshot copies are not new experimental runs.

Active generated metadata may be refreshed from an isolated release checkout; it records that checkout's input bytes, current source hashes and actual environment. The frozen earlier metadata remains independently verifiable. Regeneration timestamps, executable paths and Git HEAD describe the generation event; they are not historical training-device attestations or a new release identifier.

## Reproduction from a full checkout

After preparing a compatible environment, run from the repository root:

```bash
python3 scripts/statistical_analysis.py
python3 scripts/verify_layer3_fidelity.py
python3 scripts/audit_historical_delays.py
python3 scripts/verify_release_evidence.py --strict-checkout
```

These commands do not train models or download datasets. Required inputs, immutable archives and relevant Git history are included in the proposed source tree. A source ZIP or a shallow history lacking the historical commits cannot substitute for a full clone. No private Mac directories or temporary Codex files are needed. The scripts resolve project paths from their own location.

The fourth command verifies original experiment membership and bytes against historical Git, all frozen hashes, active source/output hashes and superseded archive bytes. It compares the regenerated scientific results with preserved results. Original observations, counts, identifiers and schemas are exact; per-round and per-run delay derivatives are exact. Cross-version float accumulation may differ within 1e-9 ms in cohort aggregates, 1e-14 in probability-law cells and 1e-10 in JSON float statistics. Every nonzero difference is recorded; no value is adjusted or silently rounded to force agreement. Scientific summary text must match exactly except the actual runtime and input-manifest lines.

The strict option requires committed input serialization. Without it, the verifier accepts only the exact already-recorded working-tree CRLF variants and independently checks their normalized bytes against historical Git. It rejects arbitrary data, column, round or membership changes even if another hash is supplied. This distinction is explicitly recorded in checkout_verification.json.

## Test and environment history

Task 6's record remains unchanged: 102 targeted passes and dependency-limited full collection. Task 9A later found an existing Python 3.9.6 environment with SciPy 1.13.1, NumPy 2.0.2, pandas 2.3.3, Torch 2.8.0, torchvision 0.23.0, matplotlib 3.9.4, pytest 8.4.2 and pytest-asyncio 1.2.0. It obtained 140 local passes but only 137 passes/three failures after Git normalization. Those were real release blockers, not withdrawn test evidence.

Task 9A.1 adds eight guard tests, so the complete suite is 148 (135 unit + 13 integration). The original 140 remain included. Current verification outcomes will be appended after execution; this document does not assume a pass. The available existing environment is used without package changes or new dataset downloads. Full tests may read the existing cached MNIST and train tiny synthetic fixtures; the four analysis commands need neither. Installation of the separately documented Python 3.11+/SciPy 1.16.3 pins has not been newly tested in this task.

## Limits and approval gate

CIFAR-10 remains exploratory and historical hardware attribution incomplete. Rural's dropout discrepancy remains. No code or parameter is tuned to improve apparent theoretical agreement. The corrected release must still be checked from its actual committed/tagged tree and then independently from GitHub after explicit publication authorisation. No such commit, tag or push occurs in Task 9A.1.

## Observed Task 9A.1 verification

All original 140 tests plus eight new release guards passed in the original working tree (148/148, 23.71 s) and an empty-directory export of the staged candidate (148/148, 22.38 s): zero failures, skips or import errors. Fourteen matplotlib/pyparsing deprecation warnings occurred. Tests used cached MNIST and an external network-denial guard; zero network attempts occurred. No dataset or package was downloaded. These are Codex executions, not student independent checks.

The three documented analysis commands exited zero in the fresh export with the dataset link removed. Strict release verification also exited zero: 110 historical input identities, 28 frozen records, three original archive entries and twelve active output/provenance files verified. The active metadata describes the fresh checkout's LF inputs and actual Python 3.9.6/SciPy 1.13.1 environment; older local CRLF hashes and Python 3.12.14 metadata remain in the frozen snapshot. Source hashes distinguish the earlier and repaired test/document versions.

All 10,000 genuine latency observations, empirical dropout rows, theoretical probability CSVs and 2,200 per-round/110 per-run delay derivatives agree with the preserved scientific evidence. The historical cohort aggregate has fifteen last-digit differences, at most 3e-11 ms; six JSON statistics differ at floating-point precision. The verifier records the exact old/new values and bounds. No observations or displayed scientific conclusions were altered. Statistical text matches apart from actual-runtime and input-manifest metadata. Rural's discrepancy and the unmet throughput target remain.

The strict check initially detected that exporting over existing files had retained their CRLF working copies. The test setup was corrected by exporting the staged index into an empty directory; all 110 resulting input files were independently checked against their historical Git bytes. A later test launcher stopped at its `download=False` cache check when the temporary cache link had already been removed; the link was restored for full tests and then removed before analysis. No download was attempted. Neither issue was bypassed by weakening a check.

Recommendation: conditional GO for a separately approved commit/release workflow, with the actual committed/tagged tree checked again before pushing and a remote clone checked after publication. `v1.1.0` remains proposed and must be rechecked. This task creates no commit, tag or remote change.
