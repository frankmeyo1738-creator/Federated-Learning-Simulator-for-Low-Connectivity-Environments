# Corrections 16 and 4: genuine observations and dropout theory

`engine_propagation_latency_draws.csv` contains 10,000 unmodified return values
from `NetworkImpairmentEngine._apply_latency()`: 2,500 for each profile.
Seeds 100, 101, 102 and 103 correspond respectively to Baseline, Urban,
Rural and Severe. These fixed choices were retained from the superseded
diagnostic before examining the new observations.

`layer3_validation_report.txt` is the human-readable summary.
`layer3_provenance.json` records the actual runtime, network settings, seeds,
source-file and input-file SHA-256 hashes, statistics and output hashes.
`cohort_dropout_audit.csv` records counts and rates for the existing 110 runs.
`dropout_theory_by_round.csv` records the finite-horizon expectation by round;
`dropout_theory_distribution.csv` records the complete predicted lost-count
laws for a single run and the mean of ten seed runs. These are theoretical
probabilities, not additional experimental observations.
`DROPOUT_THEORY.md` gives the source audit, derivation, assumptions, validation
and interpretation of the observed Rural discrepancy.

## Walk through the script during the defence

1. **Constants and `EXPERIMENTS`:** enumerate the released experimental groups
   and diagnostic seeds. No desired measurement or replacement network value
   is specified. The project root is resolved from the script's location.
2. **`read_cohort`:** require all ten seed CSVs and their matching YAMLs for
   each of the eleven groups. Load the actual embedded network profiles with
   `ConfigLoader`; reject unexpected seeds, differing settings and profile drift.
3. **`read_run`:** require 20 ordered rounds. Validate integer received/dropped
   counts against the selected-client count from that run's configuration.
   Read the recorded durations. Every original CSV remains unchanged.
4. **`sample_engine`:** instantiate the production engine using that loaded
   profile. Save Python's random state, seed it, and call `_apply_latency()`
   once per observation. A context manager replaces only `asyncio.sleep`
   with an async no-op. A `finally` block restores the prior random state;
   the context manager restores sleep even on an exception. This is an
   isolated, single-threaded diagnostic, because the asyncio module is shared.
5. **What is actually observed:** the engine itself draws from `random.gauss`,
   clamps to its profile bounds, requests sleep, and returns a float. The
   diagnostic stores that returned float without rescaling, shifting, rounding
   or independent sampling. It does not call client training or rerun FL.
6. **`summarize_samples`:** calculate mean, sample SD, range, midpoint
   discrepancy, boundary counts and bounds violations from those returned
   values. A violation is recorded; it is not repaired. Clamping is distinct
   from rejection sampling a truncated normal distribution.
7. **`dropout_summary`:** for each original run, divide dropped updates by
   received plus dropped updates. Report the mean and sample SD across its
   ten seeds. Keep experimental groups separate: identical seeded network
   conditions across groups do not create independent additional replications.
8. **`superseded_means`:** verify the archived evidence hashes and compute
   the old adjusted-data means solely for a transparent comparison. Those
   means never enter the new sampling or its statistics.
9. **`marginal_dropout_theory`:** derive eligibility from the mutually
   exclusive recent trigger events for a client. Use rational arithmetic
   for the 20-round mean. Packet losses enter only after disconnection losses.
10. **`finite_dropout_distribution`:** sum events over the actual population
    of ten clients, choosing exactly five without replacement. Propagate
    state counts and cumulative lost updates with dynamic programming.
    This is probability enumeration, not a Monte Carlo sample or ML run.
11. **`compare_dropout_theory`:** convolve the single-run law ten times for
    the canonical ten-seed mean. Report discrete predictive intervals,
    observed-minus-predicted differences and any interval exceedance.
    The theoretical parameters never depend on observed drop percentages.
12. **`build_report`:** print measurements and limitations. Timing differences
    are descriptive; these records do not isolate engine overhead. Rural
    remains outside the pointwise 95% predictive interval, so universal
    agreement is not claimed even though it is inside a wider exploratory
    multiplicity-aware band.
13. **`main`:** collect and validate inputs before generating the raw CSV,
    per-run audit, report and provenance JSON. Python's float-to-string output
    permits exact binary-float round trips; displayed table rounding does not
    change the underlying observations.

## Evidence boundaries and follow-up

These are fresh diagnostic samples, not recovered samples from the historical
ML experiments. They test the latency method in isolation, not the entire
FL system, a real cellular network, or an empirical normality hypothesis.
The old diagnostic used shifted NumPy draws and cannot support engine-fidelity
claims. Its generator and outputs are preserved byte for byte under
`superseded-evidence/pre-correction-16-v1.0.8/`, with a manifest and explanation.
The old root-level `propagation_latency_draws.csv` is superseded by the newly
named engine-observation file; it is retained only in that archive.

Report section 5.6 needs the corrected latency statistics and the
selection-aware dropout analysis in `DROPOUT_THEORY.md`. The old formula is a
full-selection steady-state special case. The exact mean calculation gives
26.208599% for Rural against 30.20% observed; this discrepancy must be retained.
For Severe, selection-aware steady state is 61.481481% and the 20-round mean
is 60.732510%, so the old 69.411765%-versus-59.80% gap cannot be attributed
entirely to startup. Historical bandwidth conventions and hardware
attributions are not resolved by this diagnostic.

Historical task scope: the report and AI usage log were outside Tasks 2–3's edits. Tasks 6–8 subsequently reconciled them. In the official template report,
former §5.6 is now §6.4.4. Task 9A.1 preserves pre-repair provenance separately
and verifies release-checkout serialization; see [release verification](../../../docs/RELEASE_VERIFICATION.md).
Production code, configurations and original experimental CSVs remain unchanged.
No commit, push or tag is part of this diagnostic task.
