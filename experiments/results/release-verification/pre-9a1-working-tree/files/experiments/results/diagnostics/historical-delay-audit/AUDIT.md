# Historical transmission-delay audit (Correction 14)

110 existing runs checked: 100 legacy-binary, 10 decimal, 0 unresolved.
The divisor fix is commit 95d79160601af01c0af1a8321915af9833776eaf; classification also requires unique numerical compatibility with recorded profile bounds.
Git content history is corroborating evidence, not a signed execution manifest. No classification is made from timestamps alone.

14 CSVs differ from their committed copies only in CRLF/LF line endings; all other bytes match. Both hashes are preserved in provenance.json.

## What the saved metric means

`avg_latency_ms` is the cumulative, successful-update-weighted mean of propagation plus analytical bandwidth delay through that round.
It excludes failed updates, local training, model evaluation, protocol overhead, and a separately modelled download/broadcast phase.
The engine is created per run and is not reset each round. A round with no success carries the previous mean; a prefix with no successes has a zero placeholder.
Source trace: FLClient.train computes model_size_bytes from tensor element counts and sizes; FLServer._collect_updates retains only non-None engine returns; FLServer.run sums those delivered bytes and reads get_avg_latency; AnalyticsTracker.export_csv writes the recorded values without a unit conversion.
NetworkImpairmentEngine.transmit checks dropout and packet loss before bandwidth and propagation, then appends their sum only on success. _apply_bandwidth_throttle is analytical; _apply_latency sleeps. Bandwidth normalization therefore changes no execution timing or update selection.
The tracker's summary averages these cumulative row means equally. That is not the run's per-transmission mean; the final CSV row supplies the latter.

per_run_delay_audit.csv retains the original and decimal-derived tracker-style averages as separate fields. Only that tracker-style calculation retains initial zero placeholders, matching the original implementation; derived observation columns leave unobserved means blank.

## Derivation

Let S_r be cumulative successful updates, C_r cumulative delivered bytes, w the fixed Mbps profile value, d the historical divisor, and M_r the saved cumulative mean.
For S_r>0: B_r(d)=8000*C_r/(d*w*S_r), P_r=M_r-B_r(d), and M_r(decimal)=M_r+B_r(1000000)-B_r(d).
For k_r>0 successes in a round, recover its total-delay mean as (S_r*M_r-S_(r-1)*M_(r-1))/k_r; use that round's delivered bytes for its bandwidth correction.
Bandwidth is linear in bytes: aggregate delivered byte totals suffice even with variable individual payload sizes. Constant per-client sizes are not assumed.
Do not multiply the mixed propagation-plus-bandwidth mean by 1.048576. A no-success current-round mean remains missing, not zero.
Decimal arithmetic uses precision 50; feasibility tolerance is 0.000001 ms for historical floating-point accumulation. Values are never clamped or adjusted to fit.
Reconstruction is algebraic at saved CSV precision, not recovery of individual propagation draws or a bit-identical rerun.

## Cohort figures

Each mean below gives equal weight to ten run-final means. Units are milliseconds. CSV summaries additionally contain sample SDs and separately labelled transmission-pooled means.

| Experiment | Runs | Original bandwidth mean | Decimal bandwidth mean | Original total mean | Decimal-derived total mean | Change |
|---|---:|---:|---:|---:|---:|---:|
| baseline_fedavg | 10 | 366.174927 | 383.962240 | 369.180482 | 386.967795 | +17.787313 |
| urban_zambia_fedavg | 10 | 4307.940315 | 4517.202824 | 4448.673866 | 4657.936375 | +209.262509 |
| rural_zambia_fedavg | 10 | 30514.577230 | 31996.853333 | 31063.146426 | 32545.422529 | +1482.276104 |
| severe_disruption_fedavg | 10 | 122058.308919 | 127987.413333 | 123457.830206 | 129386.934620 | +5929.104414 |
| rural_zambia_fedprox | 10 | 30514.577230 | 31996.853333 | 31063.146426 | 32545.422529 | +1482.276104 |
| severe_disruption_fedprox | 10 | 122058.308919 | 127987.413333 | 123457.830206 | 129386.934620 | +5929.104414 |
| baseline_fedavg_noniid | 10 | 383.962240 | 383.962240 | 386.967795 | 386.967795 | +0.000000 |
| rural_zambia_fedavg_noniid | 10 | 30514.577230 | 31996.853333 | 31063.146426 | 32545.422529 | +1482.276104 |
| rural_zambia_fedprox_noniid | 10 | 30514.577230 | 31996.853333 | 31063.146426 | 32545.422529 | +1482.276104 |
| severe_disruption_fedavg_noniid | 10 | 122058.308919 | 127987.413333 | 123457.830206 | 129386.934620 | +5929.104414 |
| severe_disruption_fedprox_noniid | 10 | 122058.308919 | 127987.413333 | 123457.830206 | 129386.934620 | +5929.104414 |

There are 60 no-success rounds among supported runs; their current-round delay means are left blank.
Existing decimal-convention cumulative metrics receive zero correction.

## Report changes to make later

* Chapter 3, section 3.3: define the metric as a cumulative mean over successful uploads; disclose the historical binary divisor and the separate decimal-derived outputs. Keep binary MiB for payload volumes.
* Chapter 4, sections 4.2/4.4: explain the accumulator and state that a passing current regression test does not retroactively change historical measurements.
* Chapter 5, tables/figures using `avg_latency_ms`: label original-convention values or use the explicitly derived decimal-normalized values. Do not mix the two baseline cohorts without this disclosure.
* Chapter 5, section 5.5: analytical bandwidth correction changes no recorded wall-clock duration or throughput. CPU-only timing claims need the separate timing audit.
* Chapter 5, section 5.6: pure-propagation diagnostics from Task 2 are distinct evidence and must not be replaced by cumulative total-delay means. Recovered aggregate propagation means are not raw draws.
* Chapter 5, section 5.7: delivered byte totals and binary-MiB payload figures need no correction.
* Chapter 6: normalized transmission-delay figures are derived analytical quantities, not observed cellular transfer times or evidence of changed model accuracy, dropout, arrivals or deadlines.
* Appendix B: retain the independently computed decimal-Mbps regression example. Appendix C: document this audit runner and separate derived-output paths alongside the frozen original CSVs.

Suggested disclosure: 'The original 100-run cohort records analytical transmission delay using a binary megabit divisor (1,048,576), whereas the ten matched non-IID Baseline runs use decimal Mbps (1,000,000 bits/s). Decimal-normalized values were derived separately from recorded successful-update counts and delivered payload bytes, retaining the inferred propagation component. The CSV field avg_latency_ms is a cumulative mean over successful uploads; it is not a per-round propagation mean or measured network transfer time. Original observations remain unchanged.'

## Reproduce

Dependencies: Python 3 and PyYAML; the audit does not import the simulator, Torch, or training code. Run from the repository root:

```sh
python3 scripts/audit_historical_delays.py
python3 -m pytest -q -p no:cacheprovider tests/unit/test_historical_delay_audit.py
```

The runner reads existing CSVs/configs and Git objects, then writes only this separate output directory. Tests use synthetic observations and temporary output directories, plus read-only validation of all 2,200 historical rows.

## Limits

Individual latency samples, individual payload histories, failed-attempt byte totals and protocol overhead are not recoverable. Only the supported aggregate means are reconstructed.
The recorded payload is tensor-parameter bytes (element count times element size), not a measured serialized wire payload. The reconstructed delay retains that original payload convention.
The correction assumes the inspected accumulator, one fixed bandwidth per run, and matching historical YAML. Historical source, config and CSV hashes are recorded in provenance.json.
The final thesis, AI log and all original observations are unchanged. No ML experiments, commits, pushes or tags are part of this audit.
