"""Corrections 16 and 4: genuine latency and selection-aware dropout theory.

Only asyncio.sleep is replaced during single-threaded diagnostic sampling.
The engine's random draws, clamping and returned floats remain untouched.
Existing cohort observations and configurations are read only.
"""

import asyncio
import csv
import hashlib
import importlib.metadata
import json
import math
import platform
import random
import statistics
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from fractions import Fraction
from functools import lru_cache
from itertools import product
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import ConfigLoader
from src.network import impairment
import numpy as np  # Deterministic probability arrays only, never latency sampling.

N_DRAWS = 2500
SEEDS = {"baseline": 100, "urban_zambia": 101,
         "rural_zambia": 102, "severe_disruption": 103}
COHORT_SEEDS = tuple(range(1, 11))
DIAGNOSTICS = ROOT / "experiments/results/diagnostics"
SUPERSEDED = DIAGNOSTICS / "superseded-evidence/pre-correction-16-v1.0.8"
RAW_NAME = "engine_propagation_latency_draws.csv"
AUDIT_NAME = "cohort_dropout_audit.csv"
METADATA_NAME = "layer3_provenance.json"
SUMMARY_NAME = "layer3_validation_report.txt"
THEORY_PMF_NAME = "dropout_theory_distribution.csv"
THEORY_ROUNDS_NAME = "dropout_theory_by_round.csv"

# Released strata, not replacement network settings or expected outcomes.
EXPERIMENTS = (
    ("baseline", "fedavg", False), ("urban_zambia", "fedavg", False),
    ("rural_zambia", "fedavg", False), ("severe_disruption", "fedavg", False),
    ("rural_zambia", "fedprox", False), ("severe_disruption", "fedprox", False),
    ("baseline", "fedavg", True), ("rural_zambia", "fedavg", True),
    ("rural_zambia", "fedprox", True), ("severe_disruption", "fedavg", True),
    ("severe_disruption", "fedprox", True),
)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def relative(path):
    return Path(path).resolve().relative_to(ROOT).as_posix()


async def _no_sleep(delay, result=None):
    """Discard only the wait; no RNG calls or transformations of latency."""
    return result


def sample_engine(profile, seed, count=N_DRAWS):
    """Invoke actual _apply_latency once per observation; restore RNG and sleep.

    Use in an isolated single-threaded process: asyncio.sleep is shared by
    Python modules while this short-lived patch is active.
    """
    if count < 2:
        raise ValueError("At least two samples are required for sample SD.")
    engine = impairment.NetworkImpairmentEngine(profile)

    async def collect():
        return [await engine._apply_latency() for _ in range(count)]

    previous_state = random.getstate()
    try:
        random.seed(seed)
        with patch.object(impairment.asyncio, "sleep", new=_no_sleep):
            return asyncio.run(collect())
    finally:
        random.setstate(previous_state)


def summarize_samples(samples, profile):
    """Describe returned observations, including violations; never repair them."""
    if len(samples) < 2 or not all(math.isfinite(value) for value in samples):
        raise ValueError("Samples must contain at least two finite observations.")
    lower, upper = profile.latency_min_ms, profile.latency_max_ms
    midpoint = (lower + upper) / 2
    return {
        "count": len(samples), "mean_ms": statistics.mean(samples),
        "sample_sd_ms": statistics.stdev(samples),
        "min_ms": min(samples), "max_ms": max(samples),
        "midpoint_ms": midpoint,
        "mean_minus_midpoint_ms": statistics.mean(samples) - midpoint,
        "bounds_violations": sum(not lower <= value <= upper for value in samples),
        "at_lower_bound": sum(value == lower for value in samples),
        "at_upper_bound": sum(value == upper for value in samples),
    }


def read_run(path, config):
    """Validate complete round sequences and update accounting before using a CSV."""
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"round_num", "participating_clients", "dropped_updates", "round_duration_seconds"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"Missing diagnostic columns: {path}")
        rows = list(reader)
    if len(rows) != config.num_rounds:
        raise ValueError(f"Incomplete round count: {path}")
    selected = max(1, int(config.num_clients * config.fraction_fit))
    attempts, dropped, durations = 0, 0, []
    for expected_round, row in enumerate(rows, 1):
        try:
            round_num = int(row["round_num"])
            received = int(row["participating_clients"])
            lost = int(row["dropped_updates"])
            duration = float(row["round_duration_seconds"])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid diagnostic observation: {path}") from exc
        if round_num != expected_round:
            raise ValueError(f"Missing, repeated, or unordered rounds: {path}")
        if min(received, lost) < 0 or received + lost != selected:
            raise ValueError(f"Inconsistent selected-update counts: {path}, round {round_num}")
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError(f"Invalid round duration: {path}, round {round_num}")
        attempts += received + lost
        dropped += lost
        durations.append(duration)
    return {"attempted_updates": attempts, "dropped_updates": dropped,
            "drop_rate": dropped / attempts, "round_durations_s": durations}


def read_cohort(root=ROOT):
    """Read all 110 runs and matching configs; never import the training runner."""
    runs, profiles, sources, reference_settings = [], {}, {}, {}
    for profile_name, algorithm, noniid in EXPERIMENTS:
        experiment = f"{profile_name}_{algorithm}" + ("_noniid" if noniid else "")
        results_dir = root / "experiments/results/multiseed"
        config_dir = root / "config/experiments/multiseed"
        if noniid:
            results_dir /= "noniid"
            config_dir /= "noniid"
        expected = {results_dir / f"{experiment}_seed{seed}_metrics.csv" for seed in COHORT_SEEDS}
        actual = set(results_dir.glob(f"{experiment}_seed*_metrics.csv"))
        if actual != expected:
            raise ValueError(f"Incomplete or unexpected seed cohort for {experiment}: missing={sorted(expected - actual)}, extra={sorted(actual - expected)}")
        for seed in COHORT_SEEDS:
            config_path = config_dir / f"{experiment}_seed{seed}.yaml"
            config = ConfigLoader().load(str(config_path))
            if (config.seed != seed or config.algorithm != algorithm or config.dataset != "mnist"
                    or config.num_rounds != 20 or config.network_profile.name != profile_name
                    or config.experiment_name != f"{experiment}_seed{seed}"):
                raise ValueError(f"Unexpected cohort configuration: {config_path}")
            settings = asdict(config)
            for key in ("seed", "experiment_name", "output_dir"):
                settings.pop(key)
            if experiment in reference_settings and settings != reference_settings[experiment]:
                raise ValueError(f"Configuration drift between seeds: {config_path}")
            reference_settings[experiment] = settings
            if profile_name in profiles and config.network_profile != profiles[profile_name]:
                raise ValueError(f"Network-profile drift across configurations: {config_path}")
            profiles[profile_name] = config.network_profile
            csv_path = results_dir / f"{experiment}_seed{seed}_metrics.csv"
            runs.append({"experiment": experiment, "profile": profile_name,
                         "algorithm": algorithm, "partition_label": "non-IID" if noniid else "IID",
                         "seed": seed, "round_count": config.num_rounds,
                         "num_clients": config.num_clients, "fraction_fit": config.fraction_fit,
                         "csv_path": csv_path, "config_path": config_path,
                         **read_run(csv_path, config)})
            for source in (csv_path, config_path):
                sources[source.relative_to(root).as_posix()] = sha256(source)
    return runs, profiles, sources


def dropout_summary(runs):
    """Mean and sample SD of per-run drop proportions, by experimental stratum."""
    result = {}
    for experiment in sorted({run["experiment"] for run in runs}):
        members = [run for run in runs if run["experiment"] == experiment]
        rates = [run["drop_rate"] * 100 for run in members]
        result[experiment] = {
            "n_seeds": len(members), "mean_percent": statistics.mean(rates),
            "sample_sd_pp": statistics.stdev(rates),
            "attempted_updates": sum(run["attempted_updates"] for run in members),
            "dropped_updates": sum(run["dropped_updates"] for run in members),
        }
    return result


def marginal_dropout_theory(p, duration, loss, num_clients, selected, rounds):
    """Exact rational mean for the ideal independent-random-draw model.

    A[t] is eligibility BEFORE a selected client's possible new trigger. A
    trigger at t blocks that attempt and max(duration-1, 0) future rounds.
    Past trigger windows for one client are mutually exclusive, giving
    A[t] = 1 - q*p*sum(A[t-j], j=1..min(L,t-1)). Packet loss is checked only
    when neither an existing disconnection nor a new trigger has dropped it.
    """
    if (not isinstance(duration, int) or duration < 0
            or not isinstance(num_clients, int) or num_clients < 1
            or not isinstance(selected, int) or not 1 <= selected <= num_clients
            or not isinstance(rounds, int) or rounds < 1):
        raise ValueError("Require integer D>=0, N>=1, 1<=m<=N and T>=1.")
    p, loss = Fraction(str(p)), Fraction(str(loss))
    if not 0 <= p <= 1 or not 0 <= loss <= 1:
        raise ValueError("Trigger and packet-loss probabilities must be in [0,1].")
    q, lag = Fraction(selected, num_clients), max(duration - 1, 0)
    eligible, by_round = [], []
    for t in range(rounds):
        available = 1 - q * p * sum(eligible[max(0, t - lag):t])
        existing = 1 - available
        new_trigger = available * p
        packet_only = available * (1 - p) * loss
        eligible.append(available)
        by_round.append({"round_num": t + 1, "eligible_probability": available,
                         "persistent_drop_probability": existing,
                         "new_trigger_drop_probability": new_trigger,
                         "packet_only_drop_probability": packet_only,
                         "total_drop_probability": existing + new_trigger + packet_only})
    return {
        "by_round": by_round,
        "finite_mean": sum(row["total_drop_probability"] for row in by_round) / rounds,
        "steady_mean": 1 - (1 - p) * (1 - loss) / (1 + q * p * lag),
        "old_all_selected_steady": 1 - (1 - p) * (1 - loss) / (1 + p * lag),
    }


def finite_dropout_distribution(p, duration, loss, num_clients, selected, rounds):
    """Enumerate the finite-population Markov reward law, NOT Monte Carlo.

    State n[r] counts clients with r future eligibility blocks remaining at
    round start (r=0: available). Selection counts have a multivariate
    hypergeometric law, so exactly m distinct clients are selected. New
    triggers and then packet losses have conditional binomial laws. Each
    transition also carries the number of lost updates in that round.
    The enumeration is exact under the idealized random model; numerical
    probabilities are evaluated in float64 without mass renormalization.
    """
    reference = marginal_dropout_theory(p, duration, loss, num_clients, selected, rounds)
    p, loss = float(p), float(loss)
    bins = max(duration, 1)
    total_attempts = selected * rounds

    def binomial(n, probability):
        return [math.comb(n, k) * probability ** k * (1 - probability) ** (n - k)
                for k in range(n + 1)]

    @lru_cache(maxsize=None)
    def transitions(state):
        events = {}
        for counts in product(*(range(min(n, selected) + 1) for n in state)):
            if sum(counts) != selected:
                continue
            selection_probability = math.prod(math.comb(n, k) for n, k in zip(state, counts)) / math.comb(num_clients, selected)
            for triggers, trigger_probability in enumerate(binomial(counts[0], p)):
                if trigger_probability == 0:
                    continue
                # All clocks advance, including those of unselected clients.
                following = [state[0]] + [0] * (bins - 1)
                for r in range(1, bins):
                    following[r - 1] += state[r]
                following[0] -= triggers
                following[bins - 1] += triggers
                for packets, packet_probability in enumerate(binomial(counts[0] - triggers, loss)):
                    probability = selection_probability * trigger_probability * packet_probability
                    if probability == 0:
                        continue
                    lost = sum(counts[1:]) + triggers + packets
                    key = (tuple(following), lost)
                    events[key] = events.get(key, 0.0) + probability
        if not math.isclose(sum(events.values()), 1, rel_tol=0, abs_tol=1e-12):
            raise ArithmeticError("Transition probabilities do not sum to one.")
        return tuple((state, lost, probability) for (state, lost), probability in events.items())

    initial = (num_clients,) + (0,) * (bins - 1)
    initial_mass = np.zeros(total_attempts + 1)
    initial_mass[0] = 1
    states = {initial: initial_mass}
    for _ in range(rounds):
        following = {}
        for state, mass in states.items():
            for target, lost, probability in transitions(state):
                if target not in following:
                    following[target] = np.zeros(total_attempts + 1)
                if lost:
                    following[target][lost:] += probability * mass[:-lost]
                else:
                    following[target] += probability * mass
        states = following
    pmf = sum(states.values(), np.zeros(total_attempts + 1))
    if not math.isclose(float(pmf.sum()), 1, rel_tol=0, abs_tol=1e-10):
        raise ArithmeticError("Finite-horizon probability mass was lost.")
    rate = np.arange(total_attempts + 1) / total_attempts
    mean = float(np.dot(rate, pmf))
    if not math.isclose(mean, float(reference["finite_mean"]), rel_tol=0, abs_tol=1e-12):
        raise ArithmeticError("Population distribution and rational marginal mean disagree.")
    variance = float(np.dot((rate - float(reference["finite_mean"])) ** 2, pmf))
    return {"pmf": pmf, "mean": mean, "sd": math.sqrt(variance),
            "attempts_per_run": total_attempts, "final_state_count": len(states),
            "reference": reference}


def predictive_interval(pmf, denominator, alpha):
    """Equal-tail discrete prediction interval; not a parameter confidence CI."""
    cumulative = np.cumsum(pmf)
    lower = int(np.searchsorted(cumulative, alpha / 2, side="left"))
    upper = int(np.searchsorted(cumulative, 1 - alpha / 2, side="left"))
    if upper >= len(pmf):
        raise ArithmeticError("Prediction quantile exceeds distribution support.")
    return {"lower_percent": 100 * lower / denominator,
            "upper_percent": 100 * upper / denominator,
            "coverage": float(pmf[lower:upper + 1].sum()),
            "lower_count": lower, "upper_count": upper}


def compare_dropout_theory(profiles, runs, drops):
    """Predictions use YAML parameters only; observations enter comparison last."""
    comparisons, distributions, per_round = {}, [], []
    for name, profile in profiles.items():
        members = [run for run in runs if run["profile"] == name]
        designs = {(run["num_clients"], min(run["num_clients"], max(1, int(run["num_clients"] * run["fraction_fit"]))), run["round_count"])
                   for run in members}
        if len(designs) != 1:
            raise ValueError(f"Inconsistent selection/horizon design for {name}")
        num_clients, selected, rounds = designs.pop()
        distribution = finite_dropout_distribution(profile.dropout_probability, profile.dropout_duration_rounds,
                                                    profile.packet_loss_rate, num_clients, selected, rounds)
        # Canonical ten-seed strata selected before examining model agreement.
        observed = drops[f"{name}_fedavg"]
        n = observed["n_seeds"]
        cohort_pmf = np.array([1.0])
        for _ in range(n):
            cohort_pmf = np.convolve(cohort_pmf, distribution["pmf"])
        if not math.isclose(float(cohort_pmf.sum()), 1, rel_tol=0, abs_tol=1e-9):
            raise ArithmeticError("Cohort prediction mass does not sum to one.")
        denominator = distribution["attempts_per_run"] * n
        if denominator != observed["attempted_updates"]:
            raise ValueError(f"Observed and theoretical attempt counts disagree for {name}")
        pointwise = predictive_interval(cohort_pmf, denominator, 0.05)
        simultaneous = predictive_interval(cohort_pmf, denominator, 0.05 / len(profiles))
        reference = distribution["reference"]
        comparisons[name] = {
            "N": num_clients, "m": selected, "T": rounds, "q": selected / num_clients,
            "p": profile.dropout_probability, "D": profile.dropout_duration_rounds,
            "packet_loss": profile.packet_loss_rate, "seed_runs": n,
            "finite_mean_percent": 100 * float(reference["finite_mean"]),
            "finite_mean_exact_fraction": str(reference["finite_mean"]),
            "steady_mean_percent": 100 * float(reference["steady_mean"]),
            "old_all_selected_steady_percent": 100 * float(reference["old_all_selected_steady"]),
            "predicted_run_sd_pp": 100 * distribution["sd"],
            "predicted_mean_se_pp": 100 * distribution["sd"] / math.sqrt(n),
            "observed_mean_percent": observed["mean_percent"],
            "observed_sample_sd_pp": observed["sample_sd_pp"],
            "observed_minus_predicted_pp": observed["mean_percent"] - 100 * float(reference["finite_mean"]),
            "pointwise_95_prediction_interval": pointwise,
            "familywise_95_prediction_band": simultaneous,
            "inside_pointwise_95": pointwise["lower_count"] <= observed["dropped_updates"] <= pointwise["upper_count"],
            "inside_familywise_95": simultaneous["lower_count"] <= observed["dropped_updates"] <= simultaneous["upper_count"],
        }
        for count, probability in enumerate(distribution["pmf"]):
            distributions.append({"profile": name, "population": "single_run", "dropped_updates": count,
                                  "attempted_updates": distribution["attempts_per_run"], "probability": float(probability)})
        for count, probability in enumerate(cohort_pmf):
            distributions.append({"profile": name, "population": "ten_seed_mean", "dropped_updates": count,
                                  "attempted_updates": denominator, "probability": float(probability)})
        for row in reference["by_round"]:
            per_round.append({"profile": name, **{key: int(value) if key == "round_num" else float(value) for key, value in row.items()}})
    return comparisons, distributions, per_round


def superseded_means(archive=SUPERSEDED):
    """Compare preserved adjusted data, clearly excluded as engine evidence."""
    manifest_path = archive / "evidence_manifest.json"
    if not manifest_path.exists():
        return {}, {}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for record in manifest.values():
        if sha256(archive / record["archive_file"]) != record["sha256"]:
            raise ValueError("Superseded evidence hash mismatch; do not overwrite the archive.")
    grouped = {}
    with (archive / "propagation_latency_draws.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            grouped.setdefault(row["profile"], []).append(float(row["calibrated_latency_ms"]))
    return {name: statistics.mean(values) for name, values in grouped.items()}, manifest


def build_report(latency, drops, runs, legacy, provenance):
    lines = ["LAYER 3 DIAGNOSTICS: GENUINE OBSERVATIONS AND DROPOUT THEORY (CORRECTIONS 16 AND 4)",
             f"Sampling: {N_DRAWS} calls/profile; RNG: Python random; sleep stubbed only during sampling.",
             "Values are the unmodified, already-clamped floats returned by the production method.",
             "SD means sample standard deviation (ddof=1). Bounds checks are inclusive.",
             "The production sampler clamps Gaussian tails to the bounds; this is not rejection sampling from a truncated normal.",
             "", "1. PROPAGATION LATENCY (milliseconds)",
             "Profile | Seed | N | Mean | SD | Minimum | Maximum | Midpoint | Mean-midpoint | Violations"]
    for name, row in latency.items():
        lines.append(f"{name} | {SEEDS[name]} | {row['count']} | {row['mean_ms']:.6f} | {row['sample_sd_ms']:.6f} | {row['min_ms']:.6f} | {row['max_ms']:.6f} | {row['midpoint_ms']:.6f} | {row['mean_minus_midpoint_ms']:+.6f} | {row['bounds_violations']}")
        lines.append("  Network configuration: " + json.dumps(provenance["network_profiles"][name], sort_keys=True))
    lines.extend([f"Observed bounds violations: {sum(row['bounds_violations'] for row in latency.values())} in {sum(row['count'] for row in latency.values())} draws.",
                  "This checks observed bounds; it does not establish empirical network calibration or distributional normality.",
                  "", "2. EMPIRICAL COHORT DROPS (existing observations only)",
                  f"Completeness: {len(runs)} files, {len(drops)} configurations, seeds 1-10, 20 ordered rounds/file; update counts checked against each YAML.",
                  "Experiment | Seeds | Mean drop (%) | Sample SD (pp) | Dropped / attempted updates"])
    for name, row in drops.items():
        lines.append(f"{name} | {row['n_seeds']} | {row['mean_percent']:.6f} | {row['sample_sd_pp']:.6f} | {row['dropped_updates']} / {row['attempted_updates']}")
    lines.extend(["These drops combine persistent client dropout and packet loss; the CSVs do not separate their causes.",
                  "Strata share network seeds and may repeat outcomes; do not pool them as independent replications.",
                  "Partition labels come from filenames/directories, not a reconstructed training execution record."])
    lines.extend(["", "2B. SELECTION-AWARE FINITE-HORIZON THEORY (canonical FedAvg IID, ten seeds)",
                  "Profile | Predicted mean (%) | Observed mean (%) | Difference (pp) | 95% prediction interval for ten-seed mean (%) | Inside?",
                  "Assumptions: independent ideal random draws; uniform fixed-size subsets each round; all clients initially eligible.",
                  "D is fixed by YAML. A trigger drops the current attempt and blocks max(D-1,0) future global rounds.",
                  "r_t = 1 - A_t*(1-p)*(1-loss); A_t = 1 - q*p*sum(A_(t-j), j=1..min(D-1,t-1)), with lag zero for D=0.",
                  "Means use exact rational arithmetic. Full-population finite-state enumeration includes within-round selection dependence.",
                  "Probability distributions use float64 arithmetic, not Monte Carlo, and are checked against the rational means."])
    for name, row in provenance["dropout_theory"].items():
        interval = row["pointwise_95_prediction_interval"]
        band = row["familywise_95_prediction_band"]
        lines.append(f"{name} | {row['finite_mean_percent']:.6f} | {row['observed_mean_percent']:.6f} | {row['observed_minus_predicted_pp']:+.6f} | [{interval['lower_percent']:.1f}, {interval['upper_percent']:.1f}] | {row['inside_pointwise_95']}")
        lines.append(f"  N={row['N']}, m={row['m']}, T={row['T']}, q={row['q']}; p={row['p']}, D={row['D']}, packet loss={row['packet_loss']}; predicted run SD={row['predicted_run_sd_pp']:.6f} pp, observed sample SD={row['observed_sample_sd_pp']:.6f} pp.")
        lines.append(f"  Selection-aware steady mean={row['steady_mean_percent']:.6f}%; old q=1 steady mean={row['old_all_selected_steady_percent']:.6f}%.")
        lines.append(f"  Exploratory Bonferroni band for four profile means: [{band['lower_percent']:.1f}, {band['upper_percent']:.1f}]%; inside={row['inside_familywise_95']}.")
    lines.extend(["Intervals are model-based predictions for ten independent seed runs, NOT confidence intervals for estimated parameters.",
                  "Pointwise intervals have at least 95% discrete coverage. Each Bonferroni band targets 98.75%, giving at least 95% joint coverage across four means under the model.",
                  "The wider exploratory bands do not erase a pointwise discrepancy. No blanket empirical-theory agreement is claimed.",
                  "A deterministic PRNG is idealized as independent draws. No per-client event logs exist here to identify the cause of discrepancies.",
                  "", "3. SUPERSEDED EVIDENCE AND REPORT SECTION 5.6",
                  "Old adjusted-data means below are read from the archived CSV, not targets for new measurements.",
                  "Profile | Superseded adjusted-data mean | Genuine engine mean | Difference (ms)"])
    for name, row in latency.items():
        if name in legacy:
            lines.append(f"{name} | {legacy[name]:.6f} | {row['mean_ms']:.6f} | {row['mean_ms'] - legacy[name]:+.6f}")
    lines.extend(["Replace section 5.6's old latency observations, SDs and ranges with new engine observations and seeds.",
                  "These are fresh diagnostic observations, not recovered propagation samples from historical ML runs.",
                  "The old formula is the q=1 steady-state special case; q=0.5 and T=20 require the selection-aware calculation above.",
                  "Separate the selection effect from the initial-state transient. Do not attribute the old Severe discrepancy solely to startup.",
                  "", "4. DESCRIPTIVE ROUND DURATIONS (canonical FedAvg IID strata only)",
                  "Profile | Rounds | Mean duration (s) | Sample SD (s) | Mean difference from Baseline (s)"])
    durations = {name: [value for run in runs if run["experiment"] == f"{name}_fedavg"
                        for value in run["round_durations_s"]] for name in SEEDS}
    baseline_mean = statistics.mean(durations["baseline"])
    for name, values in durations.items():
        lines.append(f"{name} | {len(values)} | {statistics.mean(values):.6f} | {statistics.stdev(values):.6f} | {statistics.mean(values) - baseline_mean:+.6f}")
    lines.extend(["These durations do not isolate impairment overhead or establish a hardware/thermal cause.",
                  "Rounds within runs are correlated. No timing significance or compute-noise-floor claim is made.",
                  "Bandwidth conversion is not revalidated here; avg_latency_ms remains an untouched historical total-delay metric.",
                  "", "5. PROVENANCE",
                  "Runtime: " + json.dumps(provenance["runtime"], sort_keys=True),
                  f"Source revision: {provenance['git_head']} (working-tree hashes recorded separately).",
                  f"Raw observations: {RAW_NAME}; per-run audit: {AUDIT_NAME}; full provenance: {METADATA_NAME}.",
                  f"Theory laws: {THEORY_PMF_NAME}; per-round expectations: {THEORY_ROUNDS_NAME}; derivation: DROPOUT_THEORY.md.",
                  "No ML training or end-to-end FL experiment was executed."])
    return "\n".join(lines) + "\n"


def main():
    runs, profiles, sources = read_cohort()
    legacy, archive_manifest = superseded_means()
    raw_rows, latency = [], {}
    for name, seed in SEEDS.items():
        samples = sample_engine(profiles[name], seed)
        latency[name] = summarize_samples(samples, profiles[name])
        for index, value in enumerate(samples, 1):
            raw_rows.append({"profile": name, "seed": seed, "draw_index": index,
                             "propagation_latency_ms": value,
                             "min_bound_ms": profiles[name].latency_min_ms,
                             "max_bound_ms": profiles[name].latency_max_ms})
    drops = dropout_summary(runs)
    theory, theory_distributions, theory_rounds = compare_dropout_theory(profiles, runs, drops)
    for name in ("scripts/verify_layer3_fidelity.py", "src/network/impairment.py",
                 "src/config.py", "src/core/client.py", "src/core/server.py", "src/main.py",
                 "src/analytics/tracker.py", "tests/unit/test_layer3_fidelity.py",
                 "tests/unit/test_dropout_theory.py", "experiments/results/diagnostics/DROPOUT_THEORY.md",
                 "experiments/results/diagnostics/README.md"):
        sources[name] = sha256(ROOT / name)
    provenance = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "runtime": {"python": platform.python_version(), "executable": sys.executable,
                    "platform": platform.platform(), "machine": platform.machine(),
                    "packages": {name: importlib.metadata.version(name) for name in ("torch", "numpy", "PyYAML")}},
        "method": "NetworkImpairmentEngine._apply_latency; only asyncio.sleep patched to an async no-op; Python random state restored",
        "sampling_seeds": SEEDS, "samples_per_profile": N_DRAWS,
        "network_profiles": {name: asdict(profiles[name]) for name in SEEDS},
        "cohort_seeds": list(COHORT_SEEDS), "cohort_files": len(runs),
        "source_sha256": sources, "superseded_evidence": archive_manifest,
        "latency_statistics": latency, "dropout_statistics": drops,
        "dropout_theory": theory,
        "theory_method": {
            "mean": "exact Fraction recurrence, decimal YAML probabilities treated as rational design values",
            "distribution": "finite-population Markov reward enumeration with float64 evaluation; no renormalization or Monte Carlo",
            "prediction_intervals": "equal-tail quantiles of ten-fold convolution; pointwise alpha=.05; exploratory Bonferroni alpha=.05/4",
            "assumptions": ["independent ideal draws rather than bit-exact PRNG seed predictions",
                            "uniform m-of-N subsets, independent between rounds and of prior client states",
                            "all clients initially eligible; fixed YAML duration; packet loss checked only after disconnection check",
                            "ten independent seed runs per canonical profile; paired strata not pooled"],
        },
    }
    # Preserve any bounds discrepancy in the outputs rather than adjusting it.
    DIAGNOSTICS.mkdir(parents=True, exist_ok=True)
    with (DIAGNOSTICS / RAW_NAME).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(raw_rows[0]))
        writer.writeheader()
        writer.writerows(raw_rows)  # float -> str is round-trip safe; no rounding.
    fields = ["experiment", "profile", "algorithm", "partition_label", "seed", "round_count",
              "num_clients", "fraction_fit", "attempted_updates", "dropped_updates", "drop_rate",
              "csv_path", "config_path"]
    with (DIAGNOSTICS / AUDIT_NAME).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: relative(run[key]) if key.endswith("_path") else run[key]
                          for key in fields} for run in runs)
    provenance["output_sha256"] = {name: sha256(DIAGNOSTICS / name) for name in (RAW_NAME, AUDIT_NAME)}
    for filename, records in ((THEORY_PMF_NAME, theory_distributions), (THEORY_ROUNDS_NAME, theory_rounds)):
        with (DIAGNOSTICS / filename).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
        provenance["output_sha256"][filename] = sha256(DIAGNOSTICS / filename)
    report = build_report(latency, drops, runs, legacy, provenance)
    (DIAGNOSTICS / SUMMARY_NAME).write_text(report, encoding="utf-8")
    provenance["output_sha256"][SUMMARY_NAME] = sha256(DIAGNOSTICS / SUMMARY_NAME)
    (DIAGNOSTICS / METADATA_NAME).write_text(json.dumps(provenance, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
