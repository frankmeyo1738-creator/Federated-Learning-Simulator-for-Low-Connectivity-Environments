"""Analysis-only checks: synthetic fixtures stay in pytest's temporary directory.

No model imports, training, downloads, or writes to the experimental cohort.
Real-data checks calculate reference summaries independently with csv/statistics.
"""

import csv
import hashlib
import importlib.util
import math
import statistics
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("statistical_analysis", ROOT / "scripts/statistical_analysis.py")
analysis = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analysis)


def write_fixture(directory, seed, accuracies=(0.80, 0.91, 0.96), mutate=None):
    rows = [dict(round_num=i, global_accuracy=acc, global_loss=0.1,
                 participating_clients=4, dropped_updates=1,
                 total_bytes_transmitted=1048576)
            for i, acc in enumerate(accuracies, 1)]
    if mutate:
        mutate(rows)
    path = directory / f"fixture_seed{seed}_metrics.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    return path


def test_fixture_statistics_and_first_hit(tmp_path):
    write_fixture(tmp_path, 1, (0.96, 0.80, 0.94))
    write_fixture(tmp_path, 2, (0.80, 0.91, 0.98))
    result = analysis.analyze_experiment("fixture", tmp_path, n_seeds=2, n_rounds=3)
    assert result["seeds"] == [1, 2]
    assert result["mean_acc"] == pytest.approx(statistics.mean([0.94, 0.98]))
    assert result["std_acc"] == pytest.approx(statistics.stdev([0.94, 0.98]))
    assert result["mean_drop"] == pytest.approx(1 / 5)
    assert result["mean_bytes"] / analysis.BYTES_PER_MIB == 3
    assert result["r95_rounds"] == [1, 3]  # First hit, not last/sustained hit.
    assert result["r95_mean"] == 2


def test_missing_seed_is_not_silently_excluded(tmp_path):
    write_fixture(tmp_path, 1)
    with pytest.raises(ValueError, match="Missing seed 2"):
        analysis.analyze_experiment("fixture", tmp_path, n_seeds=2, n_rounds=3)


@pytest.mark.parametrize("mutation,message", [
    (lambda rows: rows[1].update(global_accuracy=float("nan")), "Non-finite"),
    (lambda rows: rows[1].update(global_accuracy=1.2), "Invalid accuracy"),
    (lambda rows: rows[1].update(round_num=1), "Expected rounds"),
    (lambda rows: rows.reverse(), "Expected rounds"),
    (lambda rows: rows.pop(), "Expected rounds"),
    (lambda rows: rows[1].update(dropped_updates=-1), "Invalid counts"),
    (lambda rows: rows[1].update(participating_clients=2.5), "Invalid counts"),
    (lambda rows: rows[1].update(participating_clients=0, dropped_updates=0), "No attempted"),
])
def test_invalid_observations_rejected(tmp_path, mutation, message):
    write_fixture(tmp_path, 1, mutate=mutation)
    write_fixture(tmp_path, 2)
    with pytest.raises(ValueError, match=message):
        analysis.analyze_experiment("fixture", tmp_path, n_seeds=2, n_rounds=3)


def test_unreached_threshold_is_reported_without_survivor_mean(tmp_path):
    write_fixture(tmp_path, 1)
    write_fixture(tmp_path, 2, (0.80, 0.91, 0.94))
    result = analysis.analyze_experiment("fixture", tmp_path, n_seeds=2, n_rounds=3)
    assert result["r95_mean"] is None
    assert result["r95_rounds"] == [3, None]
    output = []
    analysis.print_summary_table(output, {"fixture": result}, ["fixture"])
    assert "not reached within 3 rounds for seeds [2]" in "\n".join(output)
    with pytest.raises(ValueError, match="did not reach R95"):
        analysis.convergence_decomposition(result, result, {"fixture": result})


def test_decomposition_denominators_and_additivity():
    def cohort(mean):
        return {"seeds": [1, 2], "r95_mean": mean}
    result = analysis.convergence_decomposition(cohort(2), cohort(3), {"example": cohort(5)})
    row = result["profiles"]["example"]
    assert result["heterogeneity_rounds"] == 1
    assert result["heterogeneity_percent"] == 50
    assert row["network_rounds"] == 2
    assert row["network_percent"] == pytest.approx(2 / 3 * 100)
    assert row["total_percent"] == 150
    assert row["total_rounds"] == result["heterogeneity_rounds"] + row["network_rounds"]


def test_same_length_different_seed_identities_rejected():
    first = {"seeds": list(range(1, 11)), "r95_mean": 3}
    second = {"seeds": list(range(2, 12)), "r95_mean": 4}
    with pytest.raises(ValueError, match="seed identities"):
        analysis.convergence_decomposition(first, second, {})
    with pytest.raises(ValueError, match="seed identities"):
        analysis.run_comparison([], {"a": first, "b": second}, "a", "b", "fixture")


def test_undefined_effect_is_not_replaced_with_zero():
    assert math.isnan(analysis.cohen_d_paired([1, 1, 1], [0, 0, 0]))
    assert analysis.cohen_d_paired_ci([1, 1, 1], [0, 0, 0]) == (None, None)


def test_bonferroni_does_not_claim_nominal_significance_survives(monkeypatch):
    monkeypatch.setattr(analysis.stats, "wilcoxon", lambda *a, **kw: (5, 0.03))
    cohort = {"seeds": list(range(1, 11)), "missing_seeds": [], "final_accs": np.linspace(0.8, 0.9, 10)}
    other = dict(cohort, final_accs=np.linspace(0.81, 0.92, 10))
    output = []
    analysis.run_comparison(output, {"a": cohort, "b": other}, "a", "b", "synthetic")
    assert "p_adj = 0.1200" in "\n".join(output)
    assert "NOT significant after Bonferroni" in "\n".join(output)


def real_experiments():
    return ([(name, analysis.RESULTS_DIR) for name in analysis.IID_EXPERIMENTS]
            + [(name, analysis.NONIID_DIR) for name in analysis.NONIID_EXPERIMENTS]
            + [(analysis.BASELINE_NONIID, analysis.NONIID_DIR)])


@pytest.mark.parametrize("name,directory", real_experiments())
def test_real_cohort_against_independent_csv_statistics(name, directory):
    finals, losses, drops, volumes, r90, r95 = [], [], [], [], [], []
    for seed in range(1, 11):
        with (Path(directory) / f"{name}_seed{seed}_metrics.csv").open() as handle:
            rows = list(csv.DictReader(handle))
        assert [int(row["round_num"]) for row in rows] == list(range(1, 21))
        finals.append(float(rows[-1]["global_accuracy"]))
        losses.append(float(rows[-1]["global_loss"]))
        dropped = sum(int(row["dropped_updates"]) for row in rows)
        attempted = sum(int(row["dropped_updates"]) + int(row["participating_clients"]) for row in rows)
        drops.append(dropped / attempted)
        volumes.append(sum(int(row["total_bytes_transmitted"]) for row in rows))
        for threshold, hits in ((0.90, r90), (0.95, r95)):
            hits.append(next((int(row["round_num"]) for row in rows if float(row["global_accuracy"]) >= threshold), None))
    result = analysis.analyze_experiment(name, directory)
    for key, observations in (("acc", finals), ("loss", losses), ("drop", drops), ("bytes", volumes)):
        # Relative tolerance covers floating reduction order at byte-scale values
        # (tens of millions); a fixed 1e-12-byte tolerance is below one float ULP.
        assert result[f"mean_{key}"] == pytest.approx(statistics.mean(observations), rel=1e-12, abs=1e-12)
        assert result[f"std_{key}"] == pytest.approx(statistics.stdev(observations), rel=1e-12, abs=1e-12)
    assert result["r90_rounds"] == r90
    assert result["r95_rounds"] == r95
    assert result["r90_mean"] == pytest.approx(statistics.mean(r90))
    assert result["r95_mean"] == pytest.approx(statistics.mean(r95))


def test_full_summary_is_deterministic_and_input_files_unchanged():
    paths = [Path(directory) / f"{name}_seed{seed}_metrics.csv"
             for name, directory in real_experiments() for seed in range(1, 11)]
    def hashes():
        return {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    before = hashes()
    first = analysis.generate_summary()
    assert first == analysis.generate_summary()
    assert hashes() == before
    assert "110 runs, 11 configurations" in first
    assert "--- 4. NON-IID BASELINE MATCHED COMPARATOR" in first
    assert "--- 5. CONVERGENCE DELAY DECOMPOSITION" in first
    assert analysis.BASELINE_NONIID in first
    assert "correction=False" in first


@pytest.mark.parametrize("profile,non_iid", [
    ("rural_zambia", False), ("severe_disruption", False),
    ("rural_zambia", True), ("severe_disruption", True),
])
def test_paired_statistics_against_explicit_formulas(profile, non_iid):
    directory = analysis.NONIID_DIR if non_iid else analysis.RESULTS_DIR
    suffix = "_noniid" if non_iid else ""
    avg = analysis.analyze_experiment(f"{profile}_fedavg{suffix}", directory)["final_accs"]
    prox = analysis.analyze_experiment(f"{profile}_fedprox{suffix}", directory)["final_accs"]
    differences = [float(p - a) for p, a in zip(prox, avg)]
    # Independently form average ranks, split zero ranks, and apply tie variance.
    magnitudes = sorted(abs(d) for d in differences)
    ranks = {}
    tie_term = 0
    for magnitude in set(magnitudes):
        positions = [i + 1 for i, value in enumerate(magnitudes) if value == magnitude]
        ranks[magnitude] = statistics.mean(positions)
        tie_term += len(positions) ** 3 - len(positions)
    positive = sum(ranks[abs(d)] for d in differences if d > 0)
    negative = sum(ranks[abs(d)] for d in differences if d < 0)
    zero_half = sum(ranks[0] / 2 for d in differences if d == 0)
    positive += zero_half
    negative += zero_half
    n = len(differences)
    variance = n * (n + 1) * (2 * n + 1) / 24 - tie_term / 48
    z = (positive - n * (n + 1) / 4) / math.sqrt(variance)
    expected_p = math.erfc(abs(z) / math.sqrt(2))
    w, p = analysis.stats.wilcoxon(prox, avg, zero_method="zsplit",
                                  alternative="two-sided", correction=False, mode="approx")
    assert w == min(positive, negative)
    assert p == pytest.approx(expected_p, abs=1e-12)
    expected_d = statistics.mean(differences) / statistics.stdev(differences)
    assert analysis.cohen_d_paired(prox, avg) == pytest.approx(expected_d, abs=1e-12)
    generator = np.random.RandomState(42)
    boot = []
    for _ in range(2000):
        sample = [differences[i] for i in generator.randint(0, n, size=n)]
        boot.append(statistics.mean(sample) / statistics.stdev(sample))
    ordered = sorted(boot)
    def percentile(q):
        index = (len(ordered) - 1) * q
        lower = math.floor(index)
        return ordered[lower] + (index - lower) * (ordered[lower + 1] - ordered[lower])
    assert analysis.cohen_d_paired_ci(prox, avg) == pytest.approx(
        (percentile(0.025), percentile(0.975)), abs=1e-12)


def test_mde_solution_attains_target_power():
    mde = analysis.minimum_detectable_effect(10)
    assert math.isfinite(mde) and mde > 0
    assert analysis.paired_power(mde, 10) == pytest.approx(0.8, abs=1e-10)


def test_invalid_cohort_does_not_replace_existing_summary(tmp_path, monkeypatch):
    target = tmp_path / "summary.txt"
    target.write_text("previous summary", encoding="utf-8")
    monkeypatch.setattr(analysis, "OUTPUT_FILE", target)
    monkeypatch.setattr(analysis, "RESULTS_DIR", tmp_path)
    with pytest.raises(ValueError, match="Missing seed"):
        analysis.main()
    assert target.read_text(encoding="utf-8") == "previous summary"
