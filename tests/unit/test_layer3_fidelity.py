"""Diagnostic-only tests. Never run local training or modify the ML cohort."""

import asyncio
import csv
import importlib.util
import json
import math
import random
import statistics
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("layer3_diagnostic", ROOT / "scripts/verify_layer3_fidelity.py")
diagnostic = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(diagnostic)
Engine = diagnostic.impairment.NetworkImpairmentEngine


@pytest.fixture
def profile():
    return diagnostic.ConfigLoader().load(str(ROOT / "config/experiments/multiseed/urban_zambia_fedavg_seed1.yaml")).network_profile


def test_sampling_invokes_real_engine_and_only_suppresses_sleep(monkeypatch, profile):
    original = Engine._apply_latency
    observations = []

    async def recording_engine(self):
        value = await original(self)
        observations.append(value)
        return value

    sleep = AsyncMock()
    monkeypatch.setattr(Engine, "_apply_latency", recording_engine)
    monkeypatch.setattr(diagnostic, "_no_sleep", sleep)
    before_rng, before_sleep = random.getstate(), asyncio.sleep
    values = diagnostic.sample_engine(profile, 123, count=31)
    assert values == observations
    assert len(observations) == sleep.await_count == 31
    assert [call.args[0] for call in sleep.await_args_list] == [value / 1000 for value in values]
    assert random.getstate() == before_rng
    assert asyncio.sleep is before_sleep


def test_engine_return_values_are_never_adjusted(monkeypatch, profile):
    returned = [81.12345678901234, 82.23456789012345, 83.34567890123456]
    method = AsyncMock(side_effect=returned)
    monkeypatch.setattr(Engine, "_apply_latency", method)
    assert diagnostic.sample_engine(profile, 19, count=3) == returned
    assert method.await_count == 3


def test_sampling_matches_direct_production_calls_and_restores_state(profile):
    state = random.getstate()
    before_sleep = asyncio.sleep
    random.seed(789)
    engine = Engine(profile)

    async def direct_calls():
        return [await engine._apply_latency() for _ in range(40)]

    try:
        with diagnostic.patch.object(diagnostic.impairment.asyncio, "sleep", new=diagnostic._no_sleep):
            expected = asyncio.run(direct_calls())
    finally:
        random.setstate(state)
    assert diagnostic.sample_engine(profile, 789, count=40) == expected
    assert random.getstate() == state
    assert asyncio.sleep is before_sleep


def test_same_seed_repeats_and_different_seed_changes_observations(profile):
    first = diagnostic.sample_engine(profile, 100, count=50)
    assert first == diagnostic.sample_engine(profile, 100, count=50)
    assert first != diagnostic.sample_engine(profile, 101, count=50)


def test_rng_and_sleep_are_restored_on_engine_failure(monkeypatch, profile):
    async def broken(self):
        random.random()
        raise RuntimeError("diagnostic failure")

    monkeypatch.setattr(Engine, "_apply_latency", broken)
    state, sleep = random.getstate(), asyncio.sleep
    with pytest.raises(RuntimeError, match="diagnostic failure"):
        diagnostic.sample_engine(profile, 77, count=2)
    assert random.getstate() == state
    assert asyncio.sleep is sleep


def test_summary_reports_bounds_discrepancies_without_changing_data(profile):
    values = [profile.latency_min_ms - 1, 100.123456789, profile.latency_max_ms + 1]
    before = list(values)
    summary = diagnostic.summarize_samples(values, profile)
    assert values == before
    assert summary["bounds_violations"] == 2
    assert summary["mean_ms"] == statistics.mean(values)
    assert summary["sample_sd_ms"] == statistics.stdev(values)
    assert summary["min_ms"] == min(values)


@pytest.mark.parametrize("values", [[1], [1, float("nan")], [1, float("inf")]])
def test_invalid_samples_fail_explicitly(profile, values):
    with pytest.raises(ValueError, match="finite observations"):
        diagnostic.summarize_samples(values, profile)


def write_fixture_csv(path, mutate=None):
    rows = [{"round_num": index, "participating_clients": 4, "dropped_updates": 1,
             "round_duration_seconds": 2.5} for index in range(1, 21)]
    if mutate:
        mutate(rows)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)


@pytest.mark.parametrize("mutation,message", [
    (lambda rows: rows.pop(), "Incomplete round count"),
    (lambda rows: rows[1].update(round_num=1), "unordered rounds"),
    (lambda rows: rows[1].update(dropped_updates=2), "selected-update counts"),
    (lambda rows: rows[1].update(participating_clients=-1, dropped_updates=6), "selected-update counts"),
    (lambda rows: rows[1].update(round_duration_seconds=float("inf")), "Invalid round duration"),
    (lambda rows: rows[1].update(participating_clients="4.5"), "Invalid diagnostic observation"),
])
def test_incomplete_or_invalid_round_observations_rejected(tmp_path, mutation, message):
    config = diagnostic.ConfigLoader().load(str(ROOT / "config/experiments/multiseed/baseline_fedavg_seed1.yaml"))
    path = tmp_path / "synthetic.csv"
    write_fixture_csv(path, mutation)
    with pytest.raises(ValueError, match=message):
        diagnostic.read_run(path, config)


def test_dropout_is_computed_from_observations(tmp_path):
    config = diagnostic.ConfigLoader().load(str(ROOT / "config/experiments/multiseed/baseline_fedavg_seed1.yaml"))
    path = tmp_path / "synthetic.csv"
    write_fixture_csv(path)
    first = diagnostic.read_run(path, config)
    write_fixture_csv(path, lambda rows: rows[0].update(participating_clients=3, dropped_updates=2))
    second = diagnostic.read_run(path, config)
    summary = diagnostic.dropout_summary([dict(first, experiment="synthetic"), dict(second, experiment="synthetic")])["synthetic"]
    assert summary["attempted_updates"] == 200
    assert summary["dropped_updates"] == 41
    assert summary["mean_percent"] == statistics.mean([20, 21])
    assert summary["sample_sd_pp"] == statistics.stdev([20, 21])


def make_cohort_fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(diagnostic, "EXPERIMENTS", (("baseline", "fedavg", False),))
    monkeypatch.setattr(diagnostic, "COHORT_SEEDS", (1, 2))
    results = tmp_path / "experiments/results/multiseed"
    configs = tmp_path / "config/experiments/multiseed"
    results.mkdir(parents=True)
    configs.mkdir(parents=True)
    template = (ROOT / "config/experiments/multiseed/baseline_fedavg_seed1.yaml").read_text()
    for seed in (1, 2):
        (configs / f"baseline_fedavg_seed{seed}.yaml").write_text(
            template.replace("seed1", f"seed{seed}").replace("seed: 1", f"seed: {seed}"))
        write_fixture_csv(results / f"baseline_fedavg_seed{seed}_metrics.csv")
    return results, configs


@pytest.mark.parametrize("problem", ["missing_seed", "extra_seed", "profile_drift"])
def test_cohort_completeness_and_profile_drift(tmp_path, monkeypatch, problem):
    results, configs = make_cohort_fixture(tmp_path, monkeypatch)
    if problem == "missing_seed":
        (results / "baseline_fedavg_seed2_metrics.csv").unlink()
    elif problem == "extra_seed":
        write_fixture_csv(results / "baseline_fedavg_seed3_metrics.csv")
    else:
        path = configs / "baseline_fedavg_seed2.yaml"
        path.write_text(path.read_text().replace("latency_max_ms: 5", "latency_max_ms: 6"))
    with pytest.raises(ValueError, match="seed cohort|Configuration drift"):
        diagnostic.read_cohort(tmp_path)


def test_all_real_cohort_dropout_statistics_independently():
    runs, profiles, sources = diagnostic.read_cohort()
    assert len(runs) == 110
    assert len(profiles) == 4
    assert len(sources) == 220
    for experiment, summary in diagnostic.dropout_summary(runs).items():
        reference = []
        for run in runs:
            if run["experiment"] != experiment:
                continue
            with run["csv_path"].open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            dropped = sum(int(row["dropped_updates"]) for row in rows)
            attempted = sum(int(row["dropped_updates"]) + int(row["participating_clients"]) for row in rows)
            reference.append(100 * dropped / attempted)
        assert summary["n_seeds"] == len(reference) == 10
        assert summary["mean_percent"] == pytest.approx(statistics.mean(reference), abs=1e-12)
        assert summary["sample_sd_pp"] == pytest.approx(statistics.stdev(reference), abs=1e-12)


def test_superseded_evidence_is_preserved_byte_for_byte():
    _, manifest = diagnostic.superseded_means()
    assert len(manifest) == 3
    for record in manifest.values():
        assert diagnostic.sha256(diagnostic.SUPERSEDED / record["archive_file"]) == record["sha256"]


def test_full_diagnostic_round_trip_provenance_and_no_training(tmp_path, monkeypatch, capsys):
    from src.core.client import FLClient

    def forbidden(*args, **kwargs):
        raise AssertionError("Diagnostic must never train an ML client")

    monkeypatch.setattr(FLClient, "train", forbidden)
    monkeypatch.setattr(diagnostic, "DIAGNOSTICS", tmp_path)
    _, _, source_hashes = diagnostic.read_cohort()
    state, sleep = random.getstate(), asyncio.sleep
    diagnostic.main()
    capsys.readouterr()
    assert random.getstate() == state
    assert asyncio.sleep is sleep
    metadata = json.loads((tmp_path / diagnostic.METADATA_NAME).read_text())
    with (tmp_path / diagnostic.RAW_NAME).open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 10000
    for name, seed in diagnostic.SEEDS.items():
        profile = diagnostic.ConfigLoader().load(str(ROOT / f"config/experiments/multiseed/{name}_fedavg_seed1.yaml")).network_profile
        values = [float(row["propagation_latency_ms"]) for row in rows if row["profile"] == name]
        assert values == diagnostic.sample_engine(profile, seed)
        assert metadata["latency_statistics"][name] == diagnostic.summarize_samples(values, profile)
        reference_mean = math.fsum(values) / len(values)
        reference_sd = math.sqrt(math.fsum((value - reference_mean) ** 2 for value in values) / (len(values) - 1))
        assert metadata["latency_statistics"][name]["mean_ms"] == pytest.approx(reference_mean, abs=1e-12)
        assert metadata["latency_statistics"][name]["sample_sd_ms"] == pytest.approx(reference_sd, abs=1e-12)
        assert all(profile.latency_min_ms <= value <= profile.latency_max_ms for value in values)
    for name, digest in metadata["output_sha256"].items():
        assert diagnostic.sha256(tmp_path / name) == digest
    assert all(diagnostic.sha256(ROOT / path) == digest for path, digest in source_hashes.items())
    assert all(diagnostic.sha256(ROOT / path) == digest for path, digest in metadata["source_sha256"].items())
    with (tmp_path / diagnostic.AUDIT_NAME).open(newline="") as handle:
        assert len(list(csv.DictReader(handle))) == 110
    report = (tmp_path / diagnostic.SUMMARY_NAME).read_text()
    assert "do not isolate impairment overhead" in report
    assert "not recovered propagation samples" in report
