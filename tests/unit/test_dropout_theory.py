"""Independent exact event enumeration and production-lifecycle checks; no ML."""

import asyncio
import ast
import importlib.util
import math
from collections import defaultdict
from fractions import Fraction as F
from itertools import combinations
from pathlib import Path
from types import SimpleNamespace
from typing import List
from unittest.mock import AsyncMock, Mock

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("dropout_diagnostic", ROOT / "scripts/verify_layer3_fidelity.py")
d = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(d)


def exhaustive_labelled_events(N, m, T, D, p, loss):
    """Independent Fraction enumeration: actual expiry times on labelled IDs.

    This does not use the diagnostic's occupancy-state transitions/recurrence.
    """
    states = {(tuple([0] * N), 0): F(1)}
    subsets = list(combinations(range(N), m))
    for round_num in range(1, T + 1):
        following = defaultdict(F)
        for (expiry, total), mass in states.items():
            for selected in subsets:
                branches = {(expiry, total): mass / len(subsets)}
                for client in selected:
                    next_branches = defaultdict(F)
                    for (ends, lost), probability in branches.items():
                        if round_num < ends[client]:
                            next_branches[(ends, lost + 1)] += probability
                            continue
                        reset = list(ends)
                        reset[client] = 0
                        triggered = list(ends)
                        triggered[client] = round_num + D
                        next_branches[(tuple(triggered), lost + 1)] += probability * p
                        next_branches[(tuple(reset), lost + 1)] += probability * (1 - p) * loss
                        next_branches[(tuple(reset), lost)] += probability * (1 - p) * (1 - loss)
                    branches = {key: value for key, value in next_branches.items() if value}
                for key, value in branches.items():
                    following[key] += value
        states = dict(following)
    result = [F(0)] * (m * T + 1)
    for (_, lost), probability in states.items():
        result[lost] += probability
    assert sum(result) == 1
    return result


@pytest.mark.parametrize("N,m,T,D,p,loss", [
    (2, 1, 3, 2, F(1, 3), F(1, 4)),
    (3, 2, 3, 3, F(2, 5), F(1, 5)),
    (2, 2, 4, 2, F(1, 2), F(0)),
    (2, 1, 3, 0, F(1, 3), F(1, 4)),
    (2, 1, 3, 1, F(1, 3), F(1, 4)),
    (2, 1, 3, 5, F(1), F(1)),
])
def test_population_law_against_independent_labelled_enumeration(N, m, T, D, p, loss):
    reference = exhaustive_labelled_events(N, m, T, D, p, loss)
    law = d.finite_dropout_distribution(p, D, loss, N, m, T)
    assert law["pmf"] == pytest.approx([float(value) for value in reference], abs=1e-12)
    mean = sum(F(k, m * T) * probability for k, probability in enumerate(reference))
    variance = sum((F(k, m * T) - mean) ** 2 * probability for k, probability in enumerate(reference))
    assert law["reference"]["finite_mean"] == mean
    assert law["sd"] ** 2 == pytest.approx(float(variance), abs=1e-12)


@pytest.mark.parametrize("p,D,loss", [(0, 3, 0), (1, 3, 0), (0.3, 2, 1)])
def test_deterministic_boundary_distributions(p, D, loss):
    law = d.finite_dropout_distribution(p, D, loss, 10, 5, 20)
    expected_count = 0 if p == loss == 0 else 100
    expected = np.zeros(101)
    expected[expected_count] = 1
    assert law["pmf"] == pytest.approx(expected, abs=1e-12)


@pytest.mark.parametrize("p,D,loss", [(0, 3, 0.2), (0.15, 1, 0.07), (0.15, 0, 0.07)])
def test_no_persistence_cases_reduce_to_binomial(p, D, loss):
    law = d.finite_dropout_distribution(p, D, loss, 10, 5, 20)
    probability = p + (1 - p) * loss
    expected = [math.comb(100, k) * probability ** k * (1 - probability) ** (100 - k)
                for k in range(101)]
    assert law["pmf"] == pytest.approx(expected, abs=1e-12)


def test_initial_round_has_no_inherited_disconnections():
    result = d.marginal_dropout_theory("0.35", 3, "0.20", 10, 5, 1)
    assert result["finite_mean"] == F(12, 25)
    assert result["by_round"][0]["persistent_drop_probability"] == 0


def test_duration_counts_trigger_round_and_global_expiry():
    result = d.marginal_dropout_theory(1, 3, 0, 1, 1, 7)
    eligible = [row["eligible_probability"] for row in result["by_round"]]
    assert eligible == [1, 0, 0, 1, 0, 0, 1]
    assert result["finite_mean"] == 1  # Immediate retrigger on each eligible attempt.


def test_old_formula_is_full_selection_stationary_special_case():
    result = d.marginal_dropout_theory(".15", 2, ".07", 10, 10, 20)
    old_offline = F(2) * F(".15") / (1 + F(".15"))
    assert result["steady_mean"] == old_offline + (1 - old_offline) * F(".07")
    assert result["steady_mean"] == result["old_all_selected_steady"]
    assert result["finite_mean"] != result["steady_mean"]


@pytest.mark.parametrize("parameters", [
    (-0.1, 2, 0, 10, 5, 20), (0.1, -1, 0, 10, 5, 20),
    (0.1, 2, 1.1, 10, 5, 20), (0.1, 2, 0, 10, 11, 20),
    (0.1, 2, 0, 10, 0, 20), (0.1, 2, 0, 10, 5, 0),
])
def test_invalid_theory_parameters_rejected(parameters):
    with pytest.raises(ValueError):
        d.finite_dropout_distribution(*parameters)


def synthetic_engine(p=0.5, D=3, loss=0):
    from src.config import NetworkProfile
    from src.core.client import ModelUpdate
    profile = NetworkProfile("synthetic", 0, 0, 100, loss, p, D)
    update = ModelUpdate(0, {}, 1, 0, 1, model_size_bytes=0)
    return d.impairment.NetworkImpairmentEngine(profile), update


def test_production_persistence_includes_unselected_rounds_and_expiry(monkeypatch):
    engine, update = synthetic_engine(p=1, D=3)
    monkeypatch.setattr(engine, "_apply_latency", AsyncMock(return_value=0))
    assert asyncio.run(engine.transmit(update, 0, 1)) is None
    assert engine._dropout_end[0] == 4
    engine.profile.dropout_probability = 0
    assert asyncio.run(engine.transmit(update, 0, 3)) is None  # No attempt in round 2.
    assert asyncio.run(engine.transmit(update, 0, 4)) is update
    assert 0 not in engine._dropout_end
    assert engine._total_transmissions == 3 and engine._total_drops == 2


def test_production_retrigger_at_expiry_and_zero_duration(monkeypatch):
    engine, update = synthetic_engine(p=1, D=0)
    assert asyncio.run(engine.transmit(update, 0, 1)) is None
    assert engine._dropout_end[0] == 1
    assert asyncio.run(engine.transmit(update, 0, 2)) is None
    assert engine._dropout_end[0] == 2
    engine.profile.dropout_probability = 0
    monkeypatch.setattr(engine, "_apply_latency", AsyncMock(return_value=0))
    assert asyncio.run(engine.transmit(update, 0, 3)) is update


def test_packet_loss_not_checked_or_double_counted_during_disconnection(monkeypatch):
    engine, update = synthetic_engine(p=1, D=3, loss=1)
    packet = Mock(side_effect=AssertionError("Packet check must be skipped"))
    monkeypatch.setattr(engine, "_apply_packet_loss", packet)
    assert asyncio.run(engine.transmit(update, 0, 1)) is None
    assert asyncio.run(engine.transmit(update, 0, 2)) is None
    packet.assert_not_called()
    assert engine._total_transmissions == engine._total_drops == 2


def test_packet_loss_alone_does_not_create_disconnection(monkeypatch):
    engine, update = synthetic_engine(p=0, D=3, loss=1)
    latency = AsyncMock(side_effect=AssertionError("Lost updates cannot reach latency"))
    monkeypatch.setattr(engine, "_apply_latency", latency)
    assert asyncio.run(engine.transmit(update, 0, 1)) is None
    assert engine._dropout_end == {}
    assert engine._total_transmissions == engine._total_drops == 1
    latency.assert_not_called()


@pytest.mark.parametrize("fraction,expected_count", [(0.5, 5), (1.0, 10), (0.01, 1)])
def test_production_selection_body_uses_all_ids_including_disconnected(fraction, expected_count):
    """Execute the unchanged method body without constructing/training a server.

    Extracting this self-contained method avoids importing plotting/ML-runner
    dependencies. The source body is not reimplemented in this test.
    """
    source = ast.parse((ROOT / "src/core/server.py").read_text())
    server = next(node for node in source.body if isinstance(node, ast.ClassDef) and node.name == "FLServer")
    method = next(node for node in server.body if isinstance(node, ast.FunctionDef) and node.name == "select_clients")
    selected = list(range(expected_count))
    sample = Mock(return_value=selected)
    namespace = {"List": List, "random": SimpleNamespace(sample=sample)}
    exec(compile(ast.Module(body=[method], type_ignores=[]), "<production select_clients>", "exec"), namespace)
    fake_server = SimpleNamespace(config=SimpleNamespace(num_clients=10, fraction_fit=fraction),
                                  impairment_engine=SimpleNamespace(_dropout_end={i: 999 for i in range(10)}))
    assert namespace["select_clients"](fake_server, 1) == selected
    sample.assert_called_once_with(list(range(10)), expected_count)


def test_predictions_do_not_depend_on_observed_percentages():
    runs, profiles, _ = d.read_cohort()
    drops = d.dropout_summary(runs)
    original, _, _ = d.compare_dropout_theory(profiles, runs, drops)
    synthetic_drops = {key: dict(value) for key, value in drops.items()}
    synthetic_drops["rural_zambia_fedavg"].update(mean_percent=0, dropped_updates=0)
    changed, _, _ = d.compare_dropout_theory(profiles, runs, synthetic_drops)
    for profile in profiles:
        for key in ("finite_mean_exact_fraction", "predicted_run_sd_pp", "pointwise_95_prediction_interval"):
            assert changed[profile][key] == original[profile][key]


def test_real_profiles_distribution_means_and_discrete_prediction_coverage():
    runs, profiles, _ = d.read_cohort()
    comparisons, pmf_rows, round_rows = d.compare_dropout_theory(profiles, runs, d.dropout_summary(runs))
    assert len(pmf_rows) == 4408 and len(round_rows) == 80
    for name, comparison in comparisons.items():
        assert comparison["N"] == 10 and comparison["m"] == 5 and comparison["T"] == 20
        single = [row["probability"] for row in pmf_rows if row["profile"] == name and row["population"] == "single_run"]
        cohort = [row["probability"] for row in pmf_rows if row["profile"] == name and row["population"] == "ten_seed_mean"]
        assert sum(single) == pytest.approx(1, abs=1e-12)
        assert sum(cohort) == pytest.approx(1, abs=1e-12)
        mean = math.fsum(k / 100 * probability for k, probability in enumerate(single))
        assert mean == pytest.approx(float(F(comparison["finite_mean_exact_fraction"])), abs=1e-12)
        assert comparison["pointwise_95_prediction_interval"]["coverage"] >= 0.95 - 1e-12
        assert comparison["familywise_95_prediction_band"]["coverage"] >= 0.9875 - 1e-12
