"""Unit examples and accounting identities; no ML or writes to original data."""

import csv
import importlib.util
import json
import subprocess
from collections import Counter
from decimal import Decimal as D, localcontext
from fractions import Fraction as F
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("delay_audit", ROOT / "scripts/audit_historical_delays.py")
a = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(a)


def test_independently_computed_decimal_example_and_mixed_delay():
    # 524288*8 / 2000000 seconds: independently known exact decimal value.
    assert a.bandwidth_mean_ms(524288, 1, D(2), a.DECIMAL) == D("2097.152")
    assert a.bandwidth_mean_ms(524288, 1, D(2), a.BINARY) == D("2000")
    raw = [{"round_num": 1, "successes": 1, "bytes": 524288, "mean": D("2123.456")}]
    result = a.derive_rows(raw, {"bandwidth_mbps": 2}, a.BINARY)[0]
    assert result["derived_cumulative_propagation_mean_ms"] == D("123.456")
    assert result["derived_decimal_cumulative_total_mean_ms"] == D("2220.608")
    assert result["derived_decimal_cumulative_total_mean_ms"] != D("2123.456") * D("1.048576")


def test_variable_payloads_and_cumulative_weighting_against_individual_inputs():
    # Independently sum three individual propagation/payload pairs. Their sizes
    # differ both within and between rounds; no constant-size assumption works.
    payloads = [100, 300, 2000]
    propagation = [F(10), F(30), F(50)]
    old = [p + F(size * 8000, a.BINARY * 2) for size, p in zip(payloads, propagation)]
    new = [p + F(size * 8000, a.DECIMAL * 2) for size, p in zip(payloads, propagation)]
    with localcontext() as context:
        context.prec = 50
        dec = lambda value: D(value.numerator) / D(value.denominator)
        rows = [{"round_num": 1, "successes": 2, "bytes": 400, "mean": dec(sum(old[:2]) / 2)},
                {"round_num": 2, "successes": 0, "bytes": 0, "mean": dec(sum(old[:2]) / 2)},
                {"round_num": 3, "successes": 1, "bytes": 2000, "mean": dec(sum(old) / 3)}]
        derived = a.derive_rows(rows, {"bandwidth_mbps": 2}, a.BINARY)
        assert derived[-1]["derived_decimal_cumulative_total_mean_ms"] == dec(sum(new) / 3)
        assert derived[0]["derived_decimal_round_total_mean_ms"] == dec(sum(new[:2]) / 2)
        assert derived[-1]["derived_decimal_round_total_mean_ms"] == dec(new[-1])
        assert derived[-1]["derived_cumulative_propagation_mean_ms"] == D(30)
        assert derived[1]["derived_decimal_round_total_mean_ms"] is None
        assert derived[1]["derived_decimal_cumulative_total_mean_ms"] == derived[0]["derived_decimal_cumulative_total_mean_ms"]


def test_no_success_prefix_is_a_placeholder_not_a_latency_observation():
    rows = [{"round_num": 1, "successes": 0, "bytes": 0, "mean": D(0)},
            {"round_num": 2, "successes": 1, "bytes": 524288, "mean": D(2010)}]
    derived = a.derive_rows(rows, {"bandwidth_mbps": 2}, a.BINARY)
    assert derived[0]["cumulative_status"] == "no_success_yet_placeholder"
    assert derived[0]["derived_decimal_cumulative_total_mean_ms"] is None
    assert derived[0]["derived_decimal_round_total_mean_ms"] is None
    assert derived[1]["derived_decimal_cumulative_total_mean_ms"] == D("2107.152")


def test_decimal_metrics_receive_zero_correction():
    rows = [{"round_num": 1, "successes": 1, "bytes": 524288, "mean": D("2120.152")}]
    derived = a.derive_rows(rows, {"bandwidth_mbps": 2}, a.DECIMAL)[0]
    assert derived["derived_decimal_cumulative_total_mean_ms"] == rows[0]["mean"]
    assert derived["derived_cumulative_propagation_mean_ms"] == D(23)


def test_missing_success_does_not_allow_a_changed_cumulative_mean():
    rows = [{"round_num": 1, "successes": 1, "bytes": 524288, "mean": D(2010)},
            {"round_num": 2, "successes": 0, "bytes": 0, "mean": D(2020)}]
    with pytest.raises(ValueError, match="changed in a round"):
        a.derive_rows(rows, {"bandwidth_mbps": 2}, a.BINARY)


def test_nonzero_mean_without_any_success_is_rejected():
    with pytest.raises(ValueError, match="before the first"):
        a.derive_rows([{"round_num": 1, "successes": 0, "bytes": 0, "mean": D(10)}],
                      {"bandwidth_mbps": 2}, a.BINARY)


def test_ambiguous_divisor_is_not_forced_to_an_expected_value():
    rows = [{"round_num": 1, "successes": 1, "bytes": 100, "mean": D(100)}]
    assert a.compatible_divisors(rows, {"bandwidth_mbps": 100, "latency_min_ms": 1,
                                      "latency_max_ms": 1000}) == [a.BINARY, a.DECIMAL]


def test_real_git_fix_and_accumulator_contracts():
    before = a.historical_source(a.FIX_COMMIT + "^", "src/network/impairment.py")
    after = a.historical_source(a.FIX_COMMIT, "src/network/impairment.py")
    assert a.divisor_from_source(before) == a.BINARY
    assert a.divisor_from_source(after) == a.DECIMAL
    assert a.signature(before, "_apply_bandwidth_throttle") == a.signature(after, "_apply_bandwidth_throttle")
    assert a.historical_contract(a.FIX_COMMIT)["matched"]


def test_line_endings_are_disclosed_but_observation_changes_rejected():
    assert a.committed_content_match(b"x\r\n1\r\n", b"x\n1\n") == "line_endings_only"
    assert a.committed_content_match(b"x\n1\n", b"x\n1\n") == "byte_identical"
    with pytest.raises(ValueError, match="differ from committed"):
        a.committed_content_match(b"x\r\n2\r\n", b"x\n1\n")
    # Columns, round identities and membership are evidence, not formatting.
    original = b"round_num,experiment,value\n1,run_a,2.5\n"
    for changed in [b"round_num,experiment,result\n1,run_a,2.5\n",
                    b"round_num,experiment,value\n2,run_a,2.5\n",
                    b"round_num,experiment,value\n1,run_b,2.5\n",
                    b"round_num,experiment,value\n1,run_a,2.6\n"]:
        with pytest.raises(ValueError, match="differ from committed"):
            a.committed_content_match(changed, original)


@pytest.mark.parametrize("problem", ["missing_bytes", "bad_round", "bad_counts", "nonfinite", "zero_success_bytes"])
def test_bad_raw_accounting_is_rejected(tmp_path, problem):
    row = {"round_num": 1, "participating_clients": 1, "dropped_updates": 0,
           "avg_latency_ms": "2010", "total_bytes_transmitted": 524288}
    if problem == "missing_bytes":
        row.pop("total_bytes_transmitted")
    elif problem == "bad_round":
        row["round_num"] = 2
    elif problem == "bad_counts":
        row["dropped_updates"] = 1
    elif problem == "nonfinite":
        row["avg_latency_ms"] = "NaN"
    else:
        row.update(participating_clients=0, dropped_updates=1)
    path = tmp_path / "synthetic.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    with pytest.raises(ValueError):
        a.load_rows(path, {"num_rounds": 1, "num_clients": 1, "fraction_fit": 1})


def test_source_evidence_without_unique_numerical_support_is_insufficient(monkeypatch):
    monkeypatch.setattr(a, "compatible_divisors", lambda *args: [a.BINARY, a.DECIMAL])
    evidence, rows = a.audit_run(ROOT / "experiments/results/multiseed/baseline_fedavg_seed1_metrics.csv",
                               ROOT / "config/experiments/multiseed/baseline_fedavg_seed1.yaml")
    assert evidence["status"] == "unresolved_no_derived_correction"
    assert rows == []


def test_unsupported_historical_contract_produces_no_correction(monkeypatch):
    monkeypatch.setattr(a, "historical_contract", lambda ref: {"matched": False, "divisor": a.BINARY})
    evidence, rows = a.audit_run(ROOT / "experiments/results/multiseed/baseline_fedavg_seed1_metrics.csv",
                               ROOT / "config/experiments/multiseed/baseline_fedavg_seed1.yaml")
    assert evidence["status"] == "unresolved_no_derived_correction"
    assert rows == []


def test_full_audit_accounting_identities_provenance_and_input_preservation(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(a, "OUTPUT", tmp_path)
    a.main()
    capsys.readouterr()
    metadata = json.loads((tmp_path / "provenance.json").read_text())
    assert len(metadata["runs"]) == 110
    assert Counter(run.get("convention") for run in metadata["runs"]) == {"legacy_binary": 100, "decimal": 10}
    assert all(run["status"] == "supported_reconstruction" for run in metadata["runs"])
    # Independently compare every input with its actual historical Git bytes.
    # LF checkouts and the original CRLF working copy need not have the same
    # count of serialization differences. No other byte difference is accepted.
    expected_paths = {
        str(Path("experiments/results/multiseed") /
            ("noniid" if name.endswith("_noniid") else "") /
            f"{name}_seed{seed}_metrics.csv")
        for name in a.EXPERIMENTS for seed in range(1, 11)
    }
    observed_paths = [run["csv_path"] for run in metadata["runs"]]
    assert len(observed_paths) == len(set(observed_paths)) == 110
    assert set(observed_paths) == expected_paths
    for run in metadata["runs"]:
        current = (ROOT / run["csv_path"]).read_bytes()
        committed = subprocess.check_output(
            ["git", "show", f"{run['last_content_commit']}:{run['csv_path']}"], cwd=ROOT)
        assert current.replace(b"\r\n", b"\n") == committed.replace(b"\r\n", b"\n")
        expected_match = "byte_identical" if current == committed else "line_endings_only"
        assert run["committed_content_match"] == expected_match
        assert a.digest(current) == run["csv_sha256"]
        assert a.digest(committed) == run["committed_csv_sha256"]
    for path, checksum in {**metadata["inputs_sha256"], **metadata["source_sha256"]}.items():
        assert a.digest((ROOT / path).read_bytes()) == checksum
    for name, checksum in metadata["outputs_sha256"].items():
        assert a.digest((tmp_path / name).read_bytes()) == checksum
    with (tmp_path / "per_round_derived_delays.csv").open(newline="") as handle:
        derived = list(csv.DictReader(handle))
    assert len(derived) == 2200
    assert sum(row["round_status"] == "no_success_this_round" for row in derived) == 60
    with localcontext() as context:
        context.prec = 60
        for run in metadata["runs"]:
            with (ROOT / run["csv_path"]).open(newline="") as handle:
                original = list(csv.DictReader(handle))
            config = a.yaml.safe_load((ROOT / run["config_path"]).read_text())
            bandwidth = F(str(config["network_profile"]["bandwidth_mbps"]))
            experiment, seed = run["summary"]["experiment"], run["summary"]["seed"]
            rows = [row for row in derived if row["experiment"] == experiment and int(row["seed"]) == seed]
            count, size, reconstructed_sum, tracker_sum = 0, 0, D(0), D(0)
            for raw, result in zip(original, rows):
                k = int(raw["participating_clients"])
                count += k
                size += int(raw["total_bytes_transmitted"])
                old = F(raw["avg_latency_ms"])
                corrected = old + F(8000 * size, count) / bandwidth * (F(1, a.DECIMAL) - F(1, run["divisor"]))
                expected = D(corrected.numerator) / D(corrected.denominator)
                tracker_sum += expected
                assert abs(D(result["derived_decimal_cumulative_total_mean_ms"]) - expected) < D("1e-38")
                if k:
                    reconstructed_sum += k * D(result["derived_decimal_round_total_mean_ms"])
                else:
                    assert result["derived_decimal_round_total_mean_ms"] == ""
                assert abs(reconstructed_sum / count - expected) < D("1e-38")
                if run["divisor"] == a.DECIMAL:
                    assert D(result["derived_decimal_cumulative_total_mean_ms"]) == D(raw["avg_latency_ms"])
            assert abs(D(run["summary"]["decimal_tracker_mean_of_cumulative_rows_ms"]) - tracker_sum / len(original)) < D("1e-20")
    assert "not a signed execution manifest" in (tmp_path / "AUDIT.md").read_text()
