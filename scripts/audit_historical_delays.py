"""Read-only historical inputs -> separately labelled decimal-Mbps derivatives.

No production/ML modules are imported or executed. Git evidence and numerical
compatibility must agree before a correction is published. All original CSVs
and existing diagnostic observations remain untouched.
"""

import ast
import copy
import csv
import hashlib
import importlib.metadata
import json
import platform
import statistics
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from decimal import Decimal as D, localcontext
from functools import lru_cache
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "experiments/results/diagnostics/historical-delay-audit"
FIX_COMMIT = "95d79160601af01c0af1a8321915af9833776eaf"
BINARY, DECIMAL = 1048576, 1000000
TOLERANCE_MS = D("0.000001")  # 1 ns: float accumulation/CSV precision, not fitting.
EXPERIMENTS = (
    "baseline_fedavg", "urban_zambia_fedavg", "rural_zambia_fedavg",
    "severe_disruption_fedavg", "rural_zambia_fedprox", "severe_disruption_fedprox",
    "baseline_fedavg_noniid", "rural_zambia_fedavg_noniid", "rural_zambia_fedprox_noniid",
    "severe_disruption_fedavg_noniid", "severe_disruption_fedprox_noniid",
)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def committed_content_match(current, committed):
    """Accept CRLF/LF serialization differences only; never normalize values."""
    if current == committed:
        return "byte_identical"
    if current.replace(b"\r\n", b"\n") == committed.replace(b"\r\n", b"\n"):
        return "line_endings_only"
    raise ValueError("CSV contents differ from committed observations.")


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT)


@lru_cache(maxsize=None)
def historical_source(ref, path):
    return git("show", f"{ref}:{path}")


def method(source, name):
    node = next(node for node in ast.walk(ast.parse(source))
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name)
    node = copy.deepcopy(node)
    if (node.body and isinstance(node.body[0], ast.Expr)
            and isinstance(node.body[0].value, ast.Constant) and isinstance(node.body[0].value.value, str)):
        node.body.pop(0)
    return node


def constant_number(node):
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        return D(str(node.value))
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
        return constant_number(node.left) * constant_number(node.right)
    raise ValueError("Unsupported historical bandwidth-divisor expression.")


def divisor_from_source(source):
    node = method(source, "_apply_bandwidth_throttle")
    assignment = next(n for n in ast.walk(node) if isinstance(n, ast.Assign)
                      and any(isinstance(t, ast.Name) and t.id == "size_megabits" for t in n.targets))
    if not isinstance(assignment.value, ast.BinOp) or not isinstance(assignment.value.op, ast.Div):
        raise ValueError("Unsupported historical bandwidth calculation.")
    value = constant_number(assignment.value.right)
    if value not in (BINARY, DECIMAL):
        raise ValueError(f"Unknown historical divisor: {value}")
    return int(value)


def signature(source, name):
    node = method(source, name)
    if name == "transmit":
        # The September change added a separate propagation-only log; it did
        # not change the total-delay accumulator. No other statements removed.
        node.body = [n for n in node.body if "_propagation_latency_samples" not in ast.dump(n)]
    if name == "_apply_bandwidth_throttle":
        for n in ast.walk(node):
            if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "size_megabits" for t in n.targets):
                n.value.right = ast.Constant(value="DIVISOR")
    return ast.dump(node, include_attributes=False)


@lru_cache(maxsize=None)
def historical_contract(ref):
    checks = [("src/network/impairment.py", name) for name in
              ("transmit", "get_avg_latency", "_apply_latency", "_apply_bandwidth_throttle")]
    checks += [("src/core/server.py", name) for name in ("run", "_collect_updates")]
    checks += [("src/analytics/tracker.py", name) for name in ("export_csv", "generate_summary")]
    hashes, matched = {}, True
    for path, name in checks:
        old = historical_source(ref, path)
        hashes[f"{ref}:{path}"] = digest(old)
        matched = matched and signature(old, name) == signature((ROOT / path).read_bytes(), name)
    # The engine is constructed anew in each server, rather than shared across runs.
    for source in (historical_source(ref, "src/core/server.py"), (ROOT / "src/core/server.py").read_bytes()):
        init = method(source, "__init__")
        fresh = [n for n in ast.walk(init) if isinstance(n, ast.Assign)
                 and any(isinstance(t, ast.Attribute) and t.attr == "impairment_engine" for t in n.targets)]
        expected = ast.parse("NetworkImpairmentEngine(config.network_profile)", mode="eval").body
        matched = matched and len(fresh) == 1 and ast.dump(fresh[0].value) == ast.dump(expected)
    return {"matched": matched, "source_sha256": hashes,
            "divisor": divisor_from_source(historical_source(ref, "src/network/impairment.py"))}


def load_rows(path, config):
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"round_num", "participating_clients", "dropped_updates", "avg_latency_ms", "total_bytes_transmitted"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"Missing delay-audit columns: {path}")
        raw = list(reader)
    if len(raw) != config["num_rounds"]:
        raise ValueError(f"Incomplete rounds: {path}")
    selected = min(config["num_clients"], max(1, int(config["num_clients"] * config["fraction_fit"])))
    rows = []
    for number, row in enumerate(raw, 1):
        k, lost, size = int(row["participating_clients"]), int(row["dropped_updates"]), int(row["total_bytes_transmitted"])
        mean = D(row["avg_latency_ms"])
        if (int(row["round_num"]) != number or min(k, lost, size) < 0 or k + lost != selected
                or (k == 0 and size != 0) or not mean.is_finite() or mean < 0):
            raise ValueError(f"Invalid round/update accounting: {path}, round {number}")
        rows.append({"round_num": number, "successes": k, "bytes": size, "mean": mean})
    return rows


def bandwidth_mean_ms(size, successes, bandwidth, divisor):
    if successes <= 0 or bandwidth <= 0:
        raise ValueError("A bandwidth mean needs successful updates and positive Mbps.")
    return D(8000) * size / (D(divisor) * bandwidth * successes)


def derive_rows(rows, network, divisor):
    """Linear correction from aggregate bytes; individual payload sizes need not match.

    Returned values are algebraic reconstructions at original CSV precision,
    not recovered individual observations or bit-identical rerun results.
    """
    with localcontext() as context:
        context.prec = 50
        bandwidth = D(str(network["bandwidth_mbps"]))
        successes, size, previous_sum = 0, 0, D(0)
        previous_mean = D(0)
        output = []
        for row in rows:
            k, b, mean = row["successes"], row["bytes"], row["mean"]
            successes += k
            size += b
            if not successes and mean != 0:
                raise ValueError("Nonzero cumulative mean before the first successful update.")
            if k == 0 and successes and abs(mean - previous_mean) > TOLERANCE_MS:
                raise ValueError("Cumulative mean changed in a round with no successful updates.")
            weighted_sum = mean * successes
            record = {"round_num": row["round_num"], "successful_updates_this_round": k,
                      "delivered_bytes_this_round": b, "cumulative_successful_updates": successes,
                      "cumulative_delivered_bytes": size, "original_avg_latency_ms": mean,
                      "original_divisor": divisor,
                      "cumulative_status": "measured" if successes else "no_success_yet_placeholder",
                      "round_status": "measured" if k else "no_success_this_round"}
            record.update({key: None for key in (
                "original_cumulative_bandwidth_mean_ms", "derived_cumulative_propagation_mean_ms",
                "decimal_cumulative_bandwidth_mean_ms", "derived_decimal_cumulative_total_mean_ms",
                "reconstructed_original_round_total_mean_ms", "derived_round_propagation_mean_ms",
                "original_round_bandwidth_mean_ms", "decimal_round_bandwidth_mean_ms",
                "derived_decimal_round_total_mean_ms")})
            if successes:
                old = bandwidth_mean_ms(size, successes, bandwidth, divisor)
                new = bandwidth_mean_ms(size, successes, bandwidth, DECIMAL)
                record.update(original_cumulative_bandwidth_mean_ms=old,
                              derived_cumulative_propagation_mean_ms=mean - old,
                              decimal_cumulative_bandwidth_mean_ms=new,
                              derived_decimal_cumulative_total_mean_ms=mean + (new - old))
            if k:
                round_mean = (weighted_sum - previous_sum) / k
                old = bandwidth_mean_ms(b, k, bandwidth, divisor)
                new = bandwidth_mean_ms(b, k, bandwidth, DECIMAL)
                record.update(reconstructed_original_round_total_mean_ms=round_mean,
                              derived_round_propagation_mean_ms=round_mean - old,
                              original_round_bandwidth_mean_ms=old,
                              decimal_round_bandwidth_mean_ms=new,
                              derived_decimal_round_total_mean_ms=round_mean + (new - old))
            previous_sum, previous_mean = weighted_sum, mean
            output.append(record)
        return output


def compatible_divisors(rows, network):
    lower, upper = D(str(network["latency_min_ms"])), D(str(network["latency_max_ms"]))
    candidates = []
    for divisor in (BINARY, DECIMAL):
        derived = derive_rows(rows, network, divisor)
        values = [row[key] for row in derived for key in
                  ("derived_cumulative_propagation_mean_ms", "derived_round_propagation_mean_ms")
                  if row[key] is not None]
        if values and all(lower - TOLERANCE_MS <= value <= upper + TOLERANCE_MS for value in values):
            candidates.append(divisor)
    return candidates


def audit_run(csv_path, config_path):
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    rows = load_rows(csv_path, config)
    network = config["network_profile"]
    candidates = compatible_divisors(rows, network)
    evidence = {"csv_path": csv_path.relative_to(ROOT).as_posix(),
                "config_path": config_path.relative_to(ROOT).as_posix(),
                "csv_sha256": digest(csv_path.read_bytes()), "config_sha256": digest(config_path.read_bytes()),
                "numerically_compatible_divisors": candidates}
    try:
        evidence["last_content_commit"] = git("log", "-1", "--format=%H", "--", evidence["csv_path"]).decode().strip()
        ref = evidence["last_content_commit"]
        if not ref:
            raise ValueError("CSV lacks committed content history.")
        committed = historical_source(ref, evidence["csv_path"])
        evidence.update(committed_csv_sha256=digest(committed),
                        committed_content_match=committed_content_match(csv_path.read_bytes(), committed))
        contract = historical_contract(ref)
        old_config = yaml.safe_load(historical_source(ref, evidence["config_path"]))
        needed = ("network_profile", "num_rounds", "num_clients", "fraction_fit", "dataset", "seed", "algorithm")
        config_matches = all(old_config[key] == config[key] for key in needed)
        evidence.update(historical_source_contract=contract,
                        historical_config_matches=config_matches,
                        historical_config_sha256=digest(historical_source(ref, evidence["config_path"])))
        if not contract["matched"] or not config_matches:
            raise ValueError("Historical metric contract or configuration differs.")
        if candidates != [contract["divisor"]]:
            raise ValueError("Stored values do not uniquely corroborate the historical source divisor.")
    except (ValueError, KeyError, StopIteration, subprocess.CalledProcessError) as exc:
        evidence.update(status="unresolved_no_derived_correction", reason=str(exc))
        return evidence, []
    evidence.update(status="supported_reconstruction", divisor=candidates[0],
                    convention="legacy_binary" if candidates[0] == BINARY else "decimal",
                    provenance_limit="Commit/source evidence plus unique numerical compatibility; no attested execution-environment manifest.")
    derived = derive_rows(rows, network, candidates[0])
    for row in derived:
        row.update(experiment=config["experiment_name"].rsplit("_seed", 1)[0], seed=config["seed"])
    final = derived[-1]
    evidence["summary"] = {
        "experiment": derived[-1]["experiment"], "seed": config["seed"],
        "successful_updates": final["cumulative_successful_updates"],
        "delivered_bytes": final["cumulative_delivered_bytes"],
        "original_final_total_mean_ms": str(final["original_avg_latency_ms"]),
        "derived_decimal_final_total_mean_ms": str(final["derived_decimal_cumulative_total_mean_ms"]),
        "derived_final_propagation_mean_ms": str(final["derived_cumulative_propagation_mean_ms"]),
        "original_final_bandwidth_mean_ms": str(final["original_cumulative_bandwidth_mean_ms"]),
        "decimal_final_bandwidth_mean_ms": str(final["decimal_cumulative_bandwidth_mean_ms"]),
        "original_mean_of_cumulative_rows_ms": str(statistics.mean(row["original_avg_latency_ms"] for row in derived)),
        "decimal_tracker_mean_of_cumulative_rows_ms": str(statistics.mean(
            row["derived_decimal_cumulative_total_mean_ms"] if row["derived_decimal_cumulative_total_mean_ms"] is not None else D(0)
            for row in derived)),
        "zero_success_rounds": sum(row["successful_updates_this_round"] == 0 for row in derived),
    }
    return evidence, derived


def summarize_runs(evidence):
    groups = []
    for experiment in EXPERIMENTS:
        members = [item["summary"] for item in evidence if "summary" in item and item["summary"]["experiment"] == experiment]
        if not members:
            continue
        old = [float(item["original_final_total_mean_ms"]) for item in members]
        new = [float(item["derived_decimal_final_total_mean_ms"]) for item in members]
        propagation = [float(item["derived_final_propagation_mean_ms"]) for item in members]
        total_successes = sum(item["successful_updates"] for item in members)
        groups.append({"experiment": experiment, "n_runs": len(members),
                       "convention": ",".join(sorted({item["convention"] for item in evidence if item.get("summary", {}).get("experiment") == experiment})),
                       "original_mean_final_total_ms": statistics.mean(old),
                       "original_sample_sd_final_total_ms": statistics.stdev(old) if len(old) > 1 else None,
                       "derived_decimal_mean_final_total_ms": statistics.mean(new),
                       "derived_decimal_sample_sd_final_total_ms": statistics.stdev(new) if len(new) > 1 else None,
                       "derived_mean_final_propagation_ms": statistics.mean(propagation),
                       "original_mean_final_bandwidth_ms": statistics.mean(float(item["original_final_bandwidth_mean_ms"]) for item in members),
                       "decimal_mean_final_bandwidth_ms": statistics.mean(float(item["decimal_final_bandwidth_mean_ms"]) for item in members),
                       "mean_correction_ms": statistics.mean(n - o for n, o in zip(new, old)),
                       "original_pooled_transmission_mean_ms": sum(float(item["original_final_total_mean_ms"]) * item["successful_updates"] for item in members) / total_successes,
                       "derived_decimal_pooled_transmission_mean_ms": sum(float(item["derived_decimal_final_total_mean_ms"]) * item["successful_updates"] for item in members) / total_successes})
    return groups


def write_csv(path, records):
    if not records:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def build_audit(evidence, groups, derived):
    counts = Counter(item.get("convention", "unresolved") for item in evidence)
    lines = ["# Historical transmission-delay audit (Correction 14)", "",
             f"110 existing runs checked: {counts['legacy_binary']} legacy-binary, {counts['decimal']} decimal, {counts['unresolved']} unresolved.",
             f"The divisor fix is commit {FIX_COMMIT}; classification also requires unique numerical compatibility with recorded profile bounds.",
             "Git content history is corroborating evidence, not a signed execution manifest. No classification is made from timestamps alone.", "",
             f"{sum(item.get('committed_content_match') == 'line_endings_only' for item in evidence)} CSVs differ from their committed copies only in CRLF/LF line endings; all other bytes match. Both hashes are preserved in provenance.json.", "",
             "## What the saved metric means", "",
             "`avg_latency_ms` is the cumulative, successful-update-weighted mean of propagation plus analytical bandwidth delay through that round.",
             "It excludes failed updates, local training, model evaluation, protocol overhead, and a separately modelled download/broadcast phase.",
             "The engine is created per run and is not reset each round. A round with no success carries the previous mean; a prefix with no successes has a zero placeholder.",
             "Source trace: FLClient.train computes model_size_bytes from tensor element counts and sizes; FLServer._collect_updates retains only non-None engine returns; FLServer.run sums those delivered bytes and reads get_avg_latency; AnalyticsTracker.export_csv writes the recorded values without a unit conversion.",
             "NetworkImpairmentEngine.transmit checks dropout and packet loss before bandwidth and propagation, then appends their sum only on success. _apply_bandwidth_throttle is analytical; _apply_latency sleeps. Bandwidth normalization therefore changes no execution timing or update selection.",
             "The tracker's summary averages these cumulative row means equally. That is not the run's per-transmission mean; the final CSV row supplies the latter.", "",
             "per_run_delay_audit.csv retains the original and decimal-derived tracker-style averages as separate fields. Only that tracker-style calculation retains initial zero placeholders, matching the original implementation; derived observation columns leave unobserved means blank.", "",
             "## Derivation", "",
             "Let S_r be cumulative successful updates, C_r cumulative delivered bytes, w the fixed Mbps profile value, d the historical divisor, and M_r the saved cumulative mean.",
             "For S_r>0: B_r(d)=8000*C_r/(d*w*S_r), P_r=M_r-B_r(d), and M_r(decimal)=M_r+B_r(1000000)-B_r(d).",
             "For k_r>0 successes in a round, recover its total-delay mean as (S_r*M_r-S_(r-1)*M_(r-1))/k_r; use that round's delivered bytes for its bandwidth correction.",
             "Bandwidth is linear in bytes: aggregate delivered byte totals suffice even with variable individual payload sizes. Constant per-client sizes are not assumed.",
             "Do not multiply the mixed propagation-plus-bandwidth mean by 1.048576. A no-success current-round mean remains missing, not zero.",
             f"Decimal arithmetic uses precision 50; feasibility tolerance is {TOLERANCE_MS} ms for historical floating-point accumulation. Values are never clamped or adjusted to fit.",
             "Reconstruction is algebraic at saved CSV precision, not recovery of individual propagation draws or a bit-identical rerun.", "",
             "## Cohort figures", "",
             "Each mean below gives equal weight to ten run-final means. Units are milliseconds. CSV summaries additionally contain sample SDs and separately labelled transmission-pooled means.", "",
             "| Experiment | Runs | Original bandwidth mean | Decimal bandwidth mean | Original total mean | Decimal-derived total mean | Change |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for row in groups:
        lines.append(f"| {row['experiment']} | {row['n_runs']} | {row['original_mean_final_bandwidth_ms']:.6f} | {row['decimal_mean_final_bandwidth_ms']:.6f} | {row['original_mean_final_total_ms']:.6f} | {row['derived_decimal_mean_final_total_ms']:.6f} | {row['mean_correction_ms']:+.6f} |")
    lines.extend(["", f"There are {sum(row['successful_updates_this_round'] == 0 for row in derived)} no-success rounds among supported runs; their current-round delay means are left blank.",
                  "Existing decimal-convention cumulative metrics receive zero correction.", "",
                  "## Report changes to make later", "",
                  "* Chapter 3, section 3.3: define the metric as a cumulative mean over successful uploads; disclose the historical binary divisor and the separate decimal-derived outputs. Keep binary MiB for payload volumes.",
                  "* Chapter 4, sections 4.2/4.4: explain the accumulator and state that a passing current regression test does not retroactively change historical measurements.",
                  "* Chapter 5, tables/figures using `avg_latency_ms`: label original-convention values or use the explicitly derived decimal-normalized values. Do not mix the two baseline cohorts without this disclosure.",
                  "* Chapter 5, section 5.5: analytical bandwidth correction changes no recorded wall-clock duration or throughput. CPU-only timing claims need the separate timing audit.",
                  "* Chapter 5, section 5.6: pure-propagation diagnostics from Task 2 are distinct evidence and must not be replaced by cumulative total-delay means. Recovered aggregate propagation means are not raw draws.",
                  "* Chapter 5, section 5.7: delivered byte totals and binary-MiB payload figures need no correction.",
                  "* Chapter 6: normalized transmission-delay figures are derived analytical quantities, not observed cellular transfer times or evidence of changed model accuracy, dropout, arrivals or deadlines.",
                  "* Appendix B: retain the independently computed decimal-Mbps regression example. Appendix C: document this audit runner and separate derived-output paths alongside the frozen original CSVs.", "",
                  "Suggested disclosure: 'The original 100-run cohort records analytical transmission delay using a binary megabit divisor (1,048,576), whereas the ten matched non-IID Baseline runs use decimal Mbps (1,000,000 bits/s). Decimal-normalized values were derived separately from recorded successful-update counts and delivered payload bytes, retaining the inferred propagation component. The CSV field avg_latency_ms is a cumulative mean over successful uploads; it is not a per-round propagation mean or measured network transfer time. Original observations remain unchanged.'", "",
                  "## Reproduce", "",
                  "Dependencies: Python 3 and PyYAML; the audit does not import the simulator, Torch, or training code. Run from the repository root:", "",
                  "```sh", "python3 scripts/audit_historical_delays.py",
                  "python3 -m pytest -q -p no:cacheprovider tests/unit/test_historical_delay_audit.py", "```", "",
                  "The runner reads existing CSVs/configs and Git objects, then writes only this separate output directory. Tests use synthetic observations and temporary output directories, plus read-only validation of all 2,200 historical rows.", "",
                  "## Limits", "",
                  "Individual latency samples, individual payload histories, failed-attempt byte totals and protocol overhead are not recoverable. Only the supported aggregate means are reconstructed.",
                  "The recorded payload is tensor-parameter bytes (element count times element size), not a measured serialized wire payload. The reconstructed delay retains that original payload convention.",
                  "The correction assumes the inspected accumulator, one fixed bandwidth per run, and matching historical YAML. Historical source, config and CSV hashes are recorded in provenance.json.",
                  "The final thesis, AI log and all original observations are unchanged. No ML experiments, commits, pushes or tags are part of this audit."])
    for item in evidence:
        if item["status"] != "supported_reconstruction":
            lines.append(f"Unresolved: {item['csv_path']}: {item['reason']}")
    return "\n".join(lines) + "\n"


def main():
    evidence, derived, input_hashes = [], [], {}
    for experiment in EXPERIMENTS:
        subdir = "noniid" if experiment.endswith("_noniid") else ""
        directory = ROOT / "experiments/results/multiseed" / subdir
        expected = {directory / f"{experiment}_seed{seed}_metrics.csv" for seed in range(1, 11)}
        if set(directory.glob(f"{experiment}_seed*_metrics.csv")) != expected:
            raise ValueError(f"Incomplete or extra seed cohort: {experiment}")
        for seed in range(1, 11):
            csv_path = directory / f"{experiment}_seed{seed}_metrics.csv"
            config_path = ROOT / "config/experiments/multiseed" / subdir / f"{experiment}_seed{seed}.yaml"
            item, records = audit_run(csv_path, config_path)
            evidence.append(item)
            derived.extend(records)
            input_hashes[item["csv_path"]] = item["csv_sha256"]
            input_hashes[item["config_path"]] = item["config_sha256"]
    groups = summarize_runs(evidence)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    write_csv(OUTPUT / "per_round_derived_delays.csv", derived)
    write_csv(OUTPUT / "per_run_delay_audit.csv", [dict(item["summary"], convention=item["convention"], divisor=item["divisor"], last_content_commit=item["last_content_commit"])
                                                for item in evidence if "summary" in item])
    write_csv(OUTPUT / "cohort_delay_summary.csv", groups)
    report = build_audit(evidence, groups, derived)
    (OUTPUT / "AUDIT.md").write_text(report, encoding="utf-8")
    source_paths = ("scripts/audit_historical_delays.py", "tests/unit/test_historical_delay_audit.py",
                    "src/network/impairment.py", "src/core/server.py", "src/core/client.py", "src/analytics/tracker.py")
    provenance = {"generated_at_utc": datetime.now(timezone.utc).isoformat(),
                  "git_head": git("rev-parse", "HEAD").decode().strip(), "fix_commit": FIX_COMMIT,
                  "runtime": {"python": platform.python_version(), "executable": sys.executable,
                              "platform": platform.platform(), "PyYAML": importlib.metadata.version("PyYAML")},
                  "decimal_precision": 50, "numerical_feasibility_tolerance_ms": str(TOLERANCE_MS),
                  "inputs_sha256": input_hashes, "source_sha256": {path: digest((ROOT / path).read_bytes()) for path in source_paths},
                  "runs": evidence, "outputs_sha256": {path.name: digest(path.read_bytes()) for path in OUTPUT.iterdir() if path.is_file() and path.name != "provenance.json"}}
    (OUTPUT / "provenance.json").write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
