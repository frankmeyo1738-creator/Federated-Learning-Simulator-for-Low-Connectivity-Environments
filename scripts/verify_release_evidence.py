"""Verify release-checkout evidence against preserved pre-9A.1 records.

No training, downloading, output calibration or historical-hash rewriting.
Run the three documented analysis scripts first, then this verifier. All inputs
are project-relative or historical Git objects; old executable paths are data.
"""
import argparse
import csv
import hashlib
import importlib.metadata
import io
import json
import platform
import re
import subprocess
import sys
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = Path("experiments/results/release-verification/pre-9a1-working-tree")
OUTPUT = Path("experiments/results/release-verification/checkout_verification.json")
PROVENANCE = (
    "experiments/results/diagnostics/layer3_provenance.json",
    "experiments/results/diagnostics/historical-delay-audit/provenance.json",
)
STATISTICS = "experiments/results/statistical_summary.txt"
EXPERIMENTS = (
    "baseline_fedavg", "urban_zambia_fedavg", "rural_zambia_fedavg",
    "severe_disruption_fedavg", "rural_zambia_fedprox", "severe_disruption_fedprox",
    "baseline_fedavg_noniid", "rural_zambia_fedavg_noniid", "rural_zambia_fedprox_noniid",
    "severe_disruption_fedavg_noniid", "severe_disruption_fedprox_noniid",
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def safe_path(root, relative):
    p = Path(relative)
    require(not p.is_absolute() and ".." not in p.parts, "Unsafe evidence path: " + str(relative))
    resolved = (root / p).resolve()
    require(resolved.is_relative_to(root.resolve()), "Evidence path escapes repository: " + str(relative))
    return resolved


def git_bytes(root, ref, path):
    require(re.fullmatch(r"[0-9a-f]{40}", ref) is not None, "Historical ref must be a full commit ID")
    safe_path(root, path)
    return subprocess.check_output(["git", "show", ref + ":" + path], cwd=root)


def expected_paths():
    return {str(Path("experiments/results/multiseed") /
                ("noniid" if name.endswith("_noniid") else "") /
                f"{name}_seed{seed}_metrics.csv")
            for name in EXPERIMENTS for seed in range(1, 11)}


def verify_input_bytes(current, committed, record):
    """Accept only either of the two previously recorded byte identities."""
    require(digest(committed) == record["committed_sha256"], "Historical Git bytes changed")
    require(digest(current) in {record["original_working_tree_sha256"], record["committed_sha256"]},
            "Input differs from both recorded serializations")
    require(current.replace(b"\r\n", b"\n") == committed.replace(b"\r\n", b"\n"),
            "Input observations, columns or identities changed")
    return "historical_committed_bytes" if current == committed else "recorded_working_tree_CRLF_variant"


def compare_text(old, new, metadata_prefixes=()):
    a, b = old.splitlines(), new.splitlines()
    require(len(a) == len(b), "Generated text line count changed")
    for index, (x, y) in enumerate(zip(a, b), 1):
        if x == y:
            continue
        require(any(x.startswith(p) and y.startswith(p) for p in metadata_prefixes),
                "Scientific text changed at line " + str(index))


def compare_csv(old, new, tolerance="0", floating_fields=None):
    """Exact schema/order/text; bounded Decimal differences in named fields only."""
    a = list(csv.reader(io.StringIO(old)))
    b = list(csv.reader(io.StringIO(new)))
    require(len(a) == len(b) and a and a[0] == b[0], "CSV schema or row count changed")
    diffs = []
    for row, (x, y) in enumerate(zip(a[1:], b[1:]), 2):
        require(len(x) == len(y) == len(a[0]), "CSV row width changed")
        for field, left, right in zip(a[0], x, y):
            if left == right:
                continue
            require(floating_fields is not None and field in floating_fields,
                    "Exact CSV field changed: " + field)
            try:
                delta = abs(Decimal(left) - Decimal(right))
            except InvalidOperation as exc:
                raise ValueError("Non-numeric comparison field") from exc
            require(delta.is_finite() and delta <= Decimal(tolerance), "CSV precision bound exceeded: " + field)
            diffs.append({"row": row, "field": field, "preserved": left,
                          "regenerated": right, "absolute_difference": str(delta)})
    return diffs


def compare_statistics(old, new, path=""):
    """Only float64 roundoff is allowed; keys, labels, counts and booleans are exact."""
    if isinstance(old, dict):
        require(isinstance(new, dict) and old.keys() == new.keys(), "Statistic keys changed: " + path)
        return sum((compare_statistics(old[k], new[k], path + "/" + k) for k in old), [])
    if isinstance(old, list):
        require(isinstance(new, list) and len(old) == len(new), "Statistic list changed: " + path)
        return sum((compare_statistics(x, y, path + "/" + str(i)) for i, (x, y) in enumerate(zip(old, new))), [])
    if type(old) in (int, bool, str) or old is None:
        require(type(new) is type(old) and old == new, "Statistic identity changed: " + path)
        return []
    if type(old) is type(new) and old == new:
        return []
    require(isinstance(old, float) and isinstance(new, float), "Statistic identity changed: " + path)
    delta = abs(Decimal(str(old)) - Decimal(str(new)))
    require(delta.is_finite() and delta <= Decimal("1e-10"), "Statistic precision bound exceeded: " + path)
    return [{"field": path, "preserved": old, "regenerated": new, "absolute_difference": str(delta)}]


def verify(root=ROOT, strict_checkout=False):
    frozen = safe_path(root, SNAPSHOT)
    manifest = json.loads((frozen / "manifest.json").read_text())
    files = manifest["files"]
    for p, record in files.items():
        require(digest(safe_path(frozen, record["snapshot"]).read_bytes()) == record["sha256"],
                "Frozen evidence hash mismatch: " + p)
    inputs = {x["path"]: x for x in manifest["cohort_inputs"]}
    require(len(manifest["cohort_inputs"]) == 110 and set(inputs) == expected_paths(), "Cohort membership changed")
    actual = {str(p.relative_to(root)) for p in (root / "experiments/results/multiseed").rglob("*_metrics.csv")
              if "pre-seeding-fix-archive" not in p.parts}
    require(actual == set(inputs), "Missing or unexpected cohort experiment")
    layouts, raw_hashes, committed_hashes = {}, [], []
    committed = {}
    for p, record in inputs.items():
        raw = safe_path(root, p).read_bytes()
        committed[p] = git_bytes(root, record["historical_commit"], p)
        layouts[p] = verify_input_bytes(raw, committed[p], record)
        require(not strict_checkout or raw == committed[p], "Strict checkout contains a working-tree variant: " + p)
        require(digest(safe_path(root, record["config_path"]).read_bytes()) == record["config_sha256"], "Original config changed")
        raw_hashes.append((Path(p).name, digest(raw)))
        committed_hashes.append((Path(p).name, digest(committed[p])))

    def frozen_bytes(p):
        return safe_path(frozen, files[p]["snapshot"]).read_bytes()

    def check_sources(metadata, historical=False):
        for p, checksum in {**metadata.get("source_sha256", {}), **metadata.get("inputs_sha256", {})}.items():
            data = frozen_bytes(p) if historical and p in files else safe_path(root, p).read_bytes()
            if digest(data) != checksum and not historical and p in inputs:
                # Explicitly reported serialization equivalence, never numerical normalization.
                data = committed[p]
            require(digest(data) == checksum, "Source/input hash mismatch: " + p)

    output_hashes, source_paths = {}, {".gitattributes", ".gitignore", "scripts/verify_release_evidence.py",
                                      "tests/unit/test_release_evidence.py", "README.md", "docs/REPRODUCIBILITY.md",
                                      "docs/RESULTS.md", "docs/AI_USAGE_LOG.md", "docs/DOCUMENTATION_VERIFICATION.md",
                                      "docs/RELEASE_VERIFICATION.md"}
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    producer_heads = {}
    json_differences, csv_differences = {}, {}
    for p in PROVENANCE:
        old = json.loads(frozen_bytes(p))
        new = json.loads(safe_path(root, p).read_text())
        producer_heads[p] = new["git_head"]
        require(not strict_checkout or new["git_head"] == head, "Regenerate provenance at the current checkout revision")
        check_sources(old, historical=True)
        check_sources(new)
        require(new["runtime"]["python"] == platform.python_version(), "Regenerate outputs in the actual verification environment")
        require(new["runtime"]["platform"] == platform.platform(), "Recorded platform differs from current verifier")
        packages = new["runtime"].get("packages", {"PyYAML": new["runtime"].get("PyYAML")})
        for name, version in packages.items():
            require(importlib.metadata.version(name) == version, "Producer dependency differs: " + name)
        source_paths.update(new.get("source_sha256", {}))
        for metadata, historical in [(old, True), (new, False)]:
            for name, checksum in metadata.get("output_sha256", metadata.get("outputs_sha256", {})).items():
                path = str(Path(p).parent / name)
                data = frozen_bytes(path) if historical else safe_path(root, path).read_bytes()
                require(digest(data) == checksum, "Output hash mismatch: " + path)
                if not historical:
                    output_hashes[path] = checksum
        output_hashes[p] = digest(safe_path(root, p).read_bytes())
        if p == PROVENANCE[0]:
            for key in ["network_profiles", "sampling_seeds", "samples_per_profile", "cohort_files", "cohort_seeds", "theory_method"]:
                require(old[key] == new[key], "Configured method or experiment design changed: " + key)
            for profile in old["dropout_theory"]:
                for key in ["N", "m", "T", "q", "p", "D", "packet_loss"]:
                    require(old["dropout_theory"][profile][key] == new["dropout_theory"][profile][key],
                            "Configured dropout parameter changed: " + profile + "/" + key)
            for key in ["latency_statistics", "dropout_statistics", "dropout_theory"]:
                json_differences[key] = compare_statistics(old[key], new[key], key)
        else:
            require(len(new["runs"]) == 110 and {r["csv_path"] for r in new["runs"]} == set(inputs), "Audit run membership changed")
            require(all(r["status"] == "supported_reconstruction" for r in new["runs"]), "Unsupported historical reconstruction")

    archive = root / "experiments/results/diagnostics/superseded-evidence/pre-correction-16-v1.0.8"
    archive_manifest = json.loads((archive / "evidence_manifest.json").read_text())
    require(archive_manifest == json.loads(frozen_bytes(PROVENANCE[0]))["superseded_evidence"], "Historical archive manifest changed")
    archive_hashes = {}
    for entry in archive_manifest.values():
        p = safe_path(archive, entry["archive_file"])
        require(digest(p.read_bytes()) == entry["sha256"], "Superseded evidence hash mismatch")
        archive_hashes[str(p.relative_to(root))] = entry["sha256"]
    for p in files:
        if p.endswith(".csv") and p not in inputs:
            # Raw engine values, empirical audits and per-run/round derivatives are exact.
            floating = None
            tolerance = "0"
            if p.endswith("cohort_delay_summary.csv"):
                floating = set(next(csv.reader(io.StringIO(frozen_bytes(p).decode())))) - {"experiment", "n_runs", "convention"}
                tolerance = "1e-9"  # ms; <1e-14 relative at the largest reported means.
            elif p.endswith("dropout_theory_distribution.csv"):
                floating = {"probability"}
                tolerance = "1e-14"  # float64 probability evaluation, not an observed measurement.
            csv_differences[p] = compare_csv(frozen_bytes(p).decode(), safe_path(root, p).read_text(), tolerance, floating)
    summary = safe_path(root, STATISTICS).read_text()
    runtime_line = (f"Actual runtime: Python {platform.python_version()} | SciPy {importlib.metadata.version('scipy')}"
                    f" | NumPy {importlib.metadata.version('numpy')} | pandas {importlib.metadata.version('pandas')}")
    require(runtime_line in summary.splitlines(), "Statistical runtime metadata differs; regenerate in the current environment")
    compare_text(frozen_bytes(STATISTICS).decode(), summary, ("Actual runtime: ", "Input manifest SHA256 "))
    manifests = []
    for records in [raw_hashes, committed_hashes]:
        manifests.append(digest("".join(f"{n}\0{h}\n" for n, h in sorted(records)).encode()))
    checksum = re.search(r"^Input manifest SHA256 .*: ([0-9a-f]{64})$", summary, re.MULTILINE)
    require(checksum is not None and checksum[1] in manifests, "Statistical input manifest is unverifiable")
    output_hashes[STATISTICS] = digest(safe_path(root, STATISTICS).read_bytes())
    p = "experiments/results/diagnostics/layer3_validation_report.txt"
    compare_text(frozen_bytes(p).decode(), safe_path(root, p).read_text(), ("Runtime: ", "Source revision: "))
    p = "experiments/results/diagnostics/historical-delay-audit/AUDIT.md"
    # The line-ending count is producer-checkout metadata; all other prose is exact.
    old_lines, new_lines = frozen_bytes(p).decode().splitlines(), safe_path(root, p).read_text().splitlines()
    scrub = lambda lines: [re.sub(r"^\d+(?= CSVs differ from their committed copies only in CRLF/LF)", "COUNT", s) for s in lines]
    require(scrub(old_lines) == scrub(new_lines), "Historical audit scientific wording changed")
    return {
        "scope": "Release-checkout verification; not a new experiment, commit, tag or historical execution attestation",
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_head": head, "producer_git_heads": producer_heads,
        "runtime": {"python": platform.python_version(), "executable": sys.executable, "platform": platform.platform(),
                    "packages": {n: importlib.metadata.version(n) for n in ["numpy", "scipy", "pandas", "torch", "PyYAML"]}},
        "cohort_files": 110, "strict_checkout": strict_checkout, "input_serialization": layouts,
        "input_sha256": {p: digest(safe_path(root, p).read_bytes()) for p in inputs},
        "historical_committed_sha256": {p: inputs[p]["committed_sha256"] for p in inputs},
        "statistical_manifest_sha256": checksum[1],
        "frozen_manifest_sha256": digest((frozen / "manifest.json").read_bytes()),
        "historical_frozen_files_verified": len(files), "archive_sha256": archive_hashes,
        "active_outputs_sha256": output_hashes,
        "current_source_sha256": {p: digest(safe_path(root, p).read_bytes()) for p in sorted(source_paths)},
        "precision_differences": {"JSON_statistics": {k: v for k, v in json_differences.items() if v},
                                  "CSV_cells": {k: v for k, v in csv_differences.items() if v}},
        "precision_policy": "Raw engine observations and all original input identities exact; per-round/per-run derivatives exact; cohort aggregates <=1e-9 ms; probability-law cells <=1e-14; JSON float statistics <=1e-10; every nonzero difference recorded, never adjusted.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strict-checkout", action="store_true", help="Require all original inputs to match historical committed bytes exactly")
    parser.add_argument("--output", type=Path, default=ROOT / OUTPUT)
    args = parser.parse_args()
    result = verify(strict_checkout=args.strict_checkout)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print("Verified 110 inputs, frozen historical records, active source/output hashes, archive and scientific comparisons.")
    print("Recorded working-tree serialization variants:", sum(v != "historical_committed_bytes" for v in result["input_serialization"].values()))
    print("Verification record:", args.output)


if __name__ == "__main__":
    main()
