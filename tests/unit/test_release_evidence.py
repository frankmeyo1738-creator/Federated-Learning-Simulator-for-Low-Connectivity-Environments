"""Release verification guards; synthetic bytes only, no training or downloads."""
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("release_evidence", ROOT / "scripts/verify_release_evidence.py")
r = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(r)


def test_only_recorded_serializations_are_accepted():
    lf = b"round_num,value\n1,2.5\n"
    crlf = lf.replace(b"\n", b"\r\n")
    record = {"original_working_tree_sha256": r.digest(crlf), "committed_sha256": r.digest(lf)}
    assert r.verify_input_bytes(lf, lf, record) == "historical_committed_bytes"
    assert r.verify_input_bytes(crlf, lf, record) == "recorded_working_tree_CRLF_variant"
    for mutation in [lf.replace(b"2.5", b"2.6"), lf.replace(b"value", b"result"),
                     lf.replace(b"1,", b"2,"), lf + b"2,3.5\n"]:
        with pytest.raises(ValueError):
            r.verify_input_bytes(mutation, lf, record)


def test_even_declared_hash_cannot_authorize_changed_observations():
    lf, changed = b"x\n1\n", b"x\n2\n"
    forged = {"original_working_tree_sha256": r.digest(changed), "committed_sha256": r.digest(lf)}
    with pytest.raises(ValueError, match="observations"):
        r.verify_input_bytes(changed, lf, forged)


def test_cohort_membership_is_explicit_and_complete():
    paths = r.expected_paths()
    assert len(paths) == 110
    assert sum("/noniid/" in p for p in paths) == 50
    assert all("pre-seeding" not in p and "cifar" not in p for p in paths)
    assert {f"experiments/results/multiseed/noniid/baseline_fedavg_noniid_seed{k}_metrics.csv"
            for k in range(1, 11)}.issubset(paths)


def test_csv_schema_order_counts_and_identity_are_exact():
    original = "experiment,value\na,1.0\nb,2.0\n"
    for changed in ["experiment,result\na,1.0\nb,2.0\n", "experiment,value\nb,2.0\na,1.0\n",
                    "experiment,value\na,1.0\n", "experiment,value\nc,1.0\nb,2.0\n"]:
        with pytest.raises(ValueError):
            r.compare_csv(original, changed, "1e-9", {"value"})


def test_precision_difference_is_bounded_recorded_and_not_rewritten():
    original = "experiment,value\na,123460.23995204475\n"
    new = "experiment,value\na,123460.23995204478\n"
    diff = r.compare_csv(original, new, "1e-9", {"value"})
    assert diff[0]["absolute_difference"] == "3E-11"
    assert diff[0]["preserved"] == "123460.23995204475"
    assert diff[0]["regenerated"] == "123460.23995204478"
    for bad in ["experiment,value\na,123460.24\n", "experiment,value\na,NaN\n"]:
        with pytest.raises(ValueError):
            r.compare_csv(original, bad, "1e-9", {"value"})
    with pytest.raises(ValueError):
        r.compare_csv(original, new)  # No tolerance on exact observations.


def test_metadata_exceptions_do_not_allow_scientific_text_changes():
    old = "Actual runtime: old\nW=17.5 p=0.3074\n"
    r.compare_text(old, old.replace("old", "new"), ("Actual runtime: ",))
    with pytest.raises(ValueError):
        r.compare_text(old, old.replace("0.3074", "0.0001"), ("Actual runtime: ",))
    with pytest.raises(ValueError):
        r.compare_text(old, old + "extra result\n", ("Actual runtime: ",))
    with pytest.raises(ValueError):
        r.compare_statistics({"N": 10, "inside": False}, {"N": 10, "inside": 0})
    with pytest.raises(ValueError):
        r.compare_statistics({"mean": 2.9}, {"mean": 3.0})


def test_evidence_paths_cannot_escape_checkout(tmp_path):
    assert r.safe_path(tmp_path, "a/b") == tmp_path / "a/b"
    for path in ["../secret", "/absolute/private"]:
        with pytest.raises(ValueError):
            r.safe_path(tmp_path, path)
    (tmp_path / "link").symlink_to(tmp_path.parent, target_is_directory=True)
    with pytest.raises(ValueError):
        r.safe_path(tmp_path, "link/secret")


def test_historical_snapshot_hashes_and_membership_are_preserved():
    frozen = ROOT / r.SNAPSHOT
    manifest = json.loads((frozen / "manifest.json").read_text())
    assert {x["path"] for x in manifest["cohort_inputs"]} == r.expected_paths()
    for record in manifest["files"].values():
        assert r.digest((frozen / record["snapshot"]).read_bytes()) == record["sha256"]
    assert len(manifest["files"]) == 28
