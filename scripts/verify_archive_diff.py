#!/usr/bin/env python3
"""
Verification Script: Pre-Seeding-Fix Archive Audit
Compares all CSV files in experiments/results/multiseed/pre-seeding-fix-archive/
against the active experiments/results/multiseed/ directory to detect any
unregenerated legacy runs that still predate commit 7e7b8cf.
"""

import os
import filecmp
import sys

def main():
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    archive_dir = os.path.join(repo_root, "experiments/results/multiseed/pre-seeding-fix-archive")
    active_dir = os.path.join(repo_root, "experiments/results/multiseed")

    if not os.path.exists(archive_dir):
        print(f"❌ Archive directory not found: {archive_dir}")
        sys.exit(1)

    archived_files = sorted([f for f in os.listdir(archive_dir) if f.endswith(".csv")])
    print("=" * 65)
    print(f" AUDIT: Comparing {len(archived_files)} archived pre-fix CSVs against active results")
    print("=" * 65)

    identical, differing, missing = [], [], []

    for fname in archived_files:
        arch_path = os.path.join(archive_dir, fname)
        act_path = os.path.join(active_dir, fname)
        if not os.path.exists(act_path):
            missing.append(fname)
            print(f"❓ {fname:<44} : MISSING from active results")
        elif filecmp.cmp(arch_path, act_path, shallow=False):
            identical.append(fname)
            print(f"❌ {fname:<44} : IDENTICAL (pre-fix, needs regeneration)")
        else:
            differing.append(fname)
            print(f"✅ {fname:<44} : DIFFERS (properly post-fix regenerated)")

    print("=" * 65)
    print(f" SUMMARY:")
    print(f"   ✅ Post-fix regenerated : {len(differing):>2} / {len(archived_files)}")
    print(f"   ❌ Pre-fix identical    : {len(identical):>2} / {len(archived_files)}")
    print(f"   ❓ Missing from active  : {len(missing):>2} / {len(archived_files)}")
    print("=" * 65)

    if identical:
        print("\n⚠️  Action required: The files marked ❌ must be regenerated.")
        return 1
    else:
        print("\n🎉 ALL ARCHIVED RUNS ARE FULLY REGENERATED AND DIFFER FROM PRE-FIX!")
        return 0

if __name__ == "__main__":
    sys.exit(main())
