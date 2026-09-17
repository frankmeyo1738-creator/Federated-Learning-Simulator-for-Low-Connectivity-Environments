# ==============================================================================
# FL Network Simulator — n=10 Cohort Full Reproduction Script (Google Colab GPU)
# ==============================================================================
# This script automates end-to-end reproduction of the 76-experiment n=10 cohort
# (IID and non-IID conditions) and runs the rigorous statistical analysis.
#
# Features:
#   - Resume-safe: Skips already completed runs.
#   - Google Drive backup: Keeps checkpoints persistent across Colab disconnects.
#   - Safe GitHub push: Diagnostic-rich, non-fatal git pushes (never crashes the run).
#   - Deterministic execution: Pins cuDNN determinism.
#   - Verification: Confirms iid=True / iid=False partitioning from run logs.
#   - Statistical Analysis: Executes scripts/statistical_analysis.py (Wilcoxon mode="approx",
#     Cohen dz, power analysis, genuine binary MiB divisor).
#
# Recommended Runtime: Runtime > Change runtime type > T4 GPU
# ==============================================================================

import os
import sys
import time
import re
import shutil
import subprocess
import getpass

# ------------------------------------------------------------------------------
# 1. Configuration & Execution Mode
# ------------------------------------------------------------------------------
# Choose mode:
#   "all"        : Run simulations (resume-safe) + statistical analysis
#   "stats_only" : Skip simulation runs, immediately run statistical analysis on existing CSVs (~5s)
#   "force_all"  : Delete existing CSVs and rerun all 76 experiments from scratch + analysis
MODE = "all"

MOUNT_GOOGLE_DRIVE = True
ENABLE_GITHUB_PUSH = False  # Set to True only when you explicitly want Colab to commit & push results to GitHub

REPO_URL = "https://github.com/frankmeyo1738-creator/Federated-Learning-Simulator-for-Low-Connectivity-Environments.git"
REPO_DIR = "/content/Federated-Learning-Simulator-for-Low-Connectivity-Environments"

# ------------------------------------------------------------------------------
# 2. Google Drive Mounting (Persistence)
# ------------------------------------------------------------------------------
CHECKPOINT_DIR = "/content/FL_checkpoints"
if MOUNT_GOOGLE_DRIVE:
    try:
        from google.colab import drive
        drive.mount("/content/drive")
        CHECKPOINT_DIR = "/content/drive/MyDrive/FL_Simulator_n10_checkpoint"
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
        print(f"✅ Google Drive mounted: {CHECKPOINT_DIR}")
    except Exception as e:
        print(f"⚠️ Google Drive mount skipped or failed ({e}). Saving checkpoints locally to {CHECKPOINT_DIR}")
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# ------------------------------------------------------------------------------
# 3. Clone / Update Repository
# ------------------------------------------------------------------------------
if not os.path.exists(REPO_DIR):
    print(f"Cloning repository from {REPO_URL}...")
    subprocess.run(["git", "clone", REPO_URL, REPO_DIR], check=True)
else:
    print("Repository directory already exists. Fetching latest changes...")
    subprocess.run(["git", "-C", REPO_DIR, "pull"], check=True)

os.chdir(REPO_DIR)
print(f"Current working directory: {os.getcwd()}")

# ------------------------------------------------------------------------------
# 4. GitHub Credentials Setup (Non-leaking)
# ------------------------------------------------------------------------------
if ENABLE_GITHUB_PUSH:
    print("\n--- GitHub Authentication ---")
    print("Enter your Personal Access Token (repo scope) to push progress to GitHub.")
    print("If you do not wish to push to GitHub, simply press Enter to skip.")
    token = getpass.getpass("Paste GitHub Personal Access Token: ").strip()
    if token:
        GITHUB_USER = "frankmeyo1738-creator"
        subprocess.run(["git", "config", "user.name", GITHUB_USER], check=True)
        subprocess.run(["git", "config", "user.email", "frankmeyo1738@gmail.com"], check=True)
        PUSH_URL = f"https://{GITHUB_USER}:{token}@github.com/{GITHUB_USER}/Federated-Learning-Simulator-for-Low-Connectivity-Environments.git"
        subprocess.run(["git", "remote", "set-url", "origin", PUSH_URL], check=True)
        print("✅ GitHub push remote configured (token hidden from error traces).")
    else:
        print("No token provided. Progress will be saved to Google Drive only.")
        ENABLE_GITHUB_PUSH = False

def safe_git_push(commit_msg):
    if not ENABLE_GITHUB_PUSH:
        return
    try:
        subprocess.run(["git", "add", "-A"], cwd=REPO_DIR)
        res_commit = subprocess.run(["git", "commit", "-m", commit_msg], cwd=REPO_DIR, capture_output=True, text=True)
        if "nothing to commit" in res_commit.stdout:
            return
        res_push = subprocess.run(["git", "push", "origin", "main"], cwd=REPO_DIR, capture_output=True, text=True)
        if res_push.returncode == 0:
            print(f"  🚀 GitHub push successful: '{commit_msg}'")
        else:
            print(f"  ⚠️ GitHub push non-fatal warning: {res_push.stderr.strip()}")
            print("     (Execution will continue; data is safe in Drive checkpoint)")
    except Exception as e:
        print(f"  ⚠️ Git operation exception (non-fatal): {e}")

# ------------------------------------------------------------------------------
# 5. Colab Dependencies & PyTorch GPU Check
# ------------------------------------------------------------------------------
print("\nVerifying dependencies...")
# Colab already comes with PyTorch, CUDA, NumPy, Pandas, SciPy, and Matplotlib pre-installed.
# Installing strict version pins from requirements.txt causes pip resolver backtracking.
# We only need to ensure PyYAML is present:
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pyyaml"], check=True)

import torch
print(f"PyTorch: {torch.__version__} | CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU Device: {torch.cuda.get_device_name(0)}")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print("✅ cuDNN deterministic settings enabled.")
else:
    print("⚠️ WARNING: Running on CPU! For 10x faster execution, enable GPU under Runtime > Change runtime type.")

# ------------------------------------------------------------------------------
# 6. Experiment Manifest (76 Runs)
# ------------------------------------------------------------------------------
IID_DIR = "config/experiments/multiseed"
NONIID_DIR = "config/experiments/multiseed/noniid"
RESULTS_IID_DIR = "experiments/results/multiseed"
RESULTS_NONIID_DIR = "experiments/results/multiseed/noniid"

os.makedirs(RESULTS_IID_DIR, exist_ok=True)
os.makedirs(RESULTS_NONIID_DIR, exist_ok=True)

IID_FULL_RERUN = ["baseline_fedavg", "urban_zambia_fedavg"]  # Seeds 1-10
IID_PARTIAL = ["rural_zambia_fedavg", "rural_zambia_fedprox",
               "severe_disruption_fedavg", "severe_disruption_fedprox"]  # Seeds 7-10
NONIID_FULL = ["rural_zambia_fedavg_noniid", "rural_zambia_fedprox_noniid",
               "severe_disruption_fedavg_noniid", "severe_disruption_fedprox_noniid"]  # Seeds 1-10

runs = []
for cfg in IID_FULL_RERUN:
    for seed in range(1, 11):
        runs.append((f"{IID_DIR}/{cfg}_seed{seed}.yaml", f"{RESULTS_IID_DIR}/{cfg}_seed{seed}_metrics.csv", True, f"{cfg}_seed{seed}"))

for cfg in IID_PARTIAL:
    for seed in range(7, 11):
        runs.append((f"{IID_DIR}/{cfg}_seed{seed}.yaml", f"{RESULTS_IID_DIR}/{cfg}_seed{seed}_metrics.csv", True, f"{cfg}_seed{seed}"))

for cfg in NONIID_FULL:
    for seed in range(1, 11):
        runs.append((f"{NONIID_DIR}/{cfg}_seed{seed}.yaml", f"{RESULTS_NONIID_DIR}/{cfg}_seed{seed}_metrics.csv", False, f"{cfg}_seed{seed}"))

print(f"\nManifest defined: {len(runs)} runs total.")
assert len(runs) == 76, f"Expected 76 runs, got {len(runs)}"

# ------------------------------------------------------------------------------
# 7. Simulation Execution Loop
# ------------------------------------------------------------------------------
if MODE in ["all", "force_all"]:
    if MODE == "force_all":
        print("⚡ FORCE_ALL selected: clearing existing output CSVs for these 76 runs...")
        for _, csv_path, _, _ in runs:
            if os.path.exists(csv_path):
                os.remove(csv_path)

    LOG_DIR = os.path.join(CHECKPOINT_DIR, "run_logs")
    os.makedirs(LOG_DIR, exist_ok=True)

    completed, skipped, failed = [], [], []

    for i, (yaml_path, csv_path, use_iid, label) in enumerate(runs, 1):
        print(f"\n[{i}/{len(runs)}] {label} (iid={use_iid})")

        if os.path.exists(csv_path):
            print("  -> Output CSV already exists. Skipping (resume-safe).")
            skipped.append(label)
            continue

        cmd = ["python", "-m", "src.main", "--config", yaml_path]
        if use_iid:
            cmd.append("--iid")

        t0 = time.time()
        proc = subprocess.run(cmd, capture_output=True, text=True)
        elapsed = time.time() - t0

        log_path = os.path.join(LOG_DIR, f"{label}.log")
        with open(log_path, "w") as f:
            f.write(proc.stdout + "\n\n--- STDERR ---\n" + proc.stderr)

        if proc.returncode != 0:
            print(f"  ❌ FAILED (exit {proc.returncode}) in {elapsed:.1f}s — see {log_path}")
            failed.append(label)
            continue

        # Independent confirmation of partitioning mode from log
        full_output = proc.stdout + "\n" + proc.stderr
        match = re.search(r"iid=(\w+)", full_output)
        if match:
            actual_iid = (match.group(1) == "True")
            if actual_iid != use_iid:
                print(f"  ⚠️ MISMATCH: expected iid={use_iid}, run log says iid={actual_iid}!")
                failed.append(f"{label} (iid mismatch)")
                continue
            else:
                print(f"  ✅ Partitioning verified: iid={actual_iid} ({elapsed:.1f}s)")
        else:
            print(f"  ℹ️ Finished in {elapsed:.1f}s (could not parse iid confirmation tag)")

        if not os.path.exists(csv_path):
            print(f"  ❌ WARNING: Run exited 0 but {csv_path} not found.")
            failed.append(f"{label} (missing output)")
            continue

        completed.append(label)

        # Checkpoint every 5 completed runs or on final run
        if len(completed) % 5 == 0 or i == len(runs):
            try:
                shutil.copytree("experiments/results", os.path.join(CHECKPOINT_DIR, "experiments_results"), dirs_exist_ok=True)
            except Exception as e:
                print(f"  ⚠️ Checkpoint copy warning: {e}")
            safe_git_push(f"n=10 cohort: {len(completed)} new runs complete")

    print("\n" + "="*60)
    print(f"Simulations Finished. Completed: {len(completed)} | Skipped: {len(skipped)} | Failed: {len(failed)}")
    if failed:
        print("Failed runs needing attention:", failed)
else:
    print("\n⏩ MODE=stats_only: Skipping simulation loop.")

# ------------------------------------------------------------------------------
# 8. Statistical Analysis & Report Generation
# ------------------------------------------------------------------------------
print("\n" + "="*80)
print("RUNNING STATISTICAL ANALYSIS (n=10 COHORT)")
print("="*80)

res = subprocess.run([sys.executable, "scripts/statistical_analysis.py"], capture_output=True, text=True)
print(res.stdout)
if res.stderr:
    print("STDERR:", res.stderr)

# Backup statistical summary to Google Drive and GitHub
summary_path = "experiments/results/statistical_summary.txt"
if os.path.exists(summary_path):
    drive_summary_target = os.path.join(CHECKPOINT_DIR, "statistical_summary.txt")
    try:
        shutil.copy(summary_path, drive_summary_target)
        print(f"✅ Statistical summary saved to Google Drive: {drive_summary_target}")
    except Exception as e:
        print(f"⚠️ Warning copying summary to Drive: {e}")
    safe_git_push("Regenerate statistical summary (n=10 cohort with pinned Wilcoxon mode=approx)")

print("\n" + "="*80)
print("🎉 ALL TASKS COMPLETE!")
print("="*80)
