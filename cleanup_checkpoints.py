"""
For each Optuna config directory, keep only the best trial's checkpoints and
delete all others. Best trial number is read from logs/best_summary.json.

Before/after: ~2000 files per config → ~40 files (10 folds × 4 inner folds).
"""

import os
import glob
import json

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "Optuna_results_survival")

total_deleted = 0
total_kept = 0
total_freed_bytes = 0

config_dirs = sorted(os.listdir(RESULTS_DIR))
print(f"Found {len(config_dirs)} config directories\n")

for config_name in config_dirs:
    config_path = os.path.join(RESULTS_DIR, config_name)
    ckpt_dir = os.path.join(config_path, "checkpoints")
    summary_path = os.path.join(config_path, "logs", "best_summary.json")

    if not os.path.isdir(ckpt_dir):
        continue
    if not os.path.isfile(summary_path):
        print(f"  WARNING: no best_summary.json in {config_name}, skipping")
        continue

    with open(summary_path) as f:
        best_trial = json.load(f)["best_trial_number"]

    pth_files = glob.glob(os.path.join(ckpt_dir, "*.pth"))
    to_delete = []
    to_keep = []

    for fp in pth_files:
        fname = os.path.basename(fp)
        # filename: trial_{N}_fold_{M}_inner_folds_{K}.pth
        trial_num = int(fname.split("_")[1])
        if trial_num == best_trial:
            to_keep.append(fp)
        else:
            to_delete.append(fp)

    freed = sum(os.path.getsize(f) for f in to_delete)
    for f in to_delete:
        os.remove(f)

    total_deleted += len(to_delete)
    total_kept += len(to_keep)
    total_freed_bytes += freed
    print(f"  {config_name}: kept trial {best_trial} ({len(to_keep)} files), "
          f"deleted {len(to_delete)} files ({freed / 1e6:.1f} MB)")

print(f"\nDone.")
print(f"  Kept:    {total_kept} files")
print(f"  Deleted: {total_deleted} files")
print(f"  Freed:   {total_freed_bytes / 1e9:.2f} GB")
