"""
Delete per-trial JSON files and optuna_study.pkl from Optuna result dirs.
These intermediate files are not needed once a search is complete
(best_summary.json retains the best trial's parameters and metrics).

Usage:
    python cleanup_optuna_logs.py                    # dry run (safe, no deletion)
    python cleanup_optuna_logs.py --delete           # actually delete
    python cleanup_optuna_logs.py --delete --cohort Cohort1   # one cohort only
"""

import argparse
import os
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "Optuna_results_survival"
MODELS = ['GCN', 'GIN', 'GraphTransformer', 'GraphSAGE', 'Graphormer']


def scan(cohort_dir, delete):
    n_deleted = 0
    for model in MODELS:
        model_dir = cohort_dir / model
        if not model_dir.is_dir():
            continue
        for exp_dir in sorted(model_dir.iterdir()):
            if not exp_dir.is_dir():
                continue
            logs_dir = exp_dir / 'logs'
            if not logs_dir.is_dir():
                continue
            targets = (
                list(logs_dir.glob('trial_*_val_cindex.json')) +
                list(logs_dir.glob('optuna_study.pkl'))
            )
            for f in targets:
                if delete:
                    f.unlink()
                else:
                    print(f"  [dry-run] would delete: {f.relative_to(RESULTS_DIR)}")
                n_deleted += 1
    return n_deleted


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--delete', action='store_true', help='Actually delete files (default: dry run)')
    parser.add_argument('--cohort', help='Limit to one cohort, e.g. Cohort1')
    args = parser.parse_args()

    if not args.delete:
        print("DRY RUN — pass --delete to actually remove files\n")

    cohorts = (
        [RESULTS_DIR / args.cohort] if args.cohort
        else [d for d in sorted(RESULTS_DIR.iterdir()) if d.is_dir()]
    )

    total = 0
    for cohort_dir in cohorts:
        n = scan(cohort_dir, args.delete)
        action = 'Deleted' if args.delete else 'Would delete'
        print(f"{cohort_dir.name}: {action} {n} files")
        total += n

    print(f"\nTotal: {total} files {'deleted' if args.delete else 'would be deleted'}")


if __name__ == '__main__':
    main()
