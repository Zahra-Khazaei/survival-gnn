# optimize.py
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import argparse
import datetime
import json
import os
import time

import joblib
import optuna
import torch

from objective import get_survival_objective
from utils import load_config


def main():
    parser = argparse.ArgumentParser(description="Optuna optimization for survival GNN")
    parser.add_argument("config_path", nargs="?", default="config_survival.yaml", help="Path to YAML config")
    parser.add_argument("--k", type=int, default=None, help="Override KNN neighbors k from config (int > 0)")
    parser.add_argument("--model_name", type=str, default=None, help="Override model_name from config (e.g. GCN, GIN, GraphTransformer)")
    args = parser.parse_args()

    config = load_config(args.config_path)

    if args.k is not None:
        if args.k <= 0:
            raise ValueError(f"--k must be > 0, got {args.k}")
        config["k"] = int(args.k)

    if args.model_name is not None:
        config["model_name"] = args.model_name

    data_folder = config.get("DATA_FOLDER", "Data")

    if config.get("data_path", "") == "auto":
        config["data_path"] = f"{data_folder}/{config['LABEL_COL']}_scaled_data.csv"
    if config.get("split_path", "") == "auto":
        config["split_path"] = f"{data_folder}/{config['LABEL_COL']}_data_splits.json"

    if config.get("TASK_TYPE", "").lower() != "survival":
        config["TASK_TYPE"] = "survival"

    task_name = config["LABEL_COL"].lower()
    horizons_months = [float(h) for h in config.get("horizons_months", [60.0, 120.0])]

    timestamp = datetime.datetime.now()
    folder_timestamp = timestamp.strftime("%Y%m%d_%H%M%S")
    readable_timestamp = timestamp.strftime("%Y-%m-%d %H:%M:%S")

    k_tag = f"k{int(config.get('k', 10))}"

    cohort_tag = config.get("DATA_FOLDER", "Data/Cohort1").split("/")[-1].replace("&", "")
    model_name = config.get("model_name", "GCN")
    base_dir = os.path.join(
        "Optuna_results_survival",
        cohort_tag,
        model_name,
        f"{task_name}_{config['Graph_method']}_{config['Graph_feature']}_{k_tag}_{folder_timestamp}",
    )
    log_dir = os.path.join(base_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)

    print("torch:", torch.__version__, "cuda:", torch.cuda.is_available())
    print("Config:", args.config_path)
    print("Using k =", int(config.get("k", 10)))
    print("data_path =", config["data_path"])
    print("split_path =", config["split_path"])

    objective = get_survival_objective(base_dir, config)

    start_time = time.time()

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
    )
    study.optimize(objective, n_trials=int(config.get("n_trials", 100)))

    elapsed_time = time.time() - start_time

    print("\n=== Best Hyperparameters (Survival GNN) ===")
    print(study.best_params)

    study_path = os.path.join(log_dir, "optuna_study.pkl")
    joblib.dump(study, study_path)

    best_trial = study.best_trial

    best_summary = {
        "best_trial_number": best_trial.number,
        "k": int(config.get("k", 10)),
        "val_cindex": (
            f"{best_trial.user_attrs.get('mean_val_cindex', float('nan')):.4f} ± "
            f"{best_trial.user_attrs.get('std_val_cindex', float('nan')):.4f}"
        ),
        "best_params": best_trial.params,
        "elapsed_time_sec": round(elapsed_time, 2),
        "elapsed_time_min": round(elapsed_time / 60, 2),
        "timestamp": readable_timestamp,
        "horizons_months": horizons_months,
    }

    best_summary["val_ibs_ipcw"] = (
        f"{best_trial.user_attrs.get('mean_val_ibs_ipcw', float('nan')):.4f} ± "
        f"{best_trial.user_attrs.get('std_val_ibs_ipcw', float('nan')):.4f}"
    )

    for h in horizons_months:
        hm = int(h)

        best_summary[f"val_auc_{hm}m"] = (
            f"{best_trial.user_attrs.get(f'mean_val_auc_{hm}m', float('nan')):.4f} ± "
            f"{best_trial.user_attrs.get(f'std_val_auc_{hm}m', float('nan')):.4f}"
        )
        best_summary[f"val_auc_ipcw_{hm}m"] = (
            f"{best_trial.user_attrs.get(f'mean_val_auc_ipcw_{hm}m', float('nan')):.4f} ± "
            f"{best_trial.user_attrs.get(f'std_val_auc_ipcw_{hm}m', float('nan')):.4f}"
        )
        best_summary[f"val_bacc_{hm}m"] = (
            f"{best_trial.user_attrs.get(f'mean_val_bacc_{hm}m', float('nan')):.4f} ± "
            f"{best_trial.user_attrs.get(f'std_val_bacc_{hm}m', float('nan')):.4f}"
        )
        best_summary[f"val_brier_ipcw_{hm}m"] = (
            f"{best_trial.user_attrs.get(f'mean_val_brier_ipcw_{hm}m', float('nan')):.4f} ± "
            f"{best_trial.user_attrs.get(f'std_val_brier_ipcw_{hm}m', float('nan')):.4f}"
        )

    summary_path = os.path.join(log_dir, "best_summary.json")
    with open(summary_path, "w") as f:
        json.dump(best_summary, f, indent=4)

    print("\nBest trial summary saved to:", summary_path)


if __name__ == "__main__":
    main()