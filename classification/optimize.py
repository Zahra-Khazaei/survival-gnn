import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
torch.cuda.is_available()
import optuna
import json
import os
import sys
import numpy as np
import time
import datetime
import joblib
from objective import get_objective
from utils import load_config

config_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
config = load_config(config_path)

data_folder = config.get("DATA_FOLDER", "Data")

if config.get("data_path", "") == "auto":
    config["data_path"] = f"{data_folder}/{config['LABEL_COL']}_scaled_data.csv"

if config.get("split_path", "") == "auto":
    config["split_path"] = f"{data_folder}/{config['LABEL_COL']}_data_splits.json"

TASK_NAME = config['LABEL_COL'].lower()

timestamp = datetime.datetime.now()
folder_timestamp = timestamp.strftime("%Y%m%d_%H%M%S")
readable_timestamp = timestamp.strftime("%Y-%m-%d %H:%M:%S")

base_dir = os.path.join(
    "Optuna_results",
    f"{TASK_NAME}_{config['Graph_method']}_{config['Graph_feature']}_{folder_timestamp}"
)
log_dir = os.path.join(base_dir, "logs")
os.makedirs(log_dir, exist_ok=True)

objective = get_objective(base_dir, config)

start_time = time.time()

study = optuna.create_study(
    direction="maximize",
    sampler=optuna.samplers.TPESampler(seed=42)
)
study.optimize(objective, n_trials=25)

elapsed_time = time.time() - start_time

print("\n=== Best Hyperparameters ===")
print(study.best_params)

study_path = os.path.join(log_dir, "optuna_study.pkl")
joblib.dump(study, study_path)

best_trial = study.best_trial
best_summary = {
    "best_trial_number": best_trial.number,
    "val_f1": f"{best_trial.user_attrs.get('mean_val_f1', 0.0):.4f} ± {best_trial.user_attrs.get('std_val_f1', 0.0):.4f}",
    "val_auc": f"{best_trial.user_attrs.get('mean_val_auc', 0.0):.4f} ± {best_trial.user_attrs.get('std_val_auc', 0.0):.4f}",
    "val_acc": f"{best_trial.user_attrs.get('mean_val_acc', 0.0):.4f} ± {best_trial.user_attrs.get('std_val_acc', 0.0):.4f}",
    "val_bacc": f"{best_trial.user_attrs.get('mean_val_bacc', 0.0):.4f} ± {best_trial.user_attrs.get('std_val_bacc', 0.0):.4f}",
    "val_sens": f"{best_trial.user_attrs.get('mean_val_sens', 0.0):.4f} ± {best_trial.user_attrs.get('std_val_sens', 0.0):.4f}",
    "val_spec": f"{best_trial.user_attrs.get('mean_val_spec', 0.0):.4f} ± {best_trial.user_attrs.get('std_val_spec', 0.0):.4f}",
    "best_params": best_trial.params,
    "elapsed_time_sec": round(elapsed_time, 2),
    "elapsed_time_min": round(elapsed_time / 60, 2),
    "timestamp": readable_timestamp
}

summary_path = os.path.join(log_dir, "best_summary.json")
with open(summary_path, "w") as f:
    json.dump(best_summary, f, indent=4)

print("\nBest trial summary saved to:", summary_path)
