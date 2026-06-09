# survival/baseline_cph_rsf.py
# Non-graph baselines: Cox Proportional Hazard (CPH), Random Survival Forest (RSF), and MLP.
# Uses the same config, data splits, and evaluation metrics as the GNN pipeline.
#
# Usage (no tuning — V1 behaviour):
#   python survival/baseline_cph_rsf.py --config configs/bcr_config_euc_GG_c1.yaml --model cph
#   python survival/baseline_cph_rsf.py --config configs/bcr_config_euc_GG_c1.yaml --model rsf
#   python survival/baseline_cph_rsf.py --config configs/bcr_config_euc_GG_c1.yaml --model mlp
#
# Usage (with Optuna tuning — V2, CPH/RSF only):
#   python survival/baseline_cph_rsf.py --config configs/bcr_config_euc_GG_c1.yaml --model cph --tune
#   python survival/baseline_cph_rsf.py --config configs/bcr_config_euc_GG_c1.yaml --model rsf --tune
#
# Output (no tune): Baseline_results/<label>_<model>_baseline.json
# Output (--tune):  Baseline_Results_V2/<label>_<model>_baseline.json

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import json
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from data_loader import load_survival_data, load_splits
from utils import load_config
from metrics.survival_metrics import concordance_index, binary_metrics_at_horizon, ibs_ipcw_from_hazards


def survival_fn_to_hazard_tensor(surv_fn_df, bin_width, num_bins):
    """
    Convert a lifelines survival function DataFrame to a discrete hazard tensor.

    surv_fn_df: DataFrame with time as index, one column per patient.
    Returns: torch.FloatTensor of shape [N, num_bins]
    """
    bin_edges = np.arange(num_bins + 1) * bin_width  # exactly num_bins+1 values
    t_start = bin_edges[:-1]  # [num_bins]
    t_end = bin_edges[1:]     # [num_bins]

    n_patients = surv_fn_df.shape[1]
    hazards = np.zeros((n_patients, num_bins), dtype=np.float32)

    times_in_fn = surv_fn_df.index.values.astype(float)

    for i, col in enumerate(surv_fn_df.columns):
        s_vals = surv_fn_df[col].values.astype(float)

        s_start = np.interp(t_start, times_in_fn, s_vals, left=1.0, right=s_vals[-1])
        s_end = np.interp(t_end, times_in_fn, s_vals, left=1.0, right=s_vals[-1])

        s_start = np.clip(s_start, 1e-8, 1.0)
        s_end = np.clip(s_end, 1e-8, 1.0)

        hazards[i] = np.clip(1.0 - s_end / s_start, 0.0, 1.0 - 1e-7)

    return torch.tensor(hazards, dtype=torch.float32)


def rsf_stepfn_to_hazard_tensor(step_fns, bin_width, num_bins):
    """
    Convert scikit-survival step-function predictions to a discrete hazard tensor.

    step_fns: array of StepFunction objects from RSF.predict_survival_function()
    Returns: torch.FloatTensor of shape [N, num_bins]
    """
    bin_edges = np.arange(num_bins + 1) * bin_width  # exactly num_bins+1 values
    t_start = bin_edges[:-1]
    t_end = bin_edges[1:]

    n_patients = len(step_fns)
    hazards = np.zeros((n_patients, num_bins), dtype=np.float32)

    for i, fn in enumerate(step_fns):
        t = fn.x.astype(float)
        s = fn.y.astype(float)

        s_start = np.interp(t_start, t, s, left=1.0, right=s[-1])
        s_end = np.interp(t_end, t, s, left=1.0, right=s[-1])

        s_start = np.clip(s_start, 1e-8, 1.0)
        s_end = np.clip(s_end, 1e-8, 1.0)

        hazards[i] = np.clip(1.0 - s_end / s_start, 0.0, 1.0 - 1e-7)

    return torch.tensor(hazards, dtype=torch.float32)


def evaluate_baseline(hazards_tensor, times_np, events_np, bin_width, horizons_months, ibs_max_time):
    times_t = torch.tensor(times_np, dtype=torch.float32)
    events_t = torch.tensor(events_np, dtype=torch.long)

    risk_full = 1.0 - torch.cumprod(1.0 - hazards_tensor, dim=1)[:, -1]
    c_idx = concordance_index(times_t, events_t, risk_full)

    result = {"c_index": float(c_idx), "horizons": {}}

    for h in horizons_months:
        m = binary_metrics_at_horizon(
            hazards=hazards_tensor,
            times=times_t,
            events=events_t,
            bin_width=bin_width,
            horizon_months=float(h),
        )
        result["horizons"][int(h)] = m

    ibs_t = float(bin_width) * float(int(float(ibs_max_time) // float(bin_width)))
    if ibs_t > bin_width:
        ibs_val, _ = ibs_ipcw_from_hazards(
            hazards=hazards_tensor,
            times=times_t,
            events=events_t,
            bin_width=bin_width,
            t_max=ibs_t,
        )
        result["ibs_ipcw"] = float(ibs_val)
    else:
        result["ibs_ipcw"] = float("nan")

    return result


def mean_std(arr):
    arr = np.array(arr, dtype=float)
    return float(np.nanmean(arr)), float(np.nanstd(arr))


def run_cph(train_df, test_df, feature_cols, time_col, event_col, bin_width, num_bins, horizons_months, ibs_max_time):
    from lifelines import CoxPHFitter

    cols = feature_cols + [time_col, event_col]
    train_df = train_df[cols].dropna()
    test_df = test_df[cols].dropna()

    cph = CoxPHFitter(penalizer=0.1)
    cph.fit(train_df, duration_col=time_col, event_col=event_col)

    surv_fn_df = cph.predict_survival_function(test_df[feature_cols])

    hazards = survival_fn_to_hazard_tensor(surv_fn_df, bin_width, num_bins)
    times_np = test_df[time_col].values.astype(float)
    events_np = test_df[event_col].values.astype(int)

    return evaluate_baseline(hazards, times_np, events_np, bin_width, horizons_months, ibs_max_time)


def run_rsf(train_df, test_df, feature_cols, time_col, event_col, bin_width, num_bins, horizons_months, ibs_max_time):
    from sksurv.ensemble import RandomSurvivalForest
    from sksurv.util import Surv

    cols = feature_cols + [time_col, event_col]
    train_df = train_df[cols].dropna()
    test_df = test_df[cols].dropna()

    y_train = Surv.from_arrays(
        event=train_df[event_col].values.astype(bool),
        time=train_df[time_col].values.astype(float),
    )

    rsf = RandomSurvivalForest(n_estimators=200, min_samples_leaf=15, random_state=42, n_jobs=-1)
    rsf.fit(train_df[feature_cols].values, y_train)

    step_fns = rsf.predict_survival_function(test_df[feature_cols].values)
    hazards = rsf_stepfn_to_hazard_tensor(step_fns, bin_width, num_bins)

    times_np = test_df[time_col].values.astype(float)
    events_np = test_df[event_col].values.astype(int)

    return evaluate_baseline(hazards, times_np, events_np, bin_width, horizons_months, ibs_max_time)


class _SurvivalMLP(nn.Module):
    def __init__(self, in_features, num_bins, hidden_dim=64, num_layers=2, dropout=0.2):
        super().__init__()
        layers = []
        prev = in_features
        for _ in range(num_layers):
            layers += [nn.Linear(prev, hidden_dim), nn.ReLU(), nn.Dropout(dropout)]
            prev = hidden_dim
        layers.append(nn.Linear(prev, num_bins))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return torch.sigmoid(self.net(x))


def _train_mlp(model, X, t, e, criterion, num_epochs, lr, weight_decay, gamma,
               X_val=None, t_val=None, e_val=None, es_patience=20, es_warmup=10):
    """Train MLP with optional early stopping on val C-index. Returns model with best weights."""
    from metrics.survival_metrics import concordance_index as c_index_fn

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)

    best_state = None
    best_score = -1.0
    bad_count = 0
    use_es = (X_val is not None)

    for epoch in range(1, num_epochs + 1):
        model.train()
        optimizer.zero_grad()
        hazards = model(X)
        loss = criterion(hazards, t, e)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if use_es:
            model.eval()
            with torch.no_grad():
                h_val = model(X_val)
                risk = 1.0 - torch.cumprod(1.0 - h_val, dim=1)[:, -1]
                score = float(c_index_fn(t_val, e_val, risk))
            if epoch >= es_warmup:
                if score > best_score + 1e-4:
                    best_score = score
                    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                    bad_count = 0
                else:
                    bad_count += 1
                if bad_count >= es_patience:
                    break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def run_mlp(fold, df_full, id_to_index, all_ids, feature_cols, time_col, event_col,
            bin_width, num_bins, horizons_months, ibs_max_time,
            num_epochs, n_trials, seed, es_patience=20, es_warmup=10):
    """Nested CV MLP baseline with Optuna hyperparameter tuning — mirrors the GNN pipeline."""
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    from losses.survival_loss import DiscreteTimeSurvivalLoss

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cols = feature_cols + [time_col, event_col]

    # Outer train/test
    test_ids = fold["test"]
    test_set = set(test_ids)
    train_ids = [i for i in all_ids if i not in test_set]

    train_df = df_full.iloc[[id_to_index[i] for i in train_ids if i in id_to_index]][cols].dropna()
    test_df  = df_full.iloc[[id_to_index[i] for i in test_ids  if i in id_to_index]][cols].dropna()

    X_train_full = torch.tensor(train_df[feature_cols].values, dtype=torch.float32).to(device)
    t_train_full = torch.tensor(train_df[time_col].values, dtype=torch.float32).to(device)
    e_train_full = torch.tensor(train_df[event_col].values, dtype=torch.long).to(device)
    X_test  = torch.tensor(test_df[feature_cols].values, dtype=torch.float32).to(device)
    t_test  = test_df[time_col].values.astype(float)
    e_test  = test_df[event_col].values.astype(int)

    in_features = len(feature_cols)

    # Build inner fold tensors for Optuna
    inner_folds = []
    for ik in sorted(k for k in fold if k.startswith("inner_folds_")):
        for inner in fold[ik]:
            itr_df  = df_full.iloc[[id_to_index[i] for i in inner["train"]      if i in id_to_index]][cols].dropna()
            ival_df = df_full.iloc[[id_to_index[i] for i in inner["validation"] if i in id_to_index]][cols].dropna()
            inner_folds.append({
                "X_tr":  torch.tensor(itr_df[feature_cols].values,  dtype=torch.float32).to(device),
                "t_tr":  torch.tensor(itr_df[time_col].values,      dtype=torch.float32).to(device),
                "e_tr":  torch.tensor(itr_df[event_col].values,     dtype=torch.long).to(device),
                "X_val": torch.tensor(ival_df[feature_cols].values, dtype=torch.float32).to(device),
                "t_val": torch.tensor(ival_df[time_col].values,     dtype=torch.float32).to(device),
                "e_val": torch.tensor(ival_df[event_col].values,    dtype=torch.long).to(device),
            })

    def objective(trial):
        from metrics.survival_metrics import concordance_index as c_index_fn

        hidden_dim   = trial.suggest_categorical("hidden_dim",   [16, 32, 64, 128])
        num_layers   = trial.suggest_int("num_layers", 2, 4)
        dropout      = trial.suggest_float("dropout", 0.0, 0.4)
        lr           = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
        gamma        = trial.suggest_categorical("gamma", [0.97, 0.98, 0.99, 0.995])
        alpha        = trial.suggest_float("alpha", 0.0, 0.5)
        sigma        = trial.suggest_categorical("sigma", [0.05, 0.1, 0.25])

        torch.manual_seed(seed + trial.number)
        criterion = DiscreteTimeSurvivalLoss(bin_width=bin_width, alpha=alpha, sigma=sigma).to(device)

        scores = []
        for s in inner_folds:
            model = _SurvivalMLP(in_features, num_bins, hidden_dim, num_layers, dropout).to(device)
            model = _train_mlp(
                model, s["X_tr"], s["t_tr"], s["e_tr"], criterion,
                num_epochs, lr, weight_decay, gamma,
                X_val=s["X_val"], t_val=s["t_val"], e_val=s["e_val"],
                es_patience=es_patience, es_warmup=es_warmup,
            )
            model.eval()
            with torch.no_grad():
                h_val = model(s["X_val"])
                risk = 1.0 - torch.cumprod(1.0 - h_val, dim=1)[:, -1]
                scores.append(float(c_index_fn(s["t_val"], s["e_val"], risk)))

        return float(np.nanmean(scores))

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best = study.best_params
    print(f" | best: hd={best['hidden_dim']} nl={best['num_layers']} "
          f"lr={best['lr']:.1e} do={best['dropout']:.2f}", end="", flush=True)

    # Retrain on full outer train with best hyperparams
    torch.manual_seed(seed)
    criterion = DiscreteTimeSurvivalLoss(
        bin_width=bin_width, alpha=best["alpha"], sigma=best["sigma"]
    ).to(device)
    model = _SurvivalMLP(in_features, num_bins, best["hidden_dim"], best["num_layers"], best["dropout"]).to(device)
    model = _train_mlp(
        model, X_train_full, t_train_full, e_train_full, criterion,
        num_epochs, best["lr"], best["weight_decay"], best["gamma"],
    )

    model.eval()
    with torch.no_grad():
        hazards_test = model(X_test).cpu()

    return evaluate_baseline(hazards_test, t_test, e_test, bin_width, horizons_months, ibs_max_time)


def tune_cph(splits, df_full, id_to_index, feature_cols, time_col, event_col,
             bin_width, num_bins, n_trials, seed):
    """Run one global Optuna study for CPH across ALL outer folds × inner folds.
    Mirrors the GNN optimize.py approach: one best param set applied to all outer test folds."""
    import optuna
    from lifelines import CoxPHFitter
    from metrics.survival_metrics import concordance_index as c_index_fn

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    cols = feature_cols + [time_col, event_col]

    # Collect all inner folds across all outer folds (same as GNN objective.py)
    all_inner_folds = []
    for fold in splits.values():
        for ik in sorted(k for k in fold if k.startswith("inner_folds_")):
            for inner in fold[ik]:
                itr_df  = df_full.iloc[[id_to_index[i] for i in inner["train"]      if i in id_to_index]][cols].dropna()
                ival_df = df_full.iloc[[id_to_index[i] for i in inner["validation"] if i in id_to_index]][cols].dropna()
                all_inner_folds.append({"train": itr_df, "val": ival_df})

    def objective(trial):
        penalizer = trial.suggest_float("penalizer", 1e-4, 10.0, log=True)
        l1_ratio  = trial.suggest_float("l1_ratio", 0.0, 1.0)
        scores = []
        for s in all_inner_folds:
            try:
                cph = CoxPHFitter(penalizer=penalizer, l1_ratio=l1_ratio)
                cph.fit(s["train"], duration_col=time_col, event_col=event_col)
                surv_fn = cph.predict_survival_function(s["val"][feature_cols])
                hazards = survival_fn_to_hazard_tensor(surv_fn, bin_width, num_bins)
                t_val = torch.tensor(s["val"][time_col].values, dtype=torch.float32)
                e_val = torch.tensor(s["val"][event_col].values, dtype=torch.long)
                risk = 1.0 - torch.cumprod(1.0 - hazards, dim=1)[:, -1]
                scores.append(float(c_index_fn(t_val, e_val, risk)))
            except Exception:
                scores.append(0.0)
        return float(np.nanmean(scores))

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    return study.best_params


def run_cph_with_params(train_df, test_df, feature_cols, time_col, event_col,
                        bin_width, num_bins, horizons_months, ibs_max_time, best_params):
    from lifelines import CoxPHFitter
    cols = feature_cols + [time_col, event_col]
    train_df = train_df[cols].dropna()
    test_df  = test_df[cols].dropna()
    cph = CoxPHFitter(penalizer=best_params["penalizer"], l1_ratio=best_params["l1_ratio"])
    cph.fit(train_df, duration_col=time_col, event_col=event_col)
    surv_fn_df = cph.predict_survival_function(test_df[feature_cols])
    hazards = survival_fn_to_hazard_tensor(surv_fn_df, bin_width, num_bins)
    times_np = test_df[time_col].values.astype(float)
    events_np = test_df[event_col].values.astype(int)
    return evaluate_baseline(hazards, times_np, events_np, bin_width, horizons_months, ibs_max_time)


def tune_rsf(splits, df_full, id_to_index, feature_cols, time_col, event_col,
             bin_width, num_bins, n_trials, seed):
    """Run one global Optuna study for RSF across ALL outer folds × inner folds."""
    import optuna
    from sksurv.ensemble import RandomSurvivalForest
    from sksurv.util import Surv
    from metrics.survival_metrics import concordance_index as c_index_fn

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    cols = feature_cols + [time_col, event_col]

    all_inner_folds = []
    for fold in splits.values():
        for ik in sorted(k for k in fold if k.startswith("inner_folds_")):
            for inner in fold[ik]:
                itr_df  = df_full.iloc[[id_to_index[i] for i in inner["train"]      if i in id_to_index]][cols].dropna()
                ival_df = df_full.iloc[[id_to_index[i] for i in inner["validation"] if i in id_to_index]][cols].dropna()
                all_inner_folds.append({"train": itr_df, "val": ival_df})

    def objective(trial):
        n_estimators     = trial.suggest_categorical("n_estimators",     [100, 200, 300, 500])
        max_depth        = trial.suggest_categorical("max_depth",        [None, 3, 5, 10])
        min_samples_leaf = trial.suggest_categorical("min_samples_leaf", [5, 10, 15, 20])
        max_features     = trial.suggest_categorical("max_features",     ["sqrt", "log2", 0.5])
        scores = []
        for s in all_inner_folds:
            try:
                y_tr = Surv.from_arrays(event=s["train"][event_col].values.astype(bool),
                                        time=s["train"][time_col].values.astype(float))
                rsf = RandomSurvivalForest(n_estimators=n_estimators, max_depth=max_depth,
                                          min_samples_leaf=min_samples_leaf, max_features=max_features,
                                          random_state=seed, n_jobs=-1)
                rsf.fit(s["train"][feature_cols].values, y_tr)
                step_fns = rsf.predict_survival_function(s["val"][feature_cols].values)
                hazards = rsf_stepfn_to_hazard_tensor(step_fns, bin_width, num_bins)
                t_val = torch.tensor(s["val"][time_col].values, dtype=torch.float32)
                e_val = torch.tensor(s["val"][event_col].values, dtype=torch.long)
                risk = 1.0 - torch.cumprod(1.0 - hazards, dim=1)[:, -1]
                scores.append(float(c_index_fn(t_val, e_val, risk)))
            except Exception:
                scores.append(0.0)
        return float(np.nanmean(scores))

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    return study.best_params


def run_rsf_with_params(train_df, test_df, feature_cols, time_col, event_col,
                        bin_width, num_bins, horizons_months, ibs_max_time, best_params, seed):
    from sksurv.ensemble import RandomSurvivalForest
    from sksurv.util import Surv
    cols = feature_cols + [time_col, event_col]
    train_df = train_df[cols].dropna()
    test_df  = test_df[cols].dropna()
    y_train = Surv.from_arrays(event=train_df[event_col].values.astype(bool),
                               time=train_df[time_col].values.astype(float))
    rsf = RandomSurvivalForest(n_estimators=best_params["n_estimators"],
                               max_depth=best_params["max_depth"],
                               min_samples_leaf=best_params["min_samples_leaf"],
                               max_features=best_params["max_features"],
                               random_state=seed, n_jobs=-1)
    rsf.fit(train_df[feature_cols].values, y_train)
    step_fns = rsf.predict_survival_function(test_df[feature_cols].values)
    hazards = rsf_stepfn_to_hazard_tensor(step_fns, bin_width, num_bins)
    times_np = test_df[time_col].values.astype(float)
    events_np = test_df[event_col].values.astype(int)
    return evaluate_baseline(hazards, times_np, events_np, bin_width, horizons_months, ibs_max_time)


def main():
    parser = argparse.ArgumentParser(description="CPH/RSF survival baseline")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--model", type=str, default="cph", choices=["cph", "rsf", "mlp"],
                        help="Baseline model: 'cph' (Cox PH), 'rsf' (Random Survival Forest), or 'mlp'")
    parser.add_argument("--tune", action="store_true",
                        help="Enable Optuna hyperparameter tuning for CPH/RSF (ignored for MLP, always tuned)")
    parser.add_argument("--out_dir", type=str, default=None,
                        help="Directory to save results JSON (default: Baseline_Results_V2 if --tune, else Baseline_results)")
    args = parser.parse_args()

    if args.out_dir is None:
        args.out_dir = "Baseline_Results_V2" if args.tune else "Baseline_results"

    config = load_config(args.config)

    data_folder = config.get("DATA_FOLDER", "Data")
    if config.get("data_path", "") == "auto":
        config["data_path"] = f"{data_folder}/{config['LABEL_COL']}_scaled_data.csv"
    if config.get("split_path", "") == "auto":
        config["split_path"] = f"{data_folder}/{config['LABEL_COL']}_data_splits.json"

    DATA_PATH = config["data_path"]
    SPLIT_PATH = config["split_path"]
    LABEL_COL = config["LABEL_COL"]
    TIME_COL = config["TIME_COL"]
    ID_COL = "ID"

    bin_width = float(config.get("bin_width", 6.0))
    max_time = float(config.get("max_time", 216.0))
    num_bins = int(max_time / bin_width)
    horizons_months = [float(h) for h in config.get("horizons_months", [60.0, 120.0])]
    ibs_max_time = float(config.get("ibs_max_time", 60.0))

    num_epochs = int(config.get("num_epochs", 100))
    n_trials   = int(config.get("n_trials", 50))
    seed       = int(config.get("seed", 42))
    es_cfg     = config.get("early_stopping", {}) or {}
    es_patience = int(es_cfg.get("patience", 20))
    es_warmup   = int(es_cfg.get("warmup_epochs", 10))

    df_full = pd.read_csv(DATA_PATH)
    feature_cols = [c for c in config.get("features", []) if c in df_full.columns]
    all_ids = list(df_full[ID_COL])
    id_to_index = {pid: idx for idx, pid in enumerate(df_full[ID_COL])}

    splits = load_splits(SPLIT_PATH)

    fold_results = {}
    fold_cindex = []
    fold_ibs = []

    print(f"\nRunning {args.model.upper()} baseline for {LABEL_COL}")
    print(f"Features: {feature_cols}")

    # Run one global Optuna study before the fold loop — same approach as GNN optimize.py.
    # One best param set is found across all outer folds × inner folds, then applied uniformly.
    best_params = None
    if args.tune and args.model in ("cph", "rsf"):
        print(f"\nRunning global Optuna study ({n_trials} trials) across all folds...")
        if args.model == "cph":
            best_params = tune_cph(splits, df_full, id_to_index, feature_cols,
                                   TIME_COL, LABEL_COL, bin_width, num_bins, n_trials, seed)
            print(f"Best CPH params: penalizer={best_params['penalizer']:.4f} l1_ratio={best_params['l1_ratio']:.2f}")
        else:
            best_params = tune_rsf(splits, df_full, id_to_index, feature_cols,
                                   TIME_COL, LABEL_COL, bin_width, num_bins, n_trials, seed)
            print(f"Best RSF params: {best_params}")

    for fold_name, fold in splits.items():
        test_ids = fold["test"]
        test_set = set(test_ids)
        train_ids = [i for i in all_ids if i not in test_set]

        test_idx = [id_to_index[i] for i in test_ids if i in id_to_index]
        train_idx = [id_to_index[i] for i in train_ids if i in id_to_index]

        train_df = df_full.iloc[train_idx].reset_index(drop=True)
        test_df = df_full.iloc[test_idx].reset_index(drop=True)

        print(f"\n  {fold_name}: train={len(train_df)}, test={len(test_df)}", end="", flush=True)

        try:
            if args.model == "cph":
                if args.tune:
                    metrics = run_cph_with_params(train_df, test_df, feature_cols, TIME_COL, LABEL_COL,
                                                  bin_width, num_bins, horizons_months, ibs_max_time, best_params)
                else:
                    metrics = run_cph(train_df, test_df, feature_cols, TIME_COL, LABEL_COL,
                                      bin_width, num_bins, horizons_months, ibs_max_time)
            elif args.model == "rsf":
                if args.tune:
                    metrics = run_rsf_with_params(train_df, test_df, feature_cols, TIME_COL, LABEL_COL,
                                                  bin_width, num_bins, horizons_months, ibs_max_time, best_params, seed)
                else:
                    metrics = run_rsf(train_df, test_df, feature_cols, TIME_COL, LABEL_COL,
                                      bin_width, num_bins, horizons_months, ibs_max_time)
            else:
                metrics = run_mlp(fold, df_full, id_to_index, all_ids, feature_cols,
                                  TIME_COL, LABEL_COL, bin_width, num_bins,
                                  horizons_months, ibs_max_time,
                                  num_epochs, n_trials, seed,
                                  es_patience=es_patience, es_warmup=es_warmup)
        except Exception as e:
            print(f" ERROR: {e}")
            metrics = {"c_index": float("nan"), "ibs_ipcw": float("nan"), "horizons": {}}

        fold_results[fold_name] = metrics
        fold_cindex.append(metrics["c_index"])
        fold_ibs.append(metrics.get("ibs_ipcw", float("nan")))
        print(f" | C-index: {metrics['c_index']:.4f}")

    mean_c, std_c = mean_std(fold_cindex)
    mean_ibs, std_ibs = mean_std(fold_ibs)

    fold_results["summary"] = {
        "c_index": f"{mean_c:.4f} +/- {std_c:.4f}",
        "ibs_ipcw": f"{mean_ibs:.4f} +/- {std_ibs:.4f}",
        "model": args.model.upper(),
        "tuned": args.tune,
        "best_params": best_params,
        "features": feature_cols,
        "outcome": LABEL_COL,
    }

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, f"{LABEL_COL.lower()}_{args.model}_baseline.json")
    with open(out_path, "w") as f:
        json.dump(fold_results, f, indent=4, default=str)

    print(f"\n=== {args.model.upper()} | {LABEL_COL} ===")
    print(f"C-index:  {mean_c:.4f} ± {std_c:.4f}")
    print(f"IBS-IPCW: {mean_ibs:.4f} ± {std_ibs:.4f}")
    print(f"Saved to: {out_path}")


if __name__ == "__main__":
    main()
