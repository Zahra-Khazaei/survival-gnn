# train_ensemble.py
# Deep ensemble across random seeds, leakage-free wiring (train→val, train∪val→test)
# Ensemble = average LOGITS across independently trained models (different seeds) -> softmax.
# Optional temperature scaling is fitted on VAL logits (post-ensemble) and applied to VAL/TEST.
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import json
import argparse
from typing import Tuple, Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import (
    f1_score, roc_auc_score, accuracy_score,
    balanced_accuracy_score, confusion_matrix, roc_curve
)

from torch_geometric.data import Data
from torch_geometric.utils import to_undirected, add_self_loops

from data_loader import load_data, load_splits, build_knn_graph
from models import MODEL_REGISTRY

# Core utils from your project
from utils.core import (
    load_config, get_device, compute_class_weights, get_max_degree,
    plot_roc_curve, plot_confusion_matrix, find_best_summary_path, set_seed
)

# Calibration helpers (your separate files)
from utils.utils_calibration import (
    expected_calibration_error,
    brier_score,
    nll_loss,
    save_reliability_diagram
)
from utils.temperature_scaling import fit_temperature_on_val

import matplotlib
matplotlib.use("Agg")


# ---------------- Metrics helpers ----------------
def safe_roc_auc(y_true, y_score):
    ys = np.unique(np.asarray(y_true))
    if ys.size < 2:
        return np.nan
    return roc_auc_score(y_true, y_score)

def safe_confusion_counts(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = (cm.ravel() if cm.size == 4 else [0, 0, 0, 0])
    return tn, fp, fn, tp

def try_auc(y_true, y_score):
    try:
        return roc_auc_score(y_true, y_score)
    except Exception:
        return float("nan")


# ---------------- Graph wiring (leakage-free) ----------------
def build_eval_edges_like_testpy(
    graph_x: torch.Tensor,
    df: pd.DataFrame,
    config: dict,
    device: torch.device,
    train_idx: torch.Tensor,
    val_idx: torch.Tensor,
    test_idx: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Leakage-free wiring (correct direction):
      - TRAIN backbone (within train): undirected + self-loops
      - VAL connectors:    train -> val   (val nodes read from train)
      - TEST connectors: (train∪val) -> test (test nodes read from train+val)

    Returns:
      edge_index_train (train backbone),
      edge_eval_val    (train backbone + train->val),
      edge_eval_test   (train∪val backbone + (train∪val)->test)
    """
    # TRAIN backbone (within train)
    edge_index_train_local = build_knn_graph(
        X=graph_x[train_idx],
        df=df.iloc[train_idx.cpu().numpy()],
        k=10,
        metric=config["Graph_method"],
        graph_feature=config["Graph_feature"],
        reference=None,
        df_reference=None,
    )
    edge_index_train = train_idx[edge_index_train_local.to(device)]
    edge_index_train = to_undirected(edge_index_train, num_nodes=graph_x.size(0))
    edge_index_train, _ = add_self_loops(edge_index_train, num_nodes=graph_x.size(0))

    # VAL connectors: train -> val
    edge_index_v2t_local = build_knn_graph(
        X=graph_x[val_idx],
        df=df.iloc[val_idx.cpu().numpy()],
        k=10,
        metric=config["Graph_method"],
        graph_feature=config["Graph_feature"],
        reference=graph_x[train_idx],
        df_reference=df.iloc[train_idx.cpu().numpy()],
    )
    edge_index_train2val = torch.stack(
        [
            train_idx[edge_index_v2t_local[1].to(device)],  # src: train
            val_idx[edge_index_v2t_local[0].to(device)],    # dst: val
        ],
        dim=0,
    )
    edge_eval_val = torch.cat([edge_index_train, edge_index_train2val], dim=1)

    # TEST connectors: (train∪val) -> test
    trainval_idx = torch.cat([train_idx, val_idx], dim=0)
    edge_index_trainval_local = build_knn_graph(
        X=graph_x[trainval_idx],
        df=df.iloc[trainval_idx.cpu().numpy()],
        k=10,
        metric=config["Graph_method"],
        graph_feature=config["Graph_feature"],
        reference=None,
        df_reference=None,
    )
    edge_index_trainval = trainval_idx[edge_index_trainval_local.to(device)]
    edge_index_trainval = to_undirected(edge_index_trainval, num_nodes=graph_x.size(0))
    edge_index_trainval, _ = add_self_loops(edge_index_trainval, num_nodes=graph_x.size(0))

    edge_index_t2tv_local = build_knn_graph(
        X=graph_x[test_idx],
        df=df.iloc[test_idx.cpu().numpy()],
        k=10,
        metric=config["Graph_method"],
        graph_feature=config["Graph_feature"],
        reference=graph_x[trainval_idx],
        df_reference=df.iloc[trainval_idx.cpu().numpy()],
    )
    edge_index_trainval2test = torch.stack(
        [
            trainval_idx[edge_index_t2tv_local[1].to(device)],  # src: train∪val
            test_idx[edge_index_t2tv_local[0].to(device)],      # dst: test
        ],
        dim=0,
    )
    edge_eval_test = torch.cat([edge_index_trainval, edge_index_trainval2test], dim=1)
    return edge_index_train, edge_eval_val, edge_eval_test


# ---------------- Train one seed on a fold ----------------
def train_one_model_for_fold(
    model_class,
    best_params: dict,
    config: dict,
    device: torch.device,
    graph: Data,
    df: pd.DataFrame,
    train_idx: torch.Tensor,
    val_idx: torch.Tensor,
    edge_index_train: torch.Tensor,
    seed: int
) -> torch.nn.Module:
    """
    Train a single model instance for this fold/seed:
      - train on TRAIN-only backbone (edge_index_train)
      - early stop on validation loss (VAL reads from TRAIN via connectors)
      - returns the trained model
    """
    extra_args = (config.get("extra_args") or {}).copy()
    if config["model_name"].lower() in ["graphormer", "graphormer_full"]:
        extra_args["max_degree"] = get_max_degree(edge_index_train, graph.x.size(0))

    model = model_class(
        in_channels=graph.x.size(1),
        hidden_channels=best_params["hidden_dim"],
        out_channels=2,
        num_layers=best_params["num_layers"],
        dropout=best_params["dropout"],
        activation="prelu",
        **extra_args
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=best_params["lr"],
                                 weight_decay=best_params["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

    class_weights = compute_class_weights(graph.y[train_idx], device)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

    # Build VAL eval edges (train + train->val)
    edge_index_v2t_local = build_knn_graph(
        X=graph.x[val_idx],
        df=df.iloc[val_idx.cpu().numpy()],
        k=10,
        metric=config["Graph_method"],
        graph_feature=config["Graph_feature"],
        reference=graph.x[train_idx],
        df_reference=df.iloc[train_idx.cpu().numpy()],
    )
    edge_index_train2val = torch.stack(
        [
            train_idx[edge_index_v2t_local[1].to(device)],  # train -> val
            val_idx[edge_index_v2t_local[0].to(device)],
        ],
        dim=0,
    )
    edge_eval_val = torch.cat([edge_index_train, edge_index_train2val], dim=1)

    best_val_loss, patience, wait = float("inf"), 20, 0
    best_state = None

    set_seed(seed)
    for epoch in range(1, 101):
        model.train()
        optimizer.zero_grad()
        out_tr = model(graph.x, edge_index_train)
        loss = criterion(out_tr[train_idx], graph.y[train_idx])
        loss.backward()
        optimizer.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            out_val = model(graph.x, edge_eval_val)
            val_loss = criterion(out_val[val_idx], graph.y[val_idx])

        if val_loss.item() < best_val_loss:
            best_val_loss = val_loss.item()
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    model.load_state_dict(best_state)
    return model


# ---------------- Forward helper (logits) ----------------
@torch.no_grad()
def predict_logits(
    model: torch.nn.Module,
    graph: Data,
    edge_index_eval: torch.Tensor,
    index_subset: torch.Tensor
) -> torch.Tensor:
    """
    Return raw logits (N_subset, 2) as a torch.Tensor on the same device as model inputs.
    """
    model.eval()
    logits = model(graph.x, edge_index_eval)
    return logits[index_subset]


def main():
    parser = argparse.ArgumentParser(description="Train deep ensembles (by seeds) with leakage-free wiring, any model.")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    args = parser.parse_args()

    config = load_config(args.config)
    device = get_device()

    # Resolve auto paths
    data_folder = config.get("DATA_FOLDER", "Data")
    if config.get("data_path", "") == "auto":
        config["data_path"] = f"{data_folder}/{config['LABEL_COL']}_scaled_data.csv"
    if config.get("split_path", "") == "auto":
        config["split_path"] = f"{data_folder}/{config['LABEL_COL']}_data_splits.json"

    LABEL_COL = config["LABEL_COL"]
    ID_COL = "ID"

    # Ensemble section (from YAML)
    ens_cfg = config.get("ensemble", {}) or {}
    seeds = ens_cfg.get("seeds", [42, 43, 44, 45, 46])
    th_strategy = (ens_cfg.get("threshold_strategy") or "recall").lower()  # "recall" or "f1"
    target_recall = float(ens_cfg.get("target_recall", 0.67))

    # Calibration flag
    cal_cfg = (config.get("calibration", {}) or {}).get("temperature", {}) or {}
    use_temp_cal = bool(cal_cfg.get("enabled", False))

    # Load best hyperparams (Optuna)
    best_summary_path = find_best_summary_path(
        optuna_root="Optuna_results",
        model_name=config["model_name"],
        label_col=config["LABEL_COL"],
        graph_method=config["Graph_method"],
        graph_feature=config["Graph_feature"]
    )
    with open(best_summary_path, "r") as f:
        best_params = json.load(f)["best_params"]

    # Output dirs
    logs_dir = os.path.dirname(best_summary_path)
    ens_dir = os.path.join(logs_dir, "ensemble")
    plot_dir = os.path.join(ens_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    # Data/splits
    data_df = pd.read_csv(config["data_path"])
    FEATURES = [c for c in config.get("features", []) if c in data_df.columns]
    df, X, y, id_to_index = load_data(config["data_path"], FEATURES, LABEL_COL, ID_COL)
    splits = load_splits(config["split_path"])

    graph = Data(
        x=torch.as_tensor(X, dtype=torch.float32, device=device),
        y=torch.as_tensor(y, dtype=torch.long, device=device)
    )

    model_class = MODEL_REGISTRY[config["model_name"]]

    all_results: Dict[str, dict] = {}

    # Iterate outer folds
    for outer_name, outer_data in splits.items():
        print(f"\n=== Ensemble on {outer_name} ===")

        # Indices
        test_idx = torch.tensor(
            [id_to_index[i] for i in outer_data["test"] if i in id_to_index],
            dtype=torch.long, device=device
        )
        inner_keys = [k for k in outer_data.keys() if k.startswith("inner_folds_")]
        train_val_ids = [
            pid
            for key in inner_keys
            for fold in outer_data[key]
            for pid in (fold["train"] + fold["validation"])
        ]
        train_val_idx = torch.tensor(
            [id_to_index[i] for i in train_val_ids if i in id_to_index],
            dtype=torch.long, device=device
        )

        # Small validation split from train_val (like test.py)
        labels_tv = graph.y[train_val_idx].cpu().numpy()
        val_frac = 0.10
        tr_sub = va_sub = None
        for attempt in range(8):
            splitter = StratifiedShuffleSplit(n_splits=1, test_size=val_frac, random_state=42 + attempt)
            tr_try, va_try = next(splitter.split(np.zeros_like(labels_tv), labels_tv))
            if np.unique(labels_tv[va_try]).size == 2:
                tr_sub, va_sub = tr_try, va_try
                break
            val_frac = min(val_frac + 0.05, 0.30)
        if tr_sub is None or va_sub is None:
            splitter = StratifiedShuffleSplit(n_splits=1, test_size=val_frac, random_state=123)
            tr_sub, va_sub = next(splitter.split(np.zeros_like(labels_tv), labels_tv))
            print(f"WARNING: {outer_name}: validation remained single-class; threshold tuning may default to 0.5.")

        train_idx = train_val_idx[tr_sub]
        val_idx   = train_val_idx[va_sub]

        # Build wiring (train backbone, val eval, test eval)
        edge_index_train, edge_eval_val, edge_eval_test = build_eval_edges_like_testpy(
            graph_x=graph.x, df=df, config=config, device=device,
            train_idx=train_idx, val_idx=val_idx, test_idx=test_idx
        )

        # Train per seed & collect VAL/TEST logits
        per_seed: List[dict] = []
        for seed in seeds:
            print(f"  -> training seed {seed} ...")
            set_seed(seed)
            model = train_one_model_for_fold(
                model_class=model_class,
                best_params=best_params,
                config=config,
                device=device,
                graph=graph,
                df=df,
                train_idx=train_idx,
                val_idx=val_idx,
                edge_index_train=edge_index_train,
                seed=seed
            )

            logits_val  = predict_logits(model, graph, edge_eval_val,  val_idx)   # (Nv, 2)
            logits_test = predict_logits(model, graph, edge_eval_test, test_idx)  # (Nt, 2)

            # Move to CPU for stacking
            per_seed.append({
                "seed": seed,
                "val_logits":  logits_val.detach().cpu(),
                "test_logits": logits_test.detach().cpu()
            })

        # ----- Average LOGITS across seeds (robust & simple) -----
        val_logits_stack  = torch.stack([d["val_logits"]  for d in per_seed], dim=0)  # (S, Nv, 2)
        test_logits_stack = torch.stack([d["test_logits"] for d in per_seed], dim=0)  # (S, Nt, 2)
        val_logits_mean   = val_logits_stack.mean(dim=0).to(device)   # (Nv, 2)
        test_logits_mean  = test_logits_stack.mean(dim=0).to(device)  # (Nt, 2)

        # ----- Optional calibration (Temperature Scaling on VAL logits) -----
        if use_temp_cal:
            scaler = fit_temperature_on_val(val_logits_mean, graph.y[val_idx])
            val_logits_cal  = scaler(val_logits_mean)
            test_logits_cal = scaler(test_logits_mean)
        else:
            val_logits_cal  = val_logits_mean
            test_logits_cal = test_logits_mean

        # ----- Softmax → probabilities for metrics/thresholds -----
        val_soft_mean  = F.softmax(val_logits_cal,  dim=1).detach().cpu().numpy()   # (Nv, 2)
        test_soft_mean = F.softmax(test_logits_cal, dim=1).detach().cpu().numpy()   # (Nt, 2)

        # Decide positive column by AUC on VAL probs (post-calibration if enabled)
        val_labels = graph.y[val_idx].cpu().numpy()
        auc0 = try_auc(val_labels, val_soft_mean[:, 0])
        auc1 = try_auc(val_labels, val_soft_mean[:, 1])
        pos_col = 1 if (np.nan_to_num(auc1, nan=-1) >= np.nan_to_num(auc0, nan=-1)) else 0
        print(f"{outer_name}: pos_col={pos_col} (val AUC col0={auc0:.3f}, col1={auc1:.3f})")

        # Column-chosen ensemble probs
        val_probs_mean  = val_soft_mean[:,  pos_col]
        test_probs_mean = test_soft_mean[:, pos_col]

        # -------- Calibration metrics + reliability figure (post-calibration) --------
        fold_plot_dir = os.path.join(plot_dir, outer_name)
        os.makedirs(fold_plot_dir, exist_ok=True)
        test_labels = graph.y[test_idx].cpu().numpy()

        ece   = expected_calibration_error(test_probs_mean, test_labels, n_bins=15)
        brier = brier_score(test_probs_mean, test_labels)
        nll   = nll_loss(test_probs_mean, test_labels)
        save_reliability_diagram(
            test_probs_mean, test_labels,
            save_path=os.path.join(fold_plot_dir, f"{outer_name}_ensemble_reliability.png"),
            title=f"Reliability • {outer_name}"
        )

        # -------- Threshold tuning on VAL (ensemble) --------
        if np.unique(val_labels).size == 2:
            if th_strategy == "recall":
                fpr, tpr, thr = roc_curve(val_labels, val_probs_mean)
                ix = np.where(tpr >= float(target_recall))[0]
                if ix.size > 0:
                    best_tau = float(thr[ix[0]])
                    val_f1_at_tau = float('nan')
                    print(f"{outer_name}: τ* for recall≥{target_recall:.2f} -> {best_tau:.3f}")
                else:
                    best_tau = float(thr[-1]) if thr.size else 0.5
                    val_f1_at_tau = float('nan')
                    print(f"{outer_name}: could not meet target recall; fallback τ*={best_tau:.3f}")
            else:  # "f1"
                thresholds = np.linspace(0.01, 0.99, 99)
                f1s = [f1_score(val_labels, (val_probs_mean >= t).astype(int), zero_division=0) for t in thresholds]
                best_tau = float(thresholds[int(np.argmax(f1s))])
                val_f1_at_tau = float(np.max(f1s))
                print(f"{outer_name}: best τ*={best_tau:.3f} (val F1={val_f1_at_tau:.3f})")
        else:
            best_tau = 0.5
            val_f1_at_tau = float('nan')
            print(f"{outer_name}: single-class validation; using default τ*={best_tau}")

        # -------- Evaluate on TEST (ensemble) --------
        test_preds  = (test_probs_mean >= best_tau).astype(int)

        f1   = f1_score(test_labels, test_preds, zero_division=0)
        auc  = safe_roc_auc(test_labels, test_probs_mean)
        acc  = accuracy_score(test_labels, test_preds)
        bacc = balanced_accuracy_score(test_labels, test_preds)
        tn, fp, fn, tp = safe_confusion_counts(test_labels, test_preds)
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        # Per-seed TEST AUCs (pre-calibration per-seed; reporting ensemble calibration above)
        per_seed_auc = [
            safe_roc_auc(test_labels, F.softmax(d["test_logits"].to(device), dim=1)[:, pos_col].cpu().numpy())
            for d in per_seed
        ]

        # Save fold results
        all_results[outer_name] = {
            "pos_col": int(pos_col),
            "temperature_calibration": bool(use_temp_cal),
            "calibration_metrics": {
                "ece_15": float(ece),
                "brier": float(brier),
                "nll": float(nll),
                "reliability_png": os.path.join(fold_plot_dir, f"{outer_name}_ensemble_reliability.png"),
            },
            "threshold_strategy": th_strategy,
            "target_recall": float(target_recall) if th_strategy == "recall" else None,
            "best_threshold": float(best_tau),
            "val_f1_at_best_threshold": (None if np.isnan(val_f1_at_tau) else float(val_f1_at_tau)),
            "ensemble": {
                "f1": float(f1),
                "auc": (None if np.isnan(auc) else float(auc)),
                "acc": float(acc),
                "bacc": float(bacc),
                "sensitivity": float(sens),
                "specificity": float(spec),
            },
            "per_seed_auc": [None if np.isnan(a) else float(a) for a in per_seed_auc]
        }

        # ROC + CM figures
        if not np.isnan(auc):
            plot_roc_curve(test_labels, test_probs_mean,
                           save_path=os.path.join(fold_plot_dir, f"{outer_name}_ensemble_roc.png"))
        plot_confusion_matrix(test_labels, (test_probs_mean >= best_tau).astype(int),
                              save_path=os.path.join(fold_plot_dir, f"{outer_name}_ensemble_cm.png"))

    # --------------- Aggregate Summary ---------------
    metrics = ["f1", "auc", "acc", "bacc", "sensitivity", "specificity"]
    summary = {}
    for m in metrics:
        vals = []
        for k, v in all_results.items():
            if not k.startswith("fold_"):
                continue
            val = v["ensemble"].get(m, None)
            vals.append(np.nan if val is None else float(val))
        vals = np.array(vals, dtype=float)
        mean, std = np.nanmean(vals), np.nanstd(vals)
        summary[m] = "NA" if np.isnan(mean) else f"{mean:.2f} ± {std:.2f}"

    all_results["summary"] = summary

    out_path = os.path.join(ens_dir, "ensemble_test_metrics.json")
    os.makedirs(ens_dir, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=4)

    print("\nEnsemble test results saved to:", out_path)


if __name__ == "__main__":
    main()
