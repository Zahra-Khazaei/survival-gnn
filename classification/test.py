# test.py — leakage-free evaluation (per-fold KNN; train-only train; val←train; test←(train∪val))
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import json
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import (
    f1_score, roc_auc_score, accuracy_score,
    balanced_accuracy_score, confusion_matrix, roc_curve
)

from torch_geometric.data import Data
from torch_geometric.utils import to_undirected, add_self_loops

from data_loader import (
    load_data, load_splits, build_knn_graph
)
from utils import (
    load_config, get_device, compute_class_weights, get_max_degree,
    plot_roc_curve, plot_confusion_matrix, find_best_summary_path,
    FocalLoss, mc_predict_probs
)
from models import MODEL_REGISTRY

# =================== Threshold strategy ===================
#   "f1"     -> maximize F1 on validation (classic)
#   "recall" -> lowest threshold that achieves TARGET_RECALL on validation
THRESH_STRATEGY = "recall"   # "recall" or "f1"
TARGET_RECALL   = 0.67
# ==========================================================

# ---------------- Temperature scaling helpers ----------------
class TemperatureScaler(nn.Module):
    """Single-parameter temperature scaling (Guo et al., 2017)."""
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.temperature

def fit_temperature_on_val(logits_val: torch.Tensor, labels_val: torch.Tensor) -> TemperatureScaler:
    """
    Fit temperature on validation logits to minimize NLL.
    logits_val: (N_val, C), raw (pre-softmax) logits
    labels_val: (N_val,), int64
    """
    scaler = TemperatureScaler().to(logits_val.device)
    nll = nn.CrossEntropyLoss()
    # LBFGS is stable for 1-parameter problems; Adam also works
    optimizer = torch.optim.LBFGS([scaler.temperature], lr=0.01, max_iter=100)

    def closure():
        optimizer.zero_grad()
        loss = nll(scaler(logits_val), labels_val)
        loss.backward()
        return loss

    optimizer.step(closure)
    print(f"[calibration] Optimal temperature T = {scaler.temperature.item():.4f}")
    return scaler

# ---------------- Helpers ----------------
def safe_roc_auc(y_true, y_score):
    ys = np.unique(np.asarray(y_true))
    if ys.size < 2:
        return np.nan
    return roc_auc_score(y_true, y_score)

def safe_confusion_counts(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = (cm.ravel() if cm.size == 4 else [0, 0, 0, 0])
    return tn, fp, fn, tp

def describe_labels(name, y_true):
    vals, cnts = np.unique(np.asarray(y_true), return_counts=True)
    print(f"{name} test label distribution -> {dict(zip(vals.tolist(), cnts.tolist()))}")

def try_auc(y_true, y_score):
    try:
        return roc_auc_score(y_true, y_score)
    except Exception:
        return float("nan")

# ---------------- Load Config ----------------
parser = argparse.ArgumentParser(description="Run test on best Optuna model (leakage-free)")
parser.add_argument("--config", required=True, help="Path to YAML config file")
args = parser.parse_args()

config = load_config(args.config)

# Bayesian / MC-Dropout config
mc_cfg   = (config.get("bayesian", {}) or {}).get("mc_dropout", {})
MC_ON    = bool(mc_cfg.get("enabled", False))
MC_T     = int(mc_cfg.get("mc_passes", 50))

# Calibration config (deterministic path only)
calib_cfg = (config.get("calibration", {}) or {}).get("temperature", {})
CALIBRATE = bool(calib_cfg.get("enabled", False))

device = get_device()

# Resolve auto paths
data_folder = config.get("DATA_FOLDER", "Data")
if config.get("data_path", "") == "auto":
    config["data_path"] = f"{data_folder}/{config['LABEL_COL']}_scaled_data.csv"
if config.get("split_path", "") == "auto":
    config["split_path"] = f"{data_folder}/{config['LABEL_COL']}_data_splits.json"

LABEL_COL = config["LABEL_COL"]
ID_COL = "ID"

# ---------------- Build Features ----------------
data_df = pd.read_csv(config["data_path"])
FEATURES = [c for c in config.get("features", []) if c in data_df.columns]

# ---------------- Load Best Params ----------------
best_summary_path = find_best_summary_path(
    optuna_root="Optuna_results",
    model_name=config["model_name"],
    label_col=config["LABEL_COL"],
    graph_method=config["Graph_method"],
    graph_feature=config["Graph_feature"]
)
with open(best_summary_path, "r") as f:
    best_params = json.load(f)["best_params"]

# ---------------- Load data (no global edges) ----------------
df, X, y, id_to_index = load_data(config["data_path"], FEATURES, LABEL_COL, ID_COL)
splits = load_splits(config["split_path"])

graph = Data(
    x=torch.as_tensor(X, dtype=torch.float32),
    y=torch.as_tensor(y, dtype=torch.long)
).to(device)

# ---------------- Test Each Outer Fold ----------------
results = {}
plot_dir = os.path.dirname(best_summary_path).replace("logs", "test_plots")
os.makedirs(plot_dir, exist_ok=True)

# for saving per-fold predictions
preds_out_dir = os.path.join(plot_dir, "per_fold_preds")
os.makedirs(preds_out_dir, exist_ok=True)
inv_index = {v: k for k, v in id_to_index.items()}

for outer_name, outer_data in splits.items():
    print(f"\nEvaluating {outer_name} ...")

    # Test IDs from JSON
    test_idx = torch.tensor(
        [id_to_index[i] for i in outer_data["test"] if i in id_to_index],
        dtype=torch.long, device=device
    )

    # Collect all inner train+val IDs (as per your JSON structure)
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

    # ---------------- Model ----------------
    model_class = MODEL_REGISTRY[config["model_name"]]
    extra_args = (config.get("extra_args") or {}).copy()  # (max_degree set after train edges)

    model = model_class(
        in_channels=graph.x.size(1),
        hidden_channels=best_params["hidden_dim"],
        out_channels=2,
        num_layers=best_params["num_layers"],
        dropout=best_params["dropout"],
        activation="prelu",
        **extra_args
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=best_params["lr"], weight_decay=best_params["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

    # ---------------- Create a validation subset from train_val ----------------
    train_val_labels = graph.y[train_val_idx].cpu().numpy()
    val_frac = 0.10
    tr_sub = va_sub = None
    for attempt in range(8):
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=val_frac, random_state=42 + attempt)
        tr_try, va_try = next(splitter.split(np.zeros_like(train_val_labels), train_val_labels))
        if np.unique(train_val_labels[va_try]).size == 2:
            tr_sub, va_sub = tr_try, va_try
            break
        val_frac = min(val_frac + 0.05, 0.30)
    if tr_sub is None or va_sub is None:
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=val_frac, random_state=123)
        tr_sub, va_sub = next(splitter.split(np.zeros_like(train_val_labels), train_val_labels))
        print(f"WARNING: {outer_name}: validation remained single-class; threshold tuning will default to 0.5.")

    train_idx = train_val_idx[tr_sub]
    val_idx   = train_val_idx[va_sub]

    print(f"Train size: {train_idx.numel()}, Val size: {val_idx.numel()}, Test size: {test_idx.numel()}")

    # -------- Loss selection (weights from THIS fold's train subset) --------
    if config.get("use_focal_loss", False):
        class_weights = compute_class_weights(graph.y[train_idx], device)
        criterion = FocalLoss(alpha=class_weights, gamma=2.0)
        print("Using FocalLoss (per-fold train weights).")
    else:
        class_weights = compute_class_weights(graph.y[train_idx], device)
        criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
        print("Using weighted CrossEntropyLoss (per-fold train weights).")

    # ================== Build KNN edges per phase (NO GLOBAL GRAPH) ==================
    # TRAIN-ONLY edges (within train)
    edge_index_train_local = build_knn_graph(
        X=graph.x[train_idx],
        df=df.iloc[train_idx.cpu().numpy()],
        k=10,
        metric=config["Graph_method"],
        graph_feature=config["Graph_feature"],
        reference=None,
        df_reference=None,
    )
    edge_index_train = train_idx[edge_index_train_local.to(device)]
    # Make TRAIN backbone undirected + self-loops
    edge_index_train = to_undirected(edge_index_train, num_nodes=graph.x.size(0))
    edge_index_train, _ = add_self_loops(edge_index_train, num_nodes=graph.x.size(0))

    # Graphormer max_degree (if needed) -> re-init with proper max_degree
    if config["model_name"].lower() in ["graphormer", "graphormer_full"]:
        extra_args = (config.get("extra_args") or {}).copy()
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
        optimizer = torch.optim.Adam(model.parameters(), lr=best_params["lr"], weight_decay=best_params["weight_decay"])
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

    # VAL connectors: TRAIN → VAL (messages flow into VAL)
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
            train_idx[edge_index_v2t_local[1].to(device)],  # src: train
            val_idx[edge_index_v2t_local[0].to(device)],    # dst: val
        ],
        dim=0
    )

    # TEST connectors: (TRAIN ∪ VAL) → TEST
    trainval_idx = torch.cat([train_idx, val_idx], dim=0)

    # Backbone for (TRAIN ∪ VAL) (undirected + loops)
    edge_index_trainval_local = build_knn_graph(
        X=graph.x[trainval_idx],
        df=df.iloc[trainval_idx.cpu().numpy()],
        k=10,
        metric=config["Graph_method"],
        graph_feature=config["Graph_feature"],
        reference=None,
        df_reference=None,
    )
    edge_index_trainval = trainval_idx[edge_index_trainval_local.to(device)]
    edge_index_trainval = to_undirected(edge_index_trainval, num_nodes=graph.x.size(0))
    edge_index_trainval, _ = add_self_loops(edge_index_trainval, num_nodes=graph.x.size(0))

    # TEST queries into TRAIN∪VAL -> (trainval → test)
    edge_index_t2tv_local = build_knn_graph(
        X=graph.x[test_idx],
        df=df.iloc[test_idx.cpu().numpy()],
        k=10,
        metric=config["Graph_method"],
        graph_feature=config["Graph_feature"],
        reference=graph.x[trainval_idx],
        df_reference=df.iloc[trainval_idx.cpu().numpy()],
    )
    edge_index_trainval2test = torch.stack(
        [
            trainval_idx[edge_index_t2tv_local[1].to(device)],  # src: trainval
            test_idx[edge_index_t2tv_local[0].to(device)],      # dst: test
        ],
        dim=0
    )
    # ================================================================================

    # ---------------- Early stopping based on validation loss ----------------
    best_val_loss, patience, wait = float("inf"), 20, 0
    best_model_state = None

    for epoch in range(1, 101):
        # TRAIN on train-only edges
        model.train()
        optimizer.zero_grad()
        out_tr = model(graph.x, edge_index_train)
        loss = criterion(out_tr[train_idx], graph.y[train_idx])
        loss.backward()
        optimizer.step()
        scheduler.step()

        # VAL: TRAIN backbone + TRAIN→VAL
        model.eval()
        with torch.no_grad():
            edge_eval_val = torch.cat([edge_index_train, edge_index_train2val], dim=1)
            out_val = model(graph.x, edge_eval_val)
            val_loss = criterion(out_val[val_idx], graph.y[val_idx])

        if val_loss.item() < best_val_loss:
            best_val_loss = val_loss.item()
            best_model_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"Early stopping at epoch {epoch} (no val improvement for {patience} epochs)")
                break

    # ---------------- Threshold tuning (on validation) ----------------
    model.load_state_dict(best_model_state)
    model.eval()

    # Build eval edge indices
    edge_eval_val  = torch.cat([edge_index_train, edge_index_train2val], dim=1)
    edge_eval_test = torch.cat([edge_index_trainval, edge_index_trainval2test], dim=1)

    val_labels  = graph.y[val_idx].cpu().numpy()
    test_labels = graph.y[test_idx].cpu().numpy()

    # Deterministic logits for AUC/pos_col + (maybe) calibration
    with torch.no_grad():
        logits_val_det  = model(graph.x, edge_eval_val)[val_idx]
        logits_test_det = model(graph.x, edge_eval_test)[test_idx]
    val_soft_det  = F.softmax(logits_val_det,  dim=1).cpu().numpy()
    test_soft_det = F.softmax(logits_test_det, dim=1).cpu().numpy()

    # Pick positive column by higher validation AUC (deterministic comparison)
    auc_col0 = try_auc(val_labels, val_soft_det[:, 0])
    auc_col1 = try_auc(val_labels, val_soft_det[:, 1])
    pos_col  = 1 if (np.nan_to_num(auc_col1, nan=-1) >= np.nan_to_num(auc_col0, nan=-1)) else 0
    print(f"{outer_name}: chose logit column {pos_col} as positive (val AUC col0={auc_col0:.3f}, col1={auc_col1:.3f})")

    # --- Optional Temperature Scaling (deterministic path only) ---
    scaler = None
    if CALIBRATE and not MC_ON:
        scaler = fit_temperature_on_val(logits_val_det, graph.y[val_idx])
        with torch.no_grad():
            val_soft_det  = F.softmax(scaler(logits_val_det),  dim=1).cpu().numpy()
            test_soft_det = F.softmax(scaler(logits_test_det), dim=1).cpu().numpy()
        print(f"[calibration] Applied temperature scaling to VAL/TEST (deterministic).")
    elif CALIBRATE and MC_ON:
        print("[calibration] Skipping temperature scaling because MC-Dropout is enabled (mc_predict_probs uses raw logits internally).")

    # Get probs for thresholding/metrics
    if MC_ON:
        # MC: use mean prob for metrics/thresholding (pos_col already chosen)
        mean_val_all, var_val_all, ent_val_all = mc_predict_probs(
            model, graph.x, edge_index=edge_eval_val,  T=MC_T, pos_col=pos_col
        )
        mean_test_all, var_test_all, ent_test_all = mc_predict_probs(
            model, graph.x, edge_index=edge_eval_test, T=MC_T, pos_col=pos_col
        )
        val_probs    = mean_val_all[val_idx].cpu().numpy()
        test_probs   = mean_test_all[test_idx].cpu().numpy()
        test_var     = var_test_all[test_idx].cpu().numpy()
        test_entropy = ent_test_all[test_idx].cpu().numpy()
    else:
        val_probs    = val_soft_det[:, pos_col]
        test_probs   = test_soft_det[:, pos_col]
        test_var     = None
        test_entropy = None

    print(f"{outer_name}: val probs min/mean/max  = {val_probs.min():.3f}/{val_probs.mean():.3f}/{val_probs.max():.3f}")
    print(f"{outer_name}: test probs min/mean/max = {test_probs.min():.3f}/{test_probs.mean():.3f}/{test_probs.max():.3f}")

    # Tune threshold
    if np.unique(val_labels).size == 2:
        if THRESH_STRATEGY.lower() == "recall":
            fpr, tpr, thr = roc_curve(val_labels, val_probs)
            ix = np.where(tpr >= TARGET_RECALL)[0]
            if ix.size > 0:
                best_tau = float(thr[ix[0]])
                best_val_f1 = float('nan')
                print(f"{outer_name}: τ* chosen for recall≥{TARGET_RECALL:.2f} -> τ*={best_tau:.3f}")
            else:
                best_tau = float(thr[-1]) if thr.size else 0.5
                best_val_f1 = float('nan')
                print(f"{outer_name}: could not meet target recall; fallback τ*={best_tau:.3f}")
        else:  # "f1"
            thresholds = np.linspace(0.01, 0.99, 99)
            f1s = [f1_score(val_labels, (val_probs >= t).astype(int), zero_division=0) for t in thresholds]
            best_tau = float(thresholds[int(np.argmax(f1s))])
            best_val_f1 = float(np.max(f1s))
            print(f"{outer_name}: best threshold τ* = {best_tau:.3f} (val F1 = {best_val_f1:.3f})")
    else:
        best_tau = 0.5
        best_val_f1 = float('nan')
        print(f"{outer_name}: single-class validation; using default threshold {best_tau}")

    # ---------------- Evaluate on test set ----------------
    test_preds = (test_probs >= best_tau).astype(int)
    describe_labels(outer_name, test_labels)

    pos_mask = (test_labels == 1)
    print(f"{outer_name}: test positives count = {int(pos_mask.sum())}")
    if int(pos_mask.sum()) > 0:
        print(f"{outer_name}: probs for true positives = {np.round(test_probs[pos_mask], 4)}")

    f1  = f1_score(test_labels, test_preds, zero_division=0)
    auc = safe_roc_auc(test_labels, test_probs)
    acc = accuracy_score(test_labels, test_preds)
    bacc = balanced_accuracy_score(test_labels, test_preds)
    tn, fp, fn, tp = safe_confusion_counts(test_labels, test_preds)
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    print(f"{outer_name}: predicted positives on test = {int((test_preds == 1).sum())} / {len(test_preds)}")

    results[outer_name] = {
        "f1": float(f1),
        "auc": (None if np.isnan(auc) else float(auc)),
        "acc": float(acc),
        "bacc": float(bacc),
        "sensitivity": float(sens),
        "specificity": float(spec),
        "best_threshold": float(best_tau),
        "val_f1_at_best_threshold": (None if np.isnan(best_val_f1) else float(best_val_f1)),
        "calibration": {
            "enabled": bool(CALIBRATE),
            "applied": bool(CALIBRATE and not MC_ON)
        }
    }

    # ---- Save per-patient test predictions (with uncertainty if MC_ON) ----
    test_nodes = test_idx.cpu().numpy().astype(int)
    test_pids  = [inv_index[int(n)] for n in test_nodes]
    per_fold_df = pd.DataFrame({
        "patient_id": test_pids,
        "node_idx": test_nodes,
        "y_true": test_labels.astype(int),
        "prob_pos": np.round(test_probs, 6),
        "pred": (test_probs >= best_tau).astype(int)
    })
    if MC_ON:
        per_fold_df["mc_var_prob"] = np.round(test_var, 6)
        per_fold_df["mc_entropy"]  = np.round(test_entropy, 6)
    per_fold_df.to_csv(os.path.join(preds_out_dir, f"{outer_name}_test_preds.csv"), index=False)

    # ---------------- Plots ----------------
    if not np.isnan(auc):
        plot_roc_curve(test_labels, test_probs, save_path=os.path.join(plot_dir, f"{outer_name}_roc.png"))
    plot_confusion_matrix(test_labels, test_preds, save_path=os.path.join(plot_dir, f"{outer_name}_cm.png"))

# ---------------- Aggregate Summary ----------------
summary = {}
metrics = ["f1", "auc", "acc", "bacc", "sensitivity", "specificity"]

for metric in metrics:
    vals = [float(v) if v not in [None, "NA"] else np.nan
            for v in [results[k][metric] for k in results if k.startswith("fold_")]]
    mean, std = np.nanmean(vals), np.nanstd(vals)
    summary[metric] = "NA" if np.isnan(mean) else f"{mean:.2f} ± {std:.2f}"

results["summary"] = summary

output_path = os.path.join(os.path.dirname(best_summary_path), "test_metrics.json")
with open(output_path, "w") as f:
    json.dump(results, f, indent=4)

print("\n Test results saved to:", output_path)
