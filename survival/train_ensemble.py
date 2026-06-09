# train_ensemble.py
# Deep ensemble across random seeds for survival GNNs.
# Leakage-free wiring: train backbone → val connectors → test connectors.
# Ensemble = average hazards across independently trained models (different seeds).
# No temperature scaling (not applicable to hazard outputs).
# Threshold tuning per horizon using BACC maximization on VAL set.
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import json
import argparse
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import roc_curve, roc_auc_score

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from torch_geometric.data import Data
from torch_geometric.utils import to_undirected, add_self_loops

from data_loader import load_survival_data, load_splits, build_knn_graph
from models.survival_node import SurvivalNodeGNN
from losses.survival_loss import DiscreteTimeSurvivalLoss
from metrics.survival_metrics import (
    evaluate_survival,
    find_best_threshold_for_bacc,
    risk_at_horizon_from_hazards,
    hazards_to_survival,
    concordance_index,
    binary_metrics_at_horizon,
)
from utils import load_config, get_device, set_seed


# ─────────────────── Leakage-free edge wiring ───────────────────

def build_eval_edges(
    graph_x: torch.Tensor,
    df: pd.DataFrame,
    config: dict,
    device: torch.device,
    train_idx: torch.Tensor,
    val_idx: torch.Tensor,
    test_idx: torch.Tensor,
    k: int = 10,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Leakage-free 3-phase edge construction.
      TRAIN backbone:  undirected + self-loops (train only)
      VAL connectors:  train -> val
      TEST connectors: (train+val) -> test
    Returns (edge_train, edge_eval_val, edge_eval_test).
    """

    def _undirected_selfloops(local_ei, ref_idx):
        ei = ref_idx[local_ei.to(device)]
        ei = to_undirected(ei, num_nodes=graph_x.size(0))
        ei, _ = add_self_loops(ei, num_nodes=graph_x.size(0))
        return ei

    # TRAIN backbone
    k_tr = min(k, int(train_idx.numel()) - 1)
    ei_tr = build_knn_graph(
        X=graph_x[train_idx], df=df.iloc[train_idx.cpu().numpy()],
        k=k_tr, metric=config["Graph_method"], graph_feature=config["Graph_feature"],
    )
    edge_train = _undirected_selfloops(ei_tr, train_idx)

    # VAL connectors: train -> val
    k_v = min(k, int(train_idx.numel()))
    ei_v = build_knn_graph(
        X=graph_x[val_idx], df=df.iloc[val_idx.cpu().numpy()],
        k=k_v, metric=config["Graph_method"], graph_feature=config["Graph_feature"],
        reference=graph_x[train_idx], df_reference=df.iloc[train_idx.cpu().numpy()],
    )
    edge_tr2val = torch.stack([
        train_idx[ei_v[1].to(device)],
        val_idx[ei_v[0].to(device)],
    ], dim=0)
    edge_eval_val = torch.cat([edge_train, edge_tr2val], dim=1)

    # TEST connectors: (train+val) -> test
    tv_idx = torch.cat([train_idx, val_idx], dim=0)
    k_tv = min(k, int(tv_idx.numel()) - 1)
    ei_tv = build_knn_graph(
        X=graph_x[tv_idx], df=df.iloc[tv_idx.cpu().numpy()],
        k=k_tv, metric=config["Graph_method"], graph_feature=config["Graph_feature"],
    )
    edge_trainval = _undirected_selfloops(ei_tv, tv_idx)

    k_t = min(k, int(tv_idx.numel()))
    ei_t = build_knn_graph(
        X=graph_x[test_idx], df=df.iloc[test_idx.cpu().numpy()],
        k=k_t, metric=config["Graph_method"], graph_feature=config["Graph_feature"],
        reference=graph_x[tv_idx], df_reference=df.iloc[tv_idx.cpu().numpy()],
    )
    edge_tv2test = torch.stack([
        tv_idx[ei_t[1].to(device)],
        test_idx[ei_t[0].to(device)],
    ], dim=0)
    edge_eval_test = torch.cat([edge_trainval, edge_tv2test], dim=1)

    return edge_train, edge_eval_val, edge_eval_test


# ─────────────────── Train one seed ───────────────────

def train_one_model(
    config: dict,
    best_params: dict,
    graph: Data,
    train_idx: torch.Tensor,
    val_idx: torch.Tensor,
    edge_train: torch.Tensor,
    edge_eval_val: torch.Tensor,
    bin_width: float,
    num_bins: int,
    device: torch.device,
    seed: int,
    num_epochs: int = 100,
    patience: int = 20,
    grad_clip: float = 1.0,
) -> SurvivalNodeGNN:
    """Train a single SurvivalNodeGNN for one seed with early stopping on val loss."""
    set_seed(seed)

    model_name = str(config.get("model_name", "GCN"))
    if model_name == "GraphSAGE":
        model_name = "GRAPHSAGE"

    model = SurvivalNodeGNN(
        in_channels=graph.x.size(1),
        hidden_channels=int(best_params["hidden_dim"]),
        num_layers=int(best_params["num_layers"]),
        num_bins=num_bins,
        model_name=model_name,
        dropout=float(best_params["dropout"]),
        num_heads=int(best_params.get("num_heads", 1)),
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(best_params["lr"]),
        weight_decay=float(best_params["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer, gamma=float(best_params.get("gamma", 0.99))
    )
    criterion = DiscreteTimeSurvivalLoss(
        bin_width=bin_width,
        alpha=float(best_params.get("alpha", 0.0)),
        sigma=float(best_params.get("sigma", 0.1)),
    ).to(device)

    best_val_cindex = -1.0
    best_state = None
    wait = 0

    for epoch in range(1, num_epochs + 1):
        model.train()
        optimizer.zero_grad()
        hazards_tr = model(graph, edge_index=edge_train)
        loss = criterion(hazards_tr[train_idx], graph.time[train_idx], graph.event[train_idx])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            hazards_val = model(graph, edge_index=edge_eval_val)
            risk_full = (1.0 - hazards_to_survival(hazards_val[val_idx].cpu()))[:, -1]
            val_cindex = concordance_index(
                graph.time[val_idx].cpu().numpy(),
                graph.event[val_idx].cpu().numpy(),
                risk_full.numpy(),
            )

        if not np.isnan(val_cindex) and val_cindex > best_val_cindex:
            best_val_cindex = val_cindex
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


@torch.no_grad()
def predict_hazards(
    model: SurvivalNodeGNN,
    graph: Data,
    edge_index_eval: torch.Tensor,
    index_subset: torch.Tensor,
) -> torch.Tensor:
    """Return hazards [N_subset, num_bins] on CPU."""
    model.eval()
    hazards = model(graph, edge_index=edge_index_eval)
    return hazards[index_subset].detach().cpu()


# ─────────────────── Survival-specific plots ───────────────────

def _km_estimate(times_np: np.ndarray, events_np: np.ndarray):
    """Kaplan-Meier estimator with Greenwood 95% CI (log-log transform).
    Returns (time_axis, surv_probs, ci_lower, ci_upper) starting at t=0."""
    from scipy.stats import norm
    order = np.argsort(times_np)
    t = times_np[order]
    e = events_np[order]
    S = 1.0
    greenwood_sum = 0.0
    times_out, surv_out, ci_lo_out, ci_hi_out = [0.0], [1.0], [1.0], [1.0]
    for ut in np.unique(t):
        n = int(np.sum(t >= ut))
        d = int(np.sum((t == ut) & (e == 1)))
        if n > 0:
            S *= 1.0 - d / n
        # Greenwood's formula accumulator (avoid division by zero)
        if n > d and d > 0:
            greenwood_sum += d / (n * (n - d))
        # 95% CI via log-log transform (better coverage near 0/1)
        if S > 0 and S < 1 and greenwood_sum > 0:
            log_log_S = np.log(-np.log(S))
            se_log_log = np.sqrt(greenwood_sum) / abs(np.log(S))
            z = norm.ppf(0.975)
            ci_lo = np.exp(-np.exp(log_log_S + z * se_log_log))
            ci_hi = np.exp(-np.exp(log_log_S - z * se_log_log))
        else:
            ci_lo, ci_hi = S, S
        times_out.append(float(ut))
        surv_out.append(S)
        ci_lo_out.append(float(ci_lo))
        ci_hi_out.append(float(ci_hi))
    return (np.array(times_out), np.array(surv_out),
            np.array(ci_lo_out), np.array(ci_hi_out))


def _logrank_pvalue(t1, e1, t2, e2):
    """Log-rank test p-value for two groups."""
    from scipy.stats import chi2
    all_times = np.unique(np.concatenate([t1[e1 == 1], t2[e2 == 1]]))
    O1_total = E1_total = V_total = 0.0
    for ut in all_times:
        n1 = int(np.sum(t1 >= ut))
        n2 = int(np.sum(t2 >= ut))
        d1 = int(np.sum((t1 == ut) & (e1 == 1)))
        d2 = int(np.sum((t2 == ut) & (e2 == 1)))
        n = n1 + n2
        d = d1 + d2
        if n == 0:
            continue
        E1_total += d * n1 / n
        O1_total += d1
        if n > 1:
            V_total += d * n1 * n2 * (n - d) / (n ** 2 * (n - 1))
    if V_total == 0:
        return 1.0
    stat = (O1_total - E1_total) ** 2 / V_total
    return float(chi2.sf(stat, df=1))


def plot_km_stratified(
    hazards: torch.Tensor,
    times: torch.Tensor,
    events: torch.Tensor,
    bin_width: float,
    horizons_months: list,
    save_path: str,
    title: str = "KM — Risk Stratified",
):
    """Kaplan-Meier curves stratified by median ensemble risk at the first horizon.
    Paper-quality: 95% CI bands, log-rank p-value, number-at-risk table."""
    horizon   = float(horizons_months[0])
    risk_np   = risk_at_horizon_from_hazards(hazards, bin_width, horizon).cpu().numpy()
    times_np  = times.cpu().numpy().astype(float)
    events_np = events.cpu().numpy().astype(int)

    median_risk = float(np.median(risk_np))
    hi_mask = risk_np >= median_risk
    lo_mask = ~hi_mask

    groups = [
        (hi_mask, "High risk", "#d62728"),
        (lo_mask, "Low risk",  "#1f77b4"),
    ]

    fig, (ax, ax_nar) = plt.subplots(
        2, 1, figsize=(8, 6),
        gridspec_kw={"height_ratios": [5, 1], "hspace": 0.05},
    )

    km_data = {}
    for mask, label, color in groups:
        if mask.sum() == 0:
            continue
        t_km, s_km, ci_lo, ci_hi = _km_estimate(times_np[mask], events_np[mask])
        ax.step(t_km, s_km, where="post",
                label=f"{label} (n={int(mask.sum())})", color=color, linewidth=1.8)
        # fill CI band (step-wise)
        ax.fill_between(t_km, ci_lo, ci_hi, step="post", alpha=0.15, color=color)
        km_data[label] = (mask, t_km, s_km)

    # log-rank p-value
    p = _logrank_pvalue(
        times_np[hi_mask], events_np[hi_mask],
        times_np[lo_mask], events_np[lo_mask],
    )
    p_str = f"p < 0.001" if p < 0.001 else f"p = {p:.3f}"
    ax.text(0.03, 0.03, p_str, transform=ax.transAxes,
            ha="left", va="bottom", fontsize=11,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

    for h in horizons_months:
        ax.axvline(x=float(h), color="gray", linestyle="--", linewidth=0.8, alpha=0.5)

    ax.set_ylabel("Survival probability", fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.set_xlim(left=0)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(fontsize=10, loc="upper right")
    ax.tick_params(labelbottom=False)  # hide x ticks on top panel

    # time points for number-at-risk table — use visible x range so horizon
    # dashed lines (which extend beyond max follow-up) are included
    x_right = ax.get_xlim()[1]
    nar_times = np.array([t for t in [30, 60, 90, 120, 150, 180] if t <= x_right])

    # ── Number-at-risk table ──
    ax_nar.set_xlim(ax.get_xlim())
    ax_nar.set_ylim(-0.5, len(groups) - 0.5)
    ax_nar.axis("off")

    row_labels = ["High risk", "Low risk"]
    row_colors = ["#d62728", "#1f77b4"]
    for row_i, (mask, label, color) in enumerate(groups):
        t_g = times_np[mask]
        y_pos = len(groups) - 1 - row_i
        for x_t in nar_times:
            n_at_risk = int(np.sum(t_g >= x_t))
            ax_nar.text(x_t, y_pos, str(n_at_risk),
                        ha="center", va="center", fontsize=9, color=color, fontweight="bold")
        ax_nar.text(-0.01, y_pos, label,
                    ha="right", va="center", fontsize=9, color=color, fontweight="bold",
                    transform=ax_nar.get_yaxis_transform())

    ax_nar.set_xlabel("Time (months)", fontsize=12)
    # align x-ticks of bottom panel with top panel
    ax_nar.set_xticks(nar_times)
    ax_nar.xaxis.set_visible(True)
    ax_nar.spines["top"].set_visible(False)
    ax_nar.spines["right"].set_visible(False)
    ax_nar.spines["left"].set_visible(False)
    ax_nar.spines["bottom"].set_visible(False)
    ax_nar.tick_params(left=False, labelleft=False)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_roc_at_horizon(
    hazards: torch.Tensor,
    times: torch.Tensor,
    events: torch.Tensor,
    bin_width: float,
    horizon_months: float,
    save_path: str,
    title: str = "ROC",
):
    """Time-dependent ROC curve at a specified horizon."""
    risk_np   = risk_at_horizon_from_hazards(hazards, bin_width, horizon_months).cpu().numpy()
    times_np  = times.cpu().numpy().astype(float)
    events_np = events.cpu().numpy().astype(int)

    pos  = (events_np == 1) & (times_np <= horizon_months)
    neg  = times_np > horizon_months
    keep = pos | neg

    if keep.sum() < 2 or len(np.unique(pos[keep])) < 2:
        return

    y      = pos[keep].astype(int)
    scores = risk_np[keep]

    fpr, tpr, _ = roc_curve(y, scores)
    roc_auc = roc_auc_score(y, scores)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}", color="steelblue", linewidth=2)
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=200)
    plt.close(fig)


def plot_ibs_curve(
    hazards: torch.Tensor,
    times: torch.Tensor,
    events: torch.Tensor,
    bin_width: float,
    ibs_max_time: float,
    save_path: str,
    title: str = "Brier Score over Time",
):
    """IPCW Brier score as a function of time horizon (calibration curve)."""
    horizons = np.arange(bin_width, ibs_max_time + 1e-9, bin_width)
    briers   = []
    for h in horizons:
        m = binary_metrics_at_horizon(
            hazards, times, events, bin_width, float(h), threshold=0.5
        )
        briers.append(m.get("brier_ipcw", float("nan")))
    briers = np.array(briers, dtype=float)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(horizons, briers, color="steelblue", marker="o", markersize=3, linewidth=1.5)
    ax.axhline(0.25, color="gray", linestyle="--", linewidth=0.8, label="Null (0.25)")
    ax.set_xlabel("Time horizon (months)")
    ax.set_ylabel("IPCW Brier Score")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=200)
    plt.close(fig)


def plot_cindex_per_fold(
    fold_names: list,
    cindex_values: list,
    save_path: str,
):
    """Bar chart of C-index per outer fold with mean +/- std annotation."""
    valid_vals = [v for v in cindex_values if v is not None and not np.isnan(v)]
    mean_c = float(np.mean(valid_vals)) if valid_vals else 0.0
    std_c  = float(np.std(valid_vals))  if valid_vals else 0.0
    vals   = [v if (v is not None and not np.isnan(v)) else 0.0 for v in cindex_values]

    fig, ax = plt.subplots(figsize=(max(5, len(fold_names)), 4))
    ax.bar(np.arange(len(fold_names)), vals, color="steelblue", edgecolor="white")
    ax.axhline(mean_c, color="tomato", linestyle="--", linewidth=1.2,
               label=f"Mean = {mean_c:.3f} +/- {std_c:.3f}")
    ax.set_xticks(np.arange(len(fold_names)))
    ax.set_xticklabels(fold_names, rotation=20, ha="right")
    ax.set_ylabel("C-index")
    ax.set_ylim(0, 1.05)
    ax.set_title("Ensemble C-index per Outer Fold")
    ax.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=200)
    plt.close(fig)


# ─────────────────── Main ───────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Survival deep ensemble with leakage-free wiring"
    )
    parser.add_argument("--config",     required=True, help="Path to YAML config")
    parser.add_argument("--optuna_dir", required=True,
                        help="Optuna run directory (contains logs/best_summary.json)")
    parser.add_argument("--out_dir",    default=None,
                        help="Override output base directory (default: <optuna_dir>/logs)")
    parser.add_argument("--k",          type=int, default=None,
                        help="Override k (number of KNN neighbors) from config")
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
    TIME_COL  = config["TIME_COL"]
    ID_COL    = "ID"

    set_seed(int(config.get("seed", 42)))

    # Survival binning
    bin_width       = float(config.get("bin_width", 6.0))
    max_time        = float(config.get("max_time", 216.0))
    num_bins        = int(max_time / bin_width)
    horizons_months = [float(h) for h in config.get("horizons_months", [60.0, 120.0])]
    ibs_max_time    = float(config.get("ibs_max_time", min(horizons_months)))
    k               = args.k if args.k is not None else int(config.get("k", 10))
    num_epochs      = int(config.get("num_epochs", 100))
    patience        = int(config.get("patience", 20))
    grad_clip       = float(config.get("grad_clip_norm", 1.0))

    ens_cfg = config.get("ensemble", {}) or {}
    seeds   = ens_cfg.get("seeds", [42, 43, 44, 45, 46])

    # Load best Optuna hyperparameters
    best_summary_path = os.path.join(args.optuna_dir, "logs", "best_summary.json")
    if not os.path.exists(best_summary_path):
        raise FileNotFoundError(f"best_summary.json not found: {best_summary_path}")
    with open(best_summary_path) as fh:
        best_summary = json.load(fh)
    best_params = best_summary.get("best_params", {})
    print("Loaded best params:", best_params)

    # Output directories
    base_out = args.out_dir or os.path.join(args.optuna_dir, "logs")
    ens_dir  = os.path.join(base_out, "ensemble")
    plot_dir = os.path.join(ens_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    # Load data
    df_check  = pd.read_csv(config["data_path"])
    feat_cols = [c for c in config.get("features", []) if c in df_check.columns]
    df, X, times_all, events_all, id_to_index = load_survival_data(
        config["data_path"], feat_cols, LABEL_COL, TIME_COL, ID_COL
    )
    splits = load_splits(config["split_path"])

    graph = Data(
        x=torch.as_tensor(X,           dtype=torch.float32),
        time=torch.as_tensor(times_all,  dtype=torch.float32),
        event=torch.as_tensor(events_all, dtype=torch.long),
    ).to(device)

    all_results: Dict[str, dict] = {}
    fold_names_list:  List[str]   = []
    fold_cindex_vals: List[float] = []
    ibs_vals:         List[float] = []

    # ─────── Outer fold loop ───────
    for outer_name, outer_data in splits.items():
        print(f"\n=== Ensemble on {outer_name} ===")

        test_idx = torch.tensor(
            [id_to_index[i] for i in outer_data["test"] if i in id_to_index],
            dtype=torch.long, device=device,
        )

        inner_keys    = [kk for kk in outer_data if kk.startswith("inner_folds_")]
        # dict.fromkeys preserves insertion order while deduplicating
        train_val_ids = list(dict.fromkeys(
            pid
            for kk in inner_keys
            for fold in outer_data[kk]
            for pid in (fold["train"] + fold["validation"])
        ))
        train_val_idx = torch.tensor(
            [id_to_index[i] for i in train_val_ids if i in id_to_index],
            dtype=torch.long, device=device,
        )

        # Stratified val split (stratify on event indicator)
        events_tv = graph.event[train_val_idx].cpu().numpy()
        val_frac  = 0.10
        tr_sub = va_sub = None
        for attempt in range(8):
            sss = StratifiedShuffleSplit(
                n_splits=1, test_size=val_frac, random_state=42 + attempt
            )
            try:
                tr_try, va_try = next(sss.split(np.zeros_like(events_tv), events_tv))
                if np.unique(events_tv[va_try]).size == 2:
                    tr_sub, va_sub = tr_try, va_try
                    break
            except Exception:
                pass
            val_frac = min(val_frac + 0.05, 0.30)

        if tr_sub is None:
            sss = StratifiedShuffleSplit(n_splits=1, test_size=val_frac, random_state=123)
            tr_sub, va_sub = next(sss.split(np.zeros_like(events_tv), events_tv))
            print(f"  WARNING: {outer_name}: val split may be single-class.")

        train_idx = train_val_idx[tr_sub]
        val_idx   = train_val_idx[va_sub]
        print(f"  Train={train_idx.numel()} | Val={val_idx.numel()} | Test={test_idx.numel()}")

        # Leakage-free edge wiring
        edge_train, edge_eval_val, edge_eval_test = build_eval_edges(
            graph_x=graph.x, df=df, config=config, device=device,
            train_idx=train_idx, val_idx=val_idx, test_idx=test_idx, k=k,
        )

        # Per-seed training
        per_seed_val_haz:  List[torch.Tensor] = []
        per_seed_test_haz: List[torch.Tensor] = []
        per_seed_cindex:   List[float]        = []

        for seed in seeds:
            print(f"  -> seed {seed} ...")
            model = train_one_model(
                config=config, best_params=best_params, graph=graph,
                train_idx=train_idx, val_idx=val_idx,
                edge_train=edge_train, edge_eval_val=edge_eval_val,
                bin_width=bin_width, num_bins=num_bins,
                device=device, seed=seed,
                num_epochs=num_epochs, patience=patience, grad_clip=grad_clip,
            )

            val_haz  = predict_hazards(model, graph, edge_eval_val,  val_idx)
            test_haz = predict_hazards(model, graph, edge_eval_test, test_idx)
            per_seed_val_haz.append(val_haz)
            per_seed_test_haz.append(test_haz)

            # Per-seed C-index using final-bin cumulative risk as discriminator
            risk_full = 1.0 - hazards_to_survival(test_haz)[:, -1]
            ci = concordance_index(
                graph.time[test_idx].cpu().numpy(),
                graph.event[test_idx].cpu().numpy(),
                risk_full.numpy(),
            )
            per_seed_cindex.append(float(ci) if not np.isnan(ci) else float("nan"))

        # Average hazards across seeds
        val_haz_ens  = torch.stack(per_seed_val_haz,  dim=0).mean(dim=0)  # [Nv, T]
        test_haz_ens = torch.stack(per_seed_test_haz, dim=0).mean(dim=0)  # [Nt, T]

        val_haz_t  = val_haz_ens.to(device)
        test_haz_t = test_haz_ens.to(device)
        val_times  = graph.time[val_idx]
        val_events = graph.event[val_idx]
        test_times  = graph.time[test_idx]
        test_events = graph.event[test_idx]

        # Threshold tuning per horizon on VAL (maximize BACC)
        best_thresholds: Dict[float, float] = {}
        for h in horizons_months:
            thr, bacc = find_best_threshold_for_bacc(
                val_haz_t, val_times, val_events, bin_width, float(h)
            )
            best_thresholds[float(h)] = float(thr)
            print(f"  Horizon {int(h)}m: best_thr={thr:.3f} (val BACC={bacc:.3f})")

        # Global survival evaluation on TEST
        metrics = evaluate_survival(
            hazards=test_haz_t, times=test_times, events=test_events,
            bin_width=bin_width, horizons_months=horizons_months,
            ibs_max_time=ibs_max_time,
        )

        c_idx_ens = float(metrics.get("c_index", float("nan")))
        ibs_ens   = float(metrics.get("ibs_ipcw", float("nan")))
        print(f"  Ensemble C-index={c_idx_ens:.4f} | IBS-IPCW={ibs_ens:.4f}")

        # Per-horizon metrics with tuned thresholds
        horizon_results: Dict[int, dict] = {}
        for h in horizons_months:
            h_float = float(h)
            thr_h   = best_thresholds.get(h_float, 0.5)
            bm = binary_metrics_at_horizon(
                test_haz_t, test_times, test_events, bin_width, h_float, threshold=thr_h
            )
            horizon_results[int(h)] = {
                "threshold":   thr_h,
                "auc":         bm.get("auc",              float("nan")),
                "auc_ipcw":    bm.get("auc_ipcw",         float("nan")),
                "bacc":        bm.get("balanced_accuracy", float("nan")),
                "sensitivity": bm.get("sensitivity",       float("nan")),
                "specificity": bm.get("specificity",       float("nan")),
                "brier_ipcw":  bm.get("brier_ipcw",        float("nan")),
                "n_used":      bm.get("n_used", 0),
            }
            print(
                f"  Horizon {int(h)}m | AUC={bm.get('auc', float('nan')):.3f} | "
                f"BACC={bm.get('balanced_accuracy', float('nan')):.3f}"
            )

        fold_names_list.append(outer_name)
        fold_cindex_vals.append(c_idx_ens)
        ibs_vals.append(ibs_ens)

        all_results[outer_name] = {
            "c_index":         c_idx_ens,
            "ibs_ipcw":        ibs_ens,
            "horizons":        horizon_results,
            "per_seed_cindex": per_seed_cindex,
        }

        # ── Figures ──
        fold_plot_dir = os.path.join(plot_dir, outer_name)
        os.makedirs(fold_plot_dir, exist_ok=True)

        plot_km_stratified(
            hazards=test_haz_t, times=test_times, events=test_events,
            bin_width=bin_width, horizons_months=horizons_months,
            save_path=os.path.join(fold_plot_dir, f"{outer_name}_km_risk_stratified.png"),
            title=f"KM Risk Stratified • {outer_name}",
        )

        for h in horizons_months:
            plot_roc_at_horizon(
                hazards=test_haz_t, times=test_times, events=test_events,
                bin_width=bin_width, horizon_months=float(h),
                save_path=os.path.join(fold_plot_dir, f"{outer_name}_roc_h{int(h)}m.png"),
                title=f"ROC @ {int(h)}m • {outer_name}",
            )

        if ibs_max_time > bin_width:
            plot_ibs_curve(
                hazards=test_haz_t, times=test_times, events=test_events,
                bin_width=bin_width, ibs_max_time=ibs_max_time,
                save_path=os.path.join(fold_plot_dir, f"{outer_name}_ibs_curve.png"),
                title=f"IBS Curve • {outer_name}",
            )

    # ─────── Aggregate + save ───────
    def _mean_std(vals):
        arr = np.array([v for v in vals if v is not None and not np.isnan(v)], dtype=float)
        if arr.size == 0:
            return "NA"
        return f"{float(np.nanmean(arr)):.4f} +/- {float(np.nanstd(arr)):.4f}"

    all_results["summary"] = {
        "c_index":  _mean_std(fold_cindex_vals),
        "ibs_ipcw": _mean_std(ibs_vals),
    }

    # Serialize: replace float("nan") with None for valid JSON
    def _clean(obj):
        if isinstance(obj, float) and np.isnan(obj):
            return None
        if isinstance(obj, dict):
            return {k: _clean(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_clean(v) for v in obj]
        return obj

    out_path = os.path.join(ens_dir, "ensemble_test_metrics.json")
    with open(out_path, "w") as fh:
        json.dump(_clean(all_results), fh, indent=4)

    print(f"\nEnsemble results saved to: {out_path}")
    print(f"C-index:  {_mean_std(fold_cindex_vals)}")
    print(f"IBS-IPCW: {_mean_std(ibs_vals)}")

    # C-index summary bar chart (across all folds)
    if fold_names_list:
        plot_cindex_per_fold(
            fold_names=fold_names_list,
            cindex_values=fold_cindex_vals,
            save_path=os.path.join(plot_dir, "cindex_per_fold.png"),
        )


if __name__ == "__main__":
    main()
