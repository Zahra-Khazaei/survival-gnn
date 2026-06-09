# survival_interpret_example.py
"""
Leakage-aware interpretation runner for survival GNNs with auto-candidate selection.

What it does
------------
- Loads best Optuna survival model (from --optuna_dir)
- Rebuilds fold-specific graphs (train-only backbone; train->val; train+val->test)
- Computes per-node risk scores at the chosen horizon for candidate selection
- Picks interpretation targets by strategy:
    * highest_risk             — highest predicted risk at horizon
    * lowest_risk              — lowest predicted risk at horizon
    * closest_to_median_risk   — closest to median risk score
    * early_event              — earliest observed event
    * median_by_event_status_0 / median_by_event_status_1
- Runs SHAP (deterministic) and MC-SHAP (mean +/- std) using a SurvivalRiskWrapper
  that exposes a (x, edge_index) -> [N, 2] interface compatible with explain.py
- Saves per-patient survival curve + risk distribution plots (survival-specific figures)
- Saves CSVs + PNGs to .../logs/interpret/

Usage examples
--------------
# 1) All strategies on test set of first outer fold
python survival_interpret_example.py --config configs/death_config_cos_PSA_c1.yaml \
  --optuna_dir Optuna_results_survival/gin_results/death_cosine_PSA_20260101_120000

# 2) Only highest-risk strategy on fold_2
python survival_interpret_example.py --config configs/death_config_cos_PSA_c1.yaml \
  --optuna_dir Optuna_results_survival/... --outer_fold fold_2 --strategy highest

# 3) Explicit patient
python survival_interpret_example.py --config configs/death_config_cos_PSA_c1.yaml \
  --optuna_dir Optuna_results_survival/... --patient_id PID123 --subset test

# 4) Explain risk at a different horizon
python survival_interpret_example.py --config configs/death_config_cos_PSA_c1.yaml \
  --optuna_dir Optuna_results_survival/... --horizon_months 120
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
import re
import json
from datetime import datetime

import numpy as np
import pandas as pd
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from torch_geometric.data import Data
from torch_geometric.utils import to_undirected, add_self_loops

from sklearn.model_selection import StratifiedShuffleSplit

from data_loader import load_survival_data, load_splits, build_knn_graph
from models.survival_node import SurvivalNodeGNN
from utils import load_config, get_device, set_seed
from explain import (
    explain_node_features,
    explain_node_edges,
    save_feature_importance_plot,
    save_edge_importance_plot,
    shap_with_uncertainty_features,
    shap_with_uncertainty_edges,
    save_feature_importance_errorbar,
    save_edge_importance_errorbar,
)


# ─────────────────── Survival Risk Wrapper ───────────────────

class SurvivalRiskWrapper(torch.nn.Module):
    """
    Wraps SurvivalNodeGNN to be compatible with explain.py SHAP functions.

    explain.py calls:  model(x, edge_index) -> [N, C]  and uses pos_col to select output.
    SurvivalNodeGNN:   model(data)          -> [N, num_bins]

    This wrapper: forward(x, edge_index) -> [N, 2]
      col 0 = 0  (dummy anchor)
      col 1 = logit(risk_at_horizon)

    Since softmax([0, L])[1] = sigmoid(L), and L = logit(risk):
      softmax(output)[:, 1] = risk_at_horizon  exactly.

    Always pass pos_col=1 to SHAP functions.
    MC-SHAP works because enable_dropout() walks all submodules including base_model.
    """

    def __init__(
        self,
        base_model: SurvivalNodeGNN,
        bin_width: float,
        horizon_months: float,
    ):
        super().__init__()
        self.base_model = base_model
        self.bin_width  = float(bin_width)
        T    = base_model.num_bins
        hbin = int(np.ceil(float(horizon_months) / float(bin_width))) - 1
        self.hbin = max(0, min(hbin, T - 1))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        data    = Data(x=x, edge_index=edge_index)
        hazards = self.base_model(data)                              # [N, T]
        one_m   = (1.0 - hazards).clamp(min=1e-7, max=1.0)
        surv    = torch.cumprod(one_m, dim=1)[:, self.hbin]         # [N]
        risk    = (1.0 - surv).clamp(min=1e-7, max=1.0 - 1e-7)     # [N]
        log_odds = torch.log(risk / (1.0 - risk))                   # [N]
        zeros    = torch.zeros_like(log_odds)
        return torch.stack([zeros, log_odds], dim=1)                 # [N, 2]


# ─────────────────── Helpers ───────────────────

def _indegree(edge_index: torch.Tensor, node: int) -> int:
    _, dst = edge_index
    return int((dst == node).sum().item())


def _build_eval_edges_for_subset(
    subset, graph_x, df, config, train_idx, val_idx, test_idx, device, k=10
):
    """
    Build leakage-free eval edges for the requested subset.
    TRAIN backbone: undirected + self-loops
    VAL:   train backbone + train->val connectors
    TEST:  (train+val) backbone + (train+val)->test connectors
    Returns (edge_index_eval, edge_index_train_backbone).
    """

    def _undirected_sl(local_ei, ref_idx):
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
    edge_train = _undirected_sl(ei_tr, train_idx)

    if subset == "train":
        return edge_train, edge_train

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

    if subset == "val":
        return edge_eval_val, edge_train

    # TEST connectors: (train+val) -> test
    tv_idx = torch.cat([train_idx, val_idx], dim=0)
    k_tv = min(k, int(tv_idx.numel()) - 1)
    ei_tv = build_knn_graph(
        X=graph_x[tv_idx], df=df.iloc[tv_idx.cpu().numpy()],
        k=k_tv, metric=config["Graph_method"], graph_feature=config["Graph_feature"],
    )
    edge_trainval = _undirected_sl(ei_tv, tv_idx)

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
    return edge_eval_test, edge_train


# ─────────────────── Candidate selection ───────────────────

def _select_candidates_survival(df_subset: pd.DataFrame) -> dict:
    """
    df_subset must have columns: node_idx, patient_id, time, event, risk_at_horizon.
    Returns dict of candidate rows by strategy name.
    """
    out  = {}
    risk = df_subset["risk_at_horizon"].to_numpy()
    times  = df_subset["time"].to_numpy()
    events = df_subset["event"].to_numpy()

    # 1) Highest risk at horizon
    out["highest_risk"] = df_subset.iloc[int(np.argmax(risk))]

    # 2) Lowest risk at horizon
    out["lowest_risk"] = df_subset.iloc[int(np.argmin(risk))]

    # 3) Closest to median risk
    med = float(np.median(risk))
    out["closest_to_median_risk"] = df_subset.iloc[int(np.argmin(np.abs(risk - med)))]

    # 4) Earliest observed event
    event_rows = df_subset[events == 1]
    if len(event_rows):
        out["early_event"] = event_rows.iloc[int(event_rows["time"].to_numpy().argmin())]

    # 5) Median risk by event status (0=censored, 1=event)
    for status in [0, 1]:
        grp = df_subset[events == status]
        if len(grp):
            grp_risk = grp["risk_at_horizon"].to_numpy()
            med_grp  = float(np.median(grp_risk))
            idx_in   = int(np.argmin(np.abs(grp_risk - med_grp)))
            out[f"median_by_event_status_{status}"] = grp.iloc[idx_in]

    return out


# ─────────────────── Survival-specific figures ───────────────────

def plot_survival_curve(
    surv_curve: np.ndarray,
    bin_width: float,
    patient_time: float,
    patient_event: int,
    horizons_months: list,
    save_path: str,
    title: str = "Predicted Survival Curve",
):
    """
    Predicted S(t) for one patient with their observed outcome marked.
    Events shown as red star; censoring as gray dashed vertical line.
    """
    T         = len(surv_curve)
    time_axis = np.arange(1, T + 1) * bin_width        # end of each bin (months)
    t_plot    = np.concatenate([[0.0], time_axis])
    s_plot    = np.concatenate([[1.0], surv_curve])

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(t_plot, s_plot, color="steelblue", linewidth=2, label="Predicted S(t)")

    # Mark observed outcome
    if patient_event == 1:
        s_at_event = float(np.interp(patient_time, t_plot, s_plot))
        ax.axvline(patient_time, color="tomato", linestyle="--", linewidth=1.2,
                   label=f"Event @ {patient_time:.0f}m")
        ax.plot(patient_time, s_at_event, "r*", markersize=12)
    else:
        ax.axvline(patient_time, color="gray", linestyle=":", linewidth=1.2,
                   label=f"Censored @ {patient_time:.0f}m")

    # Mark evaluation horizons and annotate S(h)
    for h in horizons_months:
        h = float(h)
        s_h = float(np.interp(h, t_plot, s_plot))
        ax.axvline(h, color="green", linestyle="--", linewidth=0.8, alpha=0.5)
        ax.annotate(
            f"S({int(h)}m)={s_h:.2f}",
            xy=(h, s_h), xytext=(h + 2, s_h + 0.05),
            fontsize=8, color="green",
        )

    ax.set_xlabel("Time (months)")
    ax.set_ylabel("Survival Probability")
    ax.set_ylim(0, 1.05)
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=200)
    plt.close(fig)


def plot_risk_distribution(
    all_risks: np.ndarray,
    patient_risk: float,
    horizon_months: float,
    save_path: str,
    title: str = "Risk Distribution",
):
    """Histogram of all-subset risks with the target patient's risk marked."""
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(all_risks, bins=20, color="steelblue", edgecolor="white", alpha=0.75,
            label=f"All patients (n={len(all_risks)})")
    ax.axvline(patient_risk, color="tomato", linewidth=2, linestyle="--",
               label=f"Patient risk = {patient_risk:.3f}")
    ax.set_xlabel(f"Risk at {int(horizon_months)}m")
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=200)
    plt.close(fig)


# ─────────────────── Core explainer ───────────────────

def _run_all_explainers_for_node(
    strategy_tag, node_idx, patient_id,
    args, config, device,
    wrapper,           # SurvivalRiskWrapper
    graph,             # Data (x, time, event on device)
    FEATURES,          # list of feature names
    interpret_dir,
    edge_index_eval,   # edge index for the chosen subset
    # survival-specific extras
    hazards_subset,    # [N_subset, num_bins] CPU tensor — full subset hazards
    subset_idx,        # [N_subset] CPU int64 — global node indices of subset
    bin_width,
    horizons_months,
):
    """
    Runs deterministic SHAP + MC-SHAP (optional) and saves all artifacts.
    Also saves survival curve and risk distribution figures.
    Returns dict of artifact paths.
    """
    ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = f"{strategy_tag}_node{node_idx}_{ts}"
    pos_col = 1  # always 1 for survival risk (see SurvivalRiskWrapper)

    # ── Feature SHAP (deterministic) ──
    shap_feat, f0_feat = explain_node_features(
        model=wrapper, x=graph.x, edge_index=edge_index_eval,
        node_idx=node_idx, pos_col=pos_col,
        n_samples=args.n_samples, baseline=None, l2=1e-3, device=device,
    )

    # ── Edge SHAP (deterministic) ──
    deg_in = _indegree(edge_index_eval, node_idx)
    if deg_in == 0:
        print(f"[warn] {strategy_tag}: node {node_idx} has 0 incoming edges; edge SHAP skipped.")
        neighbor_ids = np.array([], dtype=np.int64)
        shap_edges   = np.array([], dtype=np.float64)
        f0_edges     = float("nan")
    else:
        neighbor_ids, shap_edges, f0_edges = explain_node_edges(
            model=wrapper, x=graph.x, edge_index=edge_index_eval,
            node_idx=node_idx, pos_col=pos_col,
            n_samples=args.n_samples, l2=1e-3,
            symmetric_train_graph=True, device=device,
        )

    # ── Save deterministic CSVs ──
    feat_csv = os.path.join(interpret_dir, f"{tag}_feature_shap.csv")
    pd.DataFrame({"feature": FEATURES, "shap_value": shap_feat}).to_csv(feat_csv, index=False)

    edge_csv = os.path.join(interpret_dir, f"{tag}_edge_shap.csv")
    if neighbor_ids.size:
        pd.DataFrame({
            "neighbor_node_id": neighbor_ids.astype(int),
            "shap_value":       shap_edges,
        }).to_csv(edge_csv, index=False)
    else:
        pd.DataFrame(columns=["neighbor_node_id", "shap_value"]).to_csv(edge_csv, index=False)

    # ── Save deterministic PNGs ──
    topk    = max(1, min(args.save_topk, len(FEATURES)))
    order_f = np.argsort(-np.abs(shap_feat))[:topk]
    save_feature_importance_plot(
        shap_values=shap_feat[order_f],
        feature_names=[FEATURES[i] for i in order_f],
        save_path=os.path.join(interpret_dir, f"{tag}_feature_shap.png"),
        title=f"Feature SHAP • node={node_idx} • {strategy_tag}",
    )

    edge_png = None
    if neighbor_ids.size:
        k2      = max(1, min(args.save_topk, int(neighbor_ids.size)))
        order_e = np.argsort(-np.abs(shap_edges))[:k2]
        edge_png = os.path.join(interpret_dir, f"{tag}_edge_shap.png")
        save_edge_importance_plot(
            neighbor_ids=neighbor_ids[order_e],
            shap_values=shap_edges[order_e],
            save_path=edge_png,
            title=f"Edge SHAP • node={node_idx} • {strategy_tag}",
        )

    # ── MC-SHAP (optional) ──
    mc_root    = config.get("bayesian", {}) or {}
    mc_cfg     = mc_root.get("mc_dropout", mc_root)
    mc_enabled = bool(mc_cfg.get("enabled", False))
    mc_T       = int(mc_cfg.get("mc_passes", 50))

    feat_mc_csv = edge_mc_csv = feat_mc_png = edge_mc_png = None
    if mc_enabled:
        mean_feat, std_feat = shap_with_uncertainty_features(
            model=wrapper, x=graph.x, edge_index=edge_index_eval,
            node_idx=node_idx, pos_col=pos_col,
            n_samples=args.n_samples, T=mc_T, device=device,
        )
        feat_mc_csv = os.path.join(interpret_dir, f"{tag}_feature_shap_mc.csv")
        pd.DataFrame({
            "feature": FEATURES, "shap_mean": mean_feat, "shap_std": std_feat
        }).to_csv(feat_mc_csv, index=False)
        feat_mc_png = os.path.join(interpret_dir, f"{tag}_feature_shap_mc.png")
        save_feature_importance_errorbar(
            mean_vals=mean_feat, std_vals=std_feat, feature_names=FEATURES,
            save_path=feat_mc_png,
            title=f"Feature SHAP (MC) • node={node_idx} • {strategy_tag}",
        )

        if neighbor_ids.size:
            nbs_mc, mean_edge, std_edge = shap_with_uncertainty_edges(
                model=wrapper, x=graph.x, edge_index=edge_index_eval,
                node_idx=node_idx, pos_col=pos_col,
                n_samples=args.n_samples, T=mc_T, device=device,
            )
            edge_mc_csv = os.path.join(interpret_dir, f"{tag}_edge_shap_mc.csv")
            pd.DataFrame({
                "neighbor_node_id": nbs_mc.astype(int),
                "shap_mean": mean_edge, "shap_std": std_edge,
            }).to_csv(edge_mc_csv, index=False)
            edge_mc_png = os.path.join(interpret_dir, f"{tag}_edge_shap_mc.png")
            save_edge_importance_errorbar(
                neighbor_ids=nbs_mc, mean_vals=mean_edge, std_vals=std_edge,
                save_path=edge_mc_png,
                title=f"Edge SHAP (MC) • node={node_idx} • {strategy_tag}",
            )

    # ── Survival-specific figures ──
    # Locate this patient in the subset
    subset_np      = subset_idx.numpy() if isinstance(subset_idx, torch.Tensor) else subset_idx
    pos_in_subset  = int(np.where(subset_np == node_idx)[0][0])

    node_hazards   = hazards_subset[pos_in_subset]           # [T]
    surv_curve     = (1.0 - node_hazards).clamp(min=1e-7).cumprod(dim=0).numpy()  # [T]
    patient_time   = float(graph.time[node_idx].cpu().item())
    patient_event  = int(graph.event[node_idx].cpu().item())

    surv_png = os.path.join(interpret_dir, f"{tag}_survival_curve.png")
    plot_survival_curve(
        surv_curve=surv_curve, bin_width=bin_width,
        patient_time=patient_time, patient_event=patient_event,
        horizons_months=horizons_months,
        save_path=surv_png,
        title=f"Survival Curve • node={node_idx} • {strategy_tag}",
    )

    # Risk distribution using first horizon
    horizon = float(horizons_months[0])
    num_bins = hazards_subset.size(1)
    hbin = max(0, min(int(np.ceil(horizon / bin_width)) - 1, num_bins - 1))
    all_risks    = (1.0 - (1.0 - hazards_subset).clamp(min=1e-7).cumprod(dim=1)[:, hbin]).numpy()
    patient_risk = float(all_risks[pos_in_subset])

    risk_dist_png = os.path.join(interpret_dir, f"{tag}_risk_distribution.png")
    plot_risk_distribution(
        all_risks=all_risks, patient_risk=patient_risk,
        horizon_months=horizon,
        save_path=risk_dist_png,
        title=f"Risk Distribution @ {int(horizon)}m • {strategy_tag}",
    )

    # ── Meta JSON ──
    info_path = os.path.join(interpret_dir, f"{tag}_meta.json")
    meta = {
        "strategy":             strategy_tag,
        "node_idx":             int(node_idx),
        "patient_id":           patient_id,
        "horizon_months":       float(args.horizon_months),
        "risk_at_horizon":      float(patient_risk),
        "patient_time":         patient_time,
        "patient_event":        patient_event,
        "feature_baseline_prob": float(f0_feat),
        "edge_baseline_prob":   float(f0_edges),
        "n_samples":            int(args.n_samples),
        "mc_enabled":           mc_enabled,
        "mc_passes":            int(mc_T) if mc_enabled else 0,
    }
    with open(info_path, "w") as fh:
        json.dump(meta, fh, indent=2)

    return {
        "feat_csv":            feat_csv,
        "edge_csv":            edge_csv,
        "feat_png":            os.path.join(interpret_dir, f"{tag}_feature_shap.png"),
        "edge_png":            edge_png,
        "feat_mc_csv":         feat_mc_csv,
        "edge_mc_csv":         edge_mc_csv,
        "feat_mc_png":         feat_mc_png,
        "edge_mc_png":         edge_mc_png,
        "survival_curve_png":  surv_png,
        "risk_distribution_png": risk_dist_png,
        "meta_json":           info_path,
    }


# ─────────────────── Main ───────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Leakage-aware survival interpretation with auto-candidate selection"
    )
    parser.add_argument("--config",      required=True, help="Path to YAML config")
    parser.add_argument("--optuna_dir",  required=True,
                        help="Optuna run directory (contains logs/best_summary.json + checkpoints/)")
    parser.add_argument("--outer_fold",  default=None,
                        help="Outer fold name (e.g. fold_1). Defaults to first fold.")
    parser.add_argument("--subset",      choices=["train", "val", "test"], default="test",
                        help="Which subset graph wiring to use for candidate selection & explanation")
    parser.add_argument("--horizon_months", type=float, default=None,
                        help="Horizon (months) to explain risk at. Default: first config horizon.")

    # Explicit target OR strategy
    tgt = parser.add_mutually_exclusive_group(required=False)
    tgt.add_argument("--node_id",    type=int, help="Node index (0-based) to explain")
    tgt.add_argument("--patient_id", type=str, help="Patient ID (ID column) to explain")

    parser.add_argument(
        "--strategy",
        choices=["highest", "lowest", "closest_median", "early_event", "median", "all"],
        default=None,
        help="Candidate selection strategy (default: from config or 'all')",
    )
    parser.add_argument("--n_samples", type=int, default=300, help="SHAP coalition samples")
    parser.add_argument("--save_topk", type=int, default=25,  help="Top-K bars in plots")
    parser.add_argument("--sample_n", type=int, default=0,
                        help="Global-SHAP mode: compute feature SHAP for up to N random "
                             "subset patients (set large, e.g. 1000, to cover the whole fold) "
                             "and exit. Writes one combined CSV; skips plots/edge/MC.")
    parser.add_argument("--sample_seed", type=int, default=42,
                        help="RNG seed for --sample_n patient sampling (reproducibility)")
    parser.add_argument("--sample_edges", action="store_true",
                        help="In --sample_n mode, also compute edge SHAP per patient and "
                             "record each neighbour's event status (for edge-SHAP enrichment).")
    parser.add_argument("--k",         type=int, default=None,
                        help="Override k (number of KNN neighbors) from config")
    args = parser.parse_args()

    # ── Setup ──
    config = load_config(args.config)
    seed   = int(config.get("seed", 42))
    set_seed(seed)
    device = get_device()

    data_folder = config.get("DATA_FOLDER", "Data")
    if config.get("data_path", "") == "auto":
        config["data_path"] = f"{data_folder}/{config['LABEL_COL']}_scaled_data.csv"
    if config.get("split_path", "") == "auto":
        config["split_path"] = f"{data_folder}/{config['LABEL_COL']}_data_splits.json"

    LABEL_COL = config["LABEL_COL"]
    TIME_COL  = config["TIME_COL"]
    ID_COL    = "ID"

    # Survival binning
    bin_width       = float(config.get("bin_width", 6.0))
    max_time        = float(config.get("max_time", 216.0))
    num_bins        = int(max_time / bin_width)
    horizons_months = [float(h) for h in config.get("horizons_months", [60.0, 120.0])]
    k               = args.k if args.k is not None else int(config.get("k", 10))

    # Resolve horizon to explain
    if args.horizon_months is None:
        args.horizon_months = horizons_months[0]

    # Strategy default
    default_strategy = config.get("interpret", {}).get("strategy", "all")
    if args.strategy is None:
        args.strategy = default_strategy
    print(f"[config] Strategy: {args.strategy} | Horizon: {args.horizon_months}m")

    # ── Load Optuna best params + checkpoint info ──
    best_summary_path = os.path.join(args.optuna_dir, "logs", "best_summary.json")
    if not os.path.exists(best_summary_path):
        raise FileNotFoundError(f"best_summary.json not found: {best_summary_path}")
    with open(best_summary_path) as fh:
        best_summary = json.load(fh)
    best_params    = best_summary.get("best_params", {})
    best_trial_num = best_summary.get("best_trial_number", None)

    logs_dir     = os.path.dirname(best_summary_path)
    interpret_dir = os.path.join(logs_dir, "interpret")
    os.makedirs(interpret_dir, exist_ok=True)

    # ── Load data + splits ──
    df_check  = pd.read_csv(config["data_path"])
    FEATURES  = [c for c in config.get("features", []) if c in df_check.columns]
    df, X, times_all, events_all, id_to_index = load_survival_data(
        config["data_path"], FEATURES, LABEL_COL, TIME_COL, ID_COL
    )
    splits = load_splits(config["split_path"])

    outer_name = args.outer_fold or sorted(splits.keys())[0]
    outer      = splits[outer_name]
    print(f"[config] Using outer fold: {outer_name}")

    # ── Build index tensors ──
    graph = Data(
        x=torch.as_tensor(X,           dtype=torch.float32),
        time=torch.as_tensor(times_all,  dtype=torch.float32),
        event=torch.as_tensor(events_all, dtype=torch.long),
    ).to(device)

    test_idx = torch.tensor(
        [id_to_index[i] for i in outer["test"] if i in id_to_index],
        dtype=torch.long, device=device,
    )
    inner_keys    = [kk for kk in outer if kk.startswith("inner_folds_")]
    train_val_ids = [
        pid for kk in inner_keys for fold in outer[kk]
        for pid in (fold["train"] + fold["validation"])
    ]
    train_val_idx = torch.tensor(
        [id_to_index[i] for i in train_val_ids if i in id_to_index],
        dtype=torch.long, device=device,
    )

    # Stratified val split (stratify on event indicator, as in ensemble/test pipelines)
    events_tv = graph.event[train_val_idx].cpu().numpy()
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.10, random_state=42)
    tr_sub, va_sub = next(sss.split(np.zeros_like(events_tv), events_tv))
    train_idx = train_val_idx[tr_sub]
    val_idx   = train_val_idx[va_sub]

    # ── Build model ──
    model_name = str(config.get("model_name", "GCN"))
    if model_name == "GraphSAGE":
        model_name = "GRAPHSAGE"

    base_model = SurvivalNodeGNN(
        in_channels=graph.x.size(1),
        hidden_channels=int(best_params.get("hidden_dim", 64)),
        num_layers=int(best_params.get("num_layers", 2)),
        num_bins=num_bins,
        model_name=model_name,
        dropout=float(best_params.get("dropout", 0.2)),
    ).to(device)

    # ── Load checkpoint ──
    ckpt_dir = os.path.join(args.optuna_dir, "checkpoints")
    if os.path.isdir(ckpt_dir):
        ckpts = [f for f in os.listdir(ckpt_dir) if f.endswith(".pth")]
        pat   = re.compile(
            rf"^trial_{best_trial_num}_{re.escape(outer_name)}_inner_folds_\d+\.pth$"
        )
        candidates = sorted([f for f in ckpts if pat.match(f)])
        if not candidates:
            raise FileNotFoundError(
                f"No checkpoint for trial={best_trial_num}, outer={outer_name} in {ckpt_dir}"
            )
        ckpt_path = os.path.join(ckpt_dir, candidates[-1])
        state     = torch.load(ckpt_path, map_location=device)
        base_model.load_state_dict(state)
        print(f"Loaded checkpoint: {ckpt_path}")
    else:
        print(f"WARNING: checkpoints folder not found: {ckpt_dir}")

    # ── Build wrapper for SHAP compatibility ──
    wrapper = SurvivalRiskWrapper(
        base_model=base_model, bin_width=bin_width,
        horizon_months=args.horizon_months,
    ).to(device)

    # ── Build eval edges for requested subset ──
    edge_index_eval, _ = _build_eval_edges_for_subset(
        subset=args.subset, graph_x=graph.x, df=df, config=config,
        train_idx=train_idx, val_idx=val_idx, test_idx=test_idx,
        device=device, k=k,
    )

    # ── Compute hazards for subset (for survival figures) ──
    subset_idx = {"train": train_idx, "val": val_idx, "test": test_idx}[args.subset]
    base_model.eval()
    with torch.no_grad():
        data_eval    = Data(x=graph.x, edge_index=edge_index_eval)
        hazards_all  = base_model(data_eval)                        # [N, num_bins]
        hazards_sub  = hazards_all[subset_idx].detach().cpu()       # [N_sub, num_bins]

    subset_idx_cpu = subset_idx.cpu()
    inv_index = {v: k for k, v in id_to_index.items()}

    # ── Explicit target (--patient_id / --node_id) ──
    if args.patient_id is not None or args.node_id is not None:
        if args.patient_id is not None:
            if args.patient_id not in id_to_index:
                raise ValueError(f"Patient ID '{args.patient_id}' not found.")
            node_idx = int(id_to_index[args.patient_id])
            pid      = args.patient_id
        else:
            node_idx = int(args.node_id)
            pid      = inv_index.get(node_idx, None)

        artifacts = _run_all_explainers_for_node(
            strategy_tag=f"{outer_name}_{args.subset}_explicit",
            node_idx=node_idx, patient_id=pid,
            args=args, config=config, device=device,
            wrapper=wrapper, graph=graph,
            FEATURES=FEATURES, interpret_dir=interpret_dir,
            edge_index_eval=edge_index_eval,
            hazards_subset=hazards_sub, subset_idx=subset_idx_cpu,
            bin_width=bin_width, horizons_months=horizons_months,
        )
        print("\nSaved artifacts:", json.dumps(artifacts, indent=2))
        return

    # ── Auto candidate selection ──
    # Compute risk at target horizon for all subset nodes
    hbin_sel   = max(0, min(int(np.ceil(args.horizon_months / bin_width)) - 1, num_bins - 1))
    all_risks  = (1.0 - (1.0 - hazards_sub).clamp(min=1e-7).cumprod(dim=1)[:, hbin_sel]).numpy()

    nodes_np    = subset_idx_cpu.numpy().astype(int)
    patient_ids = [inv_index[int(n)] for n in nodes_np]
    df_subset   = pd.DataFrame({
        "patient_id":      patient_ids,
        "node_idx":        nodes_np,
        "time":            graph.time[subset_idx].cpu().numpy(),
        "event":           graph.event[subset_idx].cpu().numpy().astype(int),
        "risk_at_horizon": np.round(all_risks, 6),
    }).reset_index(drop=True)

    # Save subset prediction table
    preds_dir = os.path.join(interpret_dir, "preds")
    os.makedirs(preds_dir, exist_ok=True)
    sub_csv = os.path.join(preds_dir, f"{outer_name}_{args.subset}_subset_risks.csv")
    df_subset.to_csv(sub_csv, index=False)
    print(f"[preds] saved {sub_csv}")

    # ── Global-SHAP sampling mode: feature SHAP for many random patients ──
    if args.sample_n and args.sample_n > 0:
        rng        = np.random.default_rng(args.sample_seed)
        n_take     = min(int(args.sample_n), len(df_subset))
        sample_pos = rng.choice(len(df_subset), size=n_take, replace=False)
        print(f"[global-shap] computing feature SHAP for {n_take}/{len(df_subset)} "
              f"{args.subset} patients (fold {outer_name}) ...")
        rows_out = []
        edge_rows = []
        event_all = graph.event.detach().cpu().numpy().astype(int)
        for c, ri in enumerate(sample_pos, 1):
            row      = df_subset.iloc[int(ri)]
            node_idx = int(row["node_idx"])
            pid      = str(row["patient_id"])
            shap_feat, _ = explain_node_features(
                model=wrapper, x=graph.x, edge_index=edge_index_eval,
                node_idx=node_idx, pos_col=1,
                n_samples=args.n_samples, baseline=None, l2=1e-3, device=device,
            )
            for fname, val in zip(FEATURES, np.asarray(shap_feat).ravel()):
                rows_out.append({"patient_id": pid, "node_idx": node_idx,
                                 "event": int(row["event"]),
                                 "feature": fname, "shap_value": float(val)})
            if args.sample_edges and _indegree(edge_index_eval, node_idx) > 0:
                nbr_ids, shap_edges, _ = explain_node_edges(
                    model=wrapper, x=graph.x, edge_index=edge_index_eval,
                    node_idx=node_idx, pos_col=1,
                    n_samples=args.n_samples, l2=1e-3,
                    symmetric_train_graph=True, device=device,
                )
                for nbr, ev in zip(np.asarray(nbr_ids).ravel(), np.asarray(shap_edges).ravel()):
                    nbr = int(nbr)
                    edge_rows.append({"patient_id": pid, "patient_node": node_idx,
                                      "patient_event": int(row["event"]),
                                      "neighbor_node": nbr,
                                      "neighbor_event": int(event_all[nbr]),
                                      "edge_shap": float(ev)})
            if c % 10 == 0 or c == n_take:
                print(f"[global-shap]   {c}/{n_take}")
        out_csv = os.path.join(
            interpret_dir, f"{outer_name}_{args.subset}_global_feature_shap.csv")
        pd.DataFrame(rows_out).to_csv(out_csv, index=False)
        print(f"[global-shap] saved {out_csv}  ({len(rows_out)} rows, "
              f"{len(FEATURES)} features x {n_take} patients)")
        if args.sample_edges and edge_rows:
            edge_csv = os.path.join(
                interpret_dir, f"{outer_name}_{args.subset}_global_edge_shap.csv")
            pd.DataFrame(edge_rows).to_csv(edge_csv, index=False)
            print(f"[global-shap] saved {edge_csv}  ({len(edge_rows)} edges)")
        return

    # Pick candidates
    cand_rows = _select_candidates_survival(df_subset)

    # Which strategies to run?
    strategy_map = {
        "all":            ["highest_risk", "lowest_risk", "closest_to_median_risk",
                           "early_event",
                           "median_by_event_status_0", "median_by_event_status_1"],
        "highest":        ["highest_risk"],
        "lowest":         ["lowest_risk"],
        "closest_median": ["closest_to_median_risk"],
        "early_event":    ["early_event"],
        "median":         ["median_by_event_status_0", "median_by_event_status_1"],
    }
    wanted = strategy_map.get(args.strategy, strategy_map["all"])

    saved = {}
    for key in wanted:
        if key not in cand_rows:
            print(f"[warn] strategy '{key}' not available for this subset.")
            continue
        row      = cand_rows[key]
        node_idx = int(row["node_idx"])
        pid      = str(row["patient_id"])

        artifacts = _run_all_explainers_for_node(
            strategy_tag=f"{outer_name}_{args.subset}_{key}",
            node_idx=node_idx, patient_id=pid,
            args=args, config=config, device=device,
            wrapper=wrapper, graph=graph,
            FEATURES=FEATURES, interpret_dir=interpret_dir,
            edge_index_eval=edge_index_eval,
            hazards_subset=hazards_sub, subset_idx=subset_idx_cpu,
            bin_width=bin_width, horizons_months=horizons_months,
        )
        saved[key] = artifacts

    # Save manifest
    man_path = os.path.join(
        interpret_dir, f"{outer_name}_{args.subset}_strategies_manifest.json"
    )
    with open(man_path, "w") as fh:
        json.dump({
            "outer_fold":          outer_name,
            "subset":              args.subset,
            "horizon_months":      float(args.horizon_months),
            "strategies_requested": args.strategy,
            "strategies_ran":      list(saved.keys()),
            "artifacts":           saved,
        }, fh, indent=2)
    print(f"\n[manifest] saved {man_path}")


if __name__ == "__main__":
    main()
