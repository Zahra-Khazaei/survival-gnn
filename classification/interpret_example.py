# interpret_example.py
"""
Leakage-aware interpretation runner with auto-candidate selection.

What it does
------------
- Loads best Optuna model (using utils.find_best_summary_path)
- Rebuilds fold-specific graphs like test.py (train-only backbone; train→val; train∪val→test)
- Computes per-node probabilities for the requested subset (train/val/test)
- Picks interpretation targets by strategy:
    * misclassified_most_confident
    * highest_prob_recurrence
    * closest_to_0p5
    * median_by_true_class_0 / median_by_true_class_1
- Runs SHAP (deterministic) and MC-SHAP (mean±std) for each target
- Saves CSVs + PNGs to .../logs/interpret/

Usage examples
--------------
# 1) Auto-pick all strategies on test set of first outer fold
python interpret_example.py --config configs/bcr.yaml --strategy all

# 2) Only highest-probability recurrence on fold_3
python interpret_example.py --config configs/bcr.yaml --outer_fold fold_3 --strategy highest

# 3) Explicit target by patient_id (skips auto-selection)
python interpret_example.py --config configs/bcr.yaml --patient_id PID123 --subset test

# 4) Explicit target by node_id
python interpret_example.py --config configs/bcr.yaml --node_id 0 --subset val
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
import torch.nn.functional as F

from torch_geometric.data import Data
from torch_geometric.utils import to_undirected, add_self_loops
from sklearn.metrics import f1_score
from data_loader import load_data, load_splits, build_knn_graph
from models import MODEL_REGISTRY
from utils import (
    load_config, get_device, get_max_degree, find_best_summary_path, set_seed
)
from explain import (
    explain_node_features, explain_node_edges,
    save_feature_importance_plot, save_edge_importance_plot,
    shap_with_uncertainty_features, shap_with_uncertainty_edges,
    save_feature_importance_errorbar, save_edge_importance_errorbar,
)

# ------------- Helpers -------------
def _indegree(edge_index: torch.Tensor, node: int) -> int:
    """Count incoming edges into `node` (i.e., number of src→node)."""
    _, dst = edge_index
    return int((dst == node).sum().item())

def _safe_auc(y_true, y_score):
    from sklearn.metrics import roc_auc_score
    ys = np.unique(y_true)
    if ys.size < 2:
        return np.nan
    return roc_auc_score(y_true, y_score)

def _build_eval_edges_for_subset(
    subset, graph_x, df, config, train_idx, val_idx, test_idx, device
):
    """
    Build the eval graph with correct message directions:
      - TRAIN backbone: undirected + self-loops (train-only)
      - VAL connectors:  train  -> val
      - TEST connectors: train∪val -> test
    Returns (edge_index_eval, edge_index_train_backbone).
    """
    # ---- TRAIN backbone (undirected + self-loops) ----
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

    if subset == "train":
        return edge_index_train, edge_index_train

    # ---- VAL: connectors should be train -> val (messages flow into val nodes) ----
    edge_index_v2t_local = build_knn_graph(
        X=graph_x[val_idx],
        df=df.iloc[val_idx.cpu().numpy()],
        k=10,
        metric=config["Graph_method"],
        graph_feature=config["Graph_feature"],
        reference=graph_x[train_idx],
        df_reference=df.iloc[train_idx.cpu().numpy()],
    )
    # v2t_local is (val, train) indices into their respective blocks.
    # We want directed edges: (global train) -> (global val).
    edge_index_train2val = torch.stack(
        [
            train_idx[edge_index_v2t_local[1].to(device)],  # src: train
            val_idx[edge_index_v2t_local[0].to(device)],    # dst: val
        ],
        dim=0,
    )

    if subset == "val":
        edge_eval = torch.cat([edge_index_train, edge_index_train2val], dim=1)
        return edge_eval, edge_index_train

    # ---- TEST: connectors should be (train∪val) -> test (messages into test nodes) ----
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
    # t2tv_local is (test, trainval) inside their blocks.
    # We want directed edges: (global trainval) -> (global test).
    edge_index_trainval2test = torch.stack(
        [
            trainval_idx[edge_index_t2tv_local[1].to(device)],  # src: trainval
            test_idx[edge_index_t2tv_local[0].to(device)],      # dst: test
        ],
        dim=0,
    )

    edge_eval = torch.cat([edge_index_trainval, edge_index_trainval2test], dim=1)
    return edge_eval, edge_index_train



def _select_candidates(df_subset, outer_name):
    """Return a dict of candidate rows by strategy name."""
    out = {}
    # 1) most-confident misclassified
    wrong = df_subset[df_subset["pred"] != df_subset["y_true"]].copy()
    if len(wrong):
        conf = np.where(wrong["pred"] == 1, wrong["prob_pos"], 1.0 - wrong["prob_pos"])
        out["misclassified_most_confident"] = wrong.iloc[int(np.argmax(conf))]

    # 2) highest recurrence prob
    out["highest_prob_recurrence"] = df_subset.iloc[int(df_subset["prob_pos"].argmax())]

    # 3) closest to 0.5
    out["closest_to_0p5"] = df_subset.iloc[int((df_subset["prob_pos"] - 0.5).abs().argmin())]

    # 4) median by true class
    for cls in [0, 1]:
        dfc = df_subset[df_subset["y_true"] == cls]
        if len(dfc):
            med = dfc["prob_pos"].median()
            out[f"median_by_true_class_{cls}"] = dfc.iloc[int((dfc["prob_pos"] - med).abs().argmin())]
    return out

def _run_all_explainers_for_node(
    strategy_tag, node_idx, patient_id, pos_col, args, config, device,
    graph, df, FEATURES, interpret_dir,
    edge_index_eval
):
    """Runs deterministic SHAP + MC-SHAP, saves all artifacts; returns paths."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = f"{strategy_tag}_node{node_idx}_{ts}"

    # ---------------- Deterministic SHAP ----------------
    shap_feat, f0_feat = explain_node_features(
        model=model,
        x=graph.x,
        edge_index=edge_index_eval,
        node_idx=node_idx,
        pos_col=pos_col,
        n_samples=args.n_samples,
        baseline=None,
        l2=1e-3,
        device=device
    )

    # --- sanity: ensure node has incoming edges ---
    deg_in = _indegree(edge_index_eval, node_idx)
    if deg_in == 0:
        print(f"[warn] {strategy_tag}: node {node_idx} has 0 incoming edges; "
              f"edge SHAP will be empty.")
        neighbor_ids = np.array([], dtype=np.int64)
        shap_edges = np.array([], dtype=np.float64)
        f0_edges = float('nan')
    else:
        neighbor_ids, shap_edges, f0_edges = explain_node_edges(
            model=model,
            x=graph.x,
            edge_index=edge_index_eval,
            node_idx=node_idx,
            pos_col=pos_col,
            n_samples=args.n_samples,
            l2=1e-3,
            symmetric_train_graph=True,
            device=device
        )

    # Save deterministic CSVs/plots
    feat_csv = os.path.join(interpret_dir, f"{tag}_feature_shap.csv")
    pd.DataFrame({"feature": FEATURES, "shap_value": shap_feat}).to_csv(feat_csv, index=False)

    edge_csv = os.path.join(interpret_dir, f"{tag}_edge_shap.csv")
    if neighbor_ids.size:
        pd.DataFrame({"neighbor_node_id": neighbor_ids.astype(int), "shap_value": shap_edges}).to_csv(edge_csv, index=False)
    else:
        pd.DataFrame(columns=["neighbor_node_id", "shap_value"]).to_csv(edge_csv, index=False)

    topk = max(1, min(args.save_topk, len(FEATURES)))
    order_f = np.argsort(-np.abs(shap_feat))[:topk]
    save_feature_importance_plot(
        shap_values=shap_feat[order_f],
        feature_names=[FEATURES[i] for i in order_f],
        save_path=os.path.join(interpret_dir, f"{tag}_feature_shap.png"),
        title=f"Feature SHAP • node={node_idx} • {strategy_tag}"
    )
    if neighbor_ids.size:
        k2 = max(1, min(args.save_topk, neighbor_ids.size))
        order_e = np.argsort(-np.abs(shap_edges))[:k2]
        save_edge_importance_plot(
            neighbor_ids=neighbor_ids[order_e],
            shap_values=shap_edges[order_e],
            save_path=os.path.join(interpret_dir, f"{tag}_edge_shap.png"),
            title=f"Edge SHAP • node={node_idx} • {strategy_tag}"
        )

    # MC-SHAP (if enabled) — robustly read config (supports bayesian.mc_dropout or flat bayesian)
    mc_root = config.get("bayesian", {}) or {}
    mc_cfg = mc_root.get("mc_dropout", mc_root)
    mc_enabled = bool(mc_cfg.get("enabled", False))
    mc_T = int(mc_cfg.get("mc_passes", 50))

    feat_mc_csv = edge_mc_csv = feat_mc_png = edge_mc_png = None
    if mc_enabled:
        mean_feat, std_feat = shap_with_uncertainty_features(
            model=model,
            x=graph.x,
            edge_index=edge_index_eval,
            node_idx=node_idx,
            pos_col=pos_col,
            n_samples=args.n_samples,
            T=mc_T,
            device=device
        )
        feat_mc_csv = os.path.join(interpret_dir, f"{tag}_feature_shap_mc.csv")
        pd.DataFrame({"feature": FEATURES, "shap_mean": mean_feat, "shap_std": std_feat}).to_csv(feat_mc_csv, index=False)

        feat_mc_png = os.path.join(interpret_dir, f"{tag}_feature_shap_mc.png")
        save_feature_importance_errorbar(
            mean_vals=mean_feat, std_vals=std_feat, feature_names=FEATURES,
            save_path=feat_mc_png,
            title=f"Feature SHAP (MC) • node={node_idx} • {strategy_tag}"
        )

        if neighbor_ids.size:
            nbs_mc, mean_edge, std_edge = shap_with_uncertainty_edges(
                model=model,
                x=graph.x,
                edge_index=edge_index_eval,
                node_idx=node_idx,
                pos_col=pos_col,
                n_samples=args.n_samples,
                T=mc_T,
                device=device
            )
            edge_mc_csv = os.path.join(interpret_dir, f"{tag}_edge_shap_mc.csv")
            pd.DataFrame({"neighbor_node_id": nbs_mc.astype(int), "shap_mean": mean_edge, "shap_std": std_edge}).to_csv(edge_mc_csv, index=False)

            edge_mc_png = os.path.join(interpret_dir, f"{tag}_edge_shap_mc.png")
            save_edge_importance_errorbar(
                neighbor_ids=nbs_mc, mean_vals=mean_edge, std_vals=std_edge,
                save_path=edge_mc_png,
                title=f"Edge SHAP (MC) • node={node_idx} • {strategy_tag}"
            )

    # Meta JSON
    info_path = os.path.join(interpret_dir, f"{tag}_meta.json")
    meta = {
        "strategy": strategy_tag,
        "node_idx": int(node_idx),
        "patient_id": patient_id,
        "pos_col": int(pos_col),
        "feature_baseline_prob": float(f0_feat),
        "edge_baseline_prob": float(f0_edges),
        "n_samples": int(args.n_samples),
        "mc_enabled": mc_enabled,
        "mc_passes": int(mc_T) if mc_enabled else 0,
    }
    with open(info_path, "w") as f:
        json.dump(meta, f, indent=2)

    return {
        "feat_csv": feat_csv,
        "edge_csv": edge_csv,
        "feat_png": os.path.join(interpret_dir, f"{tag}_feature_shap.png"),
        "edge_png": (os.path.join(interpret_dir, f"{tag}_edge_shap.png") if neighbor_ids.size else None),
        "feat_mc_csv": feat_mc_csv,
        "edge_mc_csv": edge_mc_csv,
        "feat_mc_png": feat_mc_png,
        "edge_mc_png": edge_mc_png,
        "meta_json": info_path
    }

# ------------- Main -------------
def main():
    parser = argparse.ArgumentParser(description="Leakage-aware interpretation runner with auto-candidate selection")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--outer_fold", default=None, help="Outer fold name to use (e.g., fold_1). Defaults to first.")
    parser.add_argument("--subset", choices=["train", "val", "test"], default="test",
                        help="Which subset graph wiring to use for candidate selection & explanation")

    # Either explicit target OR strategy
    tgt = parser.add_mutually_exclusive_group(required=False)
    tgt.add_argument("--node_id", type=int, help="Node index (0-based) to explain")
    tgt.add_argument("--patient_id", type=str, help="Patient ID (ID column) to explain")

    parser.add_argument("--strategy", choices=["misclassified", "highest", "closest05", "median", "all"],
                        default=None,
                        help="Candidate selection strategy (default: from config or 'all')")

    parser.add_argument("--n_samples", type=int, default=300, help="SHAP coalition samples")
    parser.add_argument("--save_topk", type=int, default=25, help="Top-K bars in plots")
    args = parser.parse_args()

    # --- Setup/config/paths ---
    config = load_config(args.config)

    seed = config.get("seed", 42)
    set_seed(seed)
    print(f"[config] Global random seed set to {seed}")

    # Allow default strategy from config
    default_strategy = config.get("interpret", {}).get("strategy", "all")
    if args.strategy is None:
        args.strategy = default_strategy
    print(f"[config] Using strategy: {args.strategy}")

    device = get_device()

    data_folder = config.get("DATA_FOLDER", "Data")
    if config.get("data_path", "") == "auto":
        config["data_path"] = f"{data_folder}/{config['LABEL_COL']}_scaled_data.csv"
    if config.get("split_path", "") == "auto":
        config["split_path"] = f"{data_folder}/{config['LABEL_COL']}_data_splits.json"

    LABEL_COL = config["LABEL_COL"]
    ID_COL = "ID"

    best_summary_path = find_best_summary_path(
        optuna_root="Optuna_results",
        model_name=config["model_name"],
        label_col=config["LABEL_COL"],
        graph_method=config["Graph_method"],
        graph_feature=config["Graph_feature"]
    )
    with open(best_summary_path, "r") as f:
        best_params = json.load(f)["best_params"]

    logs_dir = os.path.dirname(best_summary_path)
    interpret_dir = os.path.join(logs_dir, "interpret")
    os.makedirs(interpret_dir, exist_ok=True)

    # --- Load data/splits ---
    df_full = pd.read_csv(config["data_path"])
    FEATURES = [c for c in config.get("features", []) if c in df_full.columns]

    df, X, y, id_to_index = load_data(config["data_path"], FEATURES, LABEL_COL, ID_COL)
    graph = Data(
        x=torch.as_tensor(X, dtype=torch.float32),
        y=torch.as_tensor(y, dtype=torch.long)
    ).to(device)

    splits = load_splits(config["split_path"])
    outer_name = args.outer_fold or sorted(splits.keys())[0]
    outer = splits[outer_name]

    # Indices
    test_idx = torch.tensor([id_to_index[i] for i in outer["test"] if i in id_to_index], dtype=torch.long, device=device)
    inner_keys = [k for k in outer.keys() if k.startswith("inner_folds_")]
    train_val_ids = [pid for key in inner_keys for fold in outer[key] for pid in (fold["train"] + fold["validation"])]
    train_val_idx = torch.tensor([id_to_index[i] for i in train_val_ids if i in id_to_index], dtype=torch.long, device=device)

    # Make small validation split from train_val (as in test.py)
    from sklearn.model_selection import StratifiedShuffleSplit
    labels_tv = graph.y[train_val_idx].cpu().numpy()
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.10, random_state=42)
    tr_sub, va_sub = next(splitter.split(np.zeros_like(labels_tv), labels_tv))
    train_idx = train_val_idx[tr_sub]
    val_idx   = train_val_idx[va_sub]

    # --- Model ---
    model_class = MODEL_REGISTRY[config["model_name"]]
    extra_args = (config.get("extra_args") or {}).copy()

    # temp backbone for Graphormer max_degree
    edge_index_train_local = build_knn_graph(
        X=graph.x[train_idx], df=df.iloc[train_idx.cpu().numpy()],
        k=10, metric=config["Graph_method"], graph_feature=config["Graph_feature"]
    )
    edge_index_train_tmp = train_idx[edge_index_train_local.to(device)]
    edge_index_train_tmp = to_undirected(edge_index_train_tmp, num_nodes=graph.x.size(0))
    edge_index_train_tmp, _ = add_self_loops(edge_index_train_tmp, num_nodes=graph.x.size(0))
    if config["model_name"].lower() in ["graphormer", "graphormer_full"]:
        extra_args["max_degree"] = get_max_degree(edge_index_train_tmp, graph.x.size(0))

    global model  # needed inside helper that calls explainers
    model = model_class(
        in_channels=graph.x.size(1),
        hidden_channels=best_params["hidden_dim"],
        out_channels=2,
        num_layers=best_params["num_layers"],
        dropout=best_params["dropout"],
        activation="prelu",
        **extra_args
    ).to(device)

    # ---------------- Load the correct checkpoint (best trial + chosen outer fold) ----------------
    with open(best_summary_path, "r") as f:
        _best = json.load(f)
    best_trial_num = _best.get("best_trial_number", None)

    ckpt_dir = os.path.join(os.path.dirname(logs_dir), "checkpoints")

    if os.path.isdir(ckpt_dir):
        ckpts = [f for f in os.listdir(ckpt_dir) if f.endswith(".pth")]

        # pattern like: trial_0_fold_3_inner_folds_1.pth
        pat = re.compile(rf"^trial_{best_trial_num}_{re.escape(outer_name)}_inner_folds_\d+\.pth$")
        candidates = sorted([f for f in ckpts if pat.match(f)])
        if not candidates:
            raise FileNotFoundError(
                f"No checkpoint found for best trial={best_trial_num} and outer={outer_name} in {ckpt_dir}"
            )

        # choose one deterministically (last inner fold); you can change to [0] if you prefer the first
        ckpt_path = os.path.join(ckpt_dir, candidates[-1])
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state)   # keep strict=True so shape mismatches surface
        print(f"Loaded checkpoint: {ckpt_path}")
    else:
        print(f"WARNING: checkpoints folder not found: {ckpt_dir}")

    # --- Build wiring for requested subset + for val (to choose pos_col) ---
    edge_index_eval, _ = _build_eval_edges_for_subset(
        subset=args.subset, graph_x=graph.x, df=df, config=config,
        train_idx=train_idx, val_idx=val_idx, test_idx=test_idx, device=device
    )
    edge_index_val_eval, _ = _build_eval_edges_for_subset(
        subset="val", graph_x=graph.x, df=df, config=config,
        train_idx=train_idx, val_idx=val_idx, test_idx=test_idx, device=device
    )

    # Choose pos_col using VAL AUC
    model.eval()
    with torch.no_grad():
        logits_val = model(graph.x, edge_index_val_eval)
        val_soft = F.softmax(logits_val[val_idx], dim=1).cpu().numpy()
        val_labels = graph.y[val_idx].cpu().numpy()
    auc0 = _safe_auc(val_labels, val_soft[:, 0])
    auc1 = _safe_auc(val_labels, val_soft[:, 1])
    pos_col = 1 if (np.nan_to_num(auc1, nan=-1) >= np.nan_to_num(auc0, nan=-1)) else 0
    print(f"[{outer_name}] positive column = {pos_col} (val AUC col0={auc0:.3f}, col1={auc1:.3f})")

    # --- If explicit target is provided, run only that ---
    inv_index = {v: k for k, v in id_to_index.items()}

    if args.patient_id is not None or args.node_id is not None:
        if args.patient_id is not None:
            if args.patient_id not in id_to_index:
                raise ValueError(f"Patient ID '{args.patient_id}' not found.")
            node_idx = int(id_to_index[args.patient_id])
            pid = args.patient_id
        else:
            node_idx = int(args.node_id)
            pid = inv_index.get(node_idx, None)

        artifacts = _run_all_explainers_for_node(
            strategy_tag=f"{outer_name}_{args.subset}_explicit",
            node_idx=node_idx,
            patient_id=pid,
            pos_col=pos_col,
            args=args, config=config, device=device,
            graph=graph, df=df, FEATURES=FEATURES,
            interpret_dir=interpret_dir,
            edge_index_eval=edge_index_eval
        )
        print("\nSaved artifacts:", json.dumps(artifacts, indent=2))
        return

    # --- Otherwise, compute probs for subset + pick candidates by strategy ---
    subset_idx = {"train": train_idx, "val": val_idx, "test": test_idx}[args.subset]
    model.eval()
    with torch.no_grad():
        logits_sub = model(graph.x, edge_index_eval)
        soft_sub = F.softmax(logits_sub[subset_idx], dim=1).cpu().numpy()
    probs_sub = soft_sub[:, pos_col]
    labels_sub = graph.y[subset_idx].cpu().numpy()

    # choose threshold by F1 on val (robust & simple)
    val_probs = val_soft[:, pos_col]
    thresholds = np.linspace(0.01, 0.99, 99)
    f1s = [f1_score(val_labels, (val_probs >= t).astype(int), zero_division=0) for t in thresholds]
    tau = float(thresholds[int(np.argmax(f1s))])
    preds_sub = (probs_sub >= tau).astype(int)

    # Build a dataframe for the subset
    nodes_sub = subset_idx.cpu().numpy().astype(int)
    patient_ids = [inv_index[int(n)] for n in nodes_sub]
    df_subset = pd.DataFrame({
        "patient_id": patient_ids,
        "node_idx": nodes_sub,
        "y_true": labels_sub.astype(int),
        "prob_pos": np.round(probs_sub, 6),
        "pred": preds_sub.astype(int),
        "threshold": tau,
        "correct": (preds_sub == labels_sub).astype(int),
    })
    # Save the subset prediction table
    preds_dir = os.path.join(interpret_dir, "preds")
    os.makedirs(preds_dir, exist_ok=True)
    sub_csv = os.path.join(preds_dir, f"{outer_name}_{args.subset}_subset_preds.csv")
    df_subset.to_csv(sub_csv, index=False)
    print(f"[preds] saved {sub_csv}")

    # Pick candidates per strategy
    cand_rows = _select_candidates(df_subset, outer_name)

    # Which strategies to run?
    wanted = []
    if args.strategy == "all":
        wanted = ["misclassified_most_confident", "highest_prob_recurrence", "closest_to_0p5",
                  "median_by_true_class_0", "median_by_true_class_1"]
    elif args.strategy == "misclassified":
        wanted = ["misclassified_most_confident"]
    elif args.strategy == "highest":
        wanted = ["highest_prob_recurrence"]
    elif args.strategy == "closest05":
        wanted = ["closest_to_0p5"]
    elif args.strategy == "median":
        wanted = ["median_by_true_class_0", "median_by_true_class_1"]

    # Run explanations for available candidates
    saved = {}
    for key in wanted:
        if key not in cand_rows:
            print(f"[warn] strategy '{key}' not available for this subset (e.g., no wrong preds or class missing).")
            continue
        row = cand_rows[key]
        node_idx = int(row["node_idx"])
        pid = str(row["patient_id"])

        artifacts = _run_all_explainers_for_node(
            strategy_tag=f"{outer_name}_{args.subset}_{key}",
            node_idx=node_idx,
            patient_id=pid,
            pos_col=pos_col,
            args=args, config=config, device=device,
            graph=graph, df=df, FEATURES=FEATURES,
            interpret_dir=interpret_dir,
            edge_index_eval=edge_index_eval
        )
        saved[key] = artifacts

    # Save a small JSON manifest of what we ran
    man_path = os.path.join(interpret_dir, f"{outer_name}_{args.subset}_strategies_manifest.json")
    with open(man_path, "w") as f:
        json.dump({
            "outer_fold": outer_name,
            "subset": args.subset,
            "threshold": tau,
            "pos_col": int(pos_col),
            "strategies_requested": args.strategy,
            "strategies_ran": list(saved.keys()),
            "artifacts": saved
        }, f, indent=2)
    print(f"\n[manifest] saved {man_path}")


if __name__ == "__main__":
    main()
