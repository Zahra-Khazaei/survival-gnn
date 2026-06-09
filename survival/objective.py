# objective.py — leakage-free nested CV for survival GNN (Optuna)
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import json
import torch
import numpy as np
import pandas as pd

from metrics.survival_metrics import (
    evaluate_survival,
    binary_metrics_at_horizon,
    find_best_threshold_for_bacc,
)

from torch_geometric.data import Data
from torch_geometric.utils import to_undirected, add_self_loops

from data_loader import load_survival_data, load_splits, build_knn_graph
from utils import set_seed, get_device, get_max_degree
from models.survival_node import SurvivalNodeGNN
from losses.survival_loss import DiscreteTimeSurvivalLoss


def get_survival_objective(base_dir, config, scratch_ckpt_dir=None, save_checkpoints=False):
    """
    Nested CV objective for discrete-time survival GNN (probability version).

    EXPECTATION:
      - model(graph) returns hazards in (0,1) with shape [N, T]
      - loss consumes hazards (probabilities)
      - metrics consume hazards (probabilities)

    Leakage-free edges:
      - TRAIN: train-only graph
      - VAL  : train-only + (val -> train) connectors

    Optimize mean Harrell C-index on validation across all inner folds.
    """

    # ===== Paths & device =====
    DATA_PATH = config["data_path"]
    SPLIT_PATH = config["split_path"]
    EVENT_COL = config["LABEL_COL"]
    TIME_COL = config["TIME_COL"]
    ID_COL = "ID"
    device = get_device()

    # ===== Feature columns (safe) =====
    data_df = pd.read_csv(DATA_PATH)
    FEATURES = [c for c in config.get("features", []) if c in data_df.columns]
    if len(FEATURES) == 0:
        raise ValueError("No valid feature columns found. Check config['features'].")

    # Warm up CUDA
    if device.type == "cuda":
        torch.randn(1, device=device)
        torch.cuda.synchronize()

    # ===== Load survival data =====
    df, X, times, events, id_to_index = load_survival_data(
        csv_path=DATA_PATH,
        feature_cols=FEATURES,
        event_col=EVENT_COL,
        time_col=TIME_COL,
        id_col=ID_COL,
    )

    graph = Data(
        x=torch.as_tensor(X, dtype=torch.float32),
        time=torch.as_tensor(times, dtype=torch.float32),
        event=torch.as_tensor(events, dtype=torch.long),
    ).to(device)

    splits = load_splits(SPLIT_PATH)

    # ===== Survival discretization =====
    bin_width = float(config.get("bin_width", 6.0))
    max_time = float(config.get("max_time", 216.0))
    if bin_width <= 0:
        raise ValueError("bin_width must be > 0.")
    if max_time <= 0:
        raise ValueError("max_time must be > 0.")

    num_bins = int(max_time / bin_width)
    if num_bins <= 1:
        raise ValueError(f"num_bins computed as {num_bins}. Increase max_time or decrease bin_width.")

    # ===== Horizons =====
    horizons_months = [float(h) for h in config.get("horizons_months", [60.0, 120.0])]
    if len(horizons_months) == 0:
        raise ValueError("horizons_months must have at least 1 value.")

    # IBS max time (your previous default behavior: max horizon unless overridden)
    ibs_max_time = float(config.get("ibs_max_time", max(horizons_months)))

    # ===== Graph hyperparams =====
    graph_method = config.get("Graph_method", "cosine")
    graph_feature = config.get("Graph_feature", "all_feature")

    k = int(config.get("k", 10))
    if k <= 0:
        raise ValueError(f"config['k'] must be > 0, got {k}")

    # ===== Early stopping config =====
    es_cfg = config.get("early_stopping", {}) or {}
    es_enabled = bool(es_cfg.get("enabled", False))
    es_patience = int(es_cfg.get("patience", 20))
    es_min_delta = float(es_cfg.get("min_delta", 1e-4))
    es_warmup = int(es_cfg.get("warmup_epochs", 10))
    es_eval_every = int(es_cfg.get("eval_every", 1))
    if es_patience < 1:
        es_patience = 1
    if es_warmup < 0:
        es_warmup = 0
    if es_eval_every < 1:
        es_eval_every = 1

    def objective(trial):
        printed_debug = False

        if save_checkpoints:
            if scratch_ckpt_dir is not None:
                checkpoint_dir = os.path.join(scratch_ckpt_dir, os.path.basename(base_dir), "checkpoints")
            else:
                checkpoint_dir = os.path.join(base_dir, "checkpoints")
            os.makedirs(checkpoint_dir, exist_ok=True)
        log_dir = os.path.join(base_dir, "logs")
        os.makedirs(log_dir, exist_ok=True)

        # ===== Seed per trial =====
        seed = int(config.get("seed", 42)) + trial.number
        set_seed(seed)

        # ===== Optuna hyperparams =====
        hidden_dim = trial.suggest_categorical("hidden_dim", [16, 32, 64, 128])
        dropout = trial.suggest_float("dropout", 0.0, 0.4)
        num_layers = trial.suggest_int("num_layers", 2, 5)
        lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
        gamma = trial.suggest_categorical("gamma", [0.97, 0.98, 0.99, 0.995])

        # num_heads: only meaningful for attention-based models
        _attention_models = {"GAT", "GRAPHTRANSFORMER", "TRANSFORMERCONV", "GRAPHORMER"}
        _model_upper = config.get("model_name", "GCN").upper()
        if _model_upper in _attention_models:
            num_heads = trial.suggest_categorical("num_heads", [2, 4, 8])
        else:
            num_heads = 1

        # DeepHit ranking loss weight and bandwidth
        alpha = trial.suggest_float("alpha", 0.0, 0.5)
        sigma = trial.suggest_categorical("sigma", [0.05, 0.1, 0.25])

        val_cindex_scores = []
        val_auc_by_h = {h: [] for h in horizons_months}
        val_auc_ipcw_by_h = {h: [] for h in horizons_months}
        val_bacc_by_h = {h: [] for h in horizons_months}
        val_brier_by_h = {h: [] for h in horizons_months}
        val_nused_by_h = {h: [] for h in horizons_months}
        val_npos_by_h = {h: [] for h in horizons_months}
        val_nneg_by_h = {h: [] for h in horizons_months}
        val_ibs_scores = []

        num_epochs = int(config.get("num_epochs", 100))

        for outer_fold_name, outer_fold in splits.items():
            inner_keys = sorted([kk for kk in outer_fold if kk.startswith("inner_folds_")])

            for inner_k in inner_keys:
                for inner in outer_fold[inner_k]:
                    train_idx = torch.tensor(
                        [id_to_index[i] for i in inner["train"] if i in id_to_index],
                        dtype=torch.long, device=device
                    )
                    val_idx = torch.tensor(
                        [id_to_index[i] for i in inner["validation"] if i in id_to_index],
                        dtype=torch.long, device=device
                    )

                    n_nodes = graph.x.size(0)
                    if train_idx.numel() == 0 or val_idx.numel() == 0:
                        raise RuntimeError(f"Empty train/val split in {outer_fold_name}/{inner_k}.")
                    if int(train_idx.max()) >= n_nodes or int(val_idx.max()) >= n_nodes:
                        raise RuntimeError("Index out of bounds in splits -> id_to_index mapping.")

                    k_train = min(int(k), int(train_idx.numel()) - 1)
                    k_val = min(int(k), int(train_idx.numel()))
                    if k_train < 1 or k_val < 1:
                        raise RuntimeError(
                            f"k too large for this fold: k={k}, n_train={train_idx.numel()} "
                            f"in {outer_fold_name}/{inner_k}"
                        )

                    # ======= Build edges (leakage-free) =======
                    edge_index_train_local = build_knn_graph(
                        X=graph.x[train_idx],
                        df=df.iloc[train_idx.detach().cpu().numpy()],
                        k=k_train,
                        metric=graph_method,
                        graph_feature=graph_feature,
                        reference=None,
                        df_reference=None,
                    )
                    if edge_index_train_local is None or edge_index_train_local.numel() == 0:
                        raise RuntimeError(
                            f"Empty train graph edges in {outer_fold_name}/{inner_k} (k_train={k_train})"
                        )
                    edge_index_train = train_idx[edge_index_train_local.to(device)]
                    edge_index_train = to_undirected(edge_index_train, num_nodes=n_nodes)
                    edge_index_train, _ = add_self_loops(edge_index_train, num_nodes=n_nodes)

                    edge_index_v2t_local = build_knn_graph(
                        X=graph.x[val_idx],
                        df=df.iloc[val_idx.detach().cpu().numpy()],
                        k=k_val,
                        metric=graph_method,
                        graph_feature=graph_feature,
                        reference=graph.x[train_idx],
                        df_reference=df.iloc[train_idx.detach().cpu().numpy()],
                    )
                    if edge_index_v2t_local is None or edge_index_v2t_local.numel() == 0:
                        raise RuntimeError(
                            f"Empty val->train edges in {outer_fold_name}/{inner_k} (k_val={k_val})"
                        )
                    src = train_idx[edge_index_v2t_local[1].to(device)]
                    dst = val_idx[edge_index_v2t_local[0].to(device)]
                    edge_index_val2train = torch.stack([src, dst], dim=0)

                    # ======= Model =======
                    model_name = config.get("model_name", "GCN")

                    extra_kwargs = {}
                    if model_name.lower() == "graphormer":
                        extra_kwargs["max_degree"] = get_max_degree(edge_index_train, graph.x.size(0))

                    model = SurvivalNodeGNN(
                        in_channels=graph.x.size(1),
                        hidden_channels=hidden_dim,
                        num_layers=num_layers,
                        num_bins=num_bins,
                        model_name=model_name,
                        dropout=dropout,
                        num_heads=num_heads,
                        **extra_kwargs,
                    ).to(device)

                    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
                    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
                    criterion = DiscreteTimeSurvivalLoss(
                        bin_width=bin_width, alpha=alpha, sigma=sigma
                    ).to(device)

                    best_state = None
                    best_val_cindex = -1.0
                    best_horizon_metrics = None
                    best_ibs = float("nan")
                    es_bad_count = 0

                    for epoch in range(1, num_epochs + 1):
                        model.train()
                        optimizer.zero_grad()

                        hazards_tr = model(graph, edge_index=edge_index_train)  # [N,T] probabilities

                        if (not printed_debug) and epoch == 1 and trial.number == 0:
                            printed_debug = True
                            ht = hazards_tr[train_idx]
                            print("[DEBUG] hazards_tr TRAIN stats:",
                                  float(ht.min().item()),
                                  float(ht.max().item()),
                                  float(ht.mean().item()))

                        loss = criterion(
                            hazards_tr[train_idx],
                            graph.time[train_idx],
                            graph.event[train_idx],
                        )
                        loss.backward()
                        grad_clip = float(config.get("grad_clip_norm", 0.0))
                        if grad_clip > 0:
                            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                        optimizer.step()
                        scheduler.step()

                        do_eval = (epoch % es_eval_every == 0) or (epoch == num_epochs)
                        if not do_eval:
                            continue

                        model.eval()
                        with torch.no_grad():
                            # TRAIN hazards (train-only edges)
                            hazards_train_only = model(graph, edge_index=edge_index_train)

                            # VAL hazards (train edges + val->train)
                            edge_eval = torch.cat([edge_index_train, edge_index_val2train], dim=1)
                            hazards_ev = model(graph, edge_index=edge_eval)

                            metrics = evaluate_survival(
                                hazards=hazards_ev[val_idx],
                                times=graph.time[val_idx],
                                events=graph.event[val_idx],
                                bin_width=bin_width,
                                horizons_months=horizons_months,
                                ibs_max_time=ibs_max_time,
                            )

                            best_thresholds = {}
                            for h in horizons_months:
                                thr, _ = find_best_threshold_for_bacc(
                                    hazards=hazards_train_only[train_idx],
                                    times=graph.time[train_idx],
                                    events=graph.event[train_idx],
                                    bin_width=bin_width,
                                    horizon_months=float(h),
                                )
                                best_thresholds[float(h)] = float(thr)

                            val_cindex = metrics.get("c_index", None)
                            if val_cindex is None or (isinstance(val_cindex, float) and np.isnan(val_cindex)):
                                val_cindex_effective = -1.0
                            else:
                                val_cindex_effective = float(val_cindex)

                            horizon_metrics = {}
                            for h in horizons_months:
                                h = float(h)
                                thr = best_thresholds[h]
                                mh = binary_metrics_at_horizon(
                                    hazards=hazards_ev[val_idx],
                                    times=graph.time[val_idx],
                                    events=graph.event[val_idx],
                                    bin_width=bin_width,
                                    horizon_months=h,
                                    threshold=thr,
                                    verbose=False,
                                )
                                horizon_metrics[h] = mh

                            improved = (val_cindex_effective > best_val_cindex + es_min_delta)
                            if improved:
                                best_val_cindex = val_cindex_effective
                                best_state = {kk: vv.detach().cpu().clone() for kk, vv in model.state_dict().items()}
                                best_horizon_metrics = horizon_metrics
                                best_ibs = float(metrics.get("ibs_ipcw", float("nan")))
                                es_bad_count = 0
                            else:
                                es_bad_count += 1

                            if es_enabled and epoch >= es_warmup:
                                if es_bad_count >= es_patience:
                                    break

                    if best_val_cindex < 0:
                        best_val_cindex = 0.0
                    val_cindex_scores.append(best_val_cindex)
                    val_ibs_scores.append(best_ibs)

                    if best_horizon_metrics is not None:
                        for h in horizons_months:
                            mh = best_horizon_metrics.get(float(h), None)
                            if mh is None:
                                val_auc_by_h[h].append(float("nan"))
                                val_auc_ipcw_by_h[h].append(float("nan"))
                                val_bacc_by_h[h].append(float("nan"))
                                val_brier_by_h[h].append(float("nan"))
                                val_nused_by_h[h].append(float("nan"))
                                val_npos_by_h[h].append(float("nan"))
                                val_nneg_by_h[h].append(float("nan"))
                                continue

                            val_auc_by_h[h].append(float(mh.get("auc", float("nan"))))
                            val_auc_ipcw_by_h[h].append(float(mh.get("auc_ipcw", float("nan"))))
                            val_bacc_by_h[h].append(float(mh.get("balanced_accuracy", float("nan"))))
                            val_brier_by_h[h].append(float(mh.get("brier_ipcw", float("nan"))))
                            val_nused_by_h[h].append(float(mh.get("n_used", float("nan"))))
                            val_npos_by_h[h].append(float(mh.get("n_pos", float("nan"))))
                            val_nneg_by_h[h].append(float(mh.get("n_neg", float("nan"))))

                    if save_checkpoints:
                        ckpt_name = f"trial_{trial.number}_{outer_fold_name}_{inner_k}.pth"
                        ckpt_path = os.path.join(checkpoint_dir, ckpt_name)
                        if best_state is not None:
                            model.load_state_dict(best_state)
                        torch.save(model.state_dict(), ckpt_path)

        mean_val_cindex = float(np.nanmean(val_cindex_scores)) if len(val_cindex_scores) else 0.0
        std_val_cindex = float(np.nanstd(val_cindex_scores)) if len(val_cindex_scores) else 0.0

        def mean_std(arr):
            if arr is None or len(arr) == 0:
                return float("nan"), float("nan")
            arr = np.array(arr, dtype=float)
            return float(np.nanmean(arr)), float(np.nanstd(arr))

        horizon_summary = {}
        for h in horizons_months:
            mean_auc, std_auc = mean_std(val_auc_by_h[h])
            mean_auc_ipcw, std_auc_ipcw = mean_std(val_auc_ipcw_by_h[h])
            mean_bacc, std_bacc = mean_std(val_bacc_by_h[h])
            mean_brier, std_brier = mean_std(val_brier_by_h[h])
            mean_nused, _ = mean_std(val_nused_by_h[h])
            mean_npos, _ = mean_std(val_npos_by_h[h])
            mean_nneg, _ = mean_std(val_nneg_by_h[h])

            horizon_summary[h] = {
                "mean_auc": mean_auc,
                "std_auc": std_auc,
                "mean_auc_ipcw": mean_auc_ipcw,
                "std_auc_ipcw": std_auc_ipcw,
                "mean_bacc": mean_bacc,
                "std_bacc": std_bacc,
                "mean_brier_ipcw": mean_brier,
                "std_brier_ipcw": std_brier,
                "mean_n_used": mean_nused,
                "mean_n_pos": mean_npos,
                "mean_n_neg": mean_nneg,
            }

        mean_ibs, std_ibs = mean_std(val_ibs_scores)

        trial.set_user_attr("mean_val_cindex", mean_val_cindex)
        trial.set_user_attr("std_val_cindex", std_val_cindex)
        trial.set_user_attr("seed", seed)

        trial.set_user_attr("mean_val_ibs_ipcw", mean_ibs)
        trial.set_user_attr("std_val_ibs_ipcw", std_ibs)

        for h in horizons_months:
            hm = int(h)
            trial.set_user_attr(f"mean_val_auc_{hm}m", horizon_summary[h]["mean_auc"])
            trial.set_user_attr(f"std_val_auc_{hm}m", horizon_summary[h]["std_auc"])
            trial.set_user_attr(f"mean_val_auc_ipcw_{hm}m", horizon_summary[h]["mean_auc_ipcw"])
            trial.set_user_attr(f"std_val_auc_ipcw_{hm}m", horizon_summary[h]["std_auc_ipcw"])
            trial.set_user_attr(f"mean_val_bacc_{hm}m", horizon_summary[h]["mean_bacc"])
            trial.set_user_attr(f"std_val_bacc_{hm}m", horizon_summary[h]["std_bacc"])
            trial.set_user_attr(f"mean_val_brier_ipcw_{hm}m", horizon_summary[h]["mean_brier_ipcw"])
            trial.set_user_attr(f"std_val_brier_ipcw_{hm}m", horizon_summary[h]["std_brier_ipcw"])
            trial.set_user_attr(f"mean_val_n_used_{hm}m", horizon_summary[h]["mean_n_used"])
            trial.set_user_attr(f"mean_val_n_pos_{hm}m", horizon_summary[h]["mean_n_pos"])
            trial.set_user_attr(f"mean_val_n_neg_{hm}m", horizon_summary[h]["mean_n_neg"])

        trial_result_path = os.path.join(log_dir, f"trial_{trial.number}_val_cindex.json")
        to_dump = {
            "mean_val_cindex": mean_val_cindex,
            "std_val_cindex": std_val_cindex,
            "mean_val_ibs_ipcw": mean_ibs,
            "std_val_ibs_ipcw": std_ibs,
            "ibs_max_time": ibs_max_time,
            "horizons_months": horizons_months,
            "horizon_summary": horizon_summary,
            "params": trial.params,
            "early_stopping": {
                "enabled": es_enabled,
                "patience": es_patience,
                "min_delta": es_min_delta,
                "warmup_epochs": es_warmup,
                "eval_every": es_eval_every,
            },
        }
        with open(trial_result_path, "w") as f:
            json.dump(to_dump, f, indent=2)

        return mean_val_cindex

    return objective