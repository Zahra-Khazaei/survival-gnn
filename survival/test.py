# test.py
# Evaluate the best survival GNN (from Optuna) on outer test folds.
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import json
import argparse

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected, add_self_loops

from data_loader import (
    load_survival_data,
    load_splits,
    build_knn_graph,
)
from utils import (
    load_config,
    set_seed,
)
from models.survival_node import SurvivalNodeGNN
from losses.survival_loss import DiscreteTimeSurvivalLoss
from metrics.survival_metrics import evaluate_survival


def build_model_from_best_params(
    best_params: dict,
    in_channels: int,
    num_bins: int,
    model_name: str,
    device: torch.device,
):
    hidden_dim = int(best_params.get("hidden_dim", 64))
    dropout = float(best_params.get("dropout", 0.2))
    num_layers = int(best_params.get("num_layers", 2))

    if model_name == "GraphSAGE":
        model_name = "GRAPHSAGE"

    model = SurvivalNodeGNN(
        in_channels=in_channels,
        hidden_channels=hidden_dim,
        num_layers=num_layers,
        num_bins=num_bins,
        model_name=model_name,
        dropout=dropout,
    ).to(device)
    return model


def mean_std(arr):
    arr = np.array(arr, dtype=float)
    return float(np.nanmean(arr)), float(np.nanstd(arr))


def main():
    parser = argparse.ArgumentParser(description="Test survival GNN on outer folds")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--optuna_dir", type=str, required=True)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--out_path",
        type=str,
        default=None,
        help="Default: <optuna_dir>/test_metrics.json",
    )
    args = parser.parse_args()

    # --------- Load config --------- #
    config = load_config(args.config)

    data_folder = config.get("DATA_FOLDER", "Data")
    if config.get("data_path", "") == "auto":
        config["data_path"] = f"{data_folder}/{config['LABEL_COL']}_scaled_data.csv"
    if config.get("split_path", "") == "auto":
        config["split_path"] = f"{data_folder}/{config['LABEL_COL']}_data_splits.json"

    if config.get("TASK_TYPE", "").lower() != "survival":
        print("⚠️  TASK_TYPE is not 'survival' in config. Forcing TASK_TYPE='survival'.")
        config["TASK_TYPE"] = "survival"

    DATA_PATH = config["data_path"]
    SPLIT_PATH = config["split_path"]
    LABEL_COL = config["LABEL_COL"]
    TIME_COL = config["TIME_COL"]
    ID_COL = "ID"
    device = torch.device(args.device)

    # --------- Load best Optuna summary --------- #
    best_summary_path = os.path.join(args.optuna_dir, "logs", "best_summary.json")
    if not os.path.exists(best_summary_path):
        raise FileNotFoundError(f"best_summary.json not found at {best_summary_path}")

    with open(best_summary_path, "r") as f:
        best_summary = json.load(f)
    best_params = best_summary.get("best_params", {})

    print("\n=== Loaded best hyperparameters from Optuna ===")
    print(best_params)

    # --------- Seed --------- #
    seed = int(config.get("seed", 42))
    set_seed(seed)

    # --------- Survival binning config --------- #
    bin_width = float(config.get("bin_width", 6.0))
    max_time = float(config.get("max_time", 216.0))
    num_bins = int(max_time / bin_width)

    # Horizons from config (months)
    horizons_months = [float(h) for h in config.get("horizons_months", [60.0, 120.0])]

    # --------- Graph construction config --------- #
    graph_method = config.get("Graph_method", "cosine")
    graph_feature = config.get("Graph_feature", "all_feature")
    # k comes from best_summary (matches what was used during optimization), with config as fallback
    k = int(best_summary.get("k", config.get("k", 10)))

    # Training hyperparameters from best params or config defaults
    lr = float(best_params.get("lr", config.get("lr", 1e-3)))
    weight_decay = float(best_params.get("weight_decay", config.get("weight_decay", 1e-4)))
    num_epochs = int(config.get("num_epochs", 100))
    gamma = float(best_params.get("gamma", 0.99))
    grad_clip_norm = float(config.get("grad_clip_norm", 1.0))
    ibs_max_time = float(config.get("ibs_max_time", max(horizons_months)))

    model_name = config.get("model_name", "GCN")

    # --------- Load data --------- #
    print("\n🔹 Loading survival data...")
    # Read once to filter features safely
    df_check = pd.read_csv(DATA_PATH)
    feature_cols = [c for c in config.get("features", []) if c in df_check.columns]

    df, X, times, events, id_to_index = load_survival_data(
        csv_path=DATA_PATH,
        feature_cols=feature_cols,
        event_col=LABEL_COL,
        time_col=TIME_COL,
        id_col=ID_COL,
    )

    graph = Data(
        x=torch.as_tensor(X, dtype=torch.float32),
        time=torch.as_tensor(times, dtype=torch.float32),
        event=torch.as_tensor(events, dtype=torch.long),
    ).to(device)

    all_ids = list(df[ID_COL])
    splits = load_splits(SPLIT_PATH)

    # --------- Containers --------- #
    outer_cindex = []
    outer_ibs = []
    outer_h = {int(h): {
        "auc": [], "auc_ipcw": [], "bacc": [],
        "brier_ipcw": [], "sensitivity": [], "specificity": [],
        "n_used": [], "n_pos": [], "n_neg": [],
    } for h in horizons_months}

    # --------- Loop over outer folds --------- #
    print("\n🔹 Evaluating on outer folds...")
    for outer_fold_name, outer_fold in splits.items():
        if "test" not in outer_fold:
            raise KeyError(f"Expected key 'test' in outer_fold {outer_fold_name}")

        test_ids = outer_fold["test"]
        test_idx = torch.tensor(
            [id_to_index[i] for i in test_ids if i in id_to_index],
            dtype=torch.long,
            device=device,
        )

        test_set = set(test_ids)
        train_ids = [i for i in all_ids if i not in test_set]
        train_idx = torch.tensor(
            [id_to_index[i] for i in train_ids if i in id_to_index],
            dtype=torch.long,
            device=device,
        )

        n_nodes = graph.x.size(0)
        assert train_idx.numel() > 0 and test_idx.numel() > 0, f"Empty train/test split in {outer_fold_name}"
        assert int(train_idx.max()) < n_nodes and int(test_idx.max()) < n_nodes, "Index out of bounds."

        print(f"\n=== {outer_fold_name} ===")
        print(f"Train size: {train_idx.numel()} | Test size: {test_idx.numel()}")

        # Clamp k to valid range (mirrors survival_objective.py)
        k_train = min(k, int(train_idx.numel()) - 1)
        k_test = min(k, int(train_idx.numel()))

        # --------- Leakage-free edges: train-only + test→train --------- #
        edge_train_local = build_knn_graph(
            X=graph.x[train_idx],
            df=df.iloc[train_idx.cpu().numpy()],
            k=k_train,
            metric=graph_method,
            graph_feature=graph_feature,
        )
        edge_train = train_idx[edge_train_local.to(device)]
        edge_train = to_undirected(edge_train, num_nodes=graph.x.size(0))
        edge_train, _ = add_self_loops(edge_train, num_nodes=graph.x.size(0))

        edge_t2tr_local = build_knn_graph(
            X=graph.x[test_idx],
            df=df.iloc[test_idx.cpu().numpy()],
            k=k_test,
            metric=graph_method,
            graph_feature=graph_feature,
            reference=graph.x[train_idx],
            df_reference=df.iloc[train_idx.cpu().numpy()],
        )
        src = train_idx[edge_t2tr_local[1].to(device)]
        dst = test_idx[edge_t2tr_local[0].to(device)]
        edge_test2train = torch.stack([src, dst], dim=0)

        # --------- Model / loss / optim --------- #
        model = build_model_from_best_params(
            best_params=best_params,
            in_channels=graph.x.size(1),
            num_bins=num_bins,
            model_name=model_name,
            device=device,
        )

        criterion = DiscreteTimeSurvivalLoss(bin_width=bin_width).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)

        # --------- Train --------- #
        print("  Training model on outer-train...")
        for epoch in range(1, num_epochs + 1):
            model.train()
            optimizer.zero_grad()

            graph.edge_index = edge_train
            hazards_tr = model(graph)

            loss = criterion(
                hazards_tr[train_idx],
                graph.time[train_idx],
                graph.event[train_idx],
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()
            scheduler.step()

            if epoch % 20 == 0 or epoch == num_epochs:
                print(f"    Epoch {epoch:03d} | train_loss = {loss.item():.4f}")

        # --------- Evaluate --------- #
        print("  Evaluating on outer-test...")
        model.eval()
        with torch.no_grad():
            edge_eval = torch.cat([edge_train, edge_test2train], dim=1)
            graph.edge_index = edge_eval
            hazards = model(graph)

            metrics = evaluate_survival(
                hazards=hazards[test_idx],
                times=graph.time[test_idx],
                events=graph.event[test_idx],
                bin_width=bin_width,
                horizons_months=horizons_months,
                ibs_max_time=ibs_max_time,
            )

        cidx = float(metrics["c_index"])
        outer_cindex.append(cidx)
        outer_ibs.append(float(metrics.get("ibs_ipcw", float("nan"))))
        print(f"  Test C-index: {cidx:.4f} | IBS-IPCW: {outer_ibs[-1]:.4f}")

        # collect per-horizon
        for h in horizons_months:
            h_key_float = float(h)
            h_key_int = int(h)

            m = metrics["horizons"].get(h_key_float)
            if m is None:
                m = metrics["horizons"].get(h_key_int)

            if m is None:
                for key in ("auc", "auc_ipcw", "bacc", "brier_ipcw",
                            "sensitivity", "specificity", "n_used", "n_pos", "n_neg"):
                    outer_h[h_key_int][key].append(float("nan"))
                continue

            auc       = float(m.get("auc",              float("nan")))
            auc_ipcw  = float(m.get("auc_ipcw",         float("nan")))
            bacc      = float(m.get("balanced_accuracy", float("nan")))
            brier     = float(m.get("brier_ipcw",        float("nan")))
            sens      = float(m.get("sensitivity",       float("nan")))
            spec      = float(m.get("specificity",       float("nan")))
            n_used    = float(m.get("n_used",            float("nan")))
            n_pos     = float(m.get("n_pos",             float("nan")))
            n_neg     = float(m.get("n_neg",             float("nan")))

            outer_h[h_key_int]["auc"].append(auc)
            outer_h[h_key_int]["auc_ipcw"].append(auc_ipcw)
            outer_h[h_key_int]["bacc"].append(bacc)
            outer_h[h_key_int]["brier_ipcw"].append(brier)
            outer_h[h_key_int]["sensitivity"].append(sens)
            outer_h[h_key_int]["specificity"].append(spec)
            outer_h[h_key_int]["n_used"].append(n_used)
            outer_h[h_key_int]["n_pos"].append(n_pos)
            outer_h[h_key_int]["n_neg"].append(n_neg)

            print(f"  Horizon {h_key_int}m | AUC: {auc:.4f} | AUC-IPCW: {auc_ipcw:.4f} | BACC: {bacc:.4f}")

    # --------- Aggregate --------- #
    mean_cidx, std_cidx = mean_std(outer_cindex)
    mean_ibs,  std_ibs  = mean_std(outer_ibs)

    results = {
        "outer_cindex": {
            "mean": mean_cidx,
            "std": std_cidx,
            "per_fold": outer_cindex,
        },
        "outer_ibs_ipcw": {
            "mean": mean_ibs,
            "std": std_ibs,
            "per_fold": outer_ibs,
        },
        "outer_horizons": {},
        "horizons_months": [int(h) for h in horizons_months],
    }

    def _agg(vals, key):
        m, s = mean_std(vals[key])
        return {"mean": m, "std": s, "per_fold": vals[key]}

    for h_int, vals in outer_h.items():
        results["outer_horizons"][str(h_int)] = {
            "auc":         _agg(vals, "auc"),
            "auc_ipcw":    _agg(vals, "auc_ipcw"),
            "bacc":        _agg(vals, "bacc"),
            "brier_ipcw":  _agg(vals, "brier_ipcw"),
            "sensitivity": _agg(vals, "sensitivity"),
            "specificity": _agg(vals, "specificity"),
            "n_used": float(np.nanmean(vals["n_used"])),
            "n_pos":  float(np.nanmean(vals["n_pos"])),
            "n_neg":  float(np.nanmean(vals["n_neg"])),
        }

    # --------- Save --------- #
    out_path = args.out_path or os.path.join(args.optuna_dir, "test_metrics.json")
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(out_path, "w") as f:
        json.dump(results, f, indent=4)

    print("\n=== Aggregated TEST results across outer folds ===")
    print(f"C-index:  {mean_cidx:.4f} ± {std_cidx:.4f}")
    print(f"IBS-IPCW: {mean_ibs:.4f} ± {std_ibs:.4f}")
    for h in horizons_months:
        h_int = int(h)
        r = results["outer_horizons"][str(h_int)]
        print(f"Horizon {h_int}m | AUC: {r['auc']['mean']:.4f} ± {r['auc']['std']:.4f} | "
              f"AUC-IPCW: {r['auc_ipcw']['mean']:.4f} ± {r['auc_ipcw']['std']:.4f} | "
              f"BACC: {r['bacc']['mean']:.4f} ± {r['bacc']['std']:.4f} | "
              f"Brier-IPCW: {r['brier_ipcw']['mean']:.4f} ± {r['brier_ipcw']['std']:.4f}")
    print("\nTest metrics saved to:", out_path)


if __name__ == "__main__":
    main()
