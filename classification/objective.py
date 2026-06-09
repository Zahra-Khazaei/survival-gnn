# objective.py  — leakage-free folds (train-only graph; val→train connectors)
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import json
import torch
import numpy as np
import pandas as pd
import torch.nn.functional as F

from torch_geometric.data import Data
from torch_geometric.utils import to_undirected, add_self_loops


from data_loader import load_data, load_splits, build_knn_graph
from utils import (
    set_seed, compute_class_weights, get_device,
    load_config, get_max_degree, FocalLoss
)
from sklearn.metrics import (
    f1_score, roc_auc_score, accuracy_score,
    confusion_matrix, balanced_accuracy_score
)
from models import MODEL_REGISTRY


def get_objective(base_dir, config):
    # ===== Paths & device =====
    DATA_PATH  = config["data_path"]
    SPLIT_PATH = config["split_path"]
    LABEL_COL  = config["LABEL_COL"]
    ID_COL     = "ID"
    device     = get_device()

    # ===== Build FEATURES safely (no accidental coupling) =====
    data = pd.read_csv(DATA_PATH)
    FEATURES = [c for c in config.get("features", []) if c in data.columns]


    # Warm up CUDA (prevents CUBLAS init error on some setups)
    if device.type == "cuda":
        torch.randn(1, device=device)
        torch.cuda.synchronize()

    # ===== Load data (no global edges here!) =====
    df, X, y, id_to_index = load_data(DATA_PATH, FEATURES, LABEL_COL, ID_COL)

    # Build a light Data container; edges will be passed per-phase
    graph = Data(
        x=torch.as_tensor(X, dtype=torch.float32),
        y=torch.as_tensor(y, dtype=torch.long)
    ).to(device)

    # Sanity on labels
    assert graph.y.min() >= 0 and graph.y.max() < 2, \
        f"Labels must be 0/1, got {torch.unique(graph.y).cpu().tolist()}"

    # Load CV splits
    splits = load_splits(SPLIT_PATH)

    def objective(trial):
        # ===== I/O =====
        checkpoint_dir = os.path.join(base_dir, "checkpoints")
        log_dir        = os.path.join(base_dir, "logs")
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)

        # ===== Repro per trial =====
        seed = 42 + trial.number
        set_seed(seed)

        # ===== Hyperparams (Optuna) =====
        hidden_dim   = trial.suggest_categorical("hidden_dim", [16, 32, 64])
        dropout      = trial.suggest_float("dropout", 0.10, 0.50)
        num_layers   = trial.suggest_int("num_layers", 2, 5)
        lr           = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True)
        gamma        = 0.99

        # ===== Accumulators across inner folds =====
        val_f1_scores, val_auc_scores = [], []
        val_acc_scores, val_bacc_scores = [], []
        val_sens_scores, val_spec_scores = [], []

        # ===== Outer / inner loops =====
        for outer_fold_name, outer_fold in splits.items():
            # iterate inner folds keys ('inner_folds_1', 'inner_folds_2', ...)
            inner_keys = sorted([k for k in outer_fold if k.startswith("inner_folds_")])

            for inner_k in inner_keys:
                # your JSON stores a list with one dict for each 'inner_folds_X'
                for inner in outer_fold[inner_k]:
                    # ---- Map patient IDs to global indices ----
                    train_idx = torch.tensor(
                        [id_to_index[i] for i in inner["train"] if i in id_to_index],
                        dtype=torch.long, device=device
                    )
                    val_idx = torch.tensor(
                        [id_to_index[i] for i in inner["validation"] if i in id_to_index],
                        dtype=torch.long, device=device
                    )

                    # ---- Safety checks ----
                    n_nodes = graph.x.size(0)
                    assert train_idx.numel() > 0 and val_idx.numel() > 0, "Empty train/val split."
                    assert int(train_idx.max()) < n_nodes and int(val_idx.max()) < n_nodes, \
                        "Index out of bounds."

                    # ======= Build edges PER PHASE (no leakage) =======

                    # TRAIN-ONLY edges
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
                    # === Make TRAIN graph undirected and add self-loops ===

                    edge_index_train = to_undirected(edge_index_train, num_nodes=graph.x.size(0))
                    edge_index_train, _ = add_self_loops(edge_index_train, num_nodes=graph.x.size(0))


                    # VAL → TRAIN edges
                    edge_index_v2t_local = build_knn_graph(
                        X=graph.x[val_idx],
                        df=df.iloc[val_idx.cpu().numpy()],
                        k=10,
                        metric=config["Graph_method"],
                        graph_feature=config["Graph_feature"],
                        reference=graph.x[train_idx],
                        df_reference=df.iloc[train_idx.cpu().numpy()],   # <— add this
                    )
                    src = train_idx[edge_index_v2t_local[1].to(device)]
                    dst = val_idx[edge_index_v2t_local[0].to(device)]
                    edge_index_val2train = torch.stack([src, dst], dim=0)
                    
                    # ======= Model =======
                    model_class = MODEL_REGISTRY[config["model_name"]]
                    extra_args = (config.get("extra_args") or {}).copy()

                    # Graphormer needs max_degree — compute on TRAIN edges
                    if config["model_name"].lower() in ["graphormer", "graphormer_full"]:
                        extra_args["max_degree"] = get_max_degree(edge_index_train, n_nodes)

                    model = model_class(
                        in_channels=graph.x.size(1),
                        hidden_channels=hidden_dim,
                        out_channels=2,
                        num_layers=num_layers,
                        dropout=dropout,
                        activation="prelu",
                        **extra_args
                    ).to(device)

                    # Quick dry-run
                    model.eval()
                    with torch.no_grad():
                        _ = model(graph.x, edge_index_train)

                    # ======= Optim =======
                    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
                    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)

                    # ======= Loss (class weights from TRAIN ONLY) =======
                    if config.get("use_focal_loss", False):
                        cw = compute_class_weights(graph.y[train_idx], device)
                        criterion = FocalLoss(alpha=cw, gamma=2.0)
                    else:
                        cw = compute_class_weights(graph.y[train_idx], device)
                        criterion = torch.nn.CrossEntropyLoss(weight=cw)

                    # ======= Train/Eval loop =======
                    best_state = None
                    best_val_f1 = -1.0
                    best_auc = best_acc = best_bacc = best_sens = best_spec = 0.0

                    for epoch in range(1, 101):
                        # ---- TRAIN on train-only graph ----
                        model.train()
                        optimizer.zero_grad()
                        out_tr = model(graph.x, edge_index_train)
                        loss = criterion(out_tr[train_idx], graph.y[train_idx])
                        loss.backward()
                        optimizer.step()
                        scheduler.step()

                        # ---- EVAL: val reads from train via val→train edges ----
                        model.eval()
                        with torch.no_grad():
                            edge_eval = torch.cat([edge_index_train, edge_index_val2train], dim=1)
                            out_ev = model(graph.x, edge_eval)

                            probs = F.softmax(out_ev[val_idx], dim=1)[:, 1].cpu().numpy()
                            preds = out_ev[val_idx].argmax(dim=1).cpu().numpy()
                            labels = graph.y[val_idx].cpu().numpy()

                            # Robust metrics
                            f1 = f1_score(labels, preds, zero_division=0)
                            try:
                                auc = roc_auc_score(labels, probs) if np.unique(labels).size == 2 else 0.0
                            except Exception:
                                auc = 0.0
                            acc  = accuracy_score(labels, preds)
                            bacc = balanced_accuracy_score(labels, preds)
                            tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
                            sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                            spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0

                            if f1 > best_val_f1:
                                best_val_f1 = f1
                                best_auc, best_acc = auc, acc
                                best_bacc, best_sens, best_spec = bacc, sens, spec
                                # save best weights (not last)
                                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

                    # ---- Log fold results ----
                    val_f1_scores.append(best_val_f1)
                    val_auc_scores.append(best_auc)
                    val_acc_scores.append(best_acc)
                    val_bacc_scores.append(best_bacc)
                    val_sens_scores.append(best_sens)
                    val_spec_scores.append(best_spec)

                    # ---- Save best checkpoint for this inner fold ----
                    ckpt_name = f"trial_{trial.number}_{outer_fold_name}_{inner_k}.pth"
                    ckpt_path = os.path.join(checkpoint_dir, ckpt_name)
                    if best_state is not None:
                        model.load_state_dict(best_state)
                    torch.save(model.state_dict(), ckpt_path)

        # ===== Aggregate across folds =====
        mean_val_f1  = float(np.mean(val_f1_scores))
        mean_val_auc = float(np.mean(val_auc_scores))
        mean_val_acc = float(np.mean(val_acc_scores))
        mean_val_bacc = float(np.mean(val_bacc_scores))
        mean_val_sens = float(np.mean(val_sens_scores))
        mean_val_spec = float(np.mean(val_spec_scores))

        std_val_f1  = float(np.std(val_f1_scores))
        std_val_auc = float(np.std(val_auc_scores))
        std_val_acc = float(np.std(val_acc_scores))
        std_val_bacc = float(np.std(val_bacc_scores))
        std_val_sens = float(np.std(val_sens_scores))
        std_val_spec = float(np.std(val_spec_scores))

        # Expose to Optuna
        trial.set_user_attr("mean_val_f1", mean_val_f1)
        trial.set_user_attr("std_val_f1", std_val_f1)

        trial.set_user_attr("mean_val_auc", mean_val_auc)
        trial.set_user_attr("std_val_auc", std_val_auc)

        trial.set_user_attr("mean_val_acc", mean_val_acc)
        trial.set_user_attr("std_val_acc", std_val_acc)

        trial.set_user_attr("mean_val_bacc", mean_val_bacc)
        trial.set_user_attr("std_val_bacc", std_val_bacc)

        trial.set_user_attr("mean_val_sens", mean_val_sens)
        trial.set_user_attr("std_val_sens", std_val_sens)

        trial.set_user_attr("mean_val_spec", mean_val_spec)
        trial.set_user_attr("std_val_spec", std_val_spec)

        trial.set_user_attr("seed", seed)

        # Persist per-trial metrics
        trial_result_path = os.path.join(log_dir, f"trial_{trial.number}_val_f1.json")
        with open(trial_result_path, "w") as f:
            json.dump({
                "mean_val_f1": mean_val_f1,
                "mean_val_auc": mean_val_auc,
                "mean_val_acc": mean_val_acc,
                "mean_val_bacc": mean_val_bacc,
                "mean_val_sens": mean_val_sens,
                "mean_val_spec": mean_val_spec,
                "std_val_f1": std_val_f1,
                "std_val_auc": std_val_auc,
                "std_val_acc": std_val_acc,
                "std_val_bacc": std_val_bacc,
                "std_val_sens": std_val_sens,
                "std_val_spec": std_val_spec,
                "params": trial.params
            }, f, indent=4)

        # Optimize F1 by design (you can switch the return metric if needed)
        return mean_val_f1

    return objective
