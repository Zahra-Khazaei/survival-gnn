# data_loader.py
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import pandas as pd
import numpy as np
import json
from torch_geometric.data import Data
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_distances
from scipy.spatial.distance import mahalanobis
from torch_geometric.utils import dense_to_sparse


def load_data(csv_path, feature_cols, label_col, id_col='ID'):
    data = pd.read_csv(csv_path)


    X = torch.tensor(data[feature_cols].values, dtype=torch.float)
    y = torch.tensor(data[label_col].values, dtype=torch.long)
    id_to_index = {pid: idx for idx, pid in enumerate(data[id_col])}
    
    return data, X, y, id_to_index


def build_knn_graph(
    X,
    df=None,
    k=10,
    metric="euclidean",
    graph_feature="all_feature",
    reference=None,
    df_reference=None,
):
    """
    Build a KNN graph.
    """
    X_np = X.cpu().numpy() if isinstance(X, torch.Tensor) else np.asarray(X)
    R_np = None
    if reference is not None:
        R_np = reference.cpu().numpy() if isinstance(reference, torch.Tensor) else np.asarray(reference)

    if graph_feature != "all_feature":
        if df is None or graph_feature not in df.columns:
            raise ValueError(f"Feature '{graph_feature}' not found in provided df for query selection.")
        X_used = df[[graph_feature]].to_numpy().astype(np.float32)
    else:
        X_used = X_np.astype(np.float32)

    if R_np is not None:
        if graph_feature != "all_feature":
            if df_reference is None or graph_feature not in df_reference.columns:
                raise ValueError(f"Feature '{graph_feature}' not found in df_reference for reference selection.")
            R_used = df_reference[[graph_feature]].to_numpy().astype(np.float32)
        else:
            R_used = R_np.astype(np.float32)

    if reference is None:
        if metric == "euclidean":
            nbrs = NearestNeighbors(n_neighbors=k + 1, metric="euclidean").fit(X_used)
            _, indices = nbrs.kneighbors(X_used)
            src_list, dst_list = [], []
            for i, neigh in enumerate(indices):
                for j in neigh[1:]:
                    src_list.append(i)
                    dst_list.append(j)
            return torch.tensor([src_list, dst_list], dtype=torch.long)

        elif metric == "cosine":
            nbrs = NearestNeighbors(n_neighbors=k + 1, metric="cosine").fit(X_used)
            _, indices = nbrs.kneighbors(X_used)
            src_list, dst_list = [], []
            for i, neigh in enumerate(indices):
                for j in neigh[1:]:
                    src_list.append(i)
                    dst_list.append(j)
            return torch.tensor([src_list, dst_list], dtype=torch.long)

        elif metric == "mahalanobis":
            VI = np.linalg.inv(np.cov(X_used.T))
            nbrs = NearestNeighbors(
                n_neighbors=k + 1,
                metric="mahalanobis",
                metric_params={"VI": VI}
            ).fit(X_used)
            _, indices = nbrs.kneighbors(X_used)
            src_list, dst_list = [], []
            for i, neigh in enumerate(indices):
                for j in neigh[1:]:
                    src_list.append(i)
                    dst_list.append(j)
            return torch.tensor([src_list, dst_list], dtype=torch.long)

        elif metric == "correlation":
            corr = np.corrcoef(X_used)
            distances = 1.0 - corr
            src_list, dst_list = [], []
            for i in range(len(X_used)):
                neigh = np.argsort(distances[i])[1:k+1]
                for j in neigh:
                    src_list.append(i)
                    dst_list.append(j)
            return torch.tensor([src_list, dst_list], dtype=torch.long)

        else:
            raise ValueError(f"Unsupported graph metric: {metric}")

    else:
        if metric == "euclidean":
            nbrs = NearestNeighbors(n_neighbors=k, metric="euclidean").fit(R_used)
            _, indices = nbrs.kneighbors(X_used)
        elif metric == "cosine":
            nbrs = NearestNeighbors(n_neighbors=k, metric="cosine").fit(R_used)
            _, indices = nbrs.kneighbors(X_used)
        elif metric == "mahalanobis":
            VI = np.linalg.inv(np.cov(R_used.T))
            nbrs = NearestNeighbors(
                n_neighbors=k,
                metric="mahalanobis",
                metric_params={"VI": VI}
            ).fit(R_used)
            _, indices = nbrs.kneighbors(X_used)
        elif metric == "correlation":
            Xc = (X_used - X_used.mean(0, keepdims=True)) / (X_used.std(0, keepdims=True) + 1e-8)
            Rc = (R_used - R_used.mean(0, keepdims=True)) / (R_used.std(0, keepdims=True) + 1e-8)
            sim = Xc @ Rc.T / (
                np.linalg.norm(Xc, axis=1, keepdims=True)
                * np.linalg.norm(Rc, axis=1, keepdims=True)
                + 1e-8
            )
            indices = np.argsort(-sim, axis=1)[:, :k]
        else:
            raise ValueError(f"Unsupported graph metric: {metric}")

        Q = X_used.shape[0]
        row = np.repeat(np.arange(Q), k)
        col = indices.reshape(-1)
        return torch.tensor(np.stack([row, col], axis=0), dtype=torch.long)


def load_splits(json_path):
    with open(json_path, 'r') as f:
        splits = json.load(f)
    return splits


def build_graph(X, edge_index, y):
    return Data(x=X, edge_index=edge_index, y=y)


def create_masks(graph, train_idx, val_idx, test_idx):
    N = graph.num_nodes
    graph.train_mask = torch.zeros(N, dtype=torch.bool)
    graph.val_mask = torch.zeros(N, dtype=torch.bool)
    graph.test_mask = torch.zeros(N, dtype=torch.bool)
    graph.train_mask[train_idx] = True
    graph.val_mask[val_idx] = True
    graph.test_mask[test_idx] = True
    return graph
