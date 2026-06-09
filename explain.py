# explain.py
import math
import os
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from utils import enable_dropout
# -------------------------
# Core helpers
# -------------------------
def _pos_prob_from_logits(logits: torch.Tensor, pos_col: int = 1) -> torch.Tensor:
    """
    Get positive-class probability from logits.
    - If logits is [C], returns scalar prob of class=pos_col
    - If logits is [N, C], returns [N] probs of class=pos_col
    """
    if logits.ndim == 1:
        return F.softmax(logits, dim=-1)[pos_col]
    return F.softmax(logits, dim=-1)[:, pos_col]


def _shapley_kernel_weight(m: int, s: int) -> float:
    """
    Shapley kernel weight for coalition size s among m items:
        w(S) = (m - 1) / (C(m, s) * s * (m - s))
    We avoid s in {0, m} (all-off and all-on) in sampling, so no infinities.
    """
    if s <= 0 or s >= m:
        return 0.0
    comb = math.comb(m, s)
    return (m - 1.0) / (comb * s * (m - s))


def _weighted_ridge(X: np.ndarray, y: np.ndarray, w: np.ndarray, l2: float = 1e-3) -> np.ndarray:
    """
    Solve weighted ridge regression for KernelSHAP:
        argmin_b ||sqrt(W) * (X b - y)||^2 + l2 ||b||^2
    Returns coefficients b.
    """
    Xw = X * w[:, None]
    A = Xw.T @ X + l2 * np.eye(X.shape[1], dtype=np.float64)
    b = Xw.T @ (y * w)
    coef = np.linalg.solve(A, b)
    return coef


def _sample_binary_masks(n_items: int, n_samples: int, rng: np.random.RandomState,
                         min_on: int = 1, max_on: Optional[int] = None) -> np.ndarray:
    """
    Random binary masks (coalitions) with #ON in [min_on, max_on].
    We exclude all-zeros and (optionally) all-ones to keep weights finite.
    """
    if max_on is None:
        max_on = n_items - 1
    masks = np.zeros((n_samples, n_items), dtype=np.int64)
    for i in range(n_samples):
        k = rng.randint(min_on, max_on + 1)
        idx = rng.choice(n_items, size=k, replace=False)
        masks[i, idx] = 1
    return masks


# -------------------------
# Feature-level GraphSHAP
# -------------------------
@torch.no_grad()
def explain_node_features(
    model: torch.nn.Module,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    node_idx: int,
    *,
    pos_col: int = 1,
    n_samples: int = 300,
    baseline: Optional[torch.Tensor] = None,
    l2: float = 1e-3,
    rng_seed: int = 123,
    device: Optional[torch.device] = None,
) -> Tuple[np.ndarray, float]:
    """
    Estimate SHAP-like values for each FEATURE of node_idx.
    Masked-OFF features are replaced by a baseline (default: feature-wise mean).
    Returns:
      shap_values: (D,) SHAP values (numpy)
      f_baseline:  float, prob when all features are OFF (node features=baseline)
    """
    model.eval()
    if device is None:
        device = next(model.parameters()).device

    x = x.clone().to(device)
    edge_index = edge_index.to(device)

    D = int(x.size(1))
    rng = np.random.RandomState(rng_seed)

    if baseline is None:
        baseline = x.mean(dim=0, keepdim=True)  # (1, D)
    baseline = baseline.to(device)

    x_node_orig = x[node_idx].clone()  # (D,)

    # Coalitions
    masks = _sample_binary_masks(D, n_samples, rng=rng, min_on=1, max_on=D - 1)  # (S, D)
    weights = np.array([_shapley_kernel_weight(D, int(m.sum())) for m in masks], dtype=np.float64)
    # Numerical safety
    weights[weights <= 0] = 1e-12

    # f(S) for each coalition
    ys = []
    for i in range(n_samples):
        m = torch.from_numpy(masks[i]).float().to(device)  # (D,)
        x_masked_node = m * x_node_orig + (1.0 - m) * baseline[0]
        x_pert = x.clone()
        x_pert[node_idx] = x_masked_node
        logits = model(x_pert, edge_index)  # [N, C]
        prob = _pos_prob_from_logits(logits[node_idx], pos_col=pos_col).item()
        ys.append(prob)
    ys = np.asarray(ys, dtype=np.float64)

    # Baseline: all OFF (x_node = baseline)
    x_off = baseline[0]
    x_pert = x.clone()
    x_pert[node_idx] = x_off
    logits_off = model(x_pert, edge_index)
    f0 = _pos_prob_from_logits(logits_off[node_idx], pos_col=pos_col).item()

    # Fit weighted ridge on (masks, ys - f0)
    y_centered = ys - f0
    shap = _weighted_ridge(masks.astype(np.float64), y_centered, weights, l2=l2)  # (D,)
    return shap, f0


# -------------------------
# Edge-level GraphSHAP (incident edges of node)
# -------------------------
@torch.no_grad()
def explain_node_edges(
    model: torch.nn.Module,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    node_idx: int,
    *,
    pos_col: int = 1,
    n_samples: int = 300,
    l2: float = 1e-3,
    symmetric_train_graph: bool = True,
    rng_seed: int = 7,
    device: Optional[torch.device] = None,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    SHAP-like values for each *incoming* edge to node_idx (neighbors that send messages into node_idx).
    OFF edges are removed during forward passes.
    Returns:
      neighbor_ids: (M,) numpy array of neighbor indices (sources of incoming edges)
      shap_values:  (M,) numpy array of SHAP values per neighbor
      f_baseline:   float, prob with all incoming edges OFF (self-loop may remain)
    """
    import numpy as np
    import torch
    import torch.nn.functional as F

    model.eval()
    if device is None:
        device = next(model.parameters()).device

    x = x.clone().to(device)
    edge_index = edge_index.to(device)
    src, dst = edge_index  # messages: src -> dst

    # --- Incoming edges into node_idx: dst == node_idx
    mask_incoming = (dst == node_idx)
    nbrs = src[mask_incoming]                  # neighbors that influence node_idx
    neighbors = torch.unique(nbrs)
    M = int(neighbors.numel())
    if M == 0:
        return np.zeros((0,), dtype=np.float64), np.zeros((0,), dtype=np.float64), float("nan")
    neighbors_np = neighbors.cpu().numpy()

    # For each neighbor, collect eid(s) where (src==nb & dst==node_idx)
    src_np = src.cpu().numpy()
    dst_np = dst.cpu().numpy()
    incident_eids = []
    for nb in neighbors_np:
        eids = np.where((src_np == nb) & (dst_np == node_idx))[0]
        incident_eids.append(eids)

    rng = np.random.RandomState(rng_seed)

    # Coalitions over incoming neighbors (allow all-on)
    masks = _sample_binary_masks(M, n_samples, rng=rng, min_on=1, max_on=M)
    weights = np.array([_shapley_kernel_weight(M, int(m.sum())) for m in masks], dtype=np.float64)
    weights[weights <= 0] = 1e-12

    ys = []
    for i in range(n_samples):
        m = masks[i]  # (M,)
        keep = torch.ones(edge_index.size(1), dtype=torch.bool, device=device)

        # Turn OFF incoming edges nb -> node_idx when mask==0
        for j, on in enumerate(m):
            if on == 0:
                keep[incident_eids[j]] = False
        edge_idx_on = edge_index[:, keep]

        if symmetric_train_graph:
            # also drop reverse edges node_idx -> nb when we turned off nb -> node_idx
            s2, d2 = edge_idx_on
            keep2 = torch.ones(edge_idx_on.size(1), dtype=torch.bool, device=device)
            for j, on in enumerate(m):
                if on == 0:
                    nb = int(neighbors_np[j])
                    keep2 &= ~((s2 == node_idx) & (d2 == nb))
            edge_idx_on = edge_idx_on[:, keep2]

        logits = model(x, edge_idx_on)
        prob = _pos_prob_from_logits(logits[node_idx], pos_col=pos_col).item()
        ys.append(prob)
    ys = np.asarray(ys, dtype=np.float64)

    # Baseline: all incoming OFF (keep self-loop if present)
    keep = torch.ones(edge_index.size(1), dtype=torch.bool, device=device)
    for eids in incident_eids:
        keep[eids] = False
    edge_idx_off = edge_index[:, keep]

    if symmetric_train_graph:
        s2, d2 = edge_idx_off
        keep2 = torch.ones(edge_idx_off.size(1), dtype=torch.bool, device=device)
        for nb in neighbors_np:
            keep2 &= ~((s2 == node_idx) & (d2 == int(nb)))
        edge_idx_off = edge_idx_off[:, keep2]

    logits_off = model(x, edge_idx_off)
    f0 = _pos_prob_from_logits(logits_off[node_idx], pos_col=pos_col).item()

    # SHAP by weighted ridge on (masks, ys - f0)
    y_centered = ys - f0
    shap = _weighted_ridge(masks.astype(np.float64), y_centered, weights, l2=l2)  # (M,)
    return neighbors_np, shap, f0


# -------------------------
# Plot + save helpers
# -------------------------
def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def save_feature_importance_plot(
    shap_values: np.ndarray,
    feature_names: list,
    save_path: str,
    title: str = "Feature SHAP (node)"
):
    order = np.argsort(-np.abs(shap_values))
    vals = shap_values[order]
    names = [feature_names[i] for i in order]

    plt.figure(figsize=(8, max(3, 0.35 * len(names))))
    y = np.arange(len(names))
    plt.barh(y, vals)
    plt.yticks(y, names)
    plt.xlabel("SHAP value (Δ prob)")
    plt.title(title, fontsize=9)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    _ensure_dir(os.path.dirname(save_path))
    plt.savefig(save_path, dpi=200)
    plt.close()


def save_edge_importance_plot(
    neighbor_ids: np.ndarray,
    shap_values: np.ndarray,
    save_path: str,
    title: str = "Edge SHAP (incident edges)"
):
    order = np.argsort(-np.abs(shap_values))
    vals = shap_values[order]
    labels = [str(int(n)) for n in neighbor_ids[order]]

    plt.figure(figsize=(8, max(3, 0.35 * len(labels))))
    y = np.arange(len(labels))
    plt.barh(y, vals)
    plt.yticks(y, labels)
    plt.xlabel("SHAP value (Δ prob)")
    plt.title(title + " • labels=neighbor node IDs", fontsize=9)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    _ensure_dir(os.path.dirname(save_path))
    plt.savefig(save_path, dpi=200)
    plt.close()




def shap_with_uncertainty_features(
    model, x, edge_index, node_idx,
    *, pos_col=1, n_samples=300, T=50, device=None, rng_seed=123
):
    """
    Run your feature SHAP T times with dropout active at inference:
      -> returns (mean_shap[D], std_shap[D]).
    """
    shap_list = []
    for _ in range(T):
        model.eval()
        with enable_dropout(model):
            sv, _ = explain_node_features(
                model, x, edge_index, node_idx,
                pos_col=pos_col, n_samples=n_samples, device=device
            )
        shap_list.append(sv)
    S = np.vstack(shap_list)  # (T, D)
    return S.mean(axis=0), S.std(axis=0)


def shap_with_uncertainty_edges(
    model, x, edge_index, node_idx,
    *, pos_col=1, n_samples=300, T=50, device=None, rng_seed=7
):
    """
    Run your edge SHAP T times with dropout active:
      -> returns (neighbor_ids[M], mean_shap[M], std_shap[M])
    """
    neigh = None
    shap_list = []
    for _ in range(T):
        model.eval()
        with enable_dropout(model):
            nbs, sv, _ = explain_node_edges(
                model, x, edge_index, node_idx,
                pos_col=pos_col, n_samples=n_samples, device=device
            )
        if neigh is None:
            neigh = nbs
        shap_list.append(sv)
    S = np.vstack(shap_list)  # (T, M)
    return neigh, S.mean(axis=0), S.std(axis=0)


# ---- error-bar plots (mean ± std) ----
def save_feature_importance_errorbar(mean_vals, std_vals, feature_names, save_path, title):
    order = np.argsort(-np.abs(mean_vals))
    mv, sv = mean_vals[order], std_vals[order]
    names  = [feature_names[i] for i in order]

    plt.figure(figsize=(8, max(3, 0.35*len(names))))
    y = np.arange(len(names))
    plt.barh(y, mv, xerr=sv, capsize=3)
    plt.yticks(y, names)
    plt.xlabel("SHAP value (Δ prob)")
    plt.title(title + " (mean ± std across MC)", fontsize=9)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    _ensure_dir(os.path.dirname(save_path))
    plt.savefig(save_path, dpi=200)
    plt.close()


def save_edge_importance_errorbar(neighbor_ids, mean_vals, std_vals, save_path, title):
    order = np.argsort(-np.abs(mean_vals))
    mv, sv = mean_vals[order], std_vals[order]
    labels = [str(int(n)) for n in neighbor_ids[order]]

    plt.figure(figsize=(8, max(3, 0.35*len(labels))))
    y = np.arange(len(labels))
    plt.barh(y, mv, xerr=sv, capsize=3)
    plt.yticks(y, labels)
    plt.xlabel("SHAP value (Δ prob)")
    plt.title(title + " (mean ± std across MC)", fontsize=9)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    _ensure_dir(os.path.dirname(save_path))
    plt.savefig(save_path, dpi=200)
    plt.close()
