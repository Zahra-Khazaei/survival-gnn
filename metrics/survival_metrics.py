#metrics/survival_metrics.py

import torch
import numpy as np
from sklearn.metrics import roc_auc_score


def find_best_threshold_for_bacc(
    hazards: torch.Tensor,
    times: torch.Tensor,
    events: torch.Tensor,
    bin_width: float,
    horizon_months: float,
    grid: np.ndarray | None = None,
):
    if grid is None:
        grid = np.linspace(0.01, 0.99, 200, dtype=np.float32)

    hazards_cpu = hazards.detach().float().cpu()
    times_cpu = times.detach().float().cpu()
    events_cpu = events.detach().long().cpu()

    n, T = hazards_cpu.shape
    if T < 1:
        return 0.5, float("nan")

    # horizon bin index (end of bin): ceil(h/w)-1
    hbin = int(np.ceil(float(horizon_months) / float(bin_width)) - 1)
    hbin = max(0, min(hbin, T - 1))

    one_minus = (1.0 - hazards_cpu).clamp(min=1e-8, max=1.0)
    surv = torch.cumprod(one_minus, dim=1)[:, hbin]
    risk = (1.0 - surv).numpy().astype(np.float32)

    horizon = float(horizon_months)
    y_pos = (events_cpu.numpy() == 1) & (times_cpu.numpy() <= horizon)
    y_neg = times_cpu.numpy() > horizon
    valid = y_pos | y_neg

    if valid.sum() == 0:
        return 0.5, float("nan")

    y = y_pos[valid].astype(np.int32)
    r = risk[valid]

    thr = grid.astype(np.float32)
    preds = (r[:, None] >= thr[None, :])

    y1 = (y == 1)[:, None]
    y0 = (y == 0)[:, None]

    tp = np.sum(preds & y1, axis=0).astype(np.float32)
    fn = np.sum((~preds) & y1, axis=0).astype(np.float32)
    tn = np.sum((~preds) & y0, axis=0).astype(np.float32)
    fp = np.sum(preds & y0, axis=0).astype(np.float32)

    tpr = tp / (tp + fn + 1e-12)
    tnr = tn / (tn + fp + 1e-12)
    bacc = 0.5 * (tpr + tnr)

    best_i = int(np.nanargmax(bacc))
    best_thr = float(thr[best_i])
    best_bacc = float(bacc[best_i])
    return best_thr, best_bacc


def hazards_to_survival(hazards: torch.Tensor) -> torch.Tensor:
    one_minus_h = 1.0 - hazards
    return torch.cumprod(one_minus_h, dim=1)


def risk_at_horizon_from_hazards(
    hazards: torch.Tensor,
    bin_width: float,
    horizon_months: float,
) -> torch.Tensor:
    _, T = hazards.shape
    S = hazards_to_survival(hazards)

    bin_idx = int(np.ceil(float(horizon_months) / float(bin_width))) - 1
    bin_idx = max(0, min(bin_idx, T - 1))

    surv_at_h = S[:, bin_idx]
    return 1.0 - surv_at_h


def concordance_index(times, events, risk_scores) -> float:
    if isinstance(times, torch.Tensor):
        times = times.detach().cpu().numpy()
    if isinstance(events, torch.Tensor):
        events = events.detach().cpu().numpy()
    if isinstance(risk_scores, torch.Tensor):
        risk_scores = risk_scores.detach().cpu().numpy()

    times = np.asarray(times, dtype=float)
    events = np.asarray(events, dtype=int)
    risk_scores = np.asarray(risk_scores, dtype=float)

    n = len(times)
    if len(events) != n or len(risk_scores) != n:
        raise ValueError("times, events, risk_scores must have same length.")

    event_mask = events == 1
    if not event_mask.any():
        return float("nan")

    # Vectorized: for each event i, compare against all j where t_j > t_i
    t_ev = times[event_mask][:, None]        # [E, 1]
    r_ev = risk_scores[event_mask][:, None]  # [E, 1]
    t_all = times[None, :]                   # [1, N]
    r_all = risk_scores[None, :]             # [1, N]

    comparable = t_ev < t_all               # [E, N]; t_i < t_i is False, so self excluded
    num_comparable = float(comparable.sum())
    if num_comparable == 0:
        return float("nan")

    concordant = comparable & (r_ev > r_all)
    tied = comparable & (r_ev == r_all)
    return float((float(concordant.sum()) + 0.5 * float(tied.sum())) / num_comparable)


def _km_fit(times: np.ndarray, event_observed: np.ndarray):
    times = np.asarray(times, dtype=float)
    event_observed = np.asarray(event_observed, dtype=int)

    order = np.argsort(times)
    t = times[order]
    e = event_observed[order]

    uniq = np.unique(t)
    S = 1.0
    surv_vals = []

    for ut in uniq:
        at_risk = np.sum(t >= ut)
        d = np.sum((t == ut) & (e == 1))
        if at_risk > 0:
            S *= (1.0 - d / at_risk)
        surv_vals.append(S)

    return uniq, np.asarray(surv_vals, dtype=float)


def _km_predict(uniq_times: np.ndarray, surv_vals: np.ndarray, t_query):
    tq = np.asarray(t_query, dtype=float)
    out = np.ones_like(tq, dtype=float)

    idx = np.searchsorted(uniq_times, tq, side="right") - 1
    valid = idx >= 0
    out[valid] = surv_vals[idx[valid]]
    out[~valid] = 1.0
    return out


def ipcw_weights_at_horizon(times_np, events_np, horizon):
    times_np = np.asarray(times_np, dtype=float)
    events_np = np.asarray(events_np, dtype=int)
    horizon = float(horizon)

    cens_event = (events_np == 0).astype(int)  # 1 = censored
    uniq, Gvals = _km_fit(times_np, cens_event)

    t_star = np.minimum(times_np, horizon)
    G_star = _km_predict(uniq, Gvals, t_star)
    G_star = np.clip(G_star, 1e-6, 1.0)

    Gh = float(np.clip(_km_predict(uniq, Gvals, np.array([horizon]))[0], 1e-6, 1.0))

    w = np.zeros_like(times_np, dtype=float)

    is_pos = (events_np == 1) & (times_np <= horizon)
    is_neg = (times_np > horizon)
    is_amb = (events_np == 0) & (times_np <= horizon)

    w[is_pos] = 1.0 / G_star[is_pos]
    w[is_neg] = 1.0 / Gh
    w[is_amb] = 0.0

    return w


def binary_metrics_at_horizon(
    hazards: torch.Tensor,
    times: torch.Tensor,
    events: torch.Tensor,
    bin_width: float,
    horizon_months: float,
    threshold: float = 0.5,
    verbose: bool = False,
):
    risk = risk_at_horizon_from_hazards(hazards, bin_width, horizon_months)
    risk_np = risk.detach().cpu().numpy().astype(float)

    times_np = times.detach().cpu().numpy().astype(float)
    events_np = events.detach().cpu().numpy().astype(int)

    pos = (events_np == 1) & (times_np <= horizon_months)
    neg = (times_np > horizon_months)
    keep = pos | neg

    n_total = int(len(times_np))
    n_used = int(keep.sum())
    n_amb_excluded = int(((events_np == 0) & (times_np <= horizon_months)).sum())

    if verbose:
        print(f"[DEBUG][{float(horizon_months)}m] total={n_total} used={n_used} amb_excluded={n_amb_excluded}")

    if n_used == 0:
        return {
            "auc": float("nan"),
            "auc_ipcw": float("nan"),
            "sensitivity": float("nan"),
            "specificity": float("nan"),
            "balanced_accuracy": float("nan"),
            "brier_ipcw": float("nan"),
            "n_used": 0,
            "n_pos": 0,
            "n_neg": 0,
            "n_total": n_total,
            "n_amb_excluded": n_amb_excluded,
        }

    y = pos[keep].astype(np.int32)
    score = risk_np[keep]

    n_pos = int(y.sum())
    n_neg = int(n_used - n_pos)

    if len(np.unique(y)) == 1:
        auc = float("nan")
    else:
        auc = float(roc_auc_score(y, score))

    w_all = ipcw_weights_at_horizon(times_np, events_np, horizon_months)
    w = w_all[keep].astype(float)

    if len(np.unique(y)) == 1 or float(np.sum(w)) <= 0.0:
        auc_ipcw = float("nan")
    else:
        auc_ipcw = float(roc_auc_score(y, score, sample_weight=w))

    y_pred = (score >= float(threshold)).astype(np.int32)

    tp = int(np.sum((y == 1) & (y_pred == 1)))
    tn = int(np.sum((y == 0) & (y_pred == 0)))
    fp = int(np.sum((y == 0) & (y_pred == 1)))
    fn = int(np.sum((y == 1) & (y_pred == 0)))

    sens = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    spec = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
    bal_acc = 0.5 * (sens + spec) if (not np.isnan(sens) and not np.isnan(spec)) else float("nan")

    w_sum = float(np.sum(w))
    if w_sum <= 0.0:
        brier_ipcw = float("nan")
    else:
        brier_ipcw = float(np.sum(w * (y.astype(float) - score.astype(float)) ** 2) / (w_sum + 1e-12))

    return {
        "auc": float(auc),
        "auc_ipcw": float(auc_ipcw),
        "sensitivity": float(sens),
        "specificity": float(spec),
        "balanced_accuracy": float(bal_acc),
        "brier_ipcw": float(brier_ipcw),
        "n_used": n_used,
        "n_pos": n_pos,
        "n_neg": n_neg,
        "n_total": n_total,
        "n_amb_excluded": n_amb_excluded,
    }


def ibs_ipcw_from_hazards(
    hazards: torch.Tensor,
    times: torch.Tensor,
    events: torch.Tensor,
    bin_width: float,
    t_max: float,
):
    bin_width = float(bin_width)
    t_max = float(t_max)

    if t_max <= bin_width:
        return float("nan"), []

    horizons = np.arange(bin_width, t_max + 1e-9, bin_width, dtype=float)

    briers = []
    for h in horizons:
        m = binary_metrics_at_horizon(
            hazards=hazards,
            times=times,
            events=events,
            bin_width=bin_width,
            horizon_months=float(h),
            threshold=0.5,
            verbose=False,
        )
        briers.append(float(m.get("brier_ipcw", float("nan"))))

    briers = np.asarray(briers, dtype=float)

    if np.all(np.isnan(briers)):
        return float("nan"), horizons.tolist()

    valid = ~np.isnan(briers)
    if valid.sum() < 2:
        return float("nan"), horizons.tolist()

    b_fill = np.interp(horizons, horizons[valid], briers[valid])
    area = float(np.trapz(b_fill, horizons))
    denom = float(horizons[-1] - horizons[0])
    if denom <= 0:
        return float("nan"), horizons.tolist()

    return float(area / denom), horizons.tolist()


def evaluate_survival(
    hazards: torch.Tensor,
    times: torch.Tensor,
    events: torch.Tensor,
    bin_width: float = 6.0,
    horizons_months=(60.0, 120.0),
    threshold: float = 0.5,
    ibs_max_time: float | None = None,
    verbose: bool = False,
):
    S = hazards_to_survival(hazards)
    risk_full = 1.0 - S[:, -1]
    c_index = concordance_index(times, events, risk_full)

    results = {"c_index": float(c_index), "horizons": {}}

    for h in horizons_months:
        h = float(h)
        results["horizons"][h] = binary_metrics_at_horizon(
            hazards=hazards,
            times=times,
            events=events,
            bin_width=float(bin_width),
            horizon_months=h,
            threshold=float(threshold),
            verbose=verbose,
        )

    if ibs_max_time is None:
        if horizons_months is None or len(horizons_months) == 0:
            ibs_max_time = None
        else:
            ibs_max_time = float(min(max([float(h) for h in horizons_months]), 60.0))

    if ibs_max_time is not None:
        ibs_max_time = float(bin_width) * float(int(float(ibs_max_time) // float(bin_width)))

    if ibs_max_time is not None and float(ibs_max_time) > float(bin_width):
        ibs_val, grid = ibs_ipcw_from_hazards(
            hazards=hazards,
            times=times,
            events=events,
            bin_width=float(bin_width),
            t_max=float(ibs_max_time),
        )
        results["ibs_ipcw"] = float(ibs_val)
        results["ibs_grid"] = grid
    else:
        results["ibs_ipcw"] = float("nan")
        results["ibs_grid"] = []

    results["ibs_max_time"] = float(ibs_max_time) if ibs_max_time is not None else None
    return results