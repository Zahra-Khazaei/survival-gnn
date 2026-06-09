#  utils/utils_calibration.py
import numpy as np
import matplotlib.pyplot as plt

def binary_calibration_bins(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15):
    """Return per-bin avg confidence, accuracy, counts."""
    probs = np.clip(probs, 1e-7, 1 - 1e-7)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.digitize(probs, bins) - 1
    idx = np.clip(idx, 0, n_bins - 1)

    bin_conf, bin_acc, bin_cnt = np.zeros(n_bins), np.zeros(n_bins), np.zeros(n_bins, dtype=int)
    for b in range(n_bins):
        m = (idx == b)
        if m.any():
            bin_conf[b] = probs[m].mean()
            bin_acc[b]  = labels[m].mean()
            bin_cnt[b]  = int(m.sum())
        else:
            bin_conf[b] = (bins[b] + bins[b+1]) / 2.0
            bin_acc[b]  = np.nan
    return bins, bin_conf, bin_acc, bin_cnt

def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    bins, bin_conf, bin_acc, bin_cnt = binary_calibration_bins(probs, labels, n_bins)
    N = len(labels)
    ece = 0.0
    for c, a, n in zip(bin_conf, bin_acc, bin_cnt):
        if n > 0 and not np.isnan(a):
            ece += (n / N) * abs(a - c)
    return float(ece)

def brier_score(probs: np.ndarray, labels: np.ndarray) -> float:
    probs = np.clip(probs, 1e-7, 1 - 1e-7)
    return float(np.mean((probs - labels) ** 2))

def nll_loss(probs: np.ndarray, labels: np.ndarray) -> float:
    probs = np.clip(probs, 1e-7, 1 - 1e-7)
    return float(-np.mean(labels * np.log(probs) + (1 - labels) * np.log(1 - probs)))

def save_reliability_diagram(probs: np.ndarray, labels: np.ndarray, save_path: str, title: str, n_bins: int = 15):
    bins, bin_conf, bin_acc, bin_cnt = binary_calibration_bins(probs, labels, n_bins)
    # Replace NaNs (empty bins) with 0 for plotting
    plot_acc = np.where(np.isnan(bin_acc), 0.0, bin_acc)
    centers = (bins[:-1] + bins[1:]) / 2.0

    plt.figure(figsize=(5, 5))
    # bars showing accuracy per bin
    plt.bar(centers, plot_acc, width=(bins[1]-bins[0]) * 0.9, alpha=0.6, edgecolor="black")
    # diagonal perfect calibration
    plt.plot([0, 1], [0, 1], linestyle="--")
    # overlay average confidence per bin as points
    plt.scatter(centers, bin_conf, s=20)
    plt.xlim(0, 1); plt.ylim(0, 1)
    plt.xlabel("Predicted probability")
    plt.ylabel("Empirical accuracy")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
