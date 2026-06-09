#!/usr/bin/env python3
"""
Generate the illustrative figures shown in the README.

IMPORTANT: every figure here is produced from *synthetic, randomly generated
data*. None of these plots contain real patient data or real experimental
results. They exist purely to illustrate the kind of output the pipeline
produces. Run with:

    python make_readme_figures.py

Outputs are written to assets/.
"""

import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ASSETS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
os.makedirs(ASSETS, exist_ok=True)

# Fixed seed so the illustrative figures are reproducible.
RNG = np.random.default_rng(42)


def workflow_diagram():
    """Schematic of the end-to-end pipeline (no data — pure diagram)."""
    fig, ax = plt.subplots(figsize=(11, 4.2))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 40)
    ax.axis("off")

    def box(x, y, w, h, text, color):
        ax.add_patch(FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.3,rounding_size=1.2",
            linewidth=1.5, edgecolor="#37474f", facecolor=color, zorder=2))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=9.5, zorder=3, wrap=True)

    def arrow(x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch(
            (x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=14,
            linewidth=1.4, color="#546e7a", zorder=1))

    blue, green, amber, grey = "#bbdefb", "#c8e6c9", "#ffe0b2", "#eceff1"
    w, h, y = 15, 9, 24

    # Main pipeline row
    box(2,  y, w, h, "Clinical + radiomic\nfeatures", blue)
    box(20, y, w, h, "Leakage-free\nKNN graph", blue)
    box(38, y, w, h, "Survival GNN\n(hazard head)", green)
    box(56, y, w, h, "Optuna tuning\n+ deep ensemble", green)
    box(74, y, w, h, "Evaluation\nC-index · IBS · AUC", amber)
    for x in (17, 35, 53, 71):
        arrow(x, y + h / 2, x + 3, y + h / 2)

    # Explainability branch
    box(74, 6, w, h, "Explainability\nfeature / edge SHAP", amber)
    arrow(63, y, 78, 6 + h)

    # Baselines feeding evaluation
    box(38, 6, w, h, "Baselines\nCPH · RSF · MLP", grey)
    arrow(53, 6 + h / 2, 74, y + 1)

    ax.set_title("Survival GNN — pipeline overview", fontsize=13, pad=8)
    fig.tight_layout()
    out = os.path.join(ASSETS, "workflow.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def data_overview():
    """Schematic overview of feature groups and the four cohorts.

    All distributions are synthetic; cohort rows describe the experimental
    *design* (feature composition, follow-up window, evaluation horizons) and
    contain no real patient statistics.
    """
    fig = plt.figure(figsize=(11, 6.4))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.9],
                          hspace=0.45, wspace=0.25)

    # ---- Panel A: feature groups -------------------------------------------
    axA = fig.add_subplot(gs[0, 0])
    axA.axis("off")
    axA.set_title("Feature groups", fontsize=12, loc="left", weight="bold")
    axA.add_patch(FancyBboxPatch((0.02, 0.05), 0.46, 0.9,
                  boxstyle="round,pad=0.02", transform=axA.transAxes,
                  facecolor="#bbdefb", edgecolor="#37474f", lw=1.3))
    axA.text(0.25, 0.86, "Clinical", ha="center", transform=axA.transAxes,
             fontsize=10.5, weight="bold")
    axA.text(0.25, 0.46,
             "• Age\n• PSA\n• Gleason\n   (primary / secondary / global)\n• Clinical stage",
             ha="center", va="center", transform=axA.transAxes, fontsize=9.5)
    axA.add_patch(FancyBboxPatch((0.52, 0.05), 0.46, 0.9,
                  boxstyle="round,pad=0.02", transform=axA.transAxes,
                  facecolor="#c8e6c9", edgecolor="#37474f", lw=1.3))
    axA.text(0.75, 0.86, "Radiomic (imaging)", ha="center",
             transform=axA.transAxes, fontsize=10.5, weight="bold")
    axA.text(0.75, 0.46,
             "• RADIOMIC_PN_1…6\n• RADIOMIC_BCR_1…6\n\n(image-derived\nquantitative features)",
             ha="center", va="center", transform=axA.transAxes, fontsize=9.5)

    # ---- Panel B: synthetic example distributions --------------------------
    axB = fig.add_subplot(gs[0, 1])
    age = RNG.normal(66, 7, 400)
    psa = RNG.lognormal(mean=2.0, sigma=0.6, size=400)  # synthetic, skewed
    axB.hist(age, bins=22, color="#64b5f6", alpha=0.85, label="Age (years)")
    axB.set_xlabel("Age (years)", color="#1565c0")
    axB.tick_params(axis="x", colors="#1565c0")
    axB.set_ylabel("Count")
    axB2 = axB.twiny()
    axB2.hist(psa, bins=22, color="#81c784", alpha=0.55, label="PSA (ng/mL)")
    axB2.set_xlabel("PSA (ng/mL)", color="#2e7d32")
    axB2.tick_params(axis="x", colors="#2e7d32")
    axB.text(0.5, -0.28, "Example feature distributions (illustrative — synthetic data)",
             transform=axB.transAxes, ha="center", fontsize=9.5, style="italic")

    # ---- Panel C: cohort design table --------------------------------------
    axC = fig.add_subplot(gs[1, :])
    axC.axis("off")
    axC.set_title("Cohort design", fontsize=12, loc="left", weight="bold")
    col_labels = ["Cohort", "Features", "Follow-up window", "Evaluation horizons"]
    rows = [
        ["Cohort 1",   "Clinical",              "~11 years",  "5-year"],
        ["Cohort 1r",  "Clinical + radiomic",   "~11 years",  "5-year"],
        ["Cohort 2",   "Clinical",              "~18 years",  "5- & 10-year"],
        ["Cohort 1&2", "Clinical (combined)",   "~18 years",  "5- & 10-year"],
    ]
    table = axC.table(cellText=rows, colLabels=col_labels,
                      cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.7)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("#b0bec5")
        if r == 0:
            cell.set_facecolor("#37474f")
            cell.set_text_props(color="white", weight="bold")
        elif rows[r - 1][1].startswith("Clinical + radiomic"):
            cell.set_facecolor("#e8f5e9")
        else:
            cell.set_facecolor("#f5f7f9" if r % 2 else "#ffffff")

    fig.suptitle("Data overview — clinical + radiomic features across four cohorts",
                 fontsize=13, y=1.02)
    out = os.path.join(ASSETS, "data_overview.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def risk_stratified_km():
    """Illustrative Kaplan-Meier-style curves for model-predicted risk groups."""
    months = np.arange(0, 121)
    # Three synthetic risk groups with different hazard rates.
    groups = {
        "Low risk": (0.0018, "#2c7fb8"),
        "Medium risk": (0.0045, "#7fcdbb"),
        "High risk": (0.0095, "#d95f0e"),
    }

    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    for label, (hazard, color) in groups.items():
        survival = np.exp(-hazard * months)
        # Add a little synthetic step-like noise so it reads as an empirical curve.
        survival = np.clip(survival + RNG.normal(0, 0.004, months.shape).cumsum() * 0.0,
                           0, 1)
        ax.step(months, survival, where="post", label=label, color=color, lw=2)

    ax.set_xlabel("Time (months)")
    ax.set_ylabel("Survival probability")
    ax.set_title("Risk-stratified survival curves\n(illustrative — synthetic data)")
    ax.set_ylim(0, 1.02)
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    out = os.path.join(ASSETS, "example_km_curves.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")


def cindex_by_model():
    """Illustrative concordance-index comparison across model families."""
    models = ["CPH", "RSF", "MLP", "GCN", "GraphSAGE", "GraphTransformer"]
    # Synthetic c-index values with synthetic error bars.
    means = np.array([0.66, 0.68, 0.67, 0.70, 0.71, 0.73])
    errs = RNG.uniform(0.015, 0.03, len(models))
    colors = ["#bdbdbd"] * 3 + ["#41ab5d"] * 3  # baselines grey, GNNs green

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.bar(models, means, yerr=errs, color=colors, capsize=4, edgecolor="white")
    ax.axhline(0.5, ls="--", color="k", lw=1, alpha=0.6)
    ax.text(0.02, 0.505, "chance (0.5)", transform=ax.get_yaxis_transform(),
            fontsize=8, alpha=0.7)
    ax.set_ylabel("Concordance index (C-index)")
    ax.set_ylim(0.45, 0.8)
    ax.set_title("Baselines vs. graph models\n(illustrative — synthetic data)")
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    fig.tight_layout()
    out = os.path.join(ASSETS, "example_cindex_comparison.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")


def knn_graph_schematic():
    """Schematic of a patient KNN similarity graph colored by predicted risk."""
    n = 60
    pts = RNG.normal(0, 1, (n, 2))
    risk = (pts[:, 0] + pts[:, 1])
    risk = (risk - risk.min()) / (np.ptp(risk) + 1e-9)

    # Connect each node to its k nearest neighbours.
    k = 4
    fig, ax = plt.subplots(figsize=(5.2, 5.0))
    for i in range(n):
        d = np.sum((pts - pts[i]) ** 2, axis=1)
        nbrs = np.argsort(d)[1:k + 1]
        for j in nbrs:
            ax.plot([pts[i, 0], pts[j, 0]], [pts[i, 1], pts[j, 1]],
                    color="#999999", lw=0.4, alpha=0.5, zorder=1)
    sc = ax.scatter(pts[:, 0], pts[:, 1], c=risk, cmap="RdYlBu_r",
                    s=70, edgecolor="white", zorder=2)
    cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("Predicted risk (illustrative)")
    ax.set_title("Patient KNN similarity graph\n(illustrative — synthetic data)")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    fig.tight_layout()
    out = os.path.join(ASSETS, "example_knn_graph.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    workflow_diagram()
    data_overview()
    risk_stratified_km()
    cindex_by_model()
    knn_graph_schematic()
    print("Done. All figures are synthetic illustrations (no real data).")
