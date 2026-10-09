"""
PCA (Principal Component Analysis) — Spotify Playlist Analytics
================================================================

Fits PCA on all 10 continuous audio features.
The number of components to RETAIN is determined automatically
from the Scree Plot elbow (largest acceleration in explained variance).

PC1 and PC2 are always the first two principal components and are used
for 2D visualisation regardless of the elbow K.

Dataset : processed_data/preprocessed_dataset.csv
Features: 10 continuous audio features (already StandardScaled — no re-scaling).
"""

import os
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


BASE_DIR          = os.path.dirname(os.path.abspath(__file__))
PREPROCESSED_PATH = os.path.join(BASE_DIR, "processed_data", "preprocessed_dataset.csv")
CHART_DIR         = os.path.join(BASE_DIR, "static", "charts")
os.makedirs(CHART_DIR, exist_ok=True)

MAX_SAMPLE = 1500

AUDIO_FEATURES = [
    "duration_ms", "danceability", "energy", "loudness",
    "speechiness", "acousticness", "instrumentalness",
    "liveness", "valence", "tempo",
]


# ──────────────────────────────────────────────────────────────
# DATA LOADING
# ──────────────────────────────────────────────────────────────

def _load_preprocessed():
    if not os.path.exists(PREPROCESSED_PATH):
        raise FileNotFoundError(
            "Preprocessed dataset not found: processed_data/preprocessed_dataset.csv"
        )
    df        = pd.read_csv(PREPROCESSED_PATH)
    available = [c for c in AUDIO_FEATURES if c in df.columns]
    if not available:
        raise ValueError(
            "None of the expected audio feature columns found in preprocessed_dataset.csv"
        )
    numeric = df[available].copy()
    numeric = numeric.replace([np.inf, -np.inf], np.nan)
    numeric = numeric.fillna(numeric.median(numeric_only=True)).fillna(0)
    if numeric.empty:
        raise ValueError("No numeric features available for PCA.")
    return numeric


# ──────────────────────────────────────────────────────────────
# ELBOW DETECTION (Scree Plot acceleration method)
# ──────────────────────────────────────────────────────────────

def _find_elbow_k(exp_var):
    """
    Determine the number of components to retain using the second-derivative
    (acceleration) method on the explained variance values.

    The elbow is the point where adding another component gives the least
    additional benefit — mathematically, where the rate-of-change of the
    scree plot values drops most sharply.

    Steps
    -----
    1. Compute first differences (Δ variance per additional PC).
    2. Compute second differences (how fast the Δ is shrinking).
    3. The elbow = index of the most negative second difference + 1
       (we keep one more component than where the curve bends).
    4. Clamp to [2, n_features].

    Returns
    -------
    k   int   number of components to retain
    """
    n = len(exp_var)
    if n < 3:
        return n

    # First and second differences on the raw variance ratios
    d1 = np.diff(exp_var)         # first derivative
    d2 = np.diff(d1)              # second derivative (acceleration)

    # The most negative second derivative = sharpest bend in the scree
    elbow_idx = int(np.argmin(d2))   # index in d2 (0-based)
    k         = elbow_idx + 2        # +2: d2 is shifted by 2 from exp_var

    return max(2, min(k, n))


# ──────────────────────────────────────────────────────────────
# CHARTS  (neutral colors — theme-agnostic)
# ──────────────────────────────────────────────────────────────

def _make_scree_plot(exp_var, cum_var, elbow_k, filename):
    """Scree plot with bars + cumulative line + elbow marker."""
    path = os.path.join(CHART_DIR, filename)
    x    = list(range(1, len(exp_var) + 1))
    pct  = [v * 100 for v in exp_var]
    cpct = [v * 100 for v in cum_var]

    fig, ax1 = plt.subplots(figsize=(9, 4.8), dpi=150)
    fig.patch.set_facecolor("#ffffff")
    ax1.set_facecolor("#f8f9fa")

    # Colour bars: accent for kept, muted for dropped
    bar_colors = ["#1db954" if i + 1 <= elbow_k else "#cccccc" for i in range(len(x))]
    bars = ax1.bar(x, pct, color=bar_colors, edgecolor="#aaaaaa", width=0.55,
                   label="Individual Variance (%)")

    for bar, val in zip(bars, pct):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.6,
                 f"{val:.1f}%", ha="center", va="bottom", fontsize=8,
                 color="#333333", fontweight="bold")

    ax1.set_xlabel("Principal Component", fontsize=10, color="#333333")
    ax1.set_ylabel("Individual Explained Variance (%)", fontsize=10, color="#333333")
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"PC{i}" for i in x])
    ax1.tick_params(colors="#333333")

    # Cumulative line on right axis
    ax2 = ax1.twinx()
    ax2.plot(x, cpct, color="#5c7cfa", marker="o", linewidth=2.2,
             markersize=6, label="Cumulative Variance (%)")
    ax2.axhline(80, color="#f59f00", linestyle=":", alpha=0.8, linewidth=1.5, label="80% threshold")
    ax2.axhline(90, color="#ff4757", linestyle=":", alpha=0.8, linewidth=1.5, label="90% threshold")
    ax2.set_ylabel("Cumulative Explained Variance (%)", fontsize=10, color="#5c7cfa")
    ax2.set_ylim(0, 108)
    ax2.tick_params(colors="#333333")

    # Elbow marker: vertical dashed line
    ax1.axvline(elbow_k + 0.5, color="#1db954", linestyle="--", linewidth=2,
                label=f"Elbow → K = {elbow_k}")

    # Annotation
    elbow_cum = cpct[elbow_k - 1]
    ax2.annotate(
        f"Elbow K={elbow_k}\n{elbow_cum:.1f}% variance",
        xy=(elbow_k, elbow_cum),
        xytext=(elbow_k + 0.6, elbow_cum - 8),
        arrowprops=dict(arrowstyle="->", color="#1db954", lw=1.5),
        fontsize=9, fontweight="bold", color="#1db954",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#f0fff4",
                  edgecolor="#1db954", alpha=0.92),
    )

    ax1.set_title("PCA Scree Plot — Elbow Determines Components to Retain",
                  fontsize=12, fontweight="bold", color="#1a1a2e", pad=12)
    ax1.grid(True, axis="y", linestyle="--", alpha=0.30, color="#cccccc")

    lines1, labs1 = ax1.get_legend_handles_labels()
    lines2, labs2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labs1 + labs2,
               loc="center right", fontsize=8, framealpha=0.92, facecolor="#ffffff")

    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _make_2d_plot(X_pca, var1, var2, filename):
    """PC1 vs PC2 scatter — always uses the first two components."""
    path = os.path.join(CHART_DIR, filename)
    fig, ax = plt.subplots(figsize=(9, 6), dpi=150)
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#f8f9fa")

    sc = ax.scatter(X_pca[:, 0], X_pca[:, 1],
                    c=X_pca[:, 0], cmap="viridis",
                    s=24, alpha=0.75, edgecolors="none")

    cb = fig.colorbar(sc, ax=ax, pad=0.02)
    cb.set_label("PC1 Projection Value", fontsize=9, color="#333333")
    cb.ax.tick_params(labelsize=8)

    ax.axhline(0, color="#888888", linestyle="--", linewidth=1, alpha=0.4)
    ax.axvline(0, color="#888888", linestyle="--", linewidth=1, alpha=0.4)

    ax.set_title(
        f"PC1 vs PC2 Projection  ({var1 + var2:.1f}% total variance)",
        fontsize=12, fontweight="bold", color="#1a1a2e", pad=12,
    )
    ax.set_xlabel(f"Principal Component 1  ({var1:.1f}% variance)", fontsize=10, color="#333333")
    ax.set_ylabel(f"Principal Component 2  ({var2:.1f}% variance)", fontsize=10, color="#333333")
    ax.tick_params(colors="#333333")
    ax.grid(True, linestyle="--", alpha=0.30, color="#cccccc")

    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _make_loadings_heatmap(loadings, feature_names, n_display_pcs, filename):
    """Feature loadings heatmap for the retained components."""
    path             = os.path.join(CHART_DIR, filename)
    matrix           = loadings[:n_display_pcs, :]
    pc_labels        = [f"PC{i + 1}" for i in range(n_display_pcs)]
    clean_feat_names = [f.replace("_", " ").title() for f in feature_names]

    fig, ax = plt.subplots(figsize=(10, max(3.5, n_display_pcs * 0.7 + 1.5)), dpi=150)
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#f8f9fa")

    im = ax.imshow(matrix, cmap="RdBu_r", aspect="auto", vmin=-0.65, vmax=0.65)
    cb = fig.colorbar(im, ax=ax, pad=0.02)
    cb.set_label("Feature Loading Weight", fontsize=9, color="#333333")
    cb.ax.tick_params(labelsize=8)

    ax.set_xticks(range(len(clean_feat_names)))
    ax.set_xticklabels(clean_feat_names, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(n_display_pcs))
    ax.set_yticklabels(pc_labels, fontsize=10, fontweight="bold")
    ax.tick_params(colors="#333333")

    for i in range(n_display_pcs):
        for j in range(len(clean_feat_names)):
            val  = matrix[i, j]
            tcol = "#ffffff" if abs(val) > 0.35 else "#111111"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    color=tcol, fontsize=8, fontweight="bold")

    ax.set_title("PCA Feature Loadings  (which features drive each component)",
                 fontsize=12, fontweight="bold", color="#1a1a2e", pad=12)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ──────────────────────────────────────────────────────────────
# PUBLIC ENTRY POINT
# ──────────────────────────────────────────────────────────────

def run_pca():
    """
    Fit PCA on all 10 audio features.
    Automatically determines the number of components to retain
    using the Scree Plot elbow (second-derivative / acceleration method).
    PC1 and PC2 are always used for 2D visualisation.
    """
    numeric       = _load_preprocessed()
    feature_names = list(numeric.columns)
    n_features    = len(feature_names)
    original_rows = len(numeric)

    sample = numeric.sample(n=min(MAX_SAMPLE, len(numeric)), random_state=42)
    X      = sample.to_numpy(dtype=float)   # already StandardScaled

    # ── Fit full PCA (all components) ────────────────────────
    model  = PCA(n_components=n_features, random_state=42)
    X_pca  = model.fit_transform(X)

    exp_var = model.explained_variance_ratio_           # array length n_features
    cum_var = np.cumsum(exp_var)

    # ── Elbow: how many components to retain ─────────────────
    elbow_k       = _find_elbow_k(exp_var)
    elbow_var_pct = round(float(cum_var[elbow_k - 1] * 100), 1)

    # ── Per-component summary ─────────────────────────────────
    components_info = []
    for i in range(n_features):
        components_info.append({
            "name":            f"PC{i + 1}",
            "variance_ratio":  round(float(exp_var[i] * 100), 2),
            "cumulative_ratio": round(float(cum_var[i] * 100), 2),
            "singular_value":  round(float(model.singular_values_[i]), 2),
            "retained":        (i + 1) <= elbow_k,
        })

    # ── Loadings table (PC1–PC4 shown) ───────────────────────
    loadings_table = []
    for j, fname in enumerate(feature_names):
        row = {"feature": fname.replace("_", " ").title()}
        for i in range(min(4, n_features)):
            row[f"pc{i + 1}"] = round(float(model.components_[i, j]), 3)
        loadings_table.append(row)

    # PC1 / PC2 variance for 2D chart labels
    var1 = round(float(exp_var[0] * 100), 1)
    var2 = round(float(exp_var[1] * 100), 1)

    # ── Charts (static filenames — no user params) ───────────
    scree_chart   = "pca_scree.png"
    proj_chart    = "pca_2d.png"
    loadings_chart = "pca_loadings.png"

    _make_scree_plot(exp_var, cum_var, elbow_k, scree_chart)
    _make_2d_plot(X_pca, var1, var2, proj_chart)
    _make_loadings_heatmap(model.components_, feature_names,
                           min(elbow_k, n_features), loadings_chart)

    # PC1/PC2 points for any canvas use
    points = [
        {"x": round(float(X_pca[i, 0]), 4), "y": round(float(X_pca[i, 1]), 4)}
        for i in range(len(X_pca))
    ]

    return {
        # ── elbow result ─────────────────────────────────────
        "elbow_k":          elbow_k,
        "elbow_var_pct":    elbow_var_pct,
        "elbow_reason":     "Scree Plot Elbow (largest acceleration in explained variance)",

        # ── kept for template compatibility ──────────────────
        "n_components":     n_features,        # total computed
        "selected_k":       elbow_k,           # retained

        # ── variance stats ───────────────────────────────────
        "total_explained_variance": round(float(cum_var[-1] * 100), 1),
        "pc1_var":          var1,
        "pc2_var":          var2,

        # ── dataset info ─────────────────────────────────────
        "input_features":   n_features,
        "features":         feature_names,
        "sample_size":      len(sample),
        "dataset_records":  original_rows,

        # ── tables ───────────────────────────────────────────
        "components_info":  components_info,
        "loadings_table":   loadings_table,
        "points":           points,

        # ── charts ───────────────────────────────────────────
        "scree_chart":      "charts/" + scree_chart,
        "proj_chart":       "charts/" + proj_chart,
        "loadings_chart":   "charts/" + loadings_chart,
    }
