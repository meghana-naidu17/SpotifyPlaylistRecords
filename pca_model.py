"""
PCA (Principal Component Analysis) module for the Spotify Playlist Analytics project.

Performs orthogonal linear dimensionality reduction, generating Scree plots,
cumulative explained variance curves, 2D component projections, and feature loadings heatmaps.
"""

import os
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PREPROCESSED_PATH = os.path.join(
    BASE_DIR, "processed_data", "preprocessed_dataset.csv"
)
CHART_DIR = os.path.join(BASE_DIR, "static", "charts")
os.makedirs(CHART_DIR, exist_ok=True)

MAX_SAMPLE = 1500


def _load_preprocessed():
    if not os.path.exists(PREPROCESSED_PATH):
        raise FileNotFoundError(
            "Preprocessed dataset not found: processed_data/preprocessed_dataset.csv"
        )

    df = pd.read_csv(PREPROCESSED_PATH)

    dummy_prefixes = (
        "track_genre_", "explicit_", "key_", "mode_", "time_signature_"
    )
    feature_cols = [
        c for c in df.columns
        if not c.startswith(dummy_prefixes)
        and c not in ["track_id", "id", "explicit"]
    ]

    numeric = df[feature_cols].select_dtypes(include=np.number).copy()
    numeric = numeric.replace([np.inf, -np.inf], np.nan)
    numeric = numeric.fillna(numeric.median(numeric_only=True))
    numeric = numeric.fillna(0)

    varying_cols = numeric.columns[numeric.nunique(dropna=False) > 1]
    numeric = numeric[varying_cols]

    if numeric.empty:
        raise ValueError("No varying numeric features available for PCA.")

    return numeric


def _make_scree_plot(exp_var, cum_var, filename):
    path = os.path.join(CHART_DIR, filename)
    fig, ax1 = plt.subplots(figsize=(9, 4.8), dpi=150)
    fig.patch.set_facecolor("#ffffff")
    ax1.set_facecolor("#fffbfc")

    x_indices = list(range(1, len(exp_var) + 1))
    pc_labels = [f"PC{i}" for i in x_indices]

    # Bar chart for individual variance
    bars = ax1.bar(
        x_indices,
        [v * 100 for v in exp_var],
        color="#f8bbd0",
        edgecolor="#e86b97",
        width=0.55,
        label="Individual Variance (%)"
    )
    ax1.set_xlabel("Principal Component", fontsize=10, fontweight="semibold", color="#5c1f3d")
    ax1.set_ylabel("Individual Explained Variance (%)", fontsize=10, fontweight="semibold", color="#5c1f3d")
    ax1.set_xticks(x_indices)
    ax1.set_xticklabels(pc_labels)

    for bar in bars:
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2, h + 0.8,
                 f"{h:.1f}%", ha="center", va="bottom", fontsize=8, color="#49122c", fontweight="bold")

    # Line plot for cumulative variance
    ax2 = ax1.twinx()
    ax2.plot(
        x_indices,
        [v * 100 for v in cum_var],
        color="#d6336c",
        marker="o",
        linewidth=2.2,
        markersize=6,
        label="Cumulative Variance (%)"
    )
    ax2.set_ylabel("Cumulative Explained Variance (%)", fontsize=10, fontweight="semibold", color="#d6336c")
    ax2.set_ylim(0, 105)

    # Threshold dashed lines at 80% and 90%
    ax2.axhline(80, color="#845ef7", linestyle=":", alpha=0.7, label="80% Threshold")
    ax2.axhline(90, color="#20c997", linestyle=":", alpha=0.7, label="90% Threshold")

    ax1.set_title("PCA Scree Plot & Cumulative Explained Variance", fontsize=12, fontweight="bold", color="#49122c", pad=12)
    ax1.grid(True, axis="y", linestyle="--", alpha=0.30, color="#f783ac")

    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right", fontsize=8, framealpha=0.92)

    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _make_2d_plot(X_pca, var1, var2, filename):
    path = os.path.join(CHART_DIR, filename)
    fig, ax = plt.subplots(figsize=(9, 6), dpi=150)
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#fffbfc")

    scatter = ax.scatter(
        X_pca[:, 0],
        X_pca[:, 1],
        c=X_pca[:, 0],
        cmap="coolwarm",
        s=24,
        alpha=0.75,
        edgecolors="none"
    )

    cbar = fig.colorbar(scatter, ax=ax, pad=0.02)
    cbar.set_label("PC1 Projection Value", fontsize=9, fontweight="bold", color="#5c1f3d")
    cbar.ax.tick_params(labelsize=8)

    ax.axhline(0, color="#ad1457", linestyle="--", linewidth=1, alpha=0.4)
    ax.axvline(0, color="#ad1457", linestyle="--", linewidth=1, alpha=0.4)

    ax.set_title(
        f"2D Principal Component Projection (PC1 vs PC2 — {var1 + var2:.1f}% Total Variance)",
        fontsize=12,
        fontweight="bold",
        color="#49122c",
        pad=12
    )
    ax.set_xlabel(f"Principal Component 1 ({var1:.1f}% variance)", fontsize=10, fontweight="semibold", color="#5c1f3d")
    ax.set_ylabel(f"Principal Component 2 ({var2:.1f}% variance)", fontsize=10, fontweight="semibold", color="#5c1f3d")
    ax.grid(True, linestyle="--", alpha=0.30, color="#f783ac")

    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _make_loadings_heatmap(loadings, feature_names, n_display_pcs, filename):
    path = os.path.join(CHART_DIR, filename)
    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=150)
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#fffbfc")

    matrix = loadings[:n_display_pcs, :]
    pc_labels = [f"PC{i+1}" for i in range(n_display_pcs)]
    clean_feature_names = [f.replace("_", " ").title() for f in feature_names]

    im = ax.imshow(matrix, cmap="PuOr_r", aspect="auto", vmin=-0.6, vmax=0.6)
    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label("Feature Loading Weight", fontsize=9, fontweight="bold", color="#5c1f3d")
    cbar.ax.tick_params(labelsize=8)

    ax.set_xticks(range(len(clean_feature_names)))
    ax.set_xticklabels(clean_feature_names, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(n_display_pcs))
    ax.set_yticklabels(pc_labels, fontsize=10, fontweight="bold")

    # Add numeric annotations inside each cell
    for i in range(n_display_pcs):
        for j in range(len(clean_feature_names)):
            val = matrix[i, j]
            text_col = "#ffffff" if abs(val) > 0.35 else "#3d1421"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    color=text_col, fontsize=8, fontweight="bold")

    ax.set_title("PCA Feature Loadings Matrix (Weights on Components)", fontsize=12, fontweight="bold", color="#49122c", pad=12)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def run_pca(
    n_components=5,
    whiten=False,
    svd_solver="auto"
):
    try:
        n_components = int(n_components)
    except (ValueError, TypeError):
        n_components = 5
    n_components = max(2, min(n_components, 11))

    if isinstance(whiten, str):
        whiten = whiten.lower() in {"true", "1", "yes"}

    svd_solver = str(svd_solver).lower().strip()
    if svd_solver not in {"auto", "full", "randomized"}:
        svd_solver = "auto"

    numeric = _load_preprocessed()
    feature_names = list(numeric.columns)
    original_rows = len(numeric)

    sample = numeric.sample(n=min(MAX_SAMPLE, len(numeric)), random_state=42)
    scaler = StandardScaler()
    X = scaler.fit_transform(sample.to_numpy(dtype=float))

    model = PCA(
        n_components=n_components,
        whiten=whiten,
        svd_solver=svd_solver,
        random_state=42
    )
    X_pca = model.fit_transform(X)

    exp_var = model.explained_variance_ratio_
    cum_var = np.cumsum(exp_var)
    var1 = round(float(exp_var[0] * 100), 1)
    var2 = round(float(exp_var[1] * 100), 1)
    total_var = round(float(cum_var[-1] * 100), 1)

    # Component summary list
    components_info = []
    for i in range(len(exp_var)):
        components_info.append({
            "name": f"PC{i+1}",
            "variance_ratio": round(float(exp_var[i] * 100), 2),
            "cumulative_ratio": round(float(cum_var[i] * 100), 2),
            "singular_value": round(float(model.singular_values_[i]), 2)
        })

    # Loadings breakdown table
    loadings_table = []
    for j, f_name in enumerate(feature_names):
        row = {"feature": f_name.replace("_", " ").title()}
        for i in range(min(4, n_components)):
            row[f"pc{i+1}"] = round(float(model.components_[i, j]), 3)
        loadings_table.append(row)

    # Chart filenames
    scree_chart = f"pca_scree_{n_components}_{svd_solver}.png"
    proj_chart = f"pca_2d_{n_components}_{svd_solver}.png"
    loadings_chart = f"pca_loadings_{n_components}_{svd_solver}.png"

    _make_scree_plot(exp_var, cum_var, scree_chart)
    _make_2d_plot(X_pca, var1, var2, proj_chart)
    _make_loadings_heatmap(model.components_, feature_names, min(3, n_components), loadings_chart)

    # Points for browser canvas
    points = []
    for i in range(len(X_pca)):
        points.append({
            "x": round(float(X_pca[i, 0]), 4),
            "y": round(float(X_pca[i, 1]), 4)
        })

    return {
        "n_components": n_components,
        "whiten": whiten,
        "svd_solver": svd_solver,
        "total_explained_variance": total_var,
        "pc1_var": var1,
        "pc2_var": var2,
        "input_features": len(feature_names),
        "features": feature_names,
        "sample_size": len(sample),
        "dataset_records": original_rows,
        "components_info": components_info,
        "loadings_table": loadings_table,
        "points": points,
        "scree_chart": "charts/" + scree_chart,
        "proj_chart": "charts/" + proj_chart,
        "loadings_chart": "charts/" + loadings_chart,
    }
