"""
DBSCAN clustering for the Spotify Playlist Analytics project.

Uses continuous preprocessed audio features from processed_data/preprocessed_dataset.csv
to discover dense clusters and explicitly identify Core, Border, and Noise points.
"""

import os
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PREPROCESSED_PATH = os.path.join(
    BASE_DIR,
    "processed_data",
    "preprocessed_dataset.csv"
)
CHART_DIR = os.path.join(BASE_DIR, "static", "charts")
os.makedirs(CHART_DIR, exist_ok=True)


def _load_preprocessed():
    if not os.path.exists(PREPROCESSED_PATH):
        raise FileNotFoundError(
            "Preprocessed dataset not found: "
            "processed_data/preprocessed_dataset.csv"
        )

    df = pd.read_csv(PREPROCESSED_PATH)

    # Core continuous Spotify audio features provide meaningful density dimensions.
    # Exclude one-hot encoded dummy columns (genres, keys, modes, etc.)
    # so DBSCAN does not suffer from high-dimensional sparse binary inflation.
    dummy_prefixes = (
        "track_genre_", "explicit_", "key_", "mode_", "time_signature_"
    )
    feature_cols = [
        c for c in df.columns
        if not c.startswith(dummy_prefixes)
        and c not in ["track_id", "id", "explicit"]
    ]

    numeric = df[feature_cols].select_dtypes(include=np.number).copy()

    if numeric.empty:
        raise ValueError("No numeric features found for DBSCAN.")

    # Clean missing / infinity
    numeric = numeric.replace([np.inf, -np.inf], np.nan)
    numeric = numeric.fillna(numeric.median(numeric_only=True))
    numeric = numeric.fillna(0)

    # Remove zero-variance columns
    varying_cols = numeric.columns[numeric.nunique(dropna=False) > 1]
    numeric = numeric[varying_cols]

    if numeric.empty:
        raise ValueError("No varying numeric features available for DBSCAN.")

    return numeric


def _make_static_plot(reduced, labels, core_mask, eps, min_samples, pca, filename):
    """
    Generate high-resolution Matplotlib scatter chart explicitly highlighting
    Core, Border, and Noise points.
    """
    path = os.path.join(CHART_DIR, filename)
    fig, ax = plt.subplots(figsize=(9.5, 6.2), dpi=150)
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#fffbfc")

    palette = [
        "#e86b97", "#5c7cfa", "#20c997", "#f59f00",
        "#845ef7", "#339af0", "#ff6b6b", "#94d82d"
    ]

    # 1. Noise points (label == -1)
    noise_mask = (labels == -1)
    if np.any(noise_mask):
        ax.scatter(
            reduced[noise_mask, 0],
            reduced[noise_mask, 1],
            c="#868e96",
            marker="x",
            s=28,
            alpha=0.60,
            label=f"Noise Points ({int(np.sum(noise_mask))})",
            zorder=2,
        )

    # 2. Clusters (Core & Border)
    unique_clusters = sorted([c for c in np.unique(labels) if c != -1])
    for cluster_id in unique_clusters:
        cluster_mask = (labels == cluster_id)
        cluster_core = cluster_mask & core_mask
        cluster_border = cluster_mask & (~core_mask)
        col = palette[cluster_id % len(palette)]

        # Core points: solid vibrant circle
        if np.any(cluster_core):
            ax.scatter(
                reduced[cluster_core, 0],
                reduced[cluster_core, 1],
                c=col,
                marker="o",
                s=36,
                edgecolors="#ffffff",
                linewidths=0.8,
                alpha=0.85,
                label=f"Cluster {cluster_id + 1} Core ({int(np.sum(cluster_core))})",
                zorder=4,
            )

        # Border points: ring/hollow circle
        if np.any(cluster_border):
            ax.scatter(
                reduced[cluster_border, 0],
                reduced[cluster_border, 1],
                facecolors="none",
                edgecolors=col,
                marker="o",
                s=40,
                linewidths=1.8,
                alpha=0.90,
                label=f"Cluster {cluster_id + 1} Border ({int(np.sum(cluster_border))})",
                zorder=3,
            )

    var1 = pca.explained_variance_ratio_[0] * 100
    var2 = pca.explained_variance_ratio_[1] * 100

    ax.set_title(
        f"DBSCAN Density Clusters — EPS={eps}, Min Samples={min_samples}",
        fontsize=12,
        fontweight="bold",
        color="#49122c",
        pad=12,
    )
    ax.set_xlabel(f"PC1 ({var1:.1f}% variance)", fontsize=10, fontweight="semibold", color="#5c1f3d")
    ax.set_ylabel(f"PC2 ({var2:.1f}% variance)", fontsize=10, fontweight="semibold", color="#5c1f3d")
    ax.grid(True, linestyle="--", alpha=0.30, color="#f783ac")
    ax.legend(loc="best", fontsize=8, framealpha=0.92, ncol=2)

    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def run_dbscan(
    eps=1.5,
    min_samples=5,
    max_rows=1200
):
    try:
        eps = float(eps)
    except (ValueError, TypeError):
        eps = 1.5

    try:
        min_samples = int(min_samples)
    except (ValueError, TypeError):
        min_samples = 5

    eps = max(0.05, eps)
    min_samples = max(2, min_samples)

    numeric = _load_preprocessed()
    original_rows = len(numeric)

    # Use reproducible sample for fast browser interaction
    if len(numeric) > max_rows:
        sample = numeric.sample(n=max_rows, random_state=42)
    else:
        sample = numeric.copy()

    # Standardize continuous features
    scaler = StandardScaler()
    X = scaler.fit_transform(sample.to_numpy(dtype=float))

    # Fit DBSCAN
    model = DBSCAN(
        eps=eps,
        min_samples=min_samples,
        n_jobs=-1
    )
    labels = model.fit_predict(X)

    # Core sample mask
    core_mask = np.zeros_like(labels, dtype=bool)
    core_mask[model.core_sample_indices_] = True

    # Point counts
    core_count = int(np.sum(core_mask))
    border_count = int(np.sum((labels != -1) & (~core_mask)))
    noise_count = int(np.sum(labels == -1))
    sample_total = int(len(sample))

    # PCA 2D for visualization
    pca = PCA(n_components=2, random_state=42)
    reduced = pca.fit_transform(X)

    # Distinct clusters (excluding noise)
    unique_labels = set(labels)
    cluster_count = len(unique_labels - {-1})

    # Cluster distribution table
    counts = pd.Series(labels).value_counts().sort_index()
    rows = []
    for cluster_id, count in counts.items():
        cluster_id = int(cluster_id)
        if cluster_id == -1:
            name = "Noise (-1)"
            c_type = "noise"
        else:
            name = f"Cluster {cluster_id + 1}"
            c_type = "cluster"

        rows.append({
            "cluster": name,
            "type": c_type,
            "records": int(count),
            "percentage": round(int(count) / sample_total * 100, 1)
        })

    # Points list for high-resolution interactive canvas
    points = []
    for point, label, is_core in zip(reduced, labels, core_mask):
        lbl = int(label)
        if lbl == -1:
            pt_type = "noise"
        elif is_core:
            pt_type = "core"
        else:
            pt_type = "border"

        points.append({
            "x": round(float(point[0]), 4),
            "y": round(float(point[1]), 4),
            "label": lbl,
            "type": pt_type
        })

    eps_str = str(eps).replace(".", "_")
    chart_filename = f"dbscan_clusters_{eps_str}_{min_samples}.png"
    _make_static_plot(reduced, labels, core_mask, eps, min_samples, pca, chart_filename)

    var1 = round(float(pca.explained_variance_ratio_[0] * 100), 1)
    var2 = round(float(pca.explained_variance_ratio_[1] * 100), 1)

    return {
        "eps": float(eps),
        "min_samples": int(min_samples),
        "clusters": int(cluster_count),
        "core_count": int(core_count),
        "border_count": int(border_count),
        "noise": int(noise_count),
        "core_percentage": round(core_count / sample_total * 100, 1),
        "border_percentage": round(border_count / sample_total * 100, 1),
        "noise_percentage": round(noise_count / sample_total * 100, 1),
        "sample_size": sample_total,
        "dataset_records": int(original_rows),
        "feature_count": int(numeric.shape[1]),
        "features": list(numeric.columns),
        "rows": rows,
        "points": points,
        "chart": "charts/" + chart_filename,
        "pc1_var": var1,
        "pc2_var": var2,
    }