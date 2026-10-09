"""
t-SNE and UMAP Manifold Learning module for the Spotify Playlist Analytics project.

Performs non-linear manifold embeddings for audio feature visualization, comparing
t-Distributed Stochastic Neighbor Embedding and Uniform Manifold Approximation and Projection.
"""

import os
import time
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
import umap


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PREPROCESSED_PATH = os.path.join(
    BASE_DIR, "processed_data", "preprocessed_dataset.csv"
)
CHART_DIR = os.path.join(BASE_DIR, "static", "charts")
os.makedirs(CHART_DIR, exist_ok=True)

MAX_SAMPLE = 800

# Exactly 10 continuous audio features, already StandardScaled.
AUDIO_FEATURES = [
    "duration_ms", "danceability", "energy", "loudness",
    "speechiness", "acousticness", "instrumentalness",
    "liveness", "valence", "tempo",
]


def _load_preprocessed():
    if not os.path.exists(PREPROCESSED_PATH):
        raise FileNotFoundError(
            "Preprocessed dataset not found: processed_data/preprocessed_dataset.csv"
        )

    df = pd.read_csv(PREPROCESSED_PATH)

    # Use exactly the 10 continuous scaled audio features.
    # Already StandardScaled — no re-scaling needed.
    available = [c for c in AUDIO_FEATURES if c in df.columns]
    if not available:
        raise ValueError(
            "None of the expected audio feature columns found in "
            "preprocessed_dataset.csv"
        )

    numeric = df[available].copy()
    numeric = numeric.replace([np.inf, -np.inf], np.nan)
    numeric = numeric.fillna(numeric.median(numeric_only=True)).fillna(0)

    if numeric.empty:
        raise ValueError("No varying numeric audio features available for Manifold Learning.")

    return numeric


def _make_comparison_plot(Z_tsne, Z_umap, c_vals, c_name, perp, neighbors, metric, filename):
    path = os.path.join(CHART_DIR, filename)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.8), dpi=150)
    fig.patch.set_facecolor("#ffffff")
    ax1.set_facecolor("#fffbfc")
    ax2.set_facecolor("#fffbfc")

    # t-SNE plot
    sc1 = ax1.scatter(
        Z_tsne[:, 0],
        Z_tsne[:, 1],
        c=c_vals,
        cmap="Spectral_r",
        s=22,
        alpha=0.75,
        edgecolors="none"
    )
    ax1.set_title(f"t-SNE Embedding (Perplexity = {perp})", fontsize=11, fontweight="bold", color="#49122c", pad=10)
    ax1.set_xlabel("t-SNE Dimension 1", fontsize=9, fontweight="semibold", color="#5c1f3d")
    ax1.set_ylabel("t-SNE Dimension 2", fontsize=9, fontweight="semibold", color="#5c1f3d")
    ax1.grid(True, linestyle="--", alpha=0.30, color="#f783ac")

    # UMAP plot
    sc2 = ax2.scatter(
        Z_umap[:, 0],
        Z_umap[:, 1],
        c=c_vals,
        cmap="Spectral_r",
        s=22,
        alpha=0.75,
        edgecolors="none"
    )
    ax2.set_title(f"UMAP Embedding (Neighbors = {neighbors}, Metric = {metric.title()})", fontsize=11, fontweight="bold", color="#49122c", pad=10)
    ax2.set_xlabel("UMAP Dimension 1", fontsize=9, fontweight="semibold", color="#5c1f3d")
    ax2.set_ylabel("UMAP Dimension 2", fontsize=9, fontweight="semibold", color="#5c1f3d")
    ax2.grid(True, linestyle="--", alpha=0.30, color="#f783ac")

    fig.subplots_adjust(right=0.88, top=0.90, bottom=0.12, wspace=0.25)
    cbar_ax = fig.add_axes([0.90, 0.18, 0.02, 0.68])
    cbar = fig.colorbar(sc2, cax=cbar_ax)
    cbar.set_label(f"Track {c_name.title()} Gradient", fontsize=9, fontweight="bold", color="#5c1f3d")
    cbar.ax.tick_params(labelsize=8)

    fig.suptitle("Manifold Projection Comparison: t-SNE vs. UMAP", fontsize=13, fontweight="bold", color="#49122c")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _make_single_plot(Z, c_vals, c_name, title, xlabel, ylabel, filename):
    path = os.path.join(CHART_DIR, filename)
    fig, ax = plt.subplots(figsize=(9, 6), dpi=150)
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#fffbfc")

    sc = ax.scatter(
        Z[:, 0],
        Z[:, 1],
        c=c_vals,
        cmap="Spectral_r",
        s=26,
        alpha=0.78,
        edgecolors="none"
    )

    cbar = fig.colorbar(sc, ax=ax, pad=0.02)
    cbar.set_label(f"Track {c_name.title()} Gradient", fontsize=9, fontweight="bold", color="#5c1f3d")
    cbar.ax.tick_params(labelsize=8)

    ax.set_title(title, fontsize=12, fontweight="bold", color="#49122c", pad=12)
    ax.set_xlabel(xlabel, fontsize=10, fontweight="semibold", color="#5c1f3d")
    ax.set_ylabel(ylabel, fontsize=10, fontweight="semibold", color="#5c1f3d")
    ax.grid(True, linestyle="--", alpha=0.30, color="#f783ac")

    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def run_tsne_umap(
    method="both",
    perplexity=30,
    learning_rate=200,
    n_iter=1000,
    n_neighbors=15,
    min_dist=0.1,
    metric="euclidean"
):
    method = str(method).lower().strip()
    if method not in {"both", "tsne", "umap"}:
        method = "both"

    try:
        perplexity = float(perplexity)
    except (ValueError, TypeError):
        perplexity = 30.0
    perplexity = max(5.0, min(perplexity, 50.0))

    try:
        learning_rate = float(learning_rate)
    except (ValueError, TypeError):
        learning_rate = 200.0
    learning_rate = max(10.0, min(learning_rate, 1000.0))

    try:
        n_iter = int(n_iter)
    except (ValueError, TypeError):
        n_iter = 1000
    n_iter = max(250, min(n_iter, 2000))

    try:
        n_neighbors = int(n_neighbors)
    except (ValueError, TypeError):
        n_neighbors = 15
    n_neighbors = max(3, min(n_neighbors, 50))

    try:
        min_dist = float(min_dist)
    except (ValueError, TypeError):
        min_dist = 0.1
    min_dist = max(0.01, min(min_dist, 0.99))

    metric = str(metric).lower().strip()
    if metric not in {"euclidean", "cosine", "manhattan"}:
        metric = "euclidean"

    numeric = _load_preprocessed()
    feature_names = list(numeric.columns)
    original_rows = len(numeric)

    sample = numeric.sample(n=min(MAX_SAMPLE, len(numeric)), random_state=42)
    # Data is already StandardScaled — use directly
    X = sample.to_numpy(dtype=float)

    # Color values by Energy or first available feature
    c_name = "energy" if "energy" in sample.columns else sample.columns[0]
    c_vals = sample[c_name].to_numpy()

    # 1. Run t-SNE
    t0 = time.time()
    tsne_model = TSNE(
        n_components=2,
        perplexity=perplexity,
        learning_rate=learning_rate,
        max_iter=n_iter,
        random_state=42,
        init="pca"
    )
    Z_tsne = tsne_model.fit_transform(X)
    tsne_duration = round(time.time() - t0, 3)

    # 2. Run UMAP
    t0 = time.time()
    umap_model = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=42
    )
    Z_umap = umap_model.fit_transform(X)
    umap_duration = round(time.time() - t0, 3)

    # Chart filenames
    perp_clean = int(perplexity)
    comparison_chart = f"tsne_umap_comparison_{perp_clean}_{n_neighbors}_{metric}.png"
    tsne_chart = f"tsne_manifold_{perp_clean}.png"
    umap_chart = f"umap_manifold_{n_neighbors}_{metric}.png"

    _make_comparison_plot(Z_tsne, Z_umap, c_vals, c_name, perp_clean, n_neighbors, metric, comparison_chart)
    _make_single_plot(
        Z_tsne, c_vals, c_name,
        f"t-SNE 2D Manifold Embedding (Perplexity = {perp_clean})",
        "t-SNE Dimension 1", "t-SNE Dimension 2",
        tsne_chart
    )
    _make_single_plot(
        Z_umap, c_vals, c_name,
        f"UMAP 2D Manifold Embedding (Neighbors = {n_neighbors}, Metric = {metric.title()})",
        "UMAP Dimension 1", "UMAP Dimension 2",
        umap_chart
    )

    # Comparison metrics table
    comparison_table = [
        {
            "criteria": "Mathematical Foundation",
            "tsne": "Kullback-Leibler divergence on Student-t distribution",
            "umap": "Fuzzy simplicial sets & Riemannian manifold topology"
        },
        {
            "criteria": "Structure Preservation",
            "tsne": "Excels at local clusters; distorts global distances",
            "umap": "Preserves both local neighborhoods and global cluster topology"
        },
        {
            "criteria": "Execution Time (Current Run)",
            "tsne": f"{tsne_duration}s ({int(n_iter)} iterations)",
            "umap": f"{umap_duration}s (~{max(1, round(tsne_duration / (umap_duration or 0.001)))}x faster)"
        },
        {
            "criteria": "Key Hyperparameters",
            "tsne": f"Perplexity ({perp_clean}), Learning Rate ({learning_rate})",
            "umap": f"N-Neighbors ({n_neighbors}), Min-Dist ({min_dist})"
        },
        {
            "criteria": "New Sample Embedding",
            "tsne": "Not supported (must re-run entire dataset)",
            "umap": "Supported (learns parametric metric mapper)"
        }
    ]

    # Points for browser canvas
    tsne_points = [
        {"x": round(float(Z_tsne[i, 0]), 4), "y": round(float(Z_tsne[i, 1]), 4)}
        for i in range(len(Z_tsne))
    ]
    umap_points = [
        {"x": round(float(Z_umap[i, 0]), 4), "y": round(float(Z_umap[i, 1]), 4)}
        for i in range(len(Z_umap))
    ]

    return {
        "method": method,
        "perplexity": perp_clean,
        "learning_rate": learning_rate,
        "n_iter": n_iter,
        "n_neighbors": n_neighbors,
        "min_dist": min_dist,
        "metric": metric,
        "tsne_duration": tsne_duration,
        "umap_duration": umap_duration,
        "speedup": round(tsne_duration / (umap_duration or 0.001), 1),
        "sample_size": len(sample),
        "dataset_records": original_rows,
        "features_count": len(feature_names),
        "comparison_table": comparison_table,
        "tsne_points": tsne_points,
        "umap_points": umap_points,
        "comparison_chart": "charts/" + comparison_chart,
        "tsne_chart": "charts/" + tsne_chart,
        "umap_chart": "charts/" + umap_chart,
    }
