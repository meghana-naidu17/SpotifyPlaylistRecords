"""
Hierarchical clustering for the Spotify Playlist Analytics project.

The clustering page intentionally reads the already preprocessed dataset:
processed_data/preprocessed_dataset.csv

Because agglomerative hierarchical clustering needs pairwise distances,
running it on all 114,000 records would require excessive memory/time.
A reproducible sample is therefore used for the clustering calculation.
"""

import os
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PREPROCESSED_PATH = os.path.join(
    BASE_DIR, "processed_data", "preprocessed_dataset.csv"
)
CHART_DIR = os.path.join(BASE_DIR, "static", "charts")
os.makedirs(CHART_DIR, exist_ok=True)

MAX_SAMPLE = 1200
MIN_K = 2
MAX_K = 8
VALID_LINKAGES = {"ward", "complete", "average", "single"}


def _load_preprocessed():
    if not os.path.exists(PREPROCESSED_PATH):
        raise FileNotFoundError(
            "Preprocessed dataset not found: processed_data/preprocessed_dataset.csv"
        )

    data = pd.read_csv(PREPROCESSED_PATH)

    # The target was already encoded during preprocessing.
    # It must not be used to form unsupervised clusters.
    feature_cols = [
        c for c in data.columns
        if not c.startswith("explicit_")
    ]

    numeric = data[feature_cols].apply(pd.to_numeric, errors="coerce")

    # Replace invalid values with column medians.
    numeric = numeric.replace([np.inf, -np.inf], np.nan)
    numeric = numeric.fillna(numeric.median(numeric_only=True))
    numeric = numeric.fillna(0)

    # Remove columns with no variation.
    varying_cols = numeric.columns[numeric.nunique(dropna=False) > 1]
    numeric = numeric[varying_cols]

    if numeric.empty:
        raise ValueError("No usable numeric features were found in the preprocessed dataset.")

    return data, numeric


def _sample_data(numeric):
    sample_n = min(MAX_SAMPLE, len(numeric))
    if sample_n < MIN_K:
        raise ValueError("Not enough preprocessed rows for hierarchical clustering.")

    return numeric.sample(n=sample_n, random_state=42)


def _make_dendrogram(sample, method, filename):
    Z = linkage(sample.to_numpy(dtype=float), method=method, metric="euclidean")

    path = os.path.join(CHART_DIR, filename)
    fig, ax = plt.subplots(figsize=(11, 5.8))

    # Showing the last 35 merged groups keeps the 1200-row dendrogram readable.
    dendrogram(
        Z,
        truncate_mode="lastp",
        p=35,
        show_leaf_counts=True,
        leaf_rotation=45,
        leaf_font_size=8,
        ax=ax,
    )
    ax.set_title(
        "Hierarchical Clustering Dendrogram — " + method.title() + " Linkage"
    )
    ax.set_xlabel("Merged groups / sample counts")
    ax.set_ylabel("Euclidean distance")
    ax.grid(axis="y", alpha=0.20)

    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _make_cluster_plot(sample, labels, method, k, filename):
    # PCA is only used to visualize the clusters in 2 dimensions.
    pca = PCA(n_components=2, random_state=42)
    reduced = pca.fit_transform(sample.to_numpy(dtype=float))

    path = os.path.join(CHART_DIR, filename)
    fig, ax = plt.subplots(figsize=(9, 6))

    for cluster_id in sorted(np.unique(labels)):
        mask = labels == cluster_id
        ax.scatter(
            reduced[mask, 0],
            reduced[mask, 1],
            s=14,
            alpha=0.65,
            label=f"Cluster {cluster_id + 1}",
        )

    ax.set_title(
        f"Hierarchical Clusters — {method.title()} Linkage, K={k}"
    )
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}% variance)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}% variance)")
    ax.legend(title="Clusters", fontsize=8)
    ax.grid(alpha=0.20)

    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def run_hierarchical_clustering(method="ward", k=4):
    method = str(method).lower().strip()
    if method not in VALID_LINKAGES:
        method = "ward"

    try:
        k = int(k)
    except (TypeError, ValueError):
        k = 4

    k = max(MIN_K, min(k, MAX_K))

    _, numeric = _load_preprocessed()
    sample = _sample_data(numeric)

    dendrogram_name = f"hierarchical_dendrogram_{method}.png"
    cluster_name = f"hierarchical_clusters_{method}_{k}.png"

    _make_dendrogram(sample, method, dendrogram_name)

    model = AgglomerativeClustering(
        n_clusters=k,
        linkage=method,
    )
    labels = model.fit_predict(sample.to_numpy(dtype=float))

    _make_cluster_plot(sample, labels, method, k, cluster_name)

    counts = pd.Series(labels).value_counts().sort_index()

    return {
        "method": method,
        "selected_k": k,
        "sample_records": int(len(sample)),
        "dataset_records": int(len(numeric)),
        "feature_count": int(numeric.shape[1]),
        "features": list(numeric.columns),
        "dendrogram": "charts/" + dendrogram_name,
        "cluster_chart": "charts/" + cluster_name,
        "cluster_counts": {
            f"Cluster {int(i) + 1}": int(v)
            for i, v in counts.items()
        },
    }
