"""
Hierarchical clustering for the Spotify Playlist Analytics project.

The clustering page reads the preprocessed dataset:
processed_data/preprocessed_dataset.csv

Automatic K calculation is performed using Silhouette Score optimization across
candidate cluster counts (K = 2 to 8), directly generating the evaluation curve,
linkage dendrogram (with automatic cutoff threshold), and PCA 2D cluster visualization.
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
from sklearn.metrics import silhouette_score


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

    # Exclude encoded target columns so clustering is purely unsupervised.
    feature_cols = [
        c for c in data.columns
        if not c.startswith("explicit_") and c not in ["track_id", "id", "explicit"]
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


def _find_optimal_k(X, method="ward", min_k=MIN_K, max_k=MAX_K):
    """
    Automatically evaluate silhouette scores across K in [min_k, max_k]
    to determine the optimal number of clusters for the selected linkage.
    """
    scores = {}
    for k in range(min_k, max_k + 1):
        try:
            model = AgglomerativeClustering(n_clusters=k, linkage=method)
            labels = model.fit_predict(X)
            if len(np.unique(labels)) > 1:
                score = silhouette_score(X, labels)
                scores[k] = round(float(score), 4)
            else:
                scores[k] = -1.0
        except Exception:
            scores[k] = -1.0

    best_k = max(scores, key=scores.get)
    return best_k, scores


def _make_silhouette_chart(scores, best_k, method, filename):
    """
    Plot silhouette scores vs. K values to visualize why optimal K was chosen.
    """
    path = os.path.join(CHART_DIR, filename)
    k_vals = list(scores.keys())
    score_vals = [scores[k] for k in k_vals]

    fig, ax = plt.subplots(figsize=(8, 4.2), dpi=150)
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#fffbfc")

    # Line plot with soft baby pink / rose styling
    ax.plot(k_vals, score_vals, color="#e86b97", marker="o", markersize=7,
            linewidth=2.2, label="Silhouette Score")

    # Highlight optimal K
    best_score = scores[best_k]
    ax.scatter([best_k], [best_score], color="#d6336c", s=140, zorder=5,
               edgecolors="#ffffff", linewidths=2, label=f"Optimal K = {best_k} ({best_score:.4f})")

    ax.annotate(
        f"Optimal K = {best_k}\n(Score: {best_score:.4f})",
        xy=(best_k, best_score),
        xytext=(best_k + 0.35, best_score + 0.02 if best_score < 0.6 else best_score - 0.05),
        arrowprops=dict(facecolor="#d6336c", shrink=0.08, width=1.5, headwidth=6),
        fontsize=9,
        fontweight="bold",
        color="#862e59",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#ffe3ec", edgecolor="#f783ac", alpha=0.9),
    )

    ax.set_title(f"Optimal K Determination — {method.title()} Linkage", fontsize=12, fontweight="bold", color="#49122c", pad=12)
    ax.set_xlabel("Number of Clusters (K)", fontsize=10, fontweight="semibold", color="#5c1f3d")
    ax.set_ylabel("Silhouette Score", fontsize=10, fontweight="semibold", color="#5c1f3d")
    ax.set_xticks(k_vals)
    ax.grid(True, linestyle="--", alpha=0.30, color="#f783ac")
    ax.legend(loc="best", fontsize=9, framealpha=0.9)

    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _make_dendrogram(sample, method, optimal_k, filename):
    X = sample.to_numpy(dtype=float)
    Z = linkage(X, method=method, metric="euclidean")

    path = os.path.join(CHART_DIR, filename)
    fig, ax = plt.subplots(figsize=(11, 5.8), dpi=150)
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#fffbfc")

    # Determine cutoff threshold corresponding to optimal_k
    if optimal_k > 1 and len(Z) >= optimal_k:
        cutoff = float(Z[-(optimal_k - 1), 2])
        if len(Z) >= optimal_k + 1:
            lower = float(Z[-optimal_k, 2])
            cutoff = (cutoff + lower) / 2.0
    else:
        cutoff = None

    dendrogram(
        Z,
        truncate_mode="lastp",
        p=35,
        color_threshold=cutoff if cutoff else 0,
        show_leaf_counts=True,
        leaf_rotation=45,
        leaf_font_size=8,
        ax=ax,
    )

    if cutoff is not None:
        ax.axhline(
            y=cutoff,
            color="#d6336c",
            linestyle="--",
            linewidth=2,
            label=f"Auto Cutoff Distance ({cutoff:.2f}) → K={optimal_k}",
        )
        ax.legend(loc="upper right", fontsize=9, framealpha=0.9)

    ax.set_title(
        f"Hierarchical Clustering Dendrogram — {method.title()} Linkage (Auto K = {optimal_k})",
        fontsize=12,
        fontweight="bold",
        color="#49122c",
        pad=12,
    )
    ax.set_xlabel("Merged track groups / Sample counts", fontsize=10, fontweight="semibold", color="#5c1f3d")
    ax.set_ylabel("Euclidean Distance", fontsize=10, fontweight="semibold", color="#5c1f3d")
    ax.grid(axis="y", alpha=0.30, linestyle="--", color="#f783ac")

    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _make_cluster_plot(sample, labels, method, k, filename):
    pca = PCA(n_components=2, random_state=42)
    reduced = pca.fit_transform(sample.to_numpy(dtype=float))

    path = os.path.join(CHART_DIR, filename)
    fig, ax = plt.subplots(figsize=(9, 6), dpi=150)
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#fffbfc")

    palette = [
        "#e86b97", "#5c7cfa", "#20c997", "#f59f00",
        "#845ef7", "#339af0", "#ff6b6b", "#94d82d"
    ]

    for i, cluster_id in enumerate(sorted(np.unique(labels))):
        mask = labels == cluster_id
        color = palette[i % len(palette)]
        ax.scatter(
            reduced[mask, 0],
            reduced[mask, 1],
            s=22,
            alpha=0.75,
            color=color,
            edgecolors="none",
            label=f"Cluster {cluster_id + 1} ({int(np.sum(mask))} tracks)",
        )

        center_x = np.mean(reduced[mask, 0])
        center_y = np.mean(reduced[mask, 1])
        ax.scatter(
            [center_x],
            [center_y],
            s=120,
            marker="X",
            color=color,
            edgecolors="#343a40",
            linewidths=1.5,
            zorder=6,
        )

    var1 = pca.explained_variance_ratio_[0] * 100
    var2 = pca.explained_variance_ratio_[1] * 100

    ax.set_title(
        f"Hierarchical Clusters (PCA 2D) — {method.title()} Linkage (Auto K = {k})",
        fontsize=12,
        fontweight="bold",
        color="#49122c",
        pad=12,
    )
    ax.set_xlabel(f"Principal Component 1 ({var1:.1f}% variance)", fontsize=10, fontweight="semibold", color="#5c1f3d")
    ax.set_ylabel(f"Principal Component 2 ({var2:.1f}% variance)", fontsize=10, fontweight="semibold", color="#5c1f3d")
    ax.legend(title="Clusters & Centroids (X)", fontsize=8, title_fontsize=9, loc="best", framealpha=0.9)
    ax.grid(True, linestyle="--", alpha=0.30, color="#f783ac")

    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def run_hierarchical_clustering(method="ward", k=None):
    """
    Run hierarchical clustering. If k is None or 'auto', automatically evaluates
    and selects the optimal K via Silhouette Score optimization.
    """
    method = str(method).lower().strip()
    if method not in VALID_LINKAGES:
        method = "ward"

    _, numeric = _load_preprocessed()
    sample = _sample_data(numeric)
    X = sample.to_numpy(dtype=float)

    # Automatically calculate optimal K via Silhouette Analysis
    optimal_k, silhouette_scores = _find_optimal_k(X, method=method, min_k=MIN_K, max_k=MAX_K)

    # Use auto optimal K unless explicitly overridden with a valid number
    if k is not None and str(k).lower() != "auto":
        try:
            chosen_k = int(k)
            chosen_k = max(MIN_K, min(chosen_k, MAX_K))
        except (TypeError, ValueError):
            chosen_k = optimal_k
    else:
        chosen_k = optimal_k

    dendrogram_name = f"hierarchical_dendrogram_{method}.png"
    cluster_name = f"hierarchical_clusters_{method}_{chosen_k}.png"
    silhouette_chart_name = f"hierarchical_silhouette_{method}.png"

    # Directly generate all 3 visualization graphs
    _make_silhouette_chart(silhouette_scores, optimal_k, method, silhouette_chart_name)
    _make_dendrogram(sample, method, chosen_k, dendrogram_name)

    model = AgglomerativeClustering(
        n_clusters=chosen_k,
        linkage=method,
    )
    labels = model.fit_predict(X)

    _make_cluster_plot(sample, labels, method, chosen_k, cluster_name)

    counts = pd.Series(labels).value_counts().sort_index()

    return {
        "method": method,
        "selected_k": chosen_k,
        "optimal_k": optimal_k,
        "is_auto_k": (chosen_k == optimal_k),
        "optimal_score": silhouette_scores.get(optimal_k, 0),
        "silhouette_scores": [
            {"k": k_val, "score": s_val, "is_best": (k_val == optimal_k)}
            for k_val, s_val in silhouette_scores.items()
        ],
        "sample_records": int(len(sample)),
        "dataset_records": int(len(numeric)),
        "feature_count": int(numeric.shape[1]),
        "features": list(numeric.columns),
        "dendrogram": "charts/" + dendrogram_name,
        "cluster_chart": "charts/" + cluster_name,
        "silhouette_chart": "charts/" + silhouette_chart_name,
        "cluster_counts": {
            f"Cluster {int(i) + 1}": int(v)
            for i, v in counts.items()
        },
    }
