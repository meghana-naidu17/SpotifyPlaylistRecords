"""
Hierarchical Clustering — Spotify Playlist Analytics
=====================================================

Linkage-driven K selection:
  Each linkage method (Ward, Complete, Average, Single) builds its own
  linkage matrix from the data.  The optimal K is derived from THAT
  linkage's dendrogram by finding the largest acceleration in the merge
  distances (the "biggest jump" in consecutive merges).

  The silhouette score is computed AFTER the clusters are formed and is
  used only as an evaluation metric — it does NOT select K.

Dataset: processed_data/preprocessed_dataset.csv
Features: 10 continuous audio features (already StandardScaled).
"""

import os
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score


BASE_DIR          = os.path.dirname(os.path.abspath(__file__))
PREPROCESSED_PATH = os.path.join(BASE_DIR, "processed_data", "preprocessed_dataset.csv")
CHART_DIR         = os.path.join(BASE_DIR, "static", "charts")
os.makedirs(CHART_DIR, exist_ok=True)

MAX_SAMPLE    = 1200
MIN_K         = 2
MAX_K         = 8
VALID_LINKAGES = {"ward", "complete", "average", "single"}

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
    data    = pd.read_csv(PREPROCESSED_PATH)
    available = [c for c in AUDIO_FEATURES if c in data.columns]
    if not available:
        raise ValueError(
            "None of the expected audio feature columns found in preprocessed_dataset.csv"
        )
    numeric = data[available].copy()
    numeric = numeric.replace([np.inf, -np.inf], np.nan)
    numeric = numeric.fillna(numeric.median(numeric_only=True)).fillna(0)
    if numeric.empty:
        raise ValueError("No usable numeric features found in the preprocessed dataset.")
    return data, numeric


def _sample_data(numeric):
    n = min(MAX_SAMPLE, len(numeric))
    if n < MIN_K:
        raise ValueError("Not enough rows for hierarchical clustering.")
    return numeric.sample(n=n, random_state=42)


# ──────────────────────────────────────────────────────────────
# LINKAGE-BASED K DETECTION
# ──────────────────────────────────────────────────────────────

def _find_k_from_dendrogram(Z, min_k=MIN_K, max_k=MAX_K):
    """
    Derive K from the linkage matrix entirely from the hierarchical structure.
    No silhouette is used.

    Strategy
    --------
    For each candidate K from min_k to max_k:
      - Compute the cutoff distance from the linkage matrix.
      - Obtain cluster labels via fcluster with that cutoff.
      - Score the partition using:
          score = (number of non-singleton clusters) / K
                  minus a penalty for the dominance of the largest cluster.

    The K with the best (lowest penalty, most non-singletons) wins.

    Why not reject singletons?
    Chaining linkages (Complete / Average / Single) ALWAYS produce at least
    one singleton on real Spotify audio data — that is a structural property
    of those algorithms, not an error.  Rejecting them would prevent any K
    from being selected.

    Returns
    -------
    k            int     optimal cluster count
    cutoff_dist  float   dendrogram cutoff distance
    """
    n_merges = len(Z)
    n_sample = n_merges + 1   # Z has n-1 rows for n points

    best_k      = min_k
    best_cutoff = float(Z[n_merges - min_k, 2]) if n_merges >= min_k else float(Z[-1, 2])
    best_score  = -1.0

    for k_try in range(min_k, max_k + 1):
        mi = n_merges - k_try
        if mi < 0 or mi >= n_merges:
            continue

        upper  = float(Z[mi, 2])
        lower  = float(Z[mi + 1, 2]) if mi + 1 < n_merges else upper * 0.8
        cutoff = (upper + lower) / 2.0

        labels = fcluster(Z, t=cutoff, criterion="distance") - 1
        unique_labels, counts = np.unique(labels, return_counts=True)
        n_actual = len(unique_labels)

        if n_actual < min_k:
            continue

        # Fraction of points in non-singleton clusters
        non_single_pts  = int(counts[counts > 1].sum())
        non_single_frac = non_single_pts / n_sample

        # Penalty for dominance of the largest cluster (closer to 1 = worse)
        dominance = float(counts.max()) / n_sample

        # Score: maximise coverage of non-singletons, minimise dominance
        score = non_single_frac - dominance

        if score > best_score:
            best_score  = score
            best_k      = n_actual
            best_cutoff = round(cutoff, 4)

    return best_k, best_cutoff


# ──────────────────────────────────────────────────────────────
# SILHOUETTE EVALUATION (evaluation only, NOT K selection)
# ──────────────────────────────────────────────────────────────

def _evaluate_silhouette(X, labels):
    """
    Compute silhouette score for the given label assignment.
    Returns 0.0 if fewer than 2 unique clusters.
    """
    unique = np.unique(labels)
    if len(unique) < 2:
        return 0.0
    try:
        return round(float(silhouette_score(X, labels)), 4)
    except Exception:
        return 0.0


def _silhouette_by_k(X, method, min_k=MIN_K, max_k=MAX_K):
    """
    Compute silhouette scores for K=min_k..max_k using the given linkage.
    Returns a dict {k: score}.
    Used ONLY for the evaluation chart — does NOT select K.
    """
    scores = {}
    for k in range(min_k, max_k + 1):
        try:
            lbl = AgglomerativeClustering(n_clusters=k, linkage=method).fit_predict(X)
            scores[k] = _evaluate_silhouette(X, lbl)
        except Exception:
            scores[k] = 0.0
    return scores


# ──────────────────────────────────────────────────────────────
# CHARTS
# ──────────────────────────────────────────────────────────────

def _make_dendrogram(X, method, k, cutoff, filename):
    """
    Build the linkage matrix for this method and plot the dendrogram
    with the linkage-specific cutoff line.
    """
    Z    = linkage(X, method=method, metric="euclidean")
    path = os.path.join(CHART_DIR, filename)

    fig, ax = plt.subplots(figsize=(11, 5.8), dpi=150)
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#f8f9fa")

    dendrogram(
        Z,
        truncate_mode="lastp",
        p=35,
        color_threshold=cutoff,
        show_leaf_counts=True,
        leaf_rotation=45,
        leaf_font_size=8,
        ax=ax,
    )

    ax.axhline(
        y=cutoff,
        color="#1db954",
        linestyle="--",
        linewidth=2.2,
        label=f"Linkage Cutoff ({cutoff:.3f})  →  K = {k}",
    )

    ax.set_title(
        f"Hierarchical Dendrogram — {method.title()} Linkage  |  K = {k}",
        fontsize=12, fontweight="bold", color="#1a1a2e", pad=12,
    )
    ax.set_xlabel("Merged track groups / Sample counts", fontsize=10, color="#333333")
    ax.set_ylabel("Euclidean Distance", fontsize=10, color="#333333")
    ax.tick_params(colors="#333333")
    ax.grid(axis="y", alpha=0.30, linestyle="--", color="#cccccc")
    ax.legend(loc="upper right", fontsize=9, framealpha=0.92, facecolor="#ffffff")

    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _make_silhouette_eval_chart(scores, linkage_k, method, filename):
    """
    Plot silhouette scores vs K for EVALUATION only.
    The vertical line marks the dendrogram-selected K —
    it is labelled clearly as the linkage-based K, not the argmax.
    """
    path     = os.path.join(CHART_DIR, filename)
    k_vals   = list(scores.keys())
    s_vals   = [scores[k] for k in k_vals]

    fig, ax = plt.subplots(figsize=(8, 4.2), dpi=150)
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#f8f9fa")

    ax.plot(k_vals, s_vals, color="#1db954", marker="o", markersize=7,
            linewidth=2.2, label="Silhouette Score (evaluation)")

    # Mark the dendrogram-selected K
    dend_score = scores.get(linkage_k, 0.0)
    ax.scatter([linkage_k], [dend_score], color="#ff4757", s=160, zorder=5,
               edgecolors="#ffffff", linewidths=2,
               label=f"Dendrogram K = {linkage_k}  (score: {dend_score:.4f})")
    ax.axvline(linkage_k, color="#ff4757", linestyle=":", linewidth=1.5, alpha=0.6)

    ax.annotate(
        f"Linkage K = {linkage_k}\nSilhouette: {dend_score:.4f}",
        xy=(linkage_k, dend_score),
        xytext=(linkage_k + 0.4, dend_score + 0.025 if dend_score < 0.55 else dend_score - 0.06),
        arrowprops=dict(facecolor="#ff4757", shrink=0.08, width=1.5, headwidth=6),
        fontsize=9, fontweight="bold", color="#222222",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#fff9e6", edgecolor="#cccccc", alpha=0.92),
    )

    ax.set_title(
        f"Silhouette Evaluation — {method.title()} Linkage  "
        f"(Dendrogram K = {linkage_k})",
        fontsize=12, fontweight="bold", color="#1a1a2e", pad=12,
    )
    ax.set_xlabel("Number of Clusters (K)", fontsize=10, color="#333333")
    ax.set_ylabel("Silhouette Score  [evaluation only]", fontsize=10, color="#333333")
    ax.set_xticks(k_vals)
    ax.tick_params(colors="#333333")
    ax.grid(True, linestyle="--", alpha=0.30, color="#cccccc")
    ax.legend(loc="best", fontsize=9, framealpha=0.92, facecolor="#ffffff")

    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _make_cluster_plot(sample, labels, method, k, filename):
    pca     = PCA(n_components=2, random_state=42)
    reduced = pca.fit_transform(sample.to_numpy(dtype=float))

    path = os.path.join(CHART_DIR, filename)
    fig, ax = plt.subplots(figsize=(9, 6), dpi=150)
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#f8f9fa")

    palette = [
        "#1db954", "#5c7cfa", "#ff4757", "#f59f00",
        "#845ef7", "#339af0", "#20c997", "#94d82d",
    ]

    for i, cid in enumerate(sorted(np.unique(labels))):
        mask   = labels == cid
        color  = palette[i % len(palette)]
        ax.scatter(
            reduced[mask, 0], reduced[mask, 1],
            s=22, alpha=0.75, color=color, edgecolors="none",
            label=f"Cluster {cid + 1} ({int(mask.sum())} tracks)",
        )
        cx, cy = np.mean(reduced[mask, 0]), np.mean(reduced[mask, 1])
        ax.scatter([cx], [cy], s=120, marker="X", color=color,
                   edgecolors="#343a40", linewidths=1.5, zorder=6)

    v1 = pca.explained_variance_ratio_[0] * 100
    v2 = pca.explained_variance_ratio_[1] * 100

    ax.set_title(
        f"Hierarchical Clusters (PCA 2D) — {method.title()} Linkage  |  K = {k}",
        fontsize=12, fontweight="bold", color="#1a1a2e", pad=12,
    )
    ax.set_xlabel(f"PC1 ({v1:.1f}% variance)", fontsize=10, color="#333333")
    ax.set_ylabel(f"PC2 ({v2:.1f}% variance)", fontsize=10, color="#333333")
    ax.tick_params(colors="#333333")
    ax.legend(title="Clusters & Centroids (X)", fontsize=8,
              title_fontsize=9, loc="best", framealpha=0.92, facecolor="#ffffff")
    ax.grid(True, linestyle="--", alpha=0.30, color="#cccccc")

    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ──────────────────────────────────────────────────────────────
# PUBLIC ENTRY POINT
# ──────────────────────────────────────────────────────────────

def run_hierarchical_clustering(method="ward", k=None):
    """
    Main function called by Flask.

    Workflow:
      1. Load & sample preprocessed data.
      2. Build the linkage matrix for the selected method.
      3. Find K from the dendrogram's largest-gap cutoff.
      4. Fit AgglomerativeClustering with that K & method.
      5. Compute silhouette score as an evaluation metric.
      6. Generate evaluation silhouette chart (evaluation only).
      7. Generate dendrogram with linkage-specific cutoff.
      8. Generate PCA cluster scatter.
    """
    method = str(method).lower().strip()
    if method not in VALID_LINKAGES:
        method = "ward"

    _, numeric = _load_preprocessed()
    sample     = _sample_data(numeric)
    X          = sample.to_numpy(dtype=float)

    # ── Step 1: build linkage matrix for THIS method ──────────
    Z = linkage(X, method=method, metric="euclidean")

    # ── Step 2: derive K from THIS dendrogram ─────────────────
    dendrogram_k, cutoff_dist = _find_k_from_dendrogram(Z, min_k=MIN_K, max_k=MAX_K)

    # ── Step 3: fit final clusters with the dendrogram K ──────
    model  = AgglomerativeClustering(n_clusters=dendrogram_k, linkage=method)
    labels = model.fit_predict(X)

    # ── Step 4: evaluate with silhouette (evaluation only) ────
    final_silhouette = _evaluate_silhouette(X, labels)

    # Silhouette across K=2..8 for evaluation chart
    sil_scores = _silhouette_by_k(X, method, min_k=MIN_K, max_k=MAX_K)

    # ── Step 5: generate charts ────────────────────────────────
    dendrogram_fn    = f"hierarchical_dendrogram_{method}.png"
    cluster_fn       = f"hierarchical_clusters_{method}_{dendrogram_k}.png"
    silhouette_fn    = f"hierarchical_silhouette_eval_{method}.png"

    _make_dendrogram(X, method, dendrogram_k, cutoff_dist, dendrogram_fn)
    _make_silhouette_eval_chart(sil_scores, dendrogram_k, method, silhouette_fn)
    _make_cluster_plot(sample, labels, method, dendrogram_k, cluster_fn)

    counts = pd.Series(labels).value_counts().sort_index()

    return {
        # ── method & K ──────────────────────────────────────
        "method":            method,
        "selected_k":        dendrogram_k,
        "cutoff_dist":       round(cutoff_dist, 4),

        # ── silhouette as EVALUATION metric only ────────────
        "final_silhouette":  final_silhouette,
        "silhouette_scores": [
            {
                "k":       k_val,
                "score":   s_val,
                "is_selected": (k_val == dendrogram_k),
            }
            for k_val, s_val in sil_scores.items()
        ],

        # ── dataset info ─────────────────────────────────────
        "sample_records":    int(len(sample)),
        "dataset_records":   int(len(numeric)),
        "feature_count":     int(numeric.shape[1]),
        "features":          list(numeric.columns),

        # ── charts ──────────────────────────────────────────
        "dendrogram":        "charts/" + dendrogram_fn,
        "cluster_chart":     "charts/" + cluster_fn,
        "silhouette_chart":  "charts/" + silhouette_fn,

        # ── cluster sizes ────────────────────────────────────
        "cluster_counts":    {
            f"Cluster {int(i) + 1}": int(v)
            for i, v in counts.items()
        },
    }
