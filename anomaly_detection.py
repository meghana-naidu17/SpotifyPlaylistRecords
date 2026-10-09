"""
anomaly_detection.py
====================
Anomaly Detection module for Spotify Playlist Analytics.

Three algorithms run on the same preprocessed audio features:
  1. Isolation Forest  — tree-based outlier isolation
  2. One-Class SVM     — kernel-based boundary estimation
  3. Autoencoder       — reconstruction-error-based detection

All anomaly scores and flags are computed against the 13
continuous audio features (no one-hot columns).
"""

import os
import time
import warnings
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.neural_network import MLPRegressor
from sklearn.svm import OneClassSVM

warnings.filterwarnings("ignore")

BASE_DIR          = os.path.dirname(os.path.abspath(__file__))
PREPROCESSED_PATH = os.path.join(BASE_DIR, "processed_data", "preprocessed_dataset.csv")
CHART_DIR         = os.path.join(BASE_DIR, "static", "charts")
os.makedirs(CHART_DIR, exist_ok=True)

MAX_SAMPLE = 2000
OCSVM_CAP  = 800

# Exactly 10 continuous audio features, already StandardScaled.
# Do NOT include popularity (not in preprocessed_dataset continuous cols),
# key or mode (they are one-hot encoded in the preprocessed CSV).
AUDIO_FEATURES = [
    "duration_ms", "danceability", "energy", "loudness",
    "speechiness", "acousticness", "instrumentalness",
    "liveness", "valence", "tempo",
]


def _load_features() -> tuple:
    """Return (X_numpy, feature_name_list) from preprocessed_dataset.csv."""
    if not os.path.exists(PREPROCESSED_PATH):
        raise FileNotFoundError(
            "Preprocessed dataset not found: "
            "processed_data/preprocessed_dataset.csv"
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

    feat = df[available].copy()
    feat = feat.replace([np.inf, -np.inf], np.nan)
    feat = feat.fillna(feat.median(numeric_only=True)).fillna(0)

    return feat, list(feat.columns)


# ──────────────────────────────────────────────────────────────
# CHART HELPERS
# ──────────────────────────────────────────────────────────────

def _scatter(reduced, is_anomaly, pca, title, algo_name, fname):
    """PCA-2D scatter: anomalies in red, normal in green."""
    path = os.path.join(CHART_DIR, fname)
    fig, ax = plt.subplots(figsize=(9, 5.8), dpi=150)
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#0d1117")

    normal  = ~is_anomaly
    anomaly =  is_anomaly

    ax.scatter(reduced[normal,  0], reduced[normal,  1],
               c="#1db954", s=14, alpha=0.45, label=f"Normal ({int(normal.sum())})",
               edgecolors="none")
    ax.scatter(reduced[anomaly, 0], reduced[anomaly, 1],
               c="#ff4757", s=30, alpha=0.90, label=f"Anomaly ({int(anomaly.sum())})",
               edgecolors="none", zorder=3)

    v1 = round(pca.explained_variance_ratio_[0] * 100, 1)
    v2 = round(pca.explained_variance_ratio_[1] * 100, 1)
    ax.set_title(title, fontsize=11, fontweight="bold", color="#ffffff", pad=10)
    ax.set_xlabel(f"PC1 ({v1}% var)", fontsize=9, color="#b3b3b3")
    ax.set_ylabel(f"PC2 ({v2}% var)", fontsize=9, color="#b3b3b3")
    ax.tick_params(colors="#b3b3b3")
    ax.grid(True, linestyle="--", alpha=0.18, color="#ffffff")
    ax.legend(fontsize=9, framealpha=0.85, facecolor="#1a1a2e", labelcolor="white")

    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _score_hist(scores, threshold, title, fname):
    """Histogram of anomaly scores with threshold line."""
    path = os.path.join(CHART_DIR, fname)
    fig, ax = plt.subplots(figsize=(8.5, 4.2), dpi=150)
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#0d1117")

    ax.hist(scores, bins=60, color="#1db954", edgecolor="none", alpha=0.75)
    ax.axvline(threshold, color="#ff4757", linewidth=2,
               linestyle="--", label=f"Threshold = {threshold:.4f}")

    ax.set_title(title, fontsize=11, fontweight="bold", color="#ffffff", pad=10)
    ax.set_xlabel("Anomaly Score", fontsize=9, color="#b3b3b3")
    ax.set_ylabel("Number of Tracks", fontsize=9, color="#b3b3b3")
    ax.tick_params(colors="#b3b3b3")
    ax.grid(True, linestyle="--", alpha=0.18, color="#ffffff")
    ax.legend(fontsize=9, framealpha=0.85, facecolor="#1a1a2e", labelcolor="white")

    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _feature_importance_bar(feature_names, importances, title, fname):
    """Horizontal bar chart showing which features drive anomaly scores."""
    path = os.path.join(CHART_DIR, fname)
    idx  = np.argsort(importances)
    names = [feature_names[i].replace("_", " ").title() for i in idx]
    vals  = [importances[i] for i in idx]

    fig, ax = plt.subplots(figsize=(8, max(3.5, len(names) * 0.4)), dpi=150)
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#0d1117")

    colors = plt.cm.RdYlGn_r(np.linspace(0.15, 0.85, len(names)))
    bars = ax.barh(names, vals, color=colors, edgecolor="none", height=0.65)
    for bar in bars:
        w = bar.get_width()
        ax.text(w + max(vals) * 0.01, bar.get_y() + bar.get_height() / 2,
                f"{w:.4f}", va="center", ha="left", fontsize=8, color="#ffffff")

    ax.set_title(title, fontsize=11, fontweight="bold", color="#ffffff", pad=10)
    ax.set_xlabel("Mean Absolute Deviation from Normal", fontsize=9, color="#b3b3b3")
    ax.tick_params(colors="#b3b3b3")
    ax.grid(True, axis="x", linestyle="--", alpha=0.18, color="#ffffff")

    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _ae_loss_curve(loss_curve, fname):
    """Training loss curve for the autoencoder."""
    path = os.path.join(CHART_DIR, fname)
    fig, ax = plt.subplots(figsize=(8.5, 4), dpi=150)
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#0d1117")

    epochs = list(range(1, len(loss_curve) + 1))
    ax.plot(epochs, loss_curve, color="#1db954", linewidth=2.2, label="Reconstruction MSE")
    min_e = int(np.argmin(loss_curve)) + 1
    ax.scatter([min_e], [loss_curve[min_e - 1]], color="#ff4757", s=90, zorder=5,
               label=f"Min loss @ epoch {min_e}")

    ax.set_title("Autoencoder Training Loss Convergence", fontsize=11,
                 fontweight="bold", color="#ffffff", pad=10)
    ax.set_xlabel("Epoch", fontsize=9, color="#b3b3b3")
    ax.set_ylabel("MSE Loss", fontsize=9, color="#b3b3b3")
    ax.tick_params(colors="#b3b3b3")
    ax.grid(True, linestyle="--", alpha=0.18, color="#ffffff")
    ax.legend(fontsize=9, framealpha=0.85, facecolor="#1a1a2e", labelcolor="white")

    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ──────────────────────────────────────────────────────────────
# ALGORITHM 1 — ISOLATION FOREST
# ──────────────────────────────────────────────────────────────

def _run_isolation_forest(X: np.ndarray, feature_names: list,
                          contamination: float, n_estimators: int,
                          reduced: np.ndarray, pca: PCA) -> dict:
    t0  = time.time()
    clf = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X)
    raw_scores = clf.decision_function(X)   # higher = more normal
    preds      = clf.predict(X)             # +1 normal, -1 anomaly
    duration   = round(time.time() - t0, 3)

    is_anomaly = (preds == -1)
    n_anomaly  = int(is_anomaly.sum())
    n_normal   = int((~is_anomaly).sum())

    # Anomaly score = negative of decision_function (higher = more anomalous)
    anom_scores = -raw_scores
    threshold   = float(np.percentile(anom_scores, (1 - contamination) * 100))

    # Feature importance: mean |z-score| for anomalous vs normal tracks
    normal_mean = X[~is_anomaly].mean(axis=0)
    anom_mean   = X[ is_anomaly].mean(axis=0) if n_anomaly > 0 else normal_mean
    importances = np.abs(anom_mean - normal_mean)

    # Top 5 most anomalous tracks
    top_idx = np.argsort(anom_scores)[-5:][::-1]
    top_tracks = []
    for i in top_idx:
        row = {f: round(float(X[i, j]), 4) for j, f in enumerate(feature_names)}
        row["anomaly_score"] = round(float(anom_scores[i]), 4)
        top_tracks.append(row)

    # Charts
    scatter_fn = "anomaly_if_scatter.png"
    hist_fn    = "anomaly_if_hist.png"
    feat_fn    = "anomaly_if_features.png"

    _scatter(reduced, is_anomaly, pca,
             f"Isolation Forest — {n_anomaly} Anomalies ({contamination*100:.0f}% contamination)",
             "IF", scatter_fn)
    _score_hist(anom_scores,
                threshold,
                "Isolation Forest — Anomaly Score Distribution",
                hist_fn)
    _feature_importance_bar(feature_names, importances,
                            "Isolation Forest — Feature Deviation (Anomaly vs Normal)",
                            feat_fn)

    return {
        "algorithm":     "Isolation Forest",
        "n_estimators":  n_estimators,
        "contamination": contamination,
        "duration":      duration,
        "n_anomaly":     n_anomaly,
        "n_normal":      n_normal,
        "anomaly_pct":   round(n_anomaly / len(X) * 100, 2),
        "mean_score_normal":  round(float(anom_scores[~is_anomaly].mean()), 4),
        "mean_score_anomaly": round(float(anom_scores[is_anomaly].mean()), 4) if n_anomaly > 0 else 0.0,
        "threshold":     round(threshold, 4),
        "top_tracks":    top_tracks,
        "feature_importance": {f: round(float(importances[j]), 4) for j, f in enumerate(feature_names)},
        "scatter_chart": "charts/" + scatter_fn,
        "hist_chart":    "charts/" + hist_fn,
        "feat_chart":    "charts/" + feat_fn,
    }


# ──────────────────────────────────────────────────────────────
# ALGORITHM 2 — ONE-CLASS SVM
# ──────────────────────────────────────────────────────────────

def _run_ocsvm(X: np.ndarray, feature_names: list,
               nu: float, kernel: str,
               reduced: np.ndarray, pca: PCA) -> dict:
    t0  = time.time()
    clf = OneClassSVM(nu=nu, kernel=kernel, gamma="scale")
    clf.fit(X)
    raw_scores = clf.decision_function(X)   # higher = more normal
    preds      = clf.predict(X)             # +1 normal, -1 anomaly
    duration   = round(time.time() - t0, 3)

    is_anomaly = (preds == -1)
    n_anomaly  = int(is_anomaly.sum())
    n_normal   = int((~is_anomaly).sum())

    anom_scores = -raw_scores
    threshold   = float(np.percentile(anom_scores, (1 - nu) * 100))

    normal_mean = X[~is_anomaly].mean(axis=0)
    anom_mean   = X[ is_anomaly].mean(axis=0) if n_anomaly > 0 else normal_mean
    importances = np.abs(anom_mean - normal_mean)

    top_idx = np.argsort(anom_scores)[-5:][::-1]
    top_tracks = []
    for i in top_idx:
        row = {f: round(float(X[i, j]), 4) for j, f in enumerate(feature_names)}
        row["anomaly_score"] = round(float(anom_scores[i]), 4)
        top_tracks.append(row)

    scatter_fn = f"anomaly_ocsvm_scatter_{kernel}.png"
    hist_fn    = f"anomaly_ocsvm_hist_{kernel}.png"
    feat_fn    = f"anomaly_ocsvm_features_{kernel}.png"

    _scatter(reduced, is_anomaly, pca,
             f"One-Class SVM ({kernel.upper()} kernel) — {n_anomaly} Anomalies",
             "OCSVM", scatter_fn)
    _score_hist(anom_scores, threshold,
                "One-Class SVM — Anomaly Score Distribution",
                hist_fn)
    _feature_importance_bar(feature_names, importances,
                            "One-Class SVM — Feature Deviation (Anomaly vs Normal)",
                            feat_fn)

    return {
        "algorithm": "One-Class SVM",
        "nu":         nu,
        "kernel":     kernel,
        "duration":   duration,
        "n_anomaly":  n_anomaly,
        "n_normal":   n_normal,
        "anomaly_pct": round(n_anomaly / len(X) * 100, 2),
        "n_support_vectors": int(clf.support_vectors_.shape[0]),
        "mean_score_normal":  round(float(anom_scores[~is_anomaly].mean()), 4),
        "mean_score_anomaly": round(float(anom_scores[is_anomaly].mean()), 4) if n_anomaly > 0 else 0.0,
        "threshold":  round(threshold, 4),
        "top_tracks": top_tracks,
        "feature_importance": {f: round(float(importances[j]), 4) for j, f in enumerate(feature_names)},
        "scatter_chart": "charts/" + scatter_fn,
        "hist_chart":    "charts/" + hist_fn,
        "feat_chart":    "charts/" + feat_fn,
    }


# ──────────────────────────────────────────────────────────────
# ALGORITHM 3 — AUTOENCODER ANOMALY DETECTION
# ──────────────────────────────────────────────────────────────

def _run_autoencoder_anomaly(X: np.ndarray, feature_names: list,
                             contamination: float, hidden_dim: int,
                             latent_dim: int, epochs: int,
                             reduced: np.ndarray, pca: PCA) -> dict:
    n_feat = X.shape[1]
    t0     = time.time()

    ae = MLPRegressor(
        hidden_layer_sizes=(hidden_dim, latent_dim, hidden_dim),
        activation="relu",
        solver="adam",
        max_iter=epochs,
        random_state=42,
        tol=1e-5,
        alpha=1e-4,
    )
    ae.fit(X, X)
    duration = round(time.time() - t0, 3)

    X_rec       = ae.predict(X)
    # Per-sample reconstruction error (MSE)
    rec_errors  = np.mean((X - X_rec) ** 2, axis=1)
    threshold   = float(np.percentile(rec_errors, (1 - contamination) * 100))
    is_anomaly  = rec_errors > threshold

    n_anomaly = int(is_anomaly.sum())
    n_normal  = int((~is_anomaly).sum())

    # Per-feature reconstruction error
    feat_errors = np.mean((X - X_rec) ** 2, axis=0)

    # Top 5 highest reconstruction error tracks
    top_idx = np.argsort(rec_errors)[-5:][::-1]
    top_tracks = []
    for i in top_idx:
        row = {f: round(float(X[i, j]), 4) for j, f in enumerate(feature_names)}
        row["reconstruction_error"] = round(float(rec_errors[i]), 4)
        top_tracks.append(row)

    scatter_fn   = "anomaly_ae_scatter.png"
    hist_fn      = "anomaly_ae_hist.png"
    feat_fn      = "anomaly_ae_features.png"
    loss_curve_fn = "anomaly_ae_loss.png"

    _scatter(reduced, is_anomaly, pca,
             f"Autoencoder Anomaly Detection — {n_anomaly} Anomalies "
             f"({contamination*100:.0f}% contamination)",
             "AE", scatter_fn)
    _score_hist(rec_errors, threshold,
                "Autoencoder — Reconstruction Error Distribution",
                hist_fn)
    _feature_importance_bar(feature_names, feat_errors,
                            "Autoencoder — Per-Feature Reconstruction Error",
                            feat_fn)
    _ae_loss_curve(ae.loss_curve_, loss_curve_fn)

    # Feature reconstruction quality
    feat_rows = []
    for j, f in enumerate(feature_names):
        err = float(feat_errors[j])
        feat_rows.append({
            "feature":     f.replace("_", " ").title(),
            "mse":         round(err, 4),
            "quality":     "High" if err < 0.05 else "Medium" if err < 0.15 else "Low",
        })
    feat_rows.sort(key=lambda r: r["mse"])

    return {
        "algorithm":    "Autoencoder",
        "hidden_dim":   hidden_dim,
        "latent_dim":   latent_dim,
        "epochs":       epochs,
        "actual_epochs": len(ae.loss_curve_),
        "final_loss":   round(float(ae.loss_curve_[-1]), 4),
        "contamination": contamination,
        "duration":     duration,
        "n_anomaly":    n_anomaly,
        "n_normal":     n_normal,
        "anomaly_pct":  round(n_anomaly / len(X) * 100, 2),
        "threshold":    round(threshold, 4),
        "mean_rec_error_normal":  round(float(rec_errors[~is_anomaly].mean()), 4),
        "mean_rec_error_anomaly": round(float(rec_errors[is_anomaly].mean()), 4) if n_anomaly > 0 else 0.0,
        "top_tracks":   top_tracks,
        "feat_rows":    feat_rows,
        "scatter_chart":    "charts/" + scatter_fn,
        "hist_chart":       "charts/" + hist_fn,
        "feat_chart":       "charts/" + feat_fn,
        "loss_curve_chart": "charts/" + loss_curve_fn,
    }


# ──────────────────────────────────────────────────────────────
# PUBLIC ENTRY POINT
# ──────────────────────────────────────────────────────────────

def run_anomaly_detection(
    contamination: float = 0.05,
    n_estimators:  int   = 100,
    nu:            float = 0.05,
    kernel:        str   = "rbf",
    hidden_dim:    int   = 32,
    latent_dim:    int   = 4,
    epochs:        int   = 80,
) -> dict:

    # ── parameter guards ──────────────────────────────────────
    try:    contamination = float(contamination)
    except: contamination = 0.05
    contamination = max(0.01, min(contamination, 0.30))

    try:    n_estimators = int(n_estimators)
    except: n_estimators = 100
    n_estimators = max(50, min(n_estimators, 300))

    try:    nu = float(nu)
    except: nu = 0.05
    nu = max(0.01, min(nu, 0.30))

    kernel = str(kernel).lower().strip()
    if kernel not in ("rbf", "linear", "poly"):
        kernel = "rbf"

    try:    hidden_dim = int(hidden_dim)
    except: hidden_dim = 32
    hidden_dim = max(8, min(hidden_dim, 64))

    try:    latent_dim = int(latent_dim)
    except: latent_dim = 4
    latent_dim = max(2, min(latent_dim, hidden_dim // 2))

    try:    epochs = int(epochs)
    except: epochs = 80
    epochs = max(20, min(epochs, 200))

    # ── load features ─────────────────────────────────────────
    feat_df, feature_names = _load_features()
    total_records = len(feat_df)

    # Sample for main algorithms (already StandardScaled)
    sample_main = feat_df.sample(n=min(MAX_SAMPLE, len(feat_df)), random_state=42)
    X_main      = sample_main.to_numpy(dtype=float)

    # Smaller sample for One-Class SVM
    sample_svm  = feat_df.sample(n=min(OCSVM_CAP, len(feat_df)), random_state=42)
    X_svm       = sample_svm.to_numpy(dtype=float)

    # Shared PCA for all scatter plots (fit on main sample)
    pca     = PCA(n_components=2, random_state=42)
    red_main = pca.fit_transform(X_main)
    red_svm  = pca.transform(X_svm)

    # ── run all three algorithms ───────────────────────────────
    iso  = _run_isolation_forest(X_main, feature_names, contamination,
                                  n_estimators, red_main, pca)
    svm  = _run_ocsvm(X_svm, feature_names, nu, kernel, red_svm, pca)
    ae   = _run_autoencoder_anomaly(X_main, feature_names, contamination,
                                     hidden_dim, latent_dim, epochs,
                                     red_main, pca)

    # ── comparison table ──────────────────────────────────────
    comparison = [
        {
            "criterion": "Approach",
            "iso":   "Randomly isolates points — anomalies need fewer splits",
            "svm":   "Learns a tight boundary around normal points",
            "ae":    "Anomalies have higher reconstruction error",
        },
        {
            "criterion": "Anomalies Found",
            "iso":   f"{iso['n_anomaly']} ({iso['anomaly_pct']}%)",
            "svm":   f"{svm['n_anomaly']} ({svm['anomaly_pct']}%)",
            "ae":    f"{ae['n_anomaly']}  ({ae['anomaly_pct']}%)",
        },
        {
            "criterion": "Runtime",
            "iso":   f"{iso['duration']}s",
            "svm":   f"{svm['duration']}s",
            "ae":    f"{ae['duration']}s",
        },
        {
            "criterion": "Scales to large data",
            "iso":   "✓ Yes — linear time",
            "svm":   "✗ Slow — quadratic in n",
            "ae":    "✓ Yes — mini-batch Adam",
        },
        {
            "criterion": "Needs label / contamination hint",
            "iso":   "Optional contamination parameter",
            "svm":   "ν controls support fraction",
            "ae":    "Threshold set at (1−contamination) percentile",
        },
        {
            "criterion": "Interpretability",
            "iso":   "Feature importance via path length",
            "svm":   "Support vectors define boundary",
            "ae":    "Per-feature reconstruction error",
        },
    ]

    return {
        # parameters echoed for template
        "contamination": contamination,
        "n_estimators":  n_estimators,
        "nu":            nu,
        "kernel":        kernel,
        "hidden_dim":    hidden_dim,
        "latent_dim":    latent_dim,
        "epochs":        epochs,
        # dataset info
        "total_records": total_records,
        "sample_size":   len(sample_main),
        "svm_sample":    len(sample_svm),
        "feature_names": feature_names,
        "n_features":    len(feature_names),
        # per-algo results
        "iso":  iso,
        "svm":  svm,
        "ae":   ae,
        # shared
        "comparison": comparison,
    }


if __name__ == "__main__":
    r = run_anomaly_detection()
    print("IF  anomalies:", r["iso"]["n_anomaly"])
    print("SVM anomalies:", r["svm"]["n_anomaly"])
    print("AE  anomalies:", r["ae"]["n_anomaly"])
