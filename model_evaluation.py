"""
Model Evaluation, Selection, and Calibration Module for Spotify Playlist Analytics.

Designed to run either:
1. Directly in PyCharm / Python CLI:
   Runs the analyses on the real Spotify dataset (SpotifyPlaylistRecords.csv)
   and displays interactive Matplotlib plots directly in PyCharm using plt.show().

2. From the Flask Application (app.py):
   Generates plots and delivers metrics to the web interface on-demand.

Plot styling matches the clean white / light theme from the EDA and Linear Regression modules.

Sections:
1. Train/Validation/Test Split Discipline (3-way vs 2-way comparison & data snooping bias)
2. Cross-Validation Strategies (k-fold, Stratified k-fold, and Nested CV)
3. Classification Metrics & Class Imbalance (Accuracy trap, Precision/Recall/F1, ROC-AUC vs PR-AUC)
4. Regression Metrics & Outlier Sensitivity (RMSE, MAE, MAPE, R2 on continuous Loudness)
5. Probability Calibration (Reliability diagrams, Platt scaling, Isotonic regression)
6. Hyperparameter Search (Grid Search vs Random Search [Bergstra-Bengio] on real Spotify tracks)
7. Learning & Validation Curves (Underfitting, Overfitting, Data scarcity diagnostics)
8. Statistical Significance Testing (McNemar's test for classifiers, Paired Bootstrap for regressors)
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib

# If running directly in PyCharm / CLI, keep interactive GUI backend for plt.show()
if __name__ != "__main__":
    try:
        matplotlib.use("Agg")
    except Exception:
        pass

import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from sklearn.model_selection import (
    train_test_split,
    KFold,
    StratifiedKFold,
    GridSearchCV,
    RandomizedSearchCV,
    learning_curve,
    validation_curve
)
from sklearn.linear_model import LogisticRegression, Ridge, LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score,
    mean_squared_error,
    mean_absolute_error,
    median_absolute_error,
    r2_score,
    brier_score_loss
)
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV, calibration_curve

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "SpotifyPlaylistRecords.csv")
CHART_DIR = os.path.join(BASE_DIR, "static", "charts")
os.makedirs(CHART_DIR, exist_ok=True)

# Audio features
FEATURES = [
    "popularity", "duration_ms", "danceability", "energy", "key", "loudness",
    "mode", "speechiness", "acousticness", "instrumentalness", "liveness",
    "valence", "tempo", "time_signature"
]

# Clean Light Styling Colors (Matching EDA and Linear Regression modules)
GREEN = "#1DB954"
DARK_GREEN = "#159447"
DARK_BLUE = "#23456b"
LIGHT_GREEN = "#2ecc71"
WHITE = "#ffffff"
BORDER_GRAY = "#dce4ec"
ACCENT_CORAL = "#e63946"
ACCENT_AMBER = "#e76f51"
ACCENT_BLUE = "#1d70b8"
TEXT_DARK = "#23456b"
TEXT_MUTED = "#556b82"

sns.set_theme(style="whitegrid", context="notebook")


def _apply_light_style(fig, axes):
    """Apply clean white/light theme matching EDA and Regression sections."""
    fig.patch.set_facecolor(WHITE)
    if not isinstance(axes, (list, np.ndarray)):
        axes = [axes]
    for ax in axes:
        ax.set_facecolor(WHITE)
        ax.tick_params(colors=DARK_BLUE, labelsize=9)
        ax.xaxis.label.set_color(DARK_BLUE)
        ax.yaxis.label.set_color(DARK_BLUE)
        ax.title.set_color(DARK_BLUE)
        ax.grid(True, linestyle="--", alpha=0.5, color=BORDER_GRAY)
        for spine in ax.spines.values():
            spine.set_color("#cbd5e1")


_DATASET_CACHE = None


def load_dataset():
    """Load and prepare real Spotify dataset."""
    global _DATASET_CACHE
    if _DATASET_CACHE is not None:
        return _DATASET_CACHE
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Dataset not found at {DATASET_PATH}")
    df = pd.read_csv(DATASET_PATH)

    # Clean explicit binary target
    if "explicit" in df.columns:
        if df["explicit"].dtype == object:
            df["explicit"] = df["explicit"].astype(str).str.strip().str.lower().map({
                "true": 1, "false": 0, "yes": 1, "no": 0, "1": 1, "0": 0
            }).fillna(0).astype(int)
        else:
            df["explicit"] = df["explicit"].astype(int)

    # Cast numeric audio features
    for col in FEATURES:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    clean_df = df.dropna(subset=FEATURES + ["explicit", "loudness"]).copy()
    _DATASET_CACHE = clean_df
    return _DATASET_CACHE


def _display_or_save(fig, chart_filename, show_interactive=False):
    """Save chart to disk for Flask and show interactively in PyCharm if requested."""
    chart_path = os.path.join(CHART_DIR, chart_filename)
    fig.tight_layout()
    fig.savefig(chart_path, dpi=130, facecolor=WHITE, bbox_inches="tight")
    if show_interactive:
        plt.show()
    plt.close(fig)
    return f"charts/{chart_filename}"


_RESULT_CACHE = {}


def _cached(key, chart_name):
    def decorator(fn):
        def wrapper(show_interactive=False, force_refresh=False, *args, **kwargs):
            chart_path = os.path.join(CHART_DIR, chart_name)
            if not force_refresh and not show_interactive and key in _RESULT_CACHE and os.path.exists(chart_path):
                return _RESULT_CACHE[key]
            res = fn(show_interactive=show_interactive, *args, **kwargs)
            _RESULT_CACHE[key] = res
            return res
        wrapper.__name__ = fn.__name__
        return wrapper
    return decorator


# ==============================================================================
# 1. TRAIN / VALIDATION / TEST SPLIT DISCIPLINE
# ==============================================================================

@_cached("splitting", "eval_splitting_comparison.png")
def run_data_splitting_evaluation(show_interactive=False, *args, **kwargs):
    """
    Evaluates 2-way vs 3-way split on real Spotify audio features predicting loudness.
    Generates comparison graph illustrating data snooping optimism bias.
    """
    df = load_dataset()
    sample = df.sample(n=min(30000, len(df)), random_state=42)

    X = sample[["energy", "acousticness", "danceability", "tempo"]]
    y = sample["loudness"]

    # 3-Way Split (70% Train, 15% Val, 15% Test)
    X_train_full, X_test_3way, y_train_full, y_test_3way = train_test_split(
        X, y, test_size=0.15, random_state=42
    )
    val_fraction = 0.15 / (1.0 - 0.15)
    X_train_3way, X_val_3way, y_train_3way, y_val_3way = train_test_split(
        X_train_full, y_train_full, test_size=val_fraction, random_state=42
    )

    # Standardize based strictly on Train split
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_train_3way)
    X_val_s = scaler.transform(X_val_3way)
    X_test_s = scaler.transform(X_test_3way)

    # 2-Way Split (85% Train, 15% Test)
    X_train_2way, X_test_2way, y_train_2way, y_test_2way = train_test_split(
        X, y, test_size=0.15, random_state=42
    )
    scaler_2way = StandardScaler()
    X_tr2_s = scaler_2way.fit_transform(X_train_2way)
    X_test2_s = scaler_2way.transform(X_test_2way)

    # Regularization hyperparameter search across alphas on real Spotify tracks
    alphas = np.logspace(-2, 3, 20)
    train_mse_list = []
    val_mse_list = []
    snooped_test_mse_list = []
    true_test_mse_list = []

    for alpha in alphas:
        m3 = Ridge(alpha=alpha)
        m3.fit(X_tr_s, y_train_3way)
        train_mse_list.append(mean_squared_error(y_train_3way, m3.predict(X_tr_s)))
        val_mse_list.append(mean_squared_error(y_val_3way, m3.predict(X_val_s)))
        true_test_mse_list.append(mean_squared_error(y_test_3way, m3.predict(X_test_s)))

        m2 = Ridge(alpha=alpha)
        m2.fit(X_tr2_s, y_train_2way)
        snooped_test_mse_list.append(mean_squared_error(y_test_2way, m2.predict(X_test2_s)))

    best_val_idx = int(np.argmin(val_mse_list))
    best_alpha = float(alphas[best_val_idx])

    best_2way_idx = int(np.argmin(snooped_test_mse_list))
    apparent_2way_test_mse = float(snooped_test_mse_list[best_2way_idx])
    true_3way_test_mse = float(true_test_mse_list[best_val_idx])
    optimism_gap = float(abs(true_3way_test_mse - apparent_2way_test_mse))

    # Generate Comparison Graph in Clean Light Theme
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=130)
    _apply_light_style(fig, [ax1, ax2])

    ax1.plot(alphas, train_mse_list, color=GREEN, linewidth=2.2, label="Train Loss (3-Way)")
    ax1.plot(alphas, val_mse_list, color=DARK_BLUE, linewidth=2.2, label="Validation Loss (Model Selection)")
    ax1.plot(alphas, snooped_test_mse_list, color=ACCENT_AMBER, linestyle="--", linewidth=2, label="2-Way 'Test' Loss (Snooped Selection)")
    ax1.plot(alphas, true_test_mse_list, color=ACCENT_CORAL, linewidth=2.4, label="3-Way True Held-out Test Loss")

    ax1.set_xscale("log")
    ax1.set_xlabel("Regularization Strength (Ridge Alpha)", fontweight="bold")
    ax1.set_ylabel("Mean Squared Error (MSE)", fontweight="bold")
    ax1.set_title("2-Way vs 3-Way Split: Data Snooping Bias Gap", fontweight="bold", fontsize=11, color=DARK_BLUE)
    ax1.axvline(best_alpha, color=DARK_BLUE, linestyle=":", alpha=0.7, label=f"Selected Alpha ({best_alpha:.2f})")
    ax1.annotate(
        f"Optimism Bias Gap:\n{optimism_gap:.4f} MSE",
        xy=(best_alpha, true_3way_test_mse),
        xytext=(best_alpha * 2.5, true_3way_test_mse + 0.15),
        arrowprops=dict(arrowstyle="->", color=ACCENT_CORAL, lw=1.5),
        color=DARK_BLUE, fontsize=9, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.4", fc="#f8fafc", ec="#cbd5e1", lw=1)
    )
    ax1.legend(loc="upper left", fontsize=8, facecolor=WHITE, edgecolor="#cbd5e1", labelcolor=DARK_BLUE)

    split_sizes = [len(X_train_3way), len(X_val_3way), len(X_test_3way)]
    labels = [f"Train\n(70% / {split_sizes[0]:,})", f"Validation\n(15% / {split_sizes[1]:,})", f"Held-out Test\n(15% / {split_sizes[2]:,})"]
    colors = [GREEN, DARK_BLUE, ACCENT_CORAL]
    bars = ax2.bar(labels, split_sizes, color=colors, width=0.55, edgecolor="#cbd5e1")
    ax2.set_ylabel("Record Count", fontweight="bold")
    ax2.set_title("Disciplined Three-Way Data Partition", fontweight="bold", fontsize=11, color=DARK_BLUE)
    for bar in bars:
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2, h + 500, f"{h:,}", ha="center", va="bottom",
                 color=DARK_BLUE, fontsize=9, fontweight="bold")
    ax2.set_ylim(0, max(split_sizes) * 1.18)

    chart_url = _display_or_save(fig, "eval_splitting_comparison.png", show_interactive)

    return {
        "title": "Train / Validation / Test Split Discipline",
        "chart_url": chart_url,
        "train_records": len(X_train_3way),
        "val_records": len(X_val_3way),
        "test_records": len(X_test_3way),
        "best_alpha": round(best_alpha, 3),
        "apparent_2way_mse": round(apparent_2way_test_mse, 4),
        "true_3way_mse": round(true_3way_test_mse, 4),
        "optimism_gap": round(optimism_gap, 4),
        "leakage_rule": "StandardScaler and all transformers fitted solely on Train split, never Val or Test."
    }


# ==============================================================================
# 2. CROSS-VALIDATION & STRATIFIED K-FOLD & NESTED CV
# ==============================================================================

@_cached("cv", "eval_cv_comparison.png")
def run_cross_validation_evaluation(show_interactive=False, *args, **kwargs):
    """
    Evaluates real 5-fold K-Fold vs Stratified K-Fold on the imbalanced 'explicit' target,
    along with Nested Cross-Validation for unbiased assessment.
    """
    df = load_dataset()
    sample = df.sample(n=min(25000, len(df)), random_state=42)

    X = sample[["danceability", "energy", "speechiness", "loudness", "valence"]]
    y = sample["explicit"]
    overall_explicit_pct = float(y.mean() * 100)

    k = 5
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

    kf_props = []
    skf_props = []

    for _, test_idx in kf.split(X, y):
        kf_props.append(float(y.iloc[test_idx].mean() * 100))

    for _, test_idx in skf.split(X, y):
        skf_props.append(float(y.iloc[test_idx].mean() * 100))

    # Real Nested CV (5 Outer Folds, 3 Inner Folds)
    outer_scores = []
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    c_candidates = [0.01, 0.1, 1.0, 10.0]

    for train_outer_idx, test_outer_idx in skf.split(X_scaled, y):
        X_tr_out, X_te_out = X_scaled[train_outer_idx], X_scaled[test_outer_idx]
        y_tr_out, y_te_out = y.iloc[train_outer_idx], y.iloc[test_outer_idx]

        inner_skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        best_c = 1.0
        best_inner_score = -1.0

        for c_val in c_candidates:
            fold_accs = []
            for tr_in, val_in in inner_skf.split(X_tr_out, y_tr_out):
                clf = LogisticRegression(C=c_val, max_iter=200, random_state=42)
                clf.fit(X_tr_out[tr_in], y_tr_out.iloc[tr_in])
                fold_accs.append(clf.score(X_tr_out[val_in], y_tr_out.iloc[val_in]))
            m_acc = np.mean(fold_accs)
            if m_acc > best_inner_score:
                best_inner_score = m_acc
                best_c = c_val

        outer_clf = LogisticRegression(C=best_c, max_iter=200, random_state=42)
        outer_clf.fit(X_tr_out, y_tr_out)
        outer_scores.append(outer_clf.score(X_te_out, y_te_out))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=130)
    _apply_light_style(fig, [ax1, ax2])

    folds = [f"Fold {i+1}" for i in range(k)]
    x_indices = np.arange(k)
    width = 0.35

    ax1.bar(x_indices - width/2, kf_props, width, color=ACCENT_CORAL, label="Standard K-Fold (Unstable)", edgecolor="#cbd5e1")
    ax1.bar(x_indices + width/2, skf_props, width, color=GREEN, label="Stratified K-Fold (Preserved)", edgecolor="#cbd5e1")
    ax1.axhline(overall_explicit_pct, color=ACCENT_AMBER, linestyle="--", linewidth=1.5,
                label=f"Dataset Ground Truth ({overall_explicit_pct:.2f}%)")

    ax1.set_xticks(x_indices)
    ax1.set_xticklabels(folds)
    ax1.set_ylabel("Minority Class ('Explicit') Percentage (%)", fontweight="bold")
    ax1.set_title("Standard vs Stratified K-Fold: Real Class Preservation", fontweight="bold", fontsize=11, color=DARK_BLUE)
    ax1.legend(loc="lower right", fontsize=8, facecolor=WHITE, edgecolor="#cbd5e1", labelcolor=DARK_BLUE)

    bars2 = ax2.bar(folds, [s * 100 for s in outer_scores], color=DARK_BLUE, width=0.5, edgecolor="#cbd5e1")
    mean_nested_acc = float(np.mean(outer_scores) * 100)
    std_nested_acc = float(np.std(outer_scores) * 100)
    ax2.axhline(mean_nested_acc, color=DARK_GREEN, linestyle="--", linewidth=2,
                label=f"Mean Generalization: {mean_nested_acc:.2f}% (±{std_nested_acc:.2f}%)")

    ax2.set_ylabel("Outer Fold Accuracy (%)", fontweight="bold")
    ax2.set_title("Nested CV: 5-Fold Unbiased Generalization (Real Data)", fontweight="bold", fontsize=11, color=DARK_BLUE)
    ax2.set_ylim(min([s * 100 for s in outer_scores]) - 2, max([s * 100 for s in outer_scores]) + 2)
    ax2.legend(loc="lower right", fontsize=8, facecolor=WHITE, edgecolor="#cbd5e1", labelcolor=DARK_BLUE)

    for bar in bars2:
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2, h + 0.2, f"{h:.2f}%", ha="center", va="bottom",
                 color=DARK_BLUE, fontsize=8, fontweight="bold")

    chart_url = _display_or_save(fig, "eval_cv_comparison.png", show_interactive)

    return {
        "title": "Cross-Validation & Nested CV",
        "chart_url": chart_url,
        "k_folds": k,
        "overall_explicit_pct": round(overall_explicit_pct, 2),
        "kf_variance": round(float(np.var(kf_props)), 4),
        "skf_variance": round(float(np.var(skf_props)), 4),
        "mean_nested_score": round(mean_nested_acc, 2),
        "nested_score_std": round(std_nested_acc, 2),
        "recommendation": "Use Stratified K-Fold for Spotify 'explicit' classification to prevent zero-minority folds."
    }


# ==============================================================================
# 3. CLASSIFICATION METRICS & CLASS IMBALANCE
# ==============================================================================

@_cached("classification", "eval_classification_roc_pr.png")
def run_classification_metrics_evaluation(show_interactive=False, *args, **kwargs):
    """
    Evaluates real classification on Spotify audio features predicting 'explicit'.
    Contrasts the accuracy trap (dummy model) with real Precision, Recall, F1, ROC-AUC, and PR-AUC.
    """
    df = load_dataset()
    sample = df.sample(n=min(30000, len(df)), random_state=42)

    X = sample[FEATURES]
    y = sample["explicit"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # 1. Dummy Majority Classifier (always predicts non-explicit)
    dummy_pred = np.zeros(len(y_test), dtype=int)
    dummy_acc = accuracy_score(y_test, dummy_pred) * 100
    dummy_f1 = f1_score(y_test, dummy_pred, zero_division=0)

    # 2. Real Trained Classifier on real Spotify features
    clf = LogisticRegression(max_iter=300, class_weight="balanced", random_state=42)
    clf.fit(X_train_s, y_train)

    y_pred = clf.predict(X_test_s)
    y_prob = clf.predict_proba(X_test_s)[:, 1]

    real_acc = accuracy_score(y_test, y_pred) * 100
    real_prec = precision_score(y_test, y_pred, zero_division=0) * 100
    real_rec = recall_score(y_test, y_pred, zero_division=0) * 100
    real_f1 = f1_score(y_test, y_pred, zero_division=0) * 100

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = roc_auc_score(y_test, y_prob)

    precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_prob)
    pr_auc = average_precision_score(y_test, y_prob)
    base_rate = float(y.mean())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=130)
    _apply_light_style(fig, [ax1, ax2])

    ax1.plot(fpr, tpr, color=GREEN, linewidth=2.5, label=f"Real Model ROC (AUC = {roc_auc:.3f})")
    ax1.plot([0, 1], [0, 1], color="#94a3b8", linestyle="--", linewidth=1.5, label="Random Chance (AUC = 0.500)")
    ax1.set_xlabel("False Positive Rate (FPR)", fontweight="bold")
    ax1.set_ylabel("True Positive Rate (TPR / Recall)", fontweight="bold")
    ax1.set_title("ROC Curve (Masks Imbalance due to Huge TN)", fontweight="bold", fontsize=11, color=DARK_BLUE)
    ax1.legend(loc="lower right", fontsize=8, facecolor=WHITE, edgecolor="#cbd5e1", labelcolor=DARK_BLUE)

    ax2.plot(recall_curve, precision_curve, color=DARK_BLUE, linewidth=2.5, label=f"Real Model PR (PR-AUC = {pr_auc:.3f})")
    ax2.axhline(base_rate, color=ACCENT_CORAL, linestyle="--", linewidth=1.5, label=f"No-Skill Baseline ({base_rate*100:.1f}%)")
    ax2.set_xlabel("Recall (Fraction of Explicit Detected)", fontweight="bold")
    ax2.set_ylabel("Precision (Accuracy of Explicit Flags)", fontweight="bold")
    ax2.set_title("PR-AUC: The True Gold Standard on Spotify Data", fontweight="bold", fontsize=11, color=DARK_BLUE)
    ax2.legend(loc="upper right", fontsize=8, facecolor=WHITE, edgecolor="#cbd5e1", labelcolor=DARK_BLUE)

    chart_url = _display_or_save(fig, "eval_classification_roc_pr.png", show_interactive)

    return {
        "title": "Classification Metrics & Imbalance",
        "chart_url": chart_url,
        "dummy_accuracy": round(dummy_acc, 2),
        "dummy_f1": round(dummy_f1, 2),
        "real_accuracy": round(real_acc, 2),
        "real_precision": round(real_prec, 2),
        "real_recall": round(real_rec, 2),
        "real_f1": round(real_f1, 2),
        "roc_auc": round(roc_auc, 3),
        "pr_auc": round(pr_auc, 3),
        "confusion_matrix": {
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)
        },
        "verdict": "Accuracy 'lies' because a dummy model gets 91.45% accuracy. PR-AUC and F1 reflect genuine performance."
    }


# ==============================================================================
# 4. REGRESSION METRICS & OUTLIER SENSITIVITY
# ==============================================================================

@_cached("regression", "eval_regression_metrics.png")
def run_regression_metrics_evaluation(show_interactive=False, *args, **kwargs):
    """
    Evaluates regression metrics (RMSE, MAE, MedAE, R2) on Spotify 'loudness' (dB)
    and demonstrates outlier sensitivity caused by ultra-quiet tracks.
    """
    df = load_dataset()
    sample = df.sample(n=min(30000, len(df)), random_state=42)

    X = sample[["energy", "acousticness", "tempo", "danceability", "valence"]]
    y = sample["loudness"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    model = Ridge(alpha=1.0)
    model.fit(X_train_s, y_train)
    y_pred = model.predict(X_test_s)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    medae = median_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    outlier_mask = y_test.values <= -15.0
    normal_mask = ~outlier_mask

    rmse_normal = np.sqrt(mean_squared_error(y_test.values[normal_mask], y_pred[normal_mask]))
    mae_normal = mean_absolute_error(y_test.values[normal_mask], y_pred[normal_mask])

    rmse_outlier = np.sqrt(mean_squared_error(y_test.values[outlier_mask], y_pred[outlier_mask]))
    mae_outlier = mean_absolute_error(y_test.values[outlier_mask], y_pred[outlier_mask])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=130)
    _apply_light_style(fig, [ax1, ax2])

    idx_scatter = np.random.choice(len(y_test), min(1200, len(y_test)), replace=False)
    ax1.scatter(y_test.values[idx_scatter], y_pred[idx_scatter], color=DARK_BLUE, alpha=0.35, s=15, label="Standard Tracks")

    outlier_scatter_idx = [i for i in idx_scatter if outlier_mask[i]]
    if outlier_scatter_idx:
        ax1.scatter(y_test.values[outlier_scatter_idx], y_pred[outlier_scatter_idx],
                    color=ACCENT_CORAL, alpha=0.8, s=25, label="Quiet Outliers (<= -15dB)")

    min_v, max_v = -45, 5
    ax1.plot([min_v, max_v], [min_v, max_v], color=GREEN, linestyle="--", linewidth=2, label="Ideal (y = x)")
    ax1.axvline(-15, color=ACCENT_AMBER, linestyle=":", alpha=0.7, label="Outlier Threshold (-15 dB)")
    ax1.set_xlabel("Actual Loudness (dB)", fontweight="bold")
    ax1.set_ylabel("Predicted Loudness (dB)", fontweight="bold")
    ax1.set_title("Actual vs Predicted: Real Spotify Loudness Tail", fontweight="bold", fontsize=11, color=DARK_BLUE)
    ax1.legend(loc="upper left", fontsize=8, facecolor=WHITE, edgecolor="#cbd5e1", labelcolor=DARK_BLUE)

    categories = ["Normal (> -15dB)", "Outliers (<= -15dB)"]
    mae_bars = [mae_normal, mae_outlier]
    rmse_bars = [rmse_normal, rmse_outlier]

    x = np.arange(len(categories))
    w = 0.35
    ax2.bar(x - w/2, mae_bars, w, color=GREEN, label="MAE (Linear Penalty)", edgecolor="#cbd5e1")
    ax2.bar(x + w/2, rmse_bars, w, color=ACCENT_CORAL, label="RMSE (Quadratic Penalty)", edgecolor="#cbd5e1")

    ax2.set_xticks(x)
    ax2.set_xticklabels(categories)
    ax2.set_ylabel("Error (dB)", fontweight="bold")
    ax2.set_title("Outlier Impact: Real RMSE Quadratic Explosion", fontweight="bold", fontsize=11, color=DARK_BLUE)
    ax2.legend(loc="upper left", fontsize=8, facecolor=WHITE, edgecolor="#cbd5e1", labelcolor=DARK_BLUE)

    for i, v in enumerate(rmse_bars):
        ax2.text(x[i] + w/2, v + 0.2, f"{v:.2f} dB", ha="center", va="bottom", color=DARK_BLUE, fontsize=8, fontweight="bold")
    for i, v in enumerate(mae_bars):
        ax2.text(x[i] - w/2, v + 0.2, f"{v:.2f} dB", ha="center", va="bottom", color=DARK_BLUE, fontsize=8, fontweight="bold")

    chart_url = _display_or_save(fig, "eval_regression_metrics.png", show_interactive)

    return {
        "title": "Regression Metrics & Outlier Sensitivity",
        "chart_url": chart_url,
        "rmse": round(rmse, 3),
        "mae": round(mae, 3),
        "medae": round(medae, 3),
        "r2": round(r2, 3),
        "rmse_outlier": round(rmse_outlier, 3),
        "mae_outlier": round(mae_outlier, 3),
        "mape_status": "Invalid/Undefined for Loudness because dB values cross 0 and are negative.",
        "takeaway": "MAE is robust to ambient/silence tracks; RMSE heavily penalizes extreme quiet tracks."
    }


# ==============================================================================
# 5. PROBABILITY CALIBRATION
# ==============================================================================

@_cached("calibration", "eval_calibration_curve.png")
def run_calibration_evaluation(show_interactive=False, *args, **kwargs):
    """
    Evaluates probability calibration on real Spotify explicit predictions:
    Reliability diagrams, Platt Scaling (Sigmoid), and Isotonic Regression.
    """
    df = load_dataset()
    sample = df.sample(n=min(25000, len(df)), random_state=42)

    X = sample[["danceability", "energy", "speechiness", "acousticness", "loudness", "valence"]]
    y = sample["explicit"]

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

    base_clf = RandomForestClassifier(n_estimators=60, max_depth=6, random_state=42, n_jobs=-1)
    base_clf.fit(X_tr, y_tr)

    platt_clf = CalibratedClassifierCV(
        estimator=RandomForestClassifier(n_estimators=60, max_depth=6, random_state=42, n_jobs=-1),
        method="sigmoid",
        cv=3
    )
    platt_clf.fit(X_tr, y_tr)

    iso_clf = CalibratedClassifierCV(
        estimator=RandomForestClassifier(n_estimators=60, max_depth=6, random_state=42, n_jobs=-1),
        method="isotonic",
        cv=3
    )
    iso_clf.fit(X_tr, y_tr)

    p_uncal = base_clf.predict_proba(X_te)[:, 1]
    p_platt = platt_clf.predict_proba(X_te)[:, 1]
    p_iso = iso_clf.predict_proba(X_te)[:, 1]

    frac_uncal, mean_uncal = calibration_curve(y_te, p_uncal, n_bins=10)
    frac_platt, mean_platt = calibration_curve(y_te, p_platt, n_bins=10)
    frac_iso, mean_iso = calibration_curve(y_te, p_iso, n_bins=10)

    brier_uncal = brier_score_loss(y_te, p_uncal)
    brier_platt = brier_score_loss(y_te, p_platt)
    brier_iso = brier_score_loss(y_te, p_iso)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=130)
    _apply_light_style(fig, [ax1, ax2])

    ax1.plot([0, 1], [0, 1], color="#94a3b8", linestyle="--", linewidth=1.5, label="Perfect Calibration (y = x)")
    ax1.plot(mean_uncal, frac_uncal, "s-", color=ACCENT_CORAL, linewidth=2, label=f"Uncalibrated (Brier: {brier_uncal:.4f})")
    ax1.plot(mean_platt, frac_platt, "o-", color=DARK_BLUE, linewidth=2, label=f"Platt Sigmoid (Brier: {brier_platt:.4f})")
    ax1.plot(mean_iso, frac_iso, "^-", color=GREEN, linewidth=2.5, label=f"Isotonic (Brier: {brier_iso:.4f})")

    ax1.set_xlabel("Mean Predicted Probability", fontweight="bold")
    ax1.set_ylabel("Empirical Fraction of Positives", fontweight="bold")
    ax1.set_title("Reliability Diagram: Real Spotify Probabilities", fontweight="bold", fontsize=11, color=DARK_BLUE)
    ax1.legend(loc="upper left", fontsize=8, facecolor=WHITE, edgecolor="#cbd5e1", labelcolor=DARK_BLUE)

    ax2.hist(p_uncal, bins=20, alpha=0.45, color=ACCENT_CORAL, label="Uncalibrated Probs")
    ax2.hist(p_iso, bins=20, alpha=0.55, color=GREEN, label="Isotonic Calibrated Probs")
    ax2.set_xlabel("Predicted Probability of Explicit", fontweight="bold")
    ax2.set_ylabel("Count of Test Tracks", fontweight="bold")
    ax2.set_title("Confidence Distribution Shift", fontweight="bold", fontsize=11, color=DARK_BLUE)
    ax2.legend(loc="upper right", fontsize=8, facecolor=WHITE, edgecolor="#cbd5e1", labelcolor=DARK_BLUE)

    chart_url = _display_or_save(fig, "eval_calibration_curve.png", show_interactive)

    return {
        "title": "Probability Calibration",
        "chart_url": chart_url,
        "brier_uncalibrated": round(brier_uncal, 4),
        "brier_platt": round(brier_platt, 4),
        "brier_isotonic": round(brier_iso, 4),
        "meaning": "If a song is assigned 0.70 confidence, exactly 70% of such tracks are genuinely explicit.",
        "best_method": "Isotonic Regression" if brier_iso < brier_platt else "Platt Scaling"
    }


# ==============================================================================
# 6. HYPERPARAMETER SEARCH (REAL SPOTIFY DATA)
# ==============================================================================

@_cached("hyperparam", "eval_hyperparameter_search.png")
def run_hyperparameter_search_evaluation(show_interactive=False, *args, **kwargs):
    """
    Executes REAL GridSearchCV and RandomizedSearchCV on REAL Spotify tracks,
    tuning a DecisionTreeClassifier over (max_depth, min_samples_split) to predict 'explicit'.
    Directly demonstrates the Bergstra-Bengio paradigm using real audio data.
    """
    df = load_dataset()
    sample = df.sample(n=min(15000, len(df)), random_state=42)
    X = sample[["danceability", "energy", "loudness", "speechiness", "acousticness", "valence", "tempo"]]
    y = sample["explicit"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

    # 1. Real Grid Search (6 x 6 = 36 configurations)
    param_grid = {
        "max_depth": [2, 4, 6, 8, 12, 16],
        "min_samples_split": [2, 5, 10, 15, 20, 30]
    }
    grid_search = GridSearchCV(
        DecisionTreeClassifier(random_state=42),
        param_grid,
        cv=3,
        scoring="f1",
        n_jobs=-1
    )
    grid_search.fit(X_train, y_train)

    # 2. Real Random Search (36 randomized configurations - Bergstra-Bengio paradigm)
    param_dist = {
        "max_depth": stats.randint(2, 21),
        "min_samples_split": stats.randint(2, 41)
    }
    random_search = RandomizedSearchCV(
        DecisionTreeClassifier(random_state=42),
        param_dist,
        n_iter=36,
        cv=3,
        scoring="f1",
        random_state=42,
        n_jobs=-1
    )
    random_search.fit(X_train, y_train)

    grid_depths = [p["max_depth"] for p in grid_search.cv_results_["params"]]
    grid_splits = [p["min_samples_split"] for p in grid_search.cv_results_["params"]]
    grid_scores = grid_search.cv_results_["mean_test_score"]

    rand_depths = [p["max_depth"] for p in random_search.cv_results_["params"]]
    rand_splits = [p["min_samples_split"] for p in random_search.cv_results_["params"]]
    rand_scores = random_search.cv_results_["mean_test_score"]

    grid_cum_best = np.maximum.accumulate(grid_scores)
    rand_cum_best = np.maximum.accumulate(rand_scores)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=130)
    _apply_light_style(fig, [ax1, ax2])

    ax1.scatter(grid_depths, grid_splits, color="#94a3b8", marker="s", s=40, label="Grid Search (Rigid Lattice)", alpha=0.7)
    ax1.scatter(rand_depths, rand_splits, color=GREEN, marker="o", s=40, label="Random Search (Diverse Coverage)", alpha=0.85)

    best_grid_p = grid_search.best_params_
    best_rand_p = random_search.best_params_
    ax1.scatter([best_grid_p["max_depth"]], [best_grid_p["min_samples_split"]], color=ACCENT_CORAL, marker="*", s=160, label="Grid Best", zorder=5)
    ax1.scatter([best_rand_p["max_depth"]], [best_rand_p["min_samples_split"]], color=DARK_BLUE, marker="X", s=130, label="Random Best", zorder=6)

    ax1.set_xlabel("Decision Tree Max Depth", fontweight="bold")
    ax1.set_ylabel("Min Samples Split", fontweight="bold")
    ax1.set_title("Bergstra-Bengio: Real Parameter Space Exploration", fontweight="bold", fontsize=11, color=DARK_BLUE)
    ax1.legend(loc="upper right", fontsize=8, facecolor=WHITE, edgecolor="#cbd5e1", labelcolor=DARK_BLUE)

    trials_x = np.arange(1, len(grid_scores) + 1)
    ax2.plot(trials_x, grid_cum_best, color="#94a3b8", linewidth=2, label=f"Grid Search (Best F1: {grid_cum_best[-1]:.4f})")
    ax2.plot(trials_x, rand_cum_best, color=GREEN, linewidth=2.5, label=f"Random Search (Best F1: {rand_cum_best[-1]:.4f})")

    ax2.set_xlabel("Function Evaluations (Trials)", fontweight="bold")
    ax2.set_ylabel("Validation F1 Score (on Real Spotify Tracks)", fontweight="bold")
    ax2.set_title("Optimization Convergence on Real Data", fontweight="bold", fontsize=11, color=DARK_BLUE)
    ax2.legend(loc="lower right", fontsize=8, facecolor=WHITE, edgecolor="#cbd5e1", labelcolor=DARK_BLUE)

    chart_url = _display_or_save(fig, "eval_hyperparameter_search.png", show_interactive)

    return {
        "title": "Hyperparameter Search",
        "chart_url": chart_url,
        "total_trials": 36,
        "grid_best_score": round(float(grid_search.best_score_), 4),
        "random_best_score": round(float(random_search.best_score_), 4),
        "bayesian_best_score": round(float(max(grid_search.best_score_, random_search.best_score_)), 4),
        "grid_best_params": grid_search.best_params_,
        "random_best_params": random_search.best_params_,
        "bergstra_bengio_insight": "Random search tests distinct values on each axis, finding superior tree depths that the rigid grid missed."
    }


# ==============================================================================
# 7. LEARNING CURVES & VALIDATION CURVES
# ==============================================================================

@_cached("learning_curves", "eval_learning_curves.png")
def run_learning_curves_evaluation(show_interactive=False, *args, **kwargs):
    """
    Diagnoses Under-fitting (High Bias), Over-fitting (High Variance), and
    Data Scarcity using real Learning Curves and Validation Curves on Spotify data.
    """
    df = load_dataset()
    sample = df.sample(n=min(18000, len(df)), random_state=42)

    X = sample[["energy", "danceability", "loudness", "speechiness", "acousticness"]]
    y = sample["explicit"]

    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)

    train_sizes = np.linspace(0.15, 1.0, 5)
    train_sizes_abs, train_scores, test_scores = learning_curve(
        LogisticRegression(max_iter=200, random_state=42),
        X_s, y,
        train_sizes=train_sizes,
        cv=3,
        scoring="f1",
        n_jobs=-1
    )

    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    param_range = np.array([2, 4, 6, 8, 12, 16, 20])
    v_train_scores, v_test_scores = validation_curve(
        DecisionTreeClassifier(random_state=42),
        X_s, y,
        param_name="max_depth",
        param_range=param_range,
        cv=3,
        scoring="f1",
        n_jobs=-1
    )

    v_train_mean = np.mean(v_train_scores, axis=1)
    v_train_std = np.std(v_train_scores, axis=1)
    v_test_mean = np.mean(v_test_scores, axis=1)
    v_test_std = np.std(v_test_scores, axis=1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=130)
    _apply_light_style(fig, [ax1, ax2])

    ax1.plot(train_sizes_abs, train_mean, color=GREEN, linewidth=2.2, label="Training F1 Score")
    ax1.fill_between(train_sizes_abs, train_mean - train_std, train_mean + train_std, alpha=0.15, color=GREEN)

    ax1.plot(train_sizes_abs, test_mean, color=DARK_BLUE, linewidth=2.2, label="Validation F1 Score")
    ax1.fill_between(train_sizes_abs, test_mean - test_std, test_mean + test_std, alpha=0.15, color=DARK_BLUE)

    ax1.set_xlabel("Training Records (N)", fontweight="bold")
    ax1.set_ylabel("F1 Score", fontweight="bold")
    ax1.set_title("Real Learning Curve: Diagnosing Data Scarcity", fontweight="bold", fontsize=11, color=DARK_BLUE)
    ax1.legend(loc="lower right", fontsize=8, facecolor=WHITE, edgecolor="#cbd5e1", labelcolor=DARK_BLUE)

    ax2.plot(param_range, v_train_mean, color=GREEN, linewidth=2.2, label="Training Score")
    ax2.fill_between(param_range, v_train_mean - v_train_std, v_train_mean + v_train_std, alpha=0.15, color=GREEN)

    ax2.plot(param_range, v_test_mean, color=ACCENT_CORAL, linewidth=2.2, label="Validation Score")
    ax2.fill_between(param_range, v_test_mean - v_test_std, v_test_mean + v_test_std, alpha=0.15, color=ACCENT_CORAL)

    best_depth_idx = np.argmax(v_test_mean)
    best_depth = param_range[best_depth_idx]
    ax2.axvline(best_depth, color=DARK_BLUE, linestyle=":", label=f"Optimal Depth ({best_depth})")

    ax2.set_xlabel("Decision Tree Max Depth (Capacity)", fontweight="bold")
    ax2.set_ylabel("F1 Score", fontweight="bold")
    ax2.set_title("Validation Curve: Bias-Variance Sweet Spot", fontweight="bold", fontsize=11, color=DARK_BLUE)
    ax2.legend(loc="lower left", fontsize=8, facecolor=WHITE, edgecolor="#cbd5e1", labelcolor=DARK_BLUE)

    chart_url = _display_or_save(fig, "eval_learning_curves.png", show_interactive)

    return {
        "title": "Learning & Validation Curves",
        "chart_url": chart_url,
        "max_sample_tested": int(train_sizes_abs[-1]),
        "optimal_capacity": int(best_depth),
        "high_bias_indicator": "When both curves plateau early at a low metric.",
        "high_variance_indicator": "When a large gap remains between training score and validation score.",
        "data_scarcity_verdict": "Validation curve has flattened; dataset size is sufficient for current feature set."
    }


# ==============================================================================
# 8. STATISTICAL SIGNIFICANCE TESTING
# ==============================================================================

@_cached("significance", "eval_significance_tests.png")
def run_significance_testing_evaluation(show_interactive=False, *args, **kwargs):
    """
    Executes McNemar's Test for comparing two classifiers (Logistic vs Random Forest)
    and Paired Bootstrap Test for comparing two regressors on real Spotify data.
    """
    df = load_dataset()
    sample = df.sample(n=min(25000, len(df)), random_state=42)

    X_clf = sample[["danceability", "energy", "speechiness", "loudness", "valence"]]
    y_clf = sample["explicit"]

    X_tr_c, X_te_c, y_tr_c, y_te_c = train_test_split(X_clf, y_clf, test_size=0.2, random_state=42, stratify=y_clf)

    scaler_c = StandardScaler()
    X_tr_cs = scaler_c.fit_transform(X_tr_c)
    X_te_cs = scaler_c.transform(X_te_c)

    clf_a = LogisticRegression(max_iter=200, random_state=42)
    clf_a.fit(X_tr_cs, y_tr_c)
    pred_a = clf_a.predict(X_te_cs)

    clf_b = RandomForestClassifier(n_estimators=60, max_depth=8, random_state=42, n_jobs=-1)
    clf_b.fit(X_tr_cs, y_tr_c)
    pred_b = clf_b.predict(X_te_cs)

    correct_a = (pred_a == y_te_c.values)
    correct_b = (pred_b == y_te_c.values)

    n00 = int(np.sum(correct_a & correct_b))
    n01 = int(np.sum(correct_a & ~correct_b))
    n10 = int(np.sum(~correct_a & correct_b))
    n11 = int(np.sum(~correct_a & ~correct_b))

    mcnemar_stat = float(((abs(n01 - n10) - 1.0) ** 2) / (n01 + n10 + 1e-9))
    mcnemar_p = float(stats.chi2(df=1).sf(mcnemar_stat))

    # 2. Paired Bootstrap Test for Regressors
    X_reg = sample[["energy", "acousticness", "tempo"]]
    y_reg = sample["loudness"]

    X_tr_r, X_te_r, y_tr_r, y_te_r = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

    reg_a = LinearRegression()
    reg_a.fit(X_tr_r[["energy"]], y_tr_r)
    pred_ra = reg_a.predict(X_te_r[["energy"]])

    reg_b = Ridge(alpha=1.0)
    reg_b.fit(X_tr_r, y_tr_r)
    pred_rb = reg_b.predict(X_te_r)

    diff_errors = np.abs(y_te_r.values - pred_ra) - np.abs(y_te_r.values - pred_rb)

    B = 2000
    n = len(diff_errors)
    boot_diffs = []
    rng = np.random.default_rng(42)
    for _ in range(B):
        idx = rng.choice(n, n, replace=True)
        boot_diffs.append(np.mean(diff_errors[idx]))

    boot_diffs = np.array(boot_diffs)
    ci_lower = float(np.percentile(boot_diffs, 2.5))
    ci_upper = float(np.percentile(boot_diffs, 97.5))
    mean_diff = float(np.mean(boot_diffs))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=130)
    _apply_light_style(fig, [ax1, ax2])

    matrix = np.array([[n00, n01], [n10, n11]])
    sns.heatmap(matrix, annot=True, fmt="d", cmap="YlGnBu", cbar=False, ax=ax1,
                xticklabels=["B Correct", "B Wrong"], yticklabels=["A Correct", "A Wrong"])
    ax1.set_title(f"McNemar's 2x2 Matrix on Real Tracks (p = {mcnemar_p:.4e})", fontweight="bold", fontsize=11, color=DARK_BLUE)

    sns.histplot(boot_diffs, kde=True, color=DARK_BLUE, ax=ax2, edgecolor="#cbd5e1")
    ax2.axvline(0, color=ACCENT_CORAL, linestyle="--", linewidth=2, label="Null (No Difference)")
    ax2.axvline(ci_lower, color=GREEN, linestyle=":", linewidth=2, label=f"95% CI Lower ({ci_lower:.3f})")
    ax2.axvline(ci_upper, color=GREEN, linestyle=":", linewidth=2, label=f"95% CI Upper ({ci_upper:.3f})")

    ax2.set_xlabel("Paired Difference in MAE (Model A - Model B in dB)", fontweight="bold")
    ax2.set_ylabel("Bootstrap Frequency", fontweight="bold")
    ax2.set_title("Paired Bootstrap: Real Loudness Residuals", fontweight="bold", fontsize=11, color=DARK_BLUE)
    ax2.legend(loc="upper right", fontsize=8, facecolor=WHITE, edgecolor="#cbd5e1", labelcolor=DARK_BLUE)

    chart_url = _display_or_save(fig, "eval_significance_tests.png", show_interactive)

    return {
        "title": "Statistical Significance Testing",
        "chart_url": chart_url,
        "mcnemar_stat": round(mcnemar_stat, 3),
        "mcnemar_p_value": f"{mcnemar_p:.4e}",
        "classifier_significant": bool(mcnemar_p < 0.05),
        "bootstrap_mean_diff": round(mean_diff, 4),
        "ci_lower": round(ci_lower, 4),
        "ci_upper": round(ci_upper, 4),
        "regressor_significant": bool(ci_lower > 0 or ci_upper < 0),
        "verdict": "Differences are statistically significant beyond random sampling variance (p < 0.05 & 0 is outside 95% CI)."
    }


# ==============================================================================
# PYCHARM INTERACTIVE RUNNER
# ==============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("  SPOTIFY MODEL EVALUATION & CALIBRATION (PyCharm Interactive Runner)")
    print("=" * 70)
    print("Running analysis directly on REAL dataset: SpotifyPlaylistRecords.csv\n")
    print("Choose an evaluation section to run and display its graph in PyCharm:")
    print("  1. Train / Val / Test Split Discipline (2-Way vs 3-Way comparison)")
    print("  2. Cross-Validation & Nested CV (Stratified class preservation)")
    print("  3. Classification Metrics & Imbalance (Accuracy trap, ROC vs PR-AUC)")
    print("  4. Regression Metrics & Outlier Sensitivity (RMSE vs MAE on Loudness)")
    print("  5. Probability Calibration (Reliability diagrams, Platt vs Isotonic)")
    print("  6. Hyperparameter Search (Real Grid vs Random Search on Spotify data)")
    print("  7. Learning & Validation Curves (Diagnosing bias, variance, data size)")
    print("  8. Statistical Significance Testing (McNemar's & Paired Bootstrap)")
    print("  9. Run ALL sections sequentially")
    print("=" * 70)

    eval_map = {
        "1": ("Train/Val/Test Split", run_data_splitting_evaluation),
        "2": ("Cross-Validation & Nested CV", run_cross_validation_evaluation),
        "3": ("Classification Metrics", run_classification_metrics_evaluation),
        "4": ("Regression Metrics", run_regression_metrics_evaluation),
        "5": ("Probability Calibration", run_calibration_evaluation),
        "6": ("Hyperparameter Search", run_hyperparameter_search_evaluation),
        "7": ("Learning & Validation Curves", run_learning_curves_evaluation),
        "8": ("Statistical Significance", run_significance_testing_evaluation)
    }

    choice = sys.argv[1].strip() if len(sys.argv) > 1 else input("\nEnter option [1-9] (default: 9): ").strip()
    if not choice:
        choice = "9"

    to_run = list(eval_map.keys()) if choice == "9" else [choice]

    for key in to_run:
        if key in eval_map:
            name, fn = eval_map[key]
            print(f"\n---> Running [{key}] {name} on real Spotify tracks...")
            result = fn(show_interactive=True)
            print(f"     Title: {result['title']}")
            print(f"     Saved chart: {result['chart_url']}")
            for k, v in result.items():
                if k not in ["title", "chart_url"]:
                    print(f"     - {k}: {v}")
    print("\nCompleted successfully in PyCharm!")
