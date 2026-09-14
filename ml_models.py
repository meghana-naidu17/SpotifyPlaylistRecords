"""Classification and clustering models used by the Flask UI."""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "SpotifyPlaylistRecords.csv")
CHART_DIR = os.path.join(BASE_DIR, "static", "charts")
os.makedirs(CHART_DIR, exist_ok=True)

FEATURES = [
    "popularity", "duration_ms", "danceability", "energy", "key", "loudness",
    "mode", "speechiness", "acousticness", "instrumentalness", "liveness",
    "valence", "tempo", "time_signature",
]


def _classification_data():
    """Return the project's numeric audio features and binary explicit target."""
    data = pd.read_csv(DATASET_PATH)
    if "explicit" not in data.columns:
        raise ValueError("The dataset must contain the 'explicit' target column.")
    X = data[FEATURES].apply(pd.to_numeric, errors="coerce")
    y = data["explicit"]
    if y.dtype == object:
        y = y.astype(str).str.strip().str.lower().map({
            "true": 1, "false": 0, "yes": 1, "no": 0, "1": 1, "0": 0
        })
    else:
        y = pd.to_numeric(y, errors="coerce")
    clean = pd.concat([X, y.rename("explicit")], axis=1).dropna()
    return clean[FEATURES], clean["explicit"].astype(int)


def _save_confusion_matrix(matrix, labels, filename, title):
    path = os.path.join(CHART_DIR, filename)
    fig, ax = plt.subplots(figsize=(6, 4.5))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Greens", cbar=False,
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("Actual label")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _save_tree_plot(model, filename, title):
    """Render a readable top section of a fitted tree, rather than an unreadable full tree."""
    path = os.path.join(CHART_DIR, filename)
    fig, ax = plt.subplots(figsize=(22, 10))
    plot_tree(model, feature_names=FEATURES, class_names=["Not Explicit", "Explicit"],
              filled=True, rounded=True, max_depth=3, fontsize=8, ax=ax)
    ax.set_title(title + " (first 4 levels)", fontsize=16, pad=18)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _build_booster(name):
    if name == "adaboost":
        return AdaBoostClassifier(n_estimators=100, random_state=42), "AdaBoost"
    if name == "gradient_boost":
        return GradientBoostingClassifier(n_estimators=100, random_state=42), "Gradient Boosting"
    if name == "lightgbm":
        try:
            from lightgbm import LGBMClassifier
            return LGBMClassifier(n_estimators=150, random_state=42, verbosity=-1), "LightGBM"
        except ImportError:
            # Keeps the page usable in installations where optional LightGBM is absent.
            return HistGradientBoostingClassifier(max_iter=150, random_state=42), "LightGBM (compatible fallback)"
    if name == "xgboost":
        try:
            from xgboost import XGBClassifier
            return XGBClassifier(n_estimators=150, max_depth=6, learning_rate=0.1,
                                 random_state=42, eval_metric="logloss"), "XGBoost"
        except ImportError:
            return HistGradientBoostingClassifier(max_iter=150, random_state=42), "XGBoost (compatible fallback)"
    raise ValueError("Unsupported boosting algorithm.")


def run_classifier(family, algorithm):
    """Train one requested classification model on an 80/20 stratified split."""
    X, y = _classification_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    tree_for_plot = None
    if family == "tree":
        if algorithm == "id3":
            # Entropy is the information-gain criterion used by ID3.
            model, name, method = DecisionTreeClassifier(criterion="entropy", random_state=42), "ID3 Decision Tree", "Information gain (entropy)"
        elif algorithm == "cart":
            model, name, method = DecisionTreeClassifier(criterion="gini", random_state=42), "CART Decision Tree", "Gini index"
        else:
            raise ValueError("Unsupported decision-tree algorithm.")
        tree_for_plot = model
    elif family == "bagging" and algorithm == "random_forest":
        model, name, method = RandomForestClassifier(n_estimators=150, random_state=42, n_jobs=-1), "Random Forest", "Bagging ensemble"
    elif family == "boosting":
        model, name = _build_booster(algorithm)
        method = "Boosting ensemble"
    else:
        raise ValueError("Unsupported algorithm.")

    model.fit(X_train, y_train)
    train_prediction = model.predict(X_train)
    test_prediction = model.predict(X_test)
    matrix = confusion_matrix(y_test, test_prediction, labels=[0, 1])
    chart_name = "{}_{}_confusion_matrix.png".format(family, algorithm)
    _save_confusion_matrix(matrix, ["Not Explicit", "Explicit"], chart_name, name + " — Confusion Matrix")
    report = classification_report(
        y_test, test_prediction, labels=[0, 1], target_names=["Not Explicit", "Explicit"],
        output_dict=True, zero_division=0
    )
    tree_chart = None
    if tree_for_plot is not None:
        tree_chart = "tree_{}_plot.png".format(algorithm)
        _save_tree_plot(model, tree_chart, name)
    elif family == "bagging":
        tree_chart = "random_forest_tree_plot.png"
        _save_tree_plot(model.estimators_[0], tree_chart, "Random Forest — estimator 1 of {}".format(len(model.estimators_)))
    return {
        "name": name, "method": method, "family": family, "algorithm": algorithm,
        "features": FEATURES, "training_records": len(X_train), "testing_records": len(X_test),
        "train_accuracy": round(accuracy_score(y_train, train_prediction) * 100, 2),
        "test_accuracy": round(accuracy_score(y_test, test_prediction) * 100, 2),
        "classification_report": report, "confusion_matrix": matrix.tolist(),
        "confusion_chart": "charts/" + chart_name,
        "tree_chart": "charts/" + tree_chart if tree_chart else None,
    }


def _clustering_data():
    data = pd.read_csv(DATASET_PATH)
    X = data[FEATURES].apply(pd.to_numeric, errors="coerce").dropna()
    return StandardScaler().fit_transform(X), len(X)


def _save_cluster_plot(X, labels, centers, k, filename):
    """Plot clusters on two standardized Spotify features with their centroids."""
    path = os.path.join(CHART_DIR, filename)
    # A sample keeps the visual readable and avoids an oversized image file.
    sample = min(12000, len(X))
    indices = np.random.default_rng(42).choice(len(X), sample, replace=False)
    x_index, y_index = FEATURES.index("danceability"), FEATURES.index("energy")
    fig, ax = plt.subplots(figsize=(9, 6.5))
    scatter = ax.scatter(X[indices, x_index], X[indices, y_index], c=labels[indices],
                         cmap="viridis", s=12, alpha=.72, linewidths=0)
    ax.scatter(centers[:, x_index], centers[:, y_index], c="#ff3b30", marker="X",
               s=220, edgecolors="black", linewidths=1.2, label="Cluster centers", zorder=3)
    ax.set_title("K-Means Clusters (K={})".format(k))
    ax.set_xlabel("Danceability (standardized)")
    ax.set_ylabel("Energy (standardized)")
    fig.colorbar(scatter, ax=ax, label="Cluster")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _select_elbow_k(k_values, wcss):
    """Select the point farthest from the endpoint line of the WCSS curve."""
    points = np.column_stack((k_values, wcss)).astype(float)
    start, end = points[0], points[-1]
    line = end - start
    offsets = points - start
    # 2-D perpendicular distance to the endpoint line (avoids NumPy's 3-D cross-product requirement).
    distances = np.abs(line[0] * offsets[:, 1] - line[1] * offsets[:, 0]) / np.linalg.norm(line)
    return int(k_values[int(np.argmax(distances))])


def run_kmeans(method="elbow", k=3):
    """Calculate K-means and its graph; clustering deliberately has no target label."""
    X, records = _clustering_data()
    method = method if method in {"manual", "elbow", "silhouette"} else "elbow"
    k = max(2, min(int(k), 10))
    chart_name = "kmeans_{}_chart.png".format(method)
    chart_path = os.path.join(CHART_DIR, chart_name)
    result = {"method": method, "records": records, "selected_k": k, "chart": "charts/" + chart_name}

    if method == "manual":
        selector_title = "Manual K-Means Clustering"
    else:
        values = list(range(2, 11))
        if method == "elbow":
            scores = [KMeans(n_clusters=value, random_state=42, n_init=10).fit(X).inertia_ for value in values]
            ylabel, title = "WCSS (Inertia)", "K-Means Elbow Method"
            result["selected_k"] = _select_elbow_k(values, scores)
        else:
            sample_size = min(5000, len(X))
            scores = []
            for value in values:
                labels = KMeans(n_clusters=value, random_state=42, n_init=10).fit_predict(X)
                scores.append(silhouette_score(X, labels, sample_size=sample_size, random_state=42))
            result["selected_k"] = values[int(np.argmax(scores))]
            ylabel, title = "Silhouette Score", "K-Means Silhouette Method"
        result["values"] = [{"k": value, "score": round(float(score), 4)} for value, score in zip(values, scores)]
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.plot(values, scores, marker="o", color="#1db954", linewidth=2)
        ax.set_xticks(values); ax.set_xlabel("Number of clusters (K)"); ax.set_ylabel(ylabel); ax.set_title(title); ax.grid(alpha=.25)
        fig.tight_layout(); fig.savefig(chart_path, dpi=150); plt.close(fig)

    fitted_model = KMeans(n_clusters=result["selected_k"], random_state=42, n_init=10).fit(X)
    labels = fitted_model.labels_
    result["cluster_counts"] = pd.Series(labels).value_counts().sort_index().to_dict()
    result["inertia"] = round(float(fitted_model.inertia_), 2)
    cluster_name = "kmeans_{}_clusters.png".format(method)
    _save_cluster_plot(X, labels, fitted_model.cluster_centers_, result["selected_k"], cluster_name)
    result["cluster_chart"] = "charts/" + cluster_name
    if method == "manual":
        result["chart"] = result["cluster_chart"]
    return result
