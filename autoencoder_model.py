"""
Autoencoder module for the Spotify Playlist Analytics project.

Uses a deep symmetric neural network to learn non-linear latent representations
from preprocessed continuous Spotify audio features.
"""

import os
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PREPROCESSED_PATH = os.path.join(
    BASE_DIR, "processed_data", "preprocessed_dataset.csv"
)
CHART_DIR = os.path.join(BASE_DIR, "static", "charts")
os.makedirs(CHART_DIR, exist_ok=True)

MAX_SAMPLE = 1200


def _load_preprocessed():
    if not os.path.exists(PREPROCESSED_PATH):
        raise FileNotFoundError(
            "Preprocessed dataset not found: processed_data/preprocessed_dataset.csv"
        )

    df = pd.read_csv(PREPROCESSED_PATH)

    # Use continuous Spotify audio features
    dummy_prefixes = (
        "track_genre_", "explicit_", "key_", "mode_", "time_signature_"
    )
    feature_cols = [
        c for c in df.columns
        if not c.startswith(dummy_prefixes)
        and c not in ["track_id", "id", "explicit"]
    ]

    numeric = df[feature_cols].select_dtypes(include=np.number).copy()

    # Clean missing / infinity
    numeric = numeric.replace([np.inf, -np.inf], np.nan)
    numeric = numeric.fillna(numeric.median(numeric_only=True))
    numeric = numeric.fillna(0)

    # Remove constant columns
    varying_cols = numeric.columns[numeric.nunique(dropna=False) > 1]
    numeric = numeric[varying_cols]

    if numeric.empty:
        raise ValueError("No varying numeric audio features available for Autoencoder.")

    return numeric


def _make_latent_plot(Z, features, X_raw, filename):
    path = os.path.join(CHART_DIR, filename)
    fig, ax = plt.subplots(figsize=(9, 6), dpi=150)
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#fffbfc")

    # Color by energy if available, else by radius
    if "energy" in features:
        c_vals = X_raw[:, features.index("energy")]
        c_label = "Track Energy (Audio Feature)"
    else:
        c_vals = np.sqrt(Z[:, 0]**2 + Z[:, 1]**2)
        c_label = "Latent Norm Distance"

    scatter = ax.scatter(
        Z[:, 0],
        Z[:, 1],
        c=c_vals,
        cmap="magma",
        s=26,
        alpha=0.78,
        edgecolors="none"
    )

    cbar = fig.colorbar(scatter, ax=ax, pad=0.02)
    cbar.set_label(c_label, fontsize=9, fontweight="bold", color="#5c1f3d")
    cbar.ax.tick_params(labelsize=8)

    ax.set_title(
        "Autoencoder 2D Latent Representation Space (Bottleneck Layer)",
        fontsize=12,
        fontweight="bold",
        color="#49122c",
        pad=12
    )
    ax.set_xlabel("Latent Dimension 1 (Z₁)", fontsize=10, fontweight="semibold", color="#5c1f3d")
    ax.set_ylabel("Latent Dimension 2 (Z₂)", fontsize=10, fontweight="semibold", color="#5c1f3d")
    ax.grid(True, linestyle="--", alpha=0.30, color="#f783ac")

    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _make_loss_plot(loss_curve, filename):
    path = os.path.join(CHART_DIR, filename)
    fig, ax = plt.subplots(figsize=(8.5, 4.4), dpi=150)
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#fffbfc")

    epochs = list(range(1, len(loss_curve) + 1))
    ax.plot(epochs, loss_curve, color="#e86b97", linewidth=2.2, label="Reconstruction MSE Loss")

    # Min loss point
    min_epoch = int(np.argmin(loss_curve)) + 1
    min_loss = float(np.min(loss_curve))
    ax.scatter([min_epoch], [min_loss], color="#d6336c", s=110, zorder=5,
               edgecolors="#ffffff", linewidths=1.8, label=f"Min Loss ({min_loss:.4f})")

    ax.set_title("Autoencoder Training Convergence (Reconstruction Loss)", fontsize=12, fontweight="bold", color="#49122c", pad=12)
    ax.set_xlabel("Epoch / Iteration", fontsize=10, fontweight="semibold", color="#5c1f3d")
    ax.set_ylabel("Mean Squared Error (MSE)", fontsize=10, fontweight="semibold", color="#5c1f3d")
    ax.grid(True, linestyle="--", alpha=0.30, color="#f783ac")
    ax.legend(loc="upper right", fontsize=9, framealpha=0.92)

    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _make_feature_error_plot(feature_errors, filename):
    path = os.path.join(CHART_DIR, filename)
    fig, ax = plt.subplots(figsize=(9, 5), dpi=150)
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#fffbfc")

    # Sort features by error
    sorted_items = sorted(feature_errors.items(), key=lambda x: x[1])
    names = [item[0].replace("_", " ").title() for item in sorted_items]
    errors = [item[1] for item in sorted_items]

    # Color gradient from low error (best preserved) to high error
    colors = plt.cm.spring(np.linspace(0.2, 0.9, len(names)))

    bars = ax.barh(names, errors, color=colors, edgecolor="#f783ac", height=0.65)
    for bar in bars:
        w = bar.get_width()
        ax.text(w + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{w:.3f}", va="center", ha="left", fontsize=8, color="#49122c", fontweight="bold")

    ax.set_title("Per-Feature Reconstruction Error (MSE)", fontsize=12, fontweight="bold", color="#49122c", pad=12)
    ax.set_xlabel("Reconstruction Mean Squared Error", fontsize=10, fontweight="semibold", color="#5c1f3d")
    ax.grid(True, axis="x", linestyle="--", alpha=0.30, color="#f783ac")

    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def run_autoencoder(
    latent_dim=2,
    hidden_dim=32,
    activation="relu",
    epochs=80,
    alpha=0.0001
):
    try:
        latent_dim = int(latent_dim)
    except (ValueError, TypeError):
        latent_dim = 2
    latent_dim = max(2, min(latent_dim, 4))

    try:
        hidden_dim = int(hidden_dim)
    except (ValueError, TypeError):
        hidden_dim = 32
    hidden_dim = max(8, min(hidden_dim, 64))

    try:
        epochs = int(epochs)
    except (ValueError, TypeError):
        epochs = 80
    epochs = max(20, min(epochs, 200))

    try:
        alpha = float(alpha)
    except (ValueError, TypeError):
        alpha = 0.0001

    activation = str(activation).lower().strip()
    if activation not in {"relu", "tanh", "logistic"}:
        activation = "relu"

    numeric = _load_preprocessed()
    feature_names = list(numeric.columns)
    original_rows = len(numeric)

    sample = numeric.sample(n=min(MAX_SAMPLE, len(numeric)), random_state=42)
    scaler = StandardScaler()
    X = scaler.fit_transform(sample.to_numpy(dtype=float))

    # Architecture: Input (D) -> Hidden (H) -> Latent (Z) -> Hidden (H) -> Output (D)
    model = MLPRegressor(
        hidden_layer_sizes=(hidden_dim, latent_dim, hidden_dim),
        activation=activation,
        solver="adam",
        max_iter=epochs,
        alpha=alpha,
        random_state=42,
        tol=1e-4
    )
    model.fit(X, X)

    # Extract 2D latent representation via encoder layers
    W0, b0 = model.coefs_[0], model.intercepts_[0]
    W1, b1 = model.coefs_[1], model.intercepts_[1]

    if activation == "relu":
        H1 = np.maximum(0, X @ W0 + b0)
    elif activation == "tanh":
        H1 = np.tanh(X @ W0 + b0)
    else:
        H1 = 1 / (1 + np.exp(-np.clip(X @ W0 + b0, -20, 20)))

    Z = H1 @ W1 + b1

    # Reconstruction & errors
    X_pred = model.predict(X)
    overall_mse = float(np.mean((X - X_pred)**2))
    total_var = float(np.var(X))
    fidelity_r2 = max(0.0, float(1.0 - (overall_mse / (total_var if total_var > 0 else 1.0))))

    feature_errors = {}
    for j, f_name in enumerate(feature_names):
        feature_errors[f_name] = round(float(np.mean((X[:, j] - X_pred[:, j])**2)), 4)

    # Chart filenames
    latent_chart = f"autoencoder_latent_{latent_dim}_{hidden_dim}_{activation}.png"
    loss_chart = f"autoencoder_loss_{latent_dim}_{hidden_dim}_{activation}.png"
    error_chart = f"autoencoder_features_{latent_dim}_{hidden_dim}_{activation}.png"

    _make_latent_plot(Z, feature_names, X, latent_chart)
    _make_loss_plot(model.loss_curve_, loss_chart)
    _make_feature_error_plot(feature_errors, error_chart)

    # Prepare sorted feature error breakdown for display
    feature_rows = []
    for f_name, err in sorted(feature_errors.items(), key=lambda x: x[1]):
        feature_rows.append({
            "feature": f_name.replace("_", " ").title(),
            "mse": err,
            "preservation": "High" if err < 0.3 else "Medium" if err < 0.7 else "Moderate"
        })

    # Sample points for browser canvas visualization
    points = []
    for i in range(len(Z)):
        points.append({
            "x": round(float(Z[i, 0]), 4),
            "y": round(float(Z[i, 1]), 4)
        })

    return {
        "latent_dim": latent_dim,
        "hidden_dim": hidden_dim,
        "activation": activation,
        "epochs": epochs,
        "actual_epochs": len(model.loss_curve_),
        "alpha": alpha,
        "final_loss": round(overall_mse, 4),
        "fidelity_r2": round(fidelity_r2 * 100, 1),
        "input_features": len(feature_names),
        "features": feature_names,
        "sample_size": len(sample),
        "dataset_records": original_rows,
        "feature_rows": feature_rows,
        "points": points,
        "latent_chart": "charts/" + latent_chart,
        "loss_chart": "charts/" + loss_chart,
        "error_chart": "charts/" + error_chart,
    }
