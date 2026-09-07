import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# =========================================================
# PATHS & STYLING
# =========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "SpotifyPlaylistRecords.csv")
CHART_DIR = os.path.join(BASE_DIR, "static", "charts")
os.makedirs(CHART_DIR, exist_ok=True)

# Spotify color scheme
GREEN = "#1DB954"
DARK_GREEN = "#159447"
DARK_BLUE = "#23456b"
WHITE = "#ffffff"
DARK_GRAY = "#121212"

sns.set_theme(style="whitegrid", context="notebook")

# =========================================================
# VALID PENALTY OPTIONS (used by app.py + template dropdown)
# =========================================================

VALID_PENALTIES = ["l2", "l1", "none"]

PENALTY_LABELS = {
    "l2": "L2 (Ridge)",
    "l1": "L1 (Lasso)",
    "none": "No Regularization"
}

# =========================================================
# BUILD LINEAR REGRESSION MODEL FOR A GIVEN PENALTY
# =========================================================

def build_model(penalty="l2"):
    penalty = (penalty or "l2").lower().strip()
    if penalty not in VALID_PENALTIES:
        penalty = "l2"
        
    if penalty == "l1":
        # Lasso penalty
        return Lasso(alpha=0.1, random_state=42)
    elif penalty == "none":
        # Ordinary Least Squares (No regularization)
        return LinearRegression()
    else:
        # L2 Ridge penalty
        return Ridge(alpha=1.0, random_state=42)

# =========================================================
# LOAD DATA
# =========================================================

def load_data():
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Dataset not found: {DATASET_PATH}")
    
    data = pd.read_csv(DATASET_PATH)
    return data

# =========================================================
# GRADIENT DESCENT IMPLEMENTATION (INTUITION)
# =========================================================

def run_gradient_descent(X_scaled, y_scaled, penalty="l2", learning_rate=0.1, iterations=100):
    """
    Manually perform gradient descent for Simple Linear Regression (y = theta_0 + theta_1 * x)
    to demonstrate cost history and parameter updates with or without regularization.
    """
    m = len(y_scaled)
    # Initialize parameters far from minimum to show trajectory
    theta_0 = -1.5
    theta_1 = -1.5
    
    cost_history = []
    theta_history = []
    
    # Regularization strength for normalized scale
    lambda_reg = 0.05
    
    for i in range(iterations):
        # Hypothesis
        h = theta_0 + theta_1 * X_scaled
        # Error
        error = h - y_scaled
        
        # Base MSE cost
        base_cost = (1.0 / m) * np.sum(error ** 2)
        
        if penalty == "l2":
            cost = base_cost + lambda_reg * (theta_1 ** 2)
            d_theta_1_reg = 2.0 * lambda_reg * theta_1
        elif penalty == "l1":
            cost = base_cost + lambda_reg * np.abs(theta_1)
            d_theta_1_reg = lambda_reg * np.sign(theta_1)
        else:
            cost = base_cost
            d_theta_1_reg = 0.0
        
        cost_history.append(cost)
        theta_history.append((theta_0, theta_1))
        
        # Gradients (bias theta_0 is unregularized)
        d_theta_0 = (2.0 / m) * np.sum(error)
        d_theta_1 = (2.0 / m) * np.sum(error * X_scaled) + d_theta_1_reg
        
        # Parameter Updates
        theta_0 = theta_0 - learning_rate * d_theta_0
        theta_1 = theta_1 - learning_rate * d_theta_1
        
    return np.array(theta_history), cost_history

# =========================================================
# PLOTTING FUNCTIONS
# =========================================================

def plot_simple_fit(X, y, model, penalty="l2", x_name="Energy", y_name="Loudness (dB)"):
    """
    Plots the scatter plot of the data with the fitted regression line.
    """
    plt.figure(figsize=(8, 5))
    
    # Scatter plot of actual points (subsampled for speed and clarity)
    sample_size = min(len(X), 1000)
    indices = np.random.choice(len(X), sample_size, replace=False)
    plt.scatter(X[indices], y[indices], color=GREEN, alpha=0.3, label="Actual Tracks", edgecolors="none")
    
    # Regression line
    x_line = pd.DataFrame(np.linspace(X.min(), X.max(), 100).reshape(-1, 1), columns=["energy"])
    y_line = model.predict(x_line)
    label_text = f"Fit ({PENALTY_LABELS.get(penalty, 'Fit')}): y = {float(model.intercept_):.2f} + {float(model.coef_[0]):.2f}*x"
    plt.plot(x_line, y_line, color="red", linewidth=3, label=label_text)
    
    plt.title(f"Simple Linear Regression: {y_name} vs {x_name} ({PENALTY_LABELS.get(penalty, 'L2')})", fontsize=14, fontweight="bold", color=DARK_BLUE)
    plt.xlabel(x_name, color=DARK_BLUE)
    plt.ylabel(y_name, color=DARK_BLUE)
    plt.legend()
    plt.tight_layout()
    
    chart_filename = f"linear_simple_fit_{penalty}.png"
    chart_path = os.path.join(CHART_DIR, chart_filename)
    plt.savefig(chart_path, dpi=130, facecolor=WHITE, bbox_inches="tight")
    plt.close()
    return f"charts/{chart_filename}"

def plot_gd_cost_history(cost_history, penalty="l2"):
    """
    Plots the Cost J(theta) vs. Iteration to show convergence.
    """
    plt.figure(figsize=(8, 4))
    plt.plot(range(len(cost_history)), cost_history, color=DARK_BLUE, linewidth=2.5, marker='o', markevery=10)
    plt.title(f"Gradient Descent: Cost Function Convergence ({PENALTY_LABELS.get(penalty, 'L2')})", fontsize=14, fontweight="bold", color=DARK_BLUE)
    plt.xlabel("Iteration / Epoch", color=DARK_BLUE)
    plt.ylabel("Cost J(theta)", color=DARK_BLUE)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    
    chart_filename = f"linear_gd_cost_history_{penalty}.png"
    chart_path = os.path.join(CHART_DIR, chart_filename)
    plt.savefig(chart_path, dpi=130, facecolor=WHITE, bbox_inches="tight")
    plt.close()
    return f"charts/{chart_filename}"

def plot_gd_contour(X_scaled, y_scaled, theta_history, penalty="l2"):
    """
    Creates a contour plot of J(theta_0, theta_1) showing the path taken by gradient descent.
    """
    m = len(y_scaled)
    # Define theta grids
    t0_vals = np.linspace(-2.0, 1.0, 100)
    t1_vals = np.linspace(-2.0, 2.0, 100)
    J_vals = np.zeros((len(t0_vals), len(t1_vals)))
    
    lambda_reg = 0.05
    for i, t0 in enumerate(t0_vals):
        for j, t1 in enumerate(t1_vals):
            h = t0 + t1 * X_scaled
            ols_cost = (1.0 / m) * np.sum((h - y_scaled) ** 2)
            if penalty == "l2":
                reg_cost = lambda_reg * (t1 ** 2)
            elif penalty == "l1":
                reg_cost = lambda_reg * np.abs(t1)
            else:
                reg_cost = 0.0
            J_vals[i, j] = ols_cost + reg_cost
            
    # Transpose to match dimensions of contour plot
    J_vals = J_vals.T
    
    plt.figure(figsize=(8, 6))
    contours = plt.contour(t0_vals, t1_vals, J_vals, levels=25, cmap="viridis")
    plt.clabel(contours, inline=True, fontsize=8)
    
    # Plot gradient descent steps
    path_x = theta_history[:, 0]
    path_y = theta_history[:, 1]
    plt.plot(path_x, path_y, "r-o", markersize=4, linewidth=1.5, label="GD Trajectory")
    plt.plot(path_x[0], path_y[0], "bo", markersize=8, label="Start (-1.5, -1.5)")
    plt.plot(path_x[-1], path_y[-1], "g*", markersize=12, label="Converged Min")
    
    plt.title(f"Cost Contour Landscape ({PENALTY_LABELS.get(penalty, 'L2')}) & GD Steps", fontsize=14, fontweight="bold", color=DARK_BLUE)
    plt.xlabel("Intercept (theta_0)", color=DARK_BLUE)
    plt.ylabel("Slope (theta_1)", color=DARK_BLUE)
    plt.legend()
    plt.tight_layout()
    
    chart_filename = f"linear_gd_trajectory_{penalty}.png"
    chart_path = os.path.join(CHART_DIR, chart_filename)
    plt.savefig(chart_path, dpi=130, facecolor=WHITE, bbox_inches="tight")
    plt.close()
    return f"charts/{chart_filename}"

def plot_multiple_fit(y_test, y_pred, penalty="l2"):
    """
    Plots Predicted vs. Actual values for multiple linear regression.
    """
    plt.figure(figsize=(8, 5))
    
    # Scatter plot
    sample_size = min(len(y_test), 1000)
    indices = np.random.choice(len(y_test), sample_size, replace=False)
    plt.scatter(y_test.values[indices], y_pred[indices], color=DARK_BLUE, alpha=0.4, label="Predicted vs Actual", edgecolors="none")
    
    # Perfect prediction line (y = x)
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], color="red", linestyle="--", linewidth=2.5, label="Perfect Prediction Reference")
    
    plt.title(f"Multiple Linear Regression ({PENALTY_LABELS.get(penalty, 'L2')}): Actual vs Predicted", fontsize=14, fontweight="bold", color=DARK_BLUE)
    plt.xlabel("Actual Loudness (dB)", color=DARK_BLUE)
    plt.ylabel("Predicted Loudness (dB)", color=DARK_BLUE)
    plt.legend()
    plt.tight_layout()
    
    chart_filename = f"linear_multiple_fit_{penalty}.png"
    chart_path = os.path.join(CHART_DIR, chart_filename)
    plt.savefig(chart_path, dpi=130, facecolor=WHITE, bbox_inches="tight")
    plt.close()
    return f"charts/{chart_filename}"

# =========================================================
# MAIN PIPELINE
# =========================================================

def run_linear_regression(penalty="l2"):
    penalty = (penalty or "l2").lower().strip()
    if penalty not in VALID_PENALTIES:
        penalty = "l2"
        
    # 1. Load dataset
    data = load_data()
    
    # Select feature and target columns (remove NaNs)
    regression_cols = ["energy", "acousticness", "tempo", "loudness"]
    for col in regression_cols:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    
    clean_df = data[regression_cols].dropna()
    
    # ==========================================
    # SIMPLE LINEAR REGRESSION
    # ==========================================
    # Predictor: energy, Target: loudness
    X_simple = clean_df[["energy"]]
    y_simple = clean_df["loudness"]
    
    X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(
        X_simple, y_simple, test_size=0.2, random_state=42
    )
    
    simple_model = build_model(penalty)
    simple_model.fit(X_train_s, y_train_s)
    
    y_pred_train_s = simple_model.predict(X_train_s)
    y_pred_test_s = simple_model.predict(X_test_s)
    
    mse_train_s = mean_squared_error(y_train_s, y_pred_train_s)
    mse_test_s = mean_squared_error(y_test_s, y_pred_test_s)
    r2_train_s = r2_score(y_train_s, y_pred_train_s)
    r2_test_s = r2_score(y_test_s, y_pred_test_s)
    
    # Generate Simple fit chart
    simple_fit_chart = plot_simple_fit(X_simple["energy"].values, y_simple.values, simple_model, penalty=penalty)
    
    # ==========================================
    # GRADIENT DESCENT SIMULATION (ON STANDARDIZED DATA)
    # ==========================================
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()
    
    X_scaled = scaler_x.fit_transform(X_simple[["energy"]]).flatten()
    y_scaled = scaler_y.fit_transform(y_simple.values.reshape(-1, 1)).flatten()
    
    theta_history, cost_history = run_gradient_descent(X_scaled, y_scaled, penalty=penalty, learning_rate=0.1, iterations=100)
    
    # Generate GD charts
    cost_history_chart = plot_gd_cost_history(cost_history, penalty=penalty)
    gd_trajectory_chart = plot_gd_contour(X_scaled, y_scaled, theta_history, penalty=penalty)
    
    # ==========================================
    # MULTIPLE LINEAR REGRESSION
    # ==========================================
    # Predictors: energy, acousticness, tempo. Target: loudness
    X_mult = clean_df[["energy", "acousticness", "tempo"]]
    y_mult = clean_df["loudness"]
    
    X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(
        X_mult, y_mult, test_size=0.2, random_state=42
    )
    
    multiple_model = build_model(penalty)
    multiple_model.fit(X_train_m, y_train_m)
    
    y_pred_train_m = multiple_model.predict(X_train_m)
    y_pred_test_m = multiple_model.predict(X_test_m)
    
    mse_train_m = mean_squared_error(y_train_m, y_pred_train_m)
    mse_test_m = mean_squared_error(y_test_m, y_pred_test_m)
    r2_train_m = r2_score(y_train_m, y_pred_train_m)
    r2_test_m = r2_score(y_test_m, y_pred_test_m)
    
    # Generate Multiple fit chart
    multiple_fit_chart = plot_multiple_fit(y_test_m, y_pred_test_m, penalty=penalty)
    
    # Extract coefficients
    coefficients = {
        "energy": round(float(multiple_model.coef_[0]), 4),
        "acousticness": round(float(multiple_model.coef_[1]), 4),
        "tempo": round(float(multiple_model.coef_[2]), 4),
        "intercept": round(float(multiple_model.intercept_), 4)
    }
    
    return {
        "penalty": penalty,
        "penalty_label": PENALTY_LABELS.get(penalty, "L2 (Ridge)"),
        "sample_size": len(clean_df),
        "simple": {
            "intercept": round(float(simple_model.intercept_), 4),
            "coefficient": round(float(simple_model.coef_[0]), 4),
            "mse_train": round(float(mse_train_s), 4),
            "mse_test": round(float(mse_test_s), 4),
            "r2_train": round(float(r2_train_s), 4),
            "r2_test": round(float(r2_test_s), 4),
            "fit_chart": simple_fit_chart
        },
        "gd": {
            "initial_cost": round(float(cost_history[0]), 4),
            "final_cost": round(float(cost_history[-1]), 4),
            "cost_chart": cost_history_chart,
            "trajectory_chart": gd_trajectory_chart
        },
        "multiple": {
            "coefficients": coefficients,
            "mse_train": round(float(mse_train_m), 4),
            "mse_test": round(float(mse_test_m), 4),
            "r2_train": round(float(r2_train_m), 4),
            "r2_test": round(float(r2_test_m), 4),
            "fit_chart": multiple_fit_chart
        }
    }

if __name__ == "__main__":
    for pen in ["l2", "l1", "none"]:
        res = run_linear_regression(penalty=pen)
        print(f"[{pen.upper()}] Intercept: {res['simple']['intercept']}, Slope: {res['simple']['coefficient']}, Test MSE: {res['simple']['mse_test']}")
