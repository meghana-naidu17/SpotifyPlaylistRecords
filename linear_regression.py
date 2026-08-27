import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
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

def run_gradient_descent(X_scaled, y_scaled, learning_rate=0.1, iterations=100):
    """
    Manually perform gradient descent for Simple Linear Regression (y = theta_0 + theta_1 * x)
    to demonstrate cost history and parameter updates.
    """
    m = len(y_scaled)
    # Initialize parameters far from minimum to show trajectory
    theta_0 = -1.5
    theta_1 = -1.5
    
    cost_history = []
    theta_history = []
    
    for i in range(iterations):
        # Hypothesis
        h = theta_0 + theta_1 * X_scaled
        # Error
        error = h - y_scaled
        # Cost J(theta) = 1/m * sum(error^2)
        cost = (1.0 / m) * np.sum(error ** 2)
        
        cost_history.append(cost)
        theta_history.append((theta_0, theta_1))
        
        # Gradients
        d_theta_0 = (2.0 / m) * np.sum(error)
        d_theta_1 = (2.0 / m) * np.sum(error * X_scaled)
        
        # Updates
        theta_0 = theta_0 - learning_rate * d_theta_0
        theta_1 = theta_1 - learning_rate * d_theta_1
        
    return np.array(theta_history), cost_history

# =========================================================
# PLOTTING FUNCTIONS
# =========================================================

def plot_simple_fit(X, y, model, x_name="Energy", y_name="Loudness (dB)"):
    """
    Plots the scatter plot of the data with the fitted regression line.
    """
    plt.figure(figsize=(8, 5))
    
    # Scatter plot of actual points (subsampled for speed and clarity if large)
    sample_size = min(len(X), 1000)
    indices = np.random.choice(len(X), sample_size, replace=False)
    plt.scatter(X[indices], y[indices], color=GREEN, alpha=0.3, label="Actual Tracks", edgecolors="none")
    
    # Regression line
    x_line = pd.DataFrame(np.linspace(X.min(), X.max(), 100).reshape(-1, 1), columns=["energy"])
    y_line = model.predict(x_line)
    plt.plot(x_line, y_line, color="red", linewidth=3, label=f"Fit: y = {model.intercept_:.2f} + {model.coef_[0]:.2f}*x")
    
    plt.title(f"Simple Linear Regression: {y_name} vs {x_name}", fontsize=14, fontweight="bold", color=DARK_BLUE)
    plt.xlabel(x_name, color=DARK_BLUE)
    plt.ylabel(y_name, color=DARK_BLUE)
    plt.legend()
    plt.tight_layout()
    
    chart_path = os.path.join(CHART_DIR, "linear_simple_fit.png")
    plt.savefig(chart_path, dpi=130, facecolor=WHITE, bbox_inches="tight")
    plt.close()
    return "charts/linear_simple_fit.png"

def plot_gd_cost_history(cost_history):
    """
    Plots the Cost J(theta) vs. Iteration to show convergence.
    """
    plt.figure(figsize=(8, 4))
    plt.plot(range(len(cost_history)), cost_history, color=DARK_BLUE, linewidth=2.5, marker='o', markevery=10)
    plt.title("Gradient Descent: Cost Function Convergence", fontsize=14, fontweight="bold", color=DARK_BLUE)
    plt.xlabel("Iteration / Epoch", color=DARK_BLUE)
    plt.ylabel("Cost J(theta)", color=DARK_BLUE)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    
    chart_path = os.path.join(CHART_DIR, "linear_gd_cost_history.png")
    plt.savefig(chart_path, dpi=130, facecolor=WHITE, bbox_inches="tight")
    plt.close()
    return "charts/linear_gd_cost_history.png"

def plot_gd_contour(X_scaled, y_scaled, theta_history):
    """
    Creates a contour plot of J(theta_0, theta_1) showing the path taken by gradient descent.
    """
    m = len(y_scaled)
    # Define theta grids
    t0_vals = np.linspace(-2.0, 1.0, 100)
    t1_vals = np.linspace(-2.0, 2.0, 100)
    J_vals = np.zeros((len(t0_vals), len(t1_vals)))
    
    for i, t0 in enumerate(t0_vals):
        for j, t1 in enumerate(t1_vals):
            h = t0 + t1 * X_scaled
            J_vals[i, j] = (1.0 / m) * np.sum((h - y_scaled) ** 2)
            
    # Note: transpose to match dimensions of contour plot
    J_vals = J_vals.T
    
    plt.figure(figsize=(8, 6))
    # Contour plot with labels
    contours = plt.contour(t0_vals, t1_vals, J_vals, levels=25, cmap="viridis")
    plt.clabel(contours, inline=True, fontsize=8)
    
    # Plot gradient descent steps
    path_x = theta_history[:, 0]
    path_y = theta_history[:, 1]
    plt.plot(path_x, path_y, "r-o", markersize=4, linewidth=1.5, label="GD Trajectory")
    plt.plot(path_x[0], path_y[0], "bo", markersize=8, label="Start (-1.5, -1.5)")
    plt.plot(path_x[-1], path_y[-1], "g*", markersize=12, label="Converged Min")
    
    plt.title("Error Cost Contour Landscape & GD Steps", fontsize=14, fontweight="bold", color=DARK_BLUE)
    plt.xlabel("Intercept (theta_0)", color=DARK_BLUE)
    plt.ylabel("Slope (theta_1)", color=DARK_BLUE)
    plt.legend()
    plt.tight_layout()
    
    chart_path = os.path.join(CHART_DIR, "linear_gd_trajectory.png")
    plt.savefig(chart_path, dpi=130, facecolor=WHITE, bbox_inches="tight")
    plt.close()
    return "charts/linear_gd_trajectory.png"

def plot_multiple_fit(y_test, y_pred):
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
    
    plt.title("Multiple Linear Regression: Actual vs Predicted Loudness", fontsize=14, fontweight="bold", color=DARK_BLUE)
    plt.xlabel("Actual Loudness (dB)", color=DARK_BLUE)
    plt.ylabel("Predicted Loudness (dB)", color=DARK_BLUE)
    plt.legend()
    plt.tight_layout()
    
    chart_path = os.path.join(CHART_DIR, "linear_multiple_fit.png")
    plt.savefig(chart_path, dpi=130, facecolor=WHITE, bbox_inches="tight")
    plt.close()
    return "charts/linear_multiple_fit.png"

# =========================================================
# MAIN PIPELINE
# =========================================================

def run_linear_regression():
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
    
    simple_model = LinearRegression()
    simple_model.fit(X_train_s, y_train_s)
    
    y_pred_train_s = simple_model.predict(X_train_s)
    y_pred_test_s = simple_model.predict(X_test_s)
    
    mse_train_s = mean_squared_error(y_train_s, y_pred_train_s)
    mse_test_s = mean_squared_error(y_test_s, y_pred_test_s)
    r2_train_s = r2_score(y_train_s, y_pred_train_s)
    r2_test_s = r2_score(y_test_s, y_pred_test_s)
    
    # Generate Simple fit chart
    simple_fit_chart = plot_simple_fit(X_simple["energy"].values, y_simple.values, simple_model)
    
    # ==========================================
    # GRADIENT DESCENT SIMULATION (ON STANDARDIZED DATA)
    # ==========================================
    # Standardize features specifically for the manual GD contour/history visualization
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()
    
    X_scaled = scaler_x.fit_transform(X_simple[["energy"]]).flatten()
    y_scaled = scaler_y.fit_transform(y_simple.values.reshape(-1, 1)).flatten()
    
    theta_history, cost_history = run_gradient_descent(X_scaled, y_scaled, learning_rate=0.1, iterations=100)
    
    # Generate GD charts
    cost_history_chart = plot_gd_cost_history(cost_history)
    gd_trajectory_chart = plot_gd_contour(X_scaled, y_scaled, theta_history)
    
    # ==========================================
    # MULTIPLE LINEAR REGRESSION
    # ==========================================
    # Predictors: energy, acousticness, tempo. Target: loudness
    X_mult = clean_df[["energy", "acousticness", "tempo"]]
    y_mult = clean_df["loudness"]
    
    X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(
        X_mult, y_mult, test_size=0.2, random_state=42
    )
    
    multiple_model = LinearRegression()
    multiple_model.fit(X_train_m, y_train_m)
    
    y_pred_train_m = multiple_model.predict(X_train_m)
    y_pred_test_m = multiple_model.predict(X_test_m)
    
    mse_train_m = mean_squared_error(y_train_m, y_pred_train_m)
    mse_test_m = mean_squared_error(y_test_m, y_pred_test_m)
    r2_train_m = r2_score(y_train_m, y_pred_train_m)
    r2_test_m = r2_score(y_test_m, y_pred_test_m)
    
    # Generate Multiple fit chart
    multiple_fit_chart = plot_multiple_fit(y_test_m, y_pred_test_m)
    
    # Extract coefficients
    coefficients = {
        "energy": round(multiple_model.coef_[0], 4),
        "acousticness": round(multiple_model.coef_[1], 4),
        "tempo": round(multiple_model.coef_[2], 4),
        "intercept": round(multiple_model.intercept_, 4)
    }
    
    return {
        "sample_size": len(clean_df),
        "simple": {
            "intercept": round(simple_model.intercept_, 4),
            "coefficient": round(simple_model.coef_[0], 4),
            "mse_train": round(mse_train_s, 4),
            "mse_test": round(mse_test_s, 4),
            "r2_train": round(r2_train_s, 4),
            "r2_test": round(r2_test_s, 4),
            "fit_chart": simple_fit_chart
        },
        "gd": {
            "initial_cost": round(cost_history[0], 4),
            "final_cost": round(cost_history[-1], 4),
            "cost_chart": cost_history_chart,
            "trajectory_chart": gd_trajectory_chart
        },
        "multiple": {
            "coefficients": coefficients,
            "mse_train": round(mse_train_m, 4),
            "mse_test": round(mse_test_m, 4),
            "r2_train": round(r2_train_m, 4),
            "r2_test": round(r2_test_m, 4),
            "fit_chart": multiple_fit_chart
        }
    }

if __name__ == "__main__":
    res = run_linear_regression()
    print("Regression Results:")
    print("Simple Regression Intercept & Slope:", res["simple"]["intercept"], res["simple"]["coefficient"])
    print("Simple Regression Test MSE:", res["simple"]["mse_test"])
    print("GD convergence cost:", res["gd"]["initial_cost"], "->", res["gd"]["final_cost"])
    print("Multiple Regression coefficients:", res["multiple"]["coefficients"])
    print("Multiple Regression Test MSE:", res["multiple"]["mse_test"])
