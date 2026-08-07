import os
import pandas as pd
import numpy as np

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns

from load_data import load_data
sns.set_style("whitegrid")
sns.set_context("paper")

CHART_DIR = os.path.join("static", "charts")
os.makedirs(CHART_DIR, exist_ok=True)


def save_chart(filename):
    plt.tight_layout()
    plt.savefig(os.path.join(CHART_DIR, filename), dpi=120)
    plt.close()


def run_eda():
    data = load_data()

    results = {}
    charts = []

    numeric_cols = data.select_dtypes(include=np.number).columns

    # Histograms
    for col in numeric_cols:
        try:
            plt.figure(figsize=(7, 4))
            sns.histplot(data[col], bins=20, kde=True)
            plt.title(col)

            safe_col = (
                col.replace(" ", "_")
                   .replace("/", "_")
                   .replace("\\", "_")
                   .replace("%", "Percent")
            )

            filename = f"hist_{safe_col}.png"
            save_chart(filename)
            charts.append(filename)

        except Exception as e:
            print(f"Histogram Error ({col}): {e}")

    # Boxplots
    for col in numeric_cols:
        try:
            plt.figure(figsize=(7, 4))
            sns.boxplot(x=data[col], color="skyblue")
            plt.title(f"Boxplot - {col}")

            safe_col = (
                col.replace(" ", "_")
                   .replace("/", "_")
                   .replace("\\", "_")
                   .replace("%", "Percent")
            )

            filename = f"box_{safe_col}.png"
            save_chart(filename)
            charts.append(filename)

        except Exception as e:
            print(f"Boxplot Error ({col}): {e}")

    # Correlation Heatmap
    corr = data[numeric_cols].corr()

    results["correlation"] = corr.round(2).to_dict()

    plt.figure(figsize=(12, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="PuOr")
    plt.title("Correlation Heatmap")

    save_chart("correlation_heatmap.png")
    charts.append("correlation_heatmap.png")

    # Relationship plots (Numeric vs Genre)
    if "track_genre" in data.columns:

        for col in numeric_cols:

            plt.figure(figsize=(12, 5))

            sns.boxplot(
                data=data,
                x="track_genre",
                y=col
            )

            plt.xticks(rotation=90)
            plt.title(f"{col} vs Track Genre")

            safe_col = (
                col.replace(" ", "_")
                   .replace("/", "_")
                   .replace("\\", "_")
                   .replace("%", "Percent")
            )

            filename = f"relationship_{safe_col}.png"

            save_chart(filename)

            charts.append(filename)

    # Categorical Countplots
    categorical_cols = [
        "explicit",
        "track_genre"
    ]

    categorical_cols = [
        col for col in categorical_cols
        if col in data.columns
    ]

    for col in categorical_cols:

        results[f"{col}_counts"] = data[col].value_counts().to_dict()

        plt.figure(figsize=(10, 5))

        sns.countplot(
            data=data,
            x=col,
            color="skyblue"
        )

        plt.xticks(rotation=45)
        plt.title(f"{col} Count")

        safe_col = (
            col.replace(" ", "_")
               .replace("/", "_")
               .replace("\\", "_")
               .replace("%", "Percent")
        )

        filename = f"count_{safe_col}.png"

        save_chart(filename)

        charts.append(filename)

    results["charts"] = charts
    results["rows"] = data.shape[0]
    results["columns"] = data.shape[1]
    results["duplicates"] = int(data.duplicated().sum())

    return results
