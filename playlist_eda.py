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

    # 1. DATASET SUMMARY
    rows = len(data)
    columns = len(data.columns)
    basic_info = pd.DataFrame({
        "Data Type": data.dtypes.astype(str),
        "Non Null Count": data.count()
    })

    numeric_summary = (
        data.describe()
        .round(2)
        .to_dict()
    )

    categorical_summary = (
        data.describe(
            include=["object", "string"]
        )
        .fillna("")
        .to_dict()
    )

    # 2. BASIC INFORMATION
    dtypes = data.dtypes.astype(str).to_dict()

    # 4. DUPLICATE ROWS

    duplicates = int(
        data.duplicated().sum()
    )

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


    #Missing values chart

    missing = data.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)

    if len(missing) > 0:
        plt.figure(figsize=(12, 5))

        sns.barplot(
            x=missing.index,
            y=missing.values,
            color="skyblue"
        )

        plt.xticks(rotation=45, ha="right")
        plt.xlabel("Columns")
        plt.ylabel("Number of Missing Values")
        plt.title("Missing Values by Column")

        save_chart("missing_values.png")
        charts.append("missing_values.png")

    # Pairplot

    pairplot_cols = [
        "popularity",
        "danceability",
        "energy",
        "loudness",
        "acousticness",
        "valence"
    ]

    pairplot_cols = [
        col for col in pairplot_cols
        if col in data.columns
    ]

    if len(pairplot_cols) >= 2:
        # Sample data for faster plotting
        pair_data = data[pairplot_cols].dropna().sample(
            min(3000, len(data)),
            random_state=42
        )

        pair_plot = sns.pairplot(
            pair_data,
            diag_kind="hist"
        )

        pair_plot.fig.suptitle(
            "Spotify Audio Features Pairplot",
            y=1.02
        )

        pair_plot.fig.savefig(
            os.path.join(CHART_DIR, "pairplot.png"),
            dpi=120,
            bbox_inches="tight"
        )

        plt.close("all")

        charts.append("pairplot.png")

    #scatter plot

    # Energy vs Popularity
    if "energy" in data.columns and "popularity" in data.columns:
        plt.figure(figsize=(8, 5))

        sns.scatterplot(
            data=data,
            x="energy",
            y="popularity",
            alpha=0.4
        )

        plt.xlabel("Energy")
        plt.ylabel("Popularity")
        plt.title("Energy vs Popularity")

        save_chart("scatter_energy_popularity.png")
        charts.append("scatter_energy_popularity.png")

    # Danceability vs Popularity

    if "danceability" in data.columns and "popularity" in data.columns:
        plt.figure(figsize=(8, 5))

        sns.scatterplot(
            data=data,
            x="danceability",
            y="popularity",
            alpha=0.4
        )

        plt.xlabel("Danceability")
        plt.ylabel("Popularity")
        plt.title("Danceability vs Popularity")

        save_chart("scatter_danceability_popularity.png")
        charts.append("scatter_danceability_popularity.png")

    # Loudness vs Energy

    if "loudness" in data.columns and "energy" in data.columns:
        plt.figure(figsize=(8, 5))

        sns.scatterplot(
            data=data,
            x="loudness",
            y="energy",
            alpha=0.4
        )

        plt.xlabel("Loudness")
        plt.ylabel("Energy")
        plt.title("Loudness vs Energy")

        save_chart("scatter_loudness_energy.png")
        charts.append("scatter_loudness_energy.png")

    # Acousticness vs Popularity

    if "acousticness" in data.columns and "popularity" in data.columns:
        plt.figure(figsize=(8, 5))

        sns.scatterplot(
            data=data,
            x="acousticness",
            y="popularity",
            alpha=0.4
        )

        plt.xlabel("Acousticness")
        plt.ylabel("Popularity")
        plt.title("Acousticness vs Popularity")

        save_chart("scatter_acousticness_popularity.png")
        charts.append("scatter_acousticness_popularity.png")

    # Valence vs Popularity

    if "valence" in data.columns and "popularity" in data.columns:
        plt.figure(figsize=(8, 5))

        sns.scatterplot(
            data=data,
            x="valence",
            y="popularity",
            alpha=0.4
        )

        plt.xlabel("Valence")
        plt.ylabel("Popularity")
        plt.title("Valence vs Popularity")

        save_chart("scatter_valence_popularity.png")
        charts.append("scatter_valence_popularity.png")

    results["rows"] = rows
    results["columns"] = columns
    results["basic_info"] = basic_info.to_dict(orient="index")
    results["numeric_summary"] = numeric_summary
    results["categorical_summary"] = categorical_summary
    results["dtypes"] = dtypes
    results["duplicates"] = duplicates
    results["charts"] = charts
    results["rows"] = data.shape[0]
    results["columns"] = data.shape[1]
    results["duplicates"] = int(data.duplicated().sum())

    return results
