import os
import pandas as pd
import numpy as np

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns

from load_data import load_data


GREEN = "#1DB954"
DARK_GREEN = "#159447"
DARK_BLUE = "#23456b"
LIGHT_GREEN = "#2ecc71"
LIGHT_GRAY = "#f5f5f5"
WHITE = "#ffffff"


sns.set_theme(
    style="whitegrid",
    context="notebook"
)

CHART_DIR = os.path.join("static", "charts")
os.makedirs(CHART_DIR, exist_ok=True)


def safe_name(column):

    return (
        str(column)
        .strip()
        .replace(" ", "_")
        .replace("/", "_")
        .replace("\\", "_")
        .replace("%", "Percent")
        .replace("(", "")
        .replace(")", "")
    )


def save_chart(filename):

    plt.tight_layout()

    plt.savefig(
        os.path.join(CHART_DIR, filename),
        dpi=130,
        facecolor=WHITE,
        bbox_inches="tight"
    )

    plt.close()


def clean_old_charts():

    if not os.path.exists(CHART_DIR):
        os.makedirs(CHART_DIR)

    for filename in os.listdir(CHART_DIR):

        if filename.lower().endswith(".png"):

            try:
                os.remove(
                    os.path.join(CHART_DIR, filename)
                )
            except Exception:
                pass


def run_eda():

    data = load_data()

    results = {}
    charts = []

    clean_old_charts()

    data.columns = data.columns.astype(str).str.strip()

    numeric_columns = [
        "popularity",
        "duration_ms",
        "danceability",
        "energy",
        "key",
        "loudness",
        "mode",
        "speechiness",
        "acousticness",
        "instrumentalness",
        "liveness",
        "valence",
        "tempo",
        "time_signature"
    ]

    numeric_cols = [
        col
        for col in numeric_columns
        if col in data.columns
    ]

    for col in numeric_cols:

        data[col] = pd.to_numeric(
            data[col],
            errors="coerce"
        )

    if "popularity" in data.columns:

        data["popularity"] = pd.to_numeric(
            data["popularity"],
            errors="coerce"
        )

    if "duration_ms" in data.columns:

        data["duration_ms"] = pd.to_numeric(
            data["duration_ms"],
            errors="coerce"
        )

    results["rows"] = int(data.shape[0])

    results["columns"] = int(data.shape[1])

    results["duplicates"] = int(
        data.duplicated().sum()
    )

    results["missing"] = (
        data.isnull()
        .sum()
        .to_dict()
    )

    results["charts"] = charts

    # =========================
    # HISTOGRAMS
    # =========================

    for col in numeric_cols:

        try:

            values = (
                pd.to_numeric(
                    data[col],
                    errors="coerce"
                )
                .replace(
                    [np.inf, -np.inf],
                    np.nan
                )
                .dropna()
            )

            if values.empty:

                print(
                    f"Skipping {col}: no valid numeric data"
                )

                continue

            if values.nunique() <= 1:

                print(
                    f"Skipping {col}: constant value"
                )

                continue

            plt.figure(
                figsize=(8, 5)
            )

            sns.histplot(
                values,
                bins=25,
                kde=True,
                color=GREEN,
                edgecolor=WHITE
            )

            plt.title(
                f"Distribution of {col}",
                color=DARK_BLUE,
                fontsize=15,
                fontweight="bold"
            )

            plt.xlabel(
                col,
                color=DARK_BLUE
            )

            plt.ylabel(
                "Count",
                color=DARK_BLUE
            )

            plt.grid(
                axis="y",
                alpha=0.25
            )

            filename = (
                f"hist_{safe_name(col)}.png"
            )

            save_chart(filename)

            charts.append(filename)

        except Exception as e:

            print(
                f"Histogram Error ({col}): {e}"
            )

    # =========================
    # BOXPLOTS
    # =========================

    for col in numeric_cols:

        try:

            values = (
                pd.to_numeric(
                    data[col],
                    errors="coerce"
                )
                .replace(
                    [np.inf, -np.inf],
                    np.nan
                )
                .dropna()
            )

            if values.empty:
                continue

            plt.figure(
                figsize=(8, 4)
            )

            sns.boxplot(
                x=values,
                color=GREEN,
                linewidth=1.5
            )

            plt.title(
                f"Boxplot of {col}",
                color=DARK_BLUE,
                fontsize=15,
                fontweight="bold"
            )

            plt.xlabel(
                col,
                color=DARK_BLUE
            )

            filename = (
                f"box_{safe_name(col)}.png"
            )

            save_chart(filename)

            charts.append(filename)

        except Exception as e:

            print(
                f"Boxplot Error ({col}): {e}"
            )

    # =========================
    # CORRELATION HEATMAP
    # =========================

    if len(numeric_cols) > 1:

        try:

            correlation = (
                data[numeric_cols]
                .corr()
            )

            results["correlation"] = (
                correlation
                .round(2)
                .to_dict()
            )

            plt.figure(
                figsize=(14, 10)
            )

            cmap = sns.diverging_palette(
                145,
                260,
                s=75,
                l=45,
                as_cmap=True
            )

            sns.heatmap(
                correlation,
                annot=True,
                fmt=".2f",
                cmap=cmap,
                linewidths=0.5,
                square=False
            )

            plt.title(
                "Correlation Heatmap",
                color=DARK_BLUE,
                fontsize=17,
                fontweight="bold"
            )

            save_chart(
                "correlation_heatmap.png"
            )

            charts.append(
                "correlation_heatmap.png"
            )

        except Exception as e:

            print(
                f"Correlation Error: {e}"
            )

    # =========================
    # EXPLICIT COUNT
    # =========================

    if "explicit" in data.columns:

        try:

            explicit_counts = (
                data["explicit"]
                .fillna("Unknown")
                .astype(str)
                .value_counts()
            )

            results["explicit_counts"] = (
                explicit_counts
                .to_dict()
            )

            plt.figure(
                figsize=(7, 5)
            )

            sns.countplot(
                data=data,
                x="explicit",
                color=GREEN
            )

            plt.title(
                "Explicit vs Non-Explicit Tracks",
                color=DARK_BLUE,
                fontsize=15,
                fontweight="bold"
            )

            plt.xlabel(
                "Explicit",
                color=DARK_BLUE
            )

            plt.ylabel(
                "Number of Tracks",
                color=DARK_BLUE
            )

            filename = "count_explicit.png"

            save_chart(filename)

            charts.append(filename)

        except Exception as e:

            print(
                f"Explicit Error: {e}"
            )

    # =========================
    # TRACK GENRE
    # =========================

    if "track_genre" in data.columns:

        try:

            genre_counts = (
                data["track_genre"]
                .fillna("Unknown")
                .value_counts()
            )

            results["track_genre_counts"] = (
                genre_counts
                .to_dict()
            )

            top_genres = (
                genre_counts
                .head(20)
                .sort_values(ascending=True)
            )

            plt.figure(
                figsize=(10, 8)
            )

            sns.barplot(
                x=top_genres.values,
                y=top_genres.index,
                color=GREEN
            )

            plt.title(
                "Top 20 Track Genres",
                color=DARK_BLUE,
                fontsize=16,
                fontweight="bold"
            )

            plt.xlabel(
                "Number of Tracks",
                color=DARK_BLUE
            )

            plt.ylabel(
                "Track Genre",
                color=DARK_BLUE
            )

            filename = (
                "count_track_genre.png"
            )

            save_chart(filename)

            charts.append(filename)

        except Exception as e:

            print(
                f"Genre Error: {e}"
            )

    # =========================
    # POPULARITY VS DANCEABILITY
    # =========================

    if (
        "popularity" in data.columns
        and "danceability" in data.columns
    ):

        try:

            plot_data = data[
                [
                    "popularity",
                    "danceability"
                ]
            ].dropna()

            if not plot_data.empty:

                plt.figure(
                    figsize=(8, 5)
                )

                sns.scatterplot(
                    data=plot_data,
                    x="danceability",
                    y="popularity",
                    color=GREEN,
                    alpha=0.5,
                    s=25
                )

                plt.title(
                    "Popularity vs Danceability",
                    color=DARK_BLUE,
                    fontsize=15,
                    fontweight="bold"
                )

                plt.xlabel(
                    "Danceability",
                    color=DARK_BLUE
                )

                plt.ylabel(
                    "Popularity",
                    color=DARK_BLUE
                )

                filename = (
                    "relationship_popularity_danceability.png"
                )

                save_chart(filename)

                charts.append(filename)

        except Exception as e:

            print(
                f"Popularity Relationship Error: {e}"
            )

    # =========================
    # POPULARITY VS ENERGY
    # =========================

    if (
        "popularity" in data.columns
        and "energy" in data.columns
    ):

        try:

            plot_data = data[
                [
                    "popularity",
                    "energy"
                ]
            ].dropna()

            if not plot_data.empty:

                plt.figure(
                    figsize=(8, 5)
                )

                sns.scatterplot(
                    data=plot_data,
                    x="energy",
                    y="popularity",
                    color=GREEN,
                    alpha=0.5,
                    s=25
                )

                plt.title(
                    "Popularity vs Energy",
                    color=DARK_BLUE,
                    fontsize=15,
                    fontweight="bold"
                )

                plt.xlabel(
                    "Energy",
                    color=DARK_BLUE
                )

                plt.ylabel(
                    "Popularity",
                    color=DARK_BLUE
                )

                filename = (
                    "relationship_popularity_energy.png"
                )

                save_chart(filename)

                charts.append(filename)

        except Exception as e:

            print(
                f"Energy Relationship Error: {e}"
            )

    # =========================
    # FINAL RESULTS
    # =========================

    results["charts"] = charts

    results["numeric_columns"] = numeric_cols

    return results