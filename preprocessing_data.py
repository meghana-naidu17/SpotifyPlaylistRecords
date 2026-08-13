import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split

from load_data import load_data

# Numerical columns to process
NUM_COLS = [
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
    "time_signature",
]

# Column used for IQR outlier demonstration
IQR_COL = "danceability"


def run_preprocessing():
    df = load_data()
    results = {}

    # --------------------------------------------------------
    # 1. Dataset shape
    # --------------------------------------------------------
    results["n_rows"]    = int(df.shape[0])
    results["n_columns"] = int(df.shape[1])

    # --------------------------------------------------------
    # 2. Keep only numeric cols that actually exist
    # --------------------------------------------------------
    numeric_cols = [c for c in NUM_COLS if c in df.columns]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    results["numeric_cols"] = numeric_cols

    # --------------------------------------------------------
    # 3. IQR Outlier Detection on danceability
    # --------------------------------------------------------
    iqr_results = {}

    if IQR_COL in df.columns:
        series = df[IQR_COL].dropna()

        Q1 = float(series.quantile(0.25))
        Q3 = float(series.quantile(0.75))
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        n_outliers = int(
            ((series < lower) | (series > upper)).sum()
        )

        iqr_results = {
            "column":      IQR_COL,
            "q1":          round(Q1, 4),
            "q3":          round(Q3, 4),
            "iqr":         round(IQR, 4),
            "lower_bound": round(lower, 4),
            "upper_bound": round(upper, 4),
            "n_outliers":  n_outliers,
            "min_before":  round(float(series.min()), 4),
            "max_before":  round(float(series.max()), 4),
        }

        # Clip outliers
        df["danceability_clipped"] = df[IQR_COL].clip(
            lower=lower, upper=upper
        )

        iqr_results["min_after"] = round(
            float(df["danceability_clipped"].min()), 4
        )
        iqr_results["max_after"] = round(
            float(df["danceability_clipped"].max()), 4
        )

        # Replace danceability with clipped version for scaling
        df[IQR_COL] = df["danceability_clipped"]

    results["iqr"] = iqr_results

    # --------------------------------------------------------
    # 4. Train-Test Split  (70/30)
    # --------------------------------------------------------
    train_df, test_df = train_test_split(
        df, test_size=0.30, random_state=42
    )

    results["split"] = {
        "total":      int(df.shape[0]),
        "train_rows": int(train_df.shape[0]),
        "test_rows":  int(test_df.shape[0]),
        "train_pct":  round(len(train_df) / len(df) * 100, 1),
        "test_pct":   round(len(test_df)  / len(df) * 100, 1),
    }

    # --------------------------------------------------------
    # 5. Min-Max Scaling
    # --------------------------------------------------------
    minmax_scaler = MinMaxScaler()
    train_mm = minmax_scaler.fit_transform(
        train_df[numeric_cols].fillna(0)
    )
    test_mm = minmax_scaler.transform(
        test_df[numeric_cols].fillna(0)
    )

    train_mm_df = pd.DataFrame(
        train_mm, columns=numeric_cols
    ).round(4)
    test_mm_df = pd.DataFrame(
        test_mm, columns=numeric_cols
    ).round(4)

    results["minmax"] = {
        "train_preview": train_mm_df.head(5).to_dict(orient="records"),
        "test_preview":  test_mm_df.head(5).to_dict(orient="records"),
        "train_stats": train_mm_df.describe().round(4).to_dict(),
        "test_stats":  test_mm_df.describe().round(4).to_dict(),
    }

    # --------------------------------------------------------
    # 6. Standard Scaling
    # --------------------------------------------------------
    std_scaler = StandardScaler()
    train_std = std_scaler.fit_transform(
        train_df[numeric_cols].fillna(0)
    )
    test_std = std_scaler.transform(
        test_df[numeric_cols].fillna(0)
    )

    train_std_df = pd.DataFrame(
        train_std, columns=numeric_cols
    ).round(4)
    test_std_df = pd.DataFrame(
        test_std, columns=numeric_cols
    ).round(4)

    results["standard"] = {
        "train_preview": train_std_df.head(5).to_dict(orient="records"),
        "test_preview":  test_std_df.head(5).to_dict(orient="records"),
        "train_stats": train_std_df.describe().round(4).to_dict(),
        "test_stats":  test_std_df.describe().round(4).to_dict(),
    }

    # --------------------------------------------------------
    # 7. Before scaling preview (original values)
    # --------------------------------------------------------
    results["before_scaling"] = (
        train_df[numeric_cols]
        .head(5)
        .round(4)
        .to_dict(orient="records")
    )

    return results
