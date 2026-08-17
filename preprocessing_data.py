import os
import pandas as pd
import numpy as np

from sklearn.preprocessing import (
    MinMaxScaler,
    StandardScaler,
    OneHotEncoder,
    OrdinalEncoder
)

from sklearn.model_selection import train_test_split

from load_data import load_data


# ============================================================
# NUMERICAL COLUMNS
# ============================================================

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

CATEGORICAL_COLS = ["track_genre"]
ORDINAL_COLS     = []
TARGET_COL       = "popularity"
IQR_COL          = "danceability"


# ============================================================
# HELPER — missing values analysis
# ============================================================

def _missing_value_analysis(df, numeric_cols):
    """
    Returns a dict with per-column missing stats and
    results of four handling strategies:
      1. row-wise deletion
      2. column-wise deletion
      3. mean imputation
      4. median imputation
    """

    total_rows = len(df)

    # Per-column missing info
    col_info = {}
    for col in df.columns:
        n_miss = int(df[col].isnull().sum())
        col_info[col] = {
            "missing": n_miss,
            "pct":     round(n_miss / total_rows * 100, 4),
            "dtype":   str(df[col].dtype),
        }

    missing_cols = {c: v for c, v in col_info.items() if v["missing"] > 0}
    total_missing_cells = sum(v["missing"] for v in col_info.values())
    rows_with_any_missing = int(df.isnull().any(axis=1).sum())

    # ---- 1. Row-wise deletion ----
    df_rowdrop = df.dropna()
    rows_dropped  = total_rows - len(df_rowdrop)
    rowdrop_pct   = round(rows_dropped / total_rows * 100, 4)
    rowdrop_applicable = rows_dropped < total_rows * 0.05  # < 5% loss

    rowdrop_result = {
        "applicable":     rowdrop_applicable,
        "rows_before":    total_rows,
        "rows_after":     int(len(df_rowdrop)),
        "rows_dropped":   rows_dropped,
        "pct_dropped":    rowdrop_pct,
        "reason": (
            f"Only {rows_dropped} row(s) ({rowdrop_pct}%) contain missing values. "
            "Dropping them causes negligible data loss — row-wise deletion is safe."
            if rowdrop_applicable else
            f"{rows_dropped} rows ({rowdrop_pct}%) would be lost. "
            "This is too high a data loss for row-wise deletion to be appropriate."
        ),
        "preview": df_rowdrop.head(3).fillna("").astype(str).to_dict(orient="records"),
    }

    # ---- 2. Column-wise deletion ----
    # Drop columns where missing% > 30% (industry rule of thumb)
    THRESHOLD = 30.0
    cols_to_drop = [
        c for c, v in col_info.items()
        if v["missing"] > 0 and v["pct"] > THRESHOLD
    ]
    cols_not_dropped = [
        c for c, v in col_info.items()
        if v["missing"] > 0 and v["pct"] <= THRESHOLD
    ]
    coldrop_applicable = len(cols_to_drop) > 0

    per_col_reasons = {}
    for c, v in col_info.items():
        if v["missing"] == 0:
            continue
        if v["pct"] > THRESHOLD:
            per_col_reasons[c] = {
                "drop": True,
                "reason": f"{v['pct']}% missing — exceeds {THRESHOLD}% threshold. Column dropped."
            }
        else:
            per_col_reasons[c] = {
                "drop": False,
                "reason": (
                    f"Only {v['pct']}% missing ({v['missing']} value(s)). "
                    f"Below {THRESHOLD}% threshold — column retained, use imputation instead."
                )
            }

    coldrop_result = {
        "applicable":      coldrop_applicable,
        "threshold_pct":   THRESHOLD,
        "cols_dropped":    cols_to_drop,
        "cols_retained":   cols_not_dropped,
        "per_col_reasons": per_col_reasons,
        "reason": (
            f"Columns with >{THRESHOLD}% missing data: {cols_to_drop}. These were dropped." 
            if coldrop_applicable else
            f"No column exceeds the {THRESHOLD}% missing threshold. "
            "Column-wise deletion is NOT applied — all columns are retained. "
            "Use row-wise deletion or imputation to handle the small number of missing values."
        ),
    }

    # ---- 3. Mean imputation (numeric columns only) ----
    mean_results = []
    df_mean = df.copy()
    for col in numeric_cols:
        n = int(df_mean[col].isnull().sum())
        if n > 0:
            mean_val = round(float(df_mean[col].mean()), 4)
            df_mean[col] = df_mean[col].fillna(mean_val)
            mean_results.append({
                "column":    col,
                "n_filled":  n,
                "fill_value": mean_val,
                "applicable": True,
                "reason":    f"Numeric column — {n} missing value(s) replaced with mean ({mean_val})."
            })

    # Non-numeric cols with missing values
    for col, v in col_info.items():
        if v["missing"] > 0 and col not in numeric_cols:
            mean_results.append({
                "column":    col,
                "n_filled":  v["missing"],
                "fill_value": None,
                "applicable": False,
                "reason":    (
                    f"Non-numeric column (dtype: {v['dtype']}) — mean imputation is not applicable. "
                    "Use mode imputation or a placeholder string instead."
                )
            })

    mean_imputation = {
        "results": mean_results,
        "preview": df_mean[numeric_cols].head(5).round(4).to_dict(orient="records")
                   if numeric_cols else [],
    }

    # ---- 4. Median imputation (numeric columns only) ----
    median_results = []
    df_median = df.copy()
    for col in numeric_cols:
        n = int(df_median[col].isnull().sum())
        if n > 0:
            median_val = round(float(df_median[col].median()), 4)
            df_median[col] = df_median[col].fillna(median_val)
            median_results.append({
                "column":    col,
                "n_filled":  n,
                "fill_value": median_val,
                "applicable": True,
                "reason":    f"Numeric column — {n} missing value(s) replaced with median ({median_val})."
            })

    for col, v in col_info.items():
        if v["missing"] > 0 and col not in numeric_cols:
            median_results.append({
                "column":    col,
                "n_filled":  v["missing"],
                "fill_value": None,
                "applicable": False,
                "reason":    (
                    f"Non-numeric column (dtype: {v['dtype']}) — median imputation is not applicable. "
                    "Use mode imputation or a placeholder string instead."
                )
            })

    median_imputation = {
        "results": median_results,
        "preview": df_median[numeric_cols].head(5).round(4).to_dict(orient="records")
                   if numeric_cols else [],
    }

    return {
        "total_rows":          total_rows,
        "total_missing_cells": total_missing_cells,
        "rows_with_missing":   rows_with_any_missing,
        "col_info":            col_info,
        "missing_cols":        missing_cols,
        "rowdrop":             rowdrop_result,
        "coldrop":             coldrop_result,
        "mean_imputation":     mean_imputation,
        "median_imputation":   median_imputation,
    }


def run_preprocessing():

    df = load_data()

    results = {}


    # ========================================================
    # 1. DATASET SHAPE
    # ========================================================

    results["n_rows"] = int(df.shape[0])
    results["n_columns"] = int(df.shape[1])

    results["all_columns"] = df.columns.tolist()


    # ========================================================
    # 2. NUMERICAL COLUMNS
    # ========================================================

    numeric_cols = [
        c for c in NUM_COLS
        if c in df.columns
    ]

    for col in numeric_cols:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    results["numeric_cols"] = numeric_cols


    # ========================================================
    # 3. CATEGORICAL COLUMNS
    # ========================================================

    categorical_cols = [
        c for c in CATEGORICAL_COLS
        if c in df.columns
    ]

    results["categorical_cols"] = categorical_cols


    # ========================================================
    # 4. IQR OUTLIER DETECTION
    # ========================================================

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

            "column": IQR_COL,

            "q1": round(Q1, 4),

            "q3": round(Q3, 4),

            "iqr": round(IQR, 4),

            "lower_bound": round(lower, 4),

            "upper_bound": round(upper, 4),

            "n_outliers": n_outliers,

            "min_before":
                round(float(series.min()), 4),

            "max_before":
                round(float(series.max()), 4)
        }


        # Clip outliers

        df["danceability_clipped"] = df[
            IQR_COL
        ].clip(
            lower=lower,
            upper=upper
        )


        iqr_results["min_after"] = round(
            float(
                df["danceability_clipped"].min()
            ),
            4
        )

        iqr_results["max_after"] = round(
            float(
                df["danceability_clipped"].max()
            ),
            4
        )


        # Replace original column

        df[IQR_COL] = df[
            "danceability_clipped"
        ]


    results["iqr"] = iqr_results


    # ========================================================
    # 5. TRAIN TEST SPLIT
    # ========================================================

    train_df, test_df = train_test_split(
        df,
        test_size=0.30,
        random_state=42
    )

    train_df = train_df.copy()
    test_df = test_df.copy()


    results["split"] = {

        "total":
            int(df.shape[0]),

        "train_rows":
            int(train_df.shape[0]),

        "test_rows":
            int(test_df.shape[0]),

        "train_pct":
            round(
                len(train_df)
                / len(df) * 100,
                1
            ),

        "test_pct":
            round(
                len(test_df)
                / len(df) * 100,
                1
            )
    }


    # ========================================================
    # 6. ONE-HOT ENCODING
    # ========================================================

    if len(categorical_cols) > 0:

        onehot_encoder = OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False
        )

        train_onehot = onehot_encoder.fit_transform(
            train_df[categorical_cols].fillna("Missing")
        )

        test_onehot = onehot_encoder.transform(
            test_df[categorical_cols].fillna("Missing")
        )


        onehot_feature_names = (
            onehot_encoder
            .get_feature_names_out(
                categorical_cols
            )
        )


        train_onehot_df = pd.DataFrame(
            train_onehot,
            columns=onehot_feature_names,
            index=train_df.index
        ).round(4)


        test_onehot_df = pd.DataFrame(
            test_onehot,
            columns=onehot_feature_names,
            index=test_df.index
        ).round(4)


        results["onehot"] = {

            "columns":
                list(onehot_feature_names),

            "train_preview":
                train_onehot_df
                .head(5)
                .to_dict(
                    orient="records"
                ),

            "test_preview":
                test_onehot_df
                .head(5)
                .to_dict(
                    orient="records"
                ),

            "number_of_features":
                len(onehot_feature_names)
        }


    # ========================================================
    # 7. ORDINAL ENCODING
    # ========================================================

    if len(ORDINAL_COLS) > 0:

        ordinal_cols = [
            c for c in ORDINAL_COLS
            if c in df.columns
        ]

        if len(ordinal_cols) > 0:

            ordinal_encoder = OrdinalEncoder(
                handle_unknown="use_encoded_value",
                unknown_value=-1
            )


            train_ordinal = (
                ordinal_encoder.fit_transform(
                    train_df[
                        ordinal_cols
                    ].fillna("Missing")
                )
            )


            test_ordinal = (
                ordinal_encoder.transform(
                    test_df[
                        ordinal_cols
                    ].fillna("Missing")
                )
            )


            train_ordinal_df = pd.DataFrame(
                train_ordinal,
                columns=ordinal_cols,
                index=train_df.index
            ).round(4)


            test_ordinal_df = pd.DataFrame(
                test_ordinal,
                columns=ordinal_cols,
                index=test_df.index
            ).round(4)


            results["ordinal"] = {

                "columns":
                    ordinal_cols,

                "train_preview":
                    train_ordinal_df
                    .head(5)
                    .to_dict(
                        orient="records"
                    ),

                "test_preview":
                    test_ordinal_df
                    .head(5)
                    .to_dict(
                        orient="records"
                    )
            }

    # ========================================================
    # TARGET ENCODING
    # Prediction targets:
    # explicit and mode
    # ========================================================

    TARGET_COLS = [
        "explicit",
        "mode"
    ]

    target_results = {}

    for target in TARGET_COLS:

        if target not in train_df.columns:
            continue

        target_results[target] = {}

        # Convert target to numeric if necessary
        train_target = pd.to_numeric(
            train_df[target],
            errors="coerce"
        )

        # Global target mean
        global_mean = train_target.mean()

        for col in categorical_cols:
            # Calculate mean target for every category
            target_map = (
                train_df
                .assign(_target=train_target)
                .groupby(col)["_target"]
                .mean()
            )

            # Encode training data
            train_encoded = (
                train_df[col]
                .map(target_map)
                .fillna(global_mean)
            )

            # Encode testing data
            test_encoded = (
                test_df[col]
                .map(target_map)
                .fillna(global_mean)
            )

            train_target_df = pd.DataFrame({
                col + "_target_" + target:
                    train_encoded
            })

            test_target_df = pd.DataFrame({
                col + "_target_" + target:
                    test_encoded
            })

            target_results[target][col] = {

                "mapping":
                    target_map
                    .round(4)
                    .to_dict(),

                "train_preview":
                    train_target_df
                    .head(5)
                    .round(4)
                    .to_dict(
                        orient="records"
                    ),

                "test_preview":
                    test_target_df
                    .head(5)
                    .round(4)
                    .to_dict(
                        orient="records"
                    )
            }

    results["target_encoding"] = target_results
    # ========================================================
    # 9. EMBEDDING-BASED ENCODING
    # ========================================================
    #
    # Embeddings are learned vectors.
    #
    # Here we create integer IDs for each category
    # and prepare them for an embedding layer.
    #
    # ========================================================

    embedding_results = {}


    for col in categorical_cols:

        categories = (
            train_df[col]
            .fillna("Missing")
            .astype(str)
            .unique()
            .tolist()
        )


        category_to_id = {
            category: index
            for index, category
            in enumerate(categories)
        }


        # Unknown test categories → -1

        train_ids = (
            train_df[col]
            .fillna("Missing")
            .astype(str)
            .map(category_to_id)
            .fillna(-1)
            .astype(int)
        )


        test_ids = (
            test_df[col]
            .fillna("Missing")
            .astype(str)
            .map(category_to_id)
            .fillna(-1)
            .astype(int)
        )


        embedding_results[col] = {

            "category_to_id":
                category_to_id,

            "embedding_input_dimension":
                len(category_to_id),

            "embedding_output_dimension":
                3,

            "train_preview":
                pd.DataFrame(
                    {
                        col + "_embedding_id":
                            train_ids
                    }
                )
                .head(5)
                .to_dict(
                    orient="records"
                ),

            "test_preview":
                pd.DataFrame(
                    {
                        col + "_embedding_id":
                            test_ids
                    }
                )
                .head(5)
                .to_dict(
                    orient="records"
                )
        }


    results["embedding"] = embedding_results


    # ========================================================
    # 10. MIN-MAX SCALING
    # ========================================================

    minmax_scaler = MinMaxScaler()


    train_mm = minmax_scaler.fit_transform(
        train_df[
            numeric_cols
        ].fillna(0)
    )


    test_mm = minmax_scaler.transform(
        test_df[
            numeric_cols
        ].fillna(0)
    )


    train_mm_df = pd.DataFrame(
        train_mm,
        columns=numeric_cols
    ).round(4)


    test_mm_df = pd.DataFrame(
        test_mm,
        columns=numeric_cols
    ).round(4)


    results["minmax"] = {

        "train_preview":
            train_mm_df
            .head(5)
            .to_dict(
                orient="records"
            ),

        "test_preview":
            test_mm_df
            .head(5)
            .to_dict(
                orient="records"
            ),

        "train_stats":
            train_mm_df
            .describe()
            .round(4)
            .to_dict(),

        "test_stats":
            test_mm_df
            .describe()
            .round(4)
            .to_dict()
    }


    # ========================================================
    # 11. STANDARD SCALING
    # ========================================================

    std_scaler = StandardScaler()


    train_std = std_scaler.fit_transform(
        train_df[
            numeric_cols
        ].fillna(0)
    )


    test_std = std_scaler.transform(
        test_df[
            numeric_cols
        ].fillna(0)
    )


    train_std_df = pd.DataFrame(
        train_std,
        columns=numeric_cols
    ).round(4)


    test_std_df = pd.DataFrame(
        test_std,
        columns=numeric_cols
    ).round(4)


    results["standard"] = {

        "train_preview":
            train_std_df
            .head(5)
            .to_dict(
                orient="records"
            ),

        "test_preview":
            test_std_df
            .head(5)
            .to_dict(
                orient="records"
            ),

        "train_stats":
            train_std_df
            .describe()
            .round(4)
            .to_dict(),

        "test_stats":
            test_std_df
            .describe()
            .round(4)
            .to_dict()
    }


    # ========================================================
    # 12. BEFORE SCALING
    # ========================================================

    results["before_scaling"] = (

        train_df[
            numeric_cols
        ]
        .head(5)
        .round(4)
        .to_dict(
            orient="records"
        )
    )

    # ========================================================
    # 13. MISSING VALUES ANALYSIS
    # ========================================================

    results["missing_analysis"] = _missing_value_analysis(
        load_data(),   # fresh copy of original df
        numeric_cols
    )

    return results