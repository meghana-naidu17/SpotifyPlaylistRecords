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


# ============================================================
# CATEGORICAL COLUMNS
# Change these according to your dataset
# ============================================================

CATEGORICAL_COLS = [
    "track_genre"
]


# ============================================================
# ORDINAL COLUMNS
# Only use this when categories have a meaningful order
# ============================================================

ORDINAL_COLS = [
    # Example:
    # "rating"
]


# ============================================================
# TARGET COLUMN
# Change this if your dataset has another target
# ============================================================

TARGET_COL = "popularity"


# ============================================================
# IQR COLUMN
# ============================================================

IQR_COL = "danceability"


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


    return results