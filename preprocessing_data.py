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
# CONFIGURATION
# ============================================================

TARGET_COL = "popularity"

# popularity is the TARGET, so it is not included here
NUM_COLS = [
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

CATEGORICAL_COLS = [
    "track_genre"
]

ORDINAL_COLS = []

IQR_COL = "danceability"

IQR_NUMERIC_COLS = [
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
# OUTPUT DIRECTORY
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "processed_data"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# MISSING VALUE ANALYSIS
# ============================================================

def _missing_value_analysis(df, numeric_cols):

    total_rows = len(df)

    col_info = {}

    # --------------------------------------------------------
    # COLUMN MISSING INFORMATION
    # --------------------------------------------------------

    for col in df.columns:

        n_missing = int(
            df[col].isnull().sum()
        )

        percentage = 0.0

        if total_rows > 0:

            percentage = round(
                n_missing / total_rows * 100,
                4
            )

        col_info[col] = {

            "missing": n_missing,

            "pct": percentage,

            "dtype": str(
                df[col].dtype
            )
        }


    missing_cols = {

        col: info

        for col, info
        in col_info.items()

        if info["missing"] > 0

    }


    total_missing_cells = sum(

        info["missing"]

        for info
        in col_info.values()

    )


    rows_with_any_missing = int(

        df.isnull()

        .any(axis=1)

        .sum()

    )


    # ========================================================
    # 1. ROW-WISE DELETION
    # ========================================================

    df_rowdrop = df.dropna()


    rows_dropped = (

        total_rows -

        len(df_rowdrop)

    )


    pct_dropped = 0.0


    if total_rows > 0:

        pct_dropped = round(

            rows_dropped /

            total_rows *

            100,

            4

        )


    rowdrop_applicable = (

        pct_dropped < 5

    )


    rowdrop_result = {

        "applicable":

            rowdrop_applicable,


        "rows_before":

            total_rows,


        "rows_after":

            int(
                len(df_rowdrop)
            ),


        "rows_dropped":

            rows_dropped,


        "pct_dropped":

            pct_dropped,


        "reason":

            (
                f"Only {rows_dropped} rows "
                f"({pct_dropped}%) would be removed."
                if rowdrop_applicable
                else
                f"{rows_dropped} rows "
                f"({pct_dropped}%) would be removed."
            ),


        "preview":

            df_rowdrop

            .head(3)

            .fillna("")

            .astype(str)

            .to_dict(
                orient="records"
            )

    }


    # ========================================================
    # 2. COLUMN-WISE DELETION
    # ========================================================

    THRESHOLD = 30.0


    cols_to_drop = [

        col

        for col, info
        in col_info.items()

        if info["missing"] > 0

        and info["pct"] > THRESHOLD

    ]


    cols_retained = [

        col

        for col
        in df.columns

        if col not in cols_to_drop

    ]


    per_col_reasons = {}


    for col, info in col_info.items():

        if info["missing"] == 0:

            continue


        if info["pct"] > THRESHOLD:

            per_col_reasons[col] = {

                "drop": True,

                "reason":

                    f"{info['pct']}% missing. "
                    f"Exceeds {THRESHOLD}% threshold."

            }


        else:

            per_col_reasons[col] = {

                "drop": False,

                "reason":

                    f"{info['pct']}% missing. "
                    f"Below {THRESHOLD}% threshold."

            }


    coldrop_result = {

        "applicable":

            len(cols_to_drop) > 0,


        "threshold_pct":

            THRESHOLD,


        "cols_dropped":

            cols_to_drop,


        "cols_retained":

            cols_retained,


        "per_col_reasons":

            per_col_reasons,


        "reason":

            (
                f"Columns dropped: {cols_to_drop}"
                if cols_to_drop
                else
                f"No columns exceed the "
                f"{THRESHOLD}% missing threshold."
            )

    }


    # ========================================================
    # 3. MEAN IMPUTATION
    # ========================================================

    df_mean = df.copy()

    mean_results = []


    for col in numeric_cols:

        if col not in df_mean.columns:

            continue


        n_missing = int(

            df_mean[col]

            .isnull()

            .sum()

        )


        if n_missing > 0:

            mean_value = round(

                float(
                    df_mean[col].mean()
                ),

                4

            )


            df_mean[col] = (

                df_mean[col]

                .fillna(mean_value)

            )


            mean_results.append({

                "column":

                    col,


                "n_filled":

                    n_missing,


                "fill_value":

                    mean_value,


                "applicable":

                    True,


                "reason":

                    f"Missing values replaced "
                    f"with mean ({mean_value})."

            })


    # Non-numeric missing values

    for col, info in col_info.items():

        if (

            info["missing"] > 0

            and

            col not in numeric_cols

        ):

            mean_results.append({

                "column":

                    col,


                "n_filled":

                    info["missing"],


                "fill_value":

                    None,


                "applicable":

                    False,


                "reason":

                    "Mean imputation is not "
                    "applicable for categorical data."

            })


    mean_imputation = {

        "results":

            mean_results,


        "preview":

            df_mean[
                numeric_cols
            ]

            .head(5)

            .round(4)

            .to_dict(
                orient="records"
            )

            if numeric_cols

            else []

    }


    # ========================================================
    # 4. MEDIAN IMPUTATION
    # ========================================================

    df_median = df.copy()

    median_results = []


    for col in numeric_cols:

        if col not in df_median.columns:

            continue


        n_missing = int(

            df_median[col]

            .isnull()

            .sum()

        )


        if n_missing > 0:

            median_value = round(

                float(
                    df_median[col].median()
                ),

                4

            )


            df_median[col] = (

                df_median[col]

                .fillna(
                    median_value
                )

            )


            median_results.append({

                "column":

                    col,


                "n_filled":

                    n_missing,


                "fill_value":

                    median_value,


                "applicable":

                    True,


                "reason":

                    f"Missing values replaced "
                    f"with median ({median_value})."

            })


    for col, info in col_info.items():

        if (

            info["missing"] > 0

            and

            col not in numeric_cols

        ):

            median_results.append({

                "column":

                    col,


                "n_filled":

                    info["missing"],


                "fill_value":

                    None,


                "applicable":

                    False,


                "reason":

                    "Median imputation is not "
                    "applicable for categorical data."

            })


    median_imputation = {

        "results":

            median_results,


        "preview":

            df_median[
                numeric_cols
            ]

            .head(5)

            .round(4)

            .to_dict(
                orient="records"
            )

            if numeric_cols

            else []

    }


    return {

        "total_rows":

            total_rows,


        "total_missing_cells":

            total_missing_cells,


        "rows_with_missing":

            rows_with_any_missing,


        "col_info":

            col_info,


        "missing_cols":

            missing_cols,


        "rowdrop":

            rowdrop_result,


        "coldrop":

            coldrop_result,


        "mean_imputation":

            mean_imputation,


        "median_imputation":

            median_imputation

    }


# ============================================================
# IQR OUTLIER ANALYSIS
# ============================================================

def _iqr_outlier_analysis(df, numeric_cols):

    column_outliers = {}

    outlier_masks = []


    for col in numeric_cols:

        if col not in df.columns:

            continue


        series = pd.to_numeric(

            df[col],

            errors="coerce"

        )


        valid = series.dropna()


        # No valid data

        if len(valid) == 0:

            column_outliers[col] = 0


            outlier_masks.append(

                pd.Series(

                    False,

                    index=df.index

                )

            )


            continue


        # Quartile 1

        q1 = float(

            valid.quantile(0.25)

        )


        # Quartile 3

        q3 = float(

            valid.quantile(0.75)

        )


        # IQR

        iqr = q3 - q1


        # Bounds

        lower_bound = (

            q1 - 1.5 * iqr

        )


        upper_bound = (

            q3 + 1.5 * iqr

        )


        # Outlier mask

        mask = (

            (series < lower_bound)

            |

            (series > upper_bound)

        )


        column_outliers[col] = int(

            mask.sum()

        )


        outlier_masks.append(

            mask.fillna(False)

        )


    # ========================================================
    # UNIQUE ROWS WITH OUTLIERS
    # ========================================================

    if outlier_masks:


        combined = np.column_stack(

            [

                mask.values

                for mask

                in outlier_masks

            ]

        )


        n_rows_with_outlier = int(

            combined

            .any(axis=1)

            .sum()

        )


    else:

        n_rows_with_outlier = 0


    # ========================================================
    # TOTAL OUTLIERS
    # ========================================================

    n_outliers = int(

        sum(
            column_outliers.values()
        )

    )


    # ========================================================
    # MAXIMUM OUTLIERS
    # IMPORTANT FOR preprocessing.html
    # ========================================================

    max_outliers = (

        max(
            column_outliers.values()
        )

        if column_outliers

        else 0

    )


    return {

        "column_outliers":

            column_outliers,


        "n_outliers":

            n_outliers,


        "n_rows_with_outlier":

            n_rows_with_outlier,


        "max_outliers":

            max_outliers,


        "n_numeric_columns":

            len(
                column_outliers
            )

    }


# ============================================================
# MAIN PREPROCESSING FUNCTION
# ============================================================

def run_preprocessing():

    # ========================================================
    # LOAD DATA
    # ========================================================

    df = load_data()

    df = df.copy()

    results = {}


    # ========================================================
    # REMOVE UNNAMED INDEX COLUMNS
    # ========================================================

    unnamed_columns = [

        col

        for col

        in df.columns

        if str(col).lower()

        .startswith("unnamed")

    ]


    if unnamed_columns:

        df.drop(

            columns=unnamed_columns,

            inplace=True

        )


    # ========================================================
    # DATASET SHAPE
    # ========================================================

    results["n_rows"] = int(

        df.shape[0]

    )


    results["n_columns"] = int(

        df.shape[1]

    )


    results["all_columns"] = (

        df.columns.tolist()

    )


    # ========================================================
    # NUMERICAL COLUMNS
    # ========================================================

    numeric_cols = [

        col

        for col

        in NUM_COLS

        if col in df.columns

    ]


    for col in numeric_cols:

        df[col] = pd.to_numeric(

            df[col],

            errors="coerce"

        )


    results["numeric_cols"] = (

        numeric_cols

    )


    results["n_numeric_features"] = (

        len(numeric_cols)

    )


    # ========================================================
    # TARGET COLUMN
    # ========================================================

    if TARGET_COL not in df.columns:

        raise ValueError(

            f"Target column '{TARGET_COL}' "

            "was not found in the dataset."

        )


    df[TARGET_COL] = pd.to_numeric(

        df[TARGET_COL],

        errors="coerce"

    )


    # ========================================================
    # CATEGORICAL COLUMNS
    # ========================================================

    categorical_cols = [

        col

        for col

        in CATEGORICAL_COLS

        if col in df.columns

    ]


    results["categorical_cols"] = (

        categorical_cols

    )


    results["n_categorical_features"] = (

        len(categorical_cols)

    )


    # ========================================================
    # MISSING VALUE ANALYSIS
    # ========================================================

    results["missing_analysis"] = (

        _missing_value_analysis(

            df.copy(),

            numeric_cols

        )

    )


    # ========================================================
    # REMOVE MISSING TARGET ROWS
    # ========================================================

    rows_before_target_drop = len(df)


    df = df.dropna(

        subset=[TARGET_COL]

    )


    results["target_missing_rows_removed"] = (

        rows_before_target_drop -

        len(df)

    )


    # ========================================================
    # IQR OUTLIER ANALYSIS
    # ========================================================

    iqr_numeric_cols = [

        col

        for col

        in IQR_NUMERIC_COLS

        if col in df.columns

    ]


    iqr_results = (

        _iqr_outlier_analysis(

            df,

            iqr_numeric_cols

        )

    )


    # ========================================================
    # CLIP DANCEABILITY OUTLIERS
    # ========================================================

    if IQR_COL in df.columns:


        series = pd.to_numeric(

            df[IQR_COL],

            errors="coerce"

        )


        valid_series = (

            series.dropna()

        )


        if len(valid_series) > 0:


            q1 = float(

                valid_series.quantile(0.25)

            )


            q3 = float(

                valid_series.quantile(0.75)

            )


            iqr = q3 - q1


            lower_bound = (

                q1 - 1.5 * iqr

            )


            upper_bound = (

                q3 + 1.5 * iqr

            )


            outlier_mask = (

                (series < lower_bound)

                |

                (series > upper_bound)

            )


            number_of_outliers = int(

                outlier_mask.sum()

            )


            min_before = float(

                valid_series.min()

            )


            max_before = float(

                valid_series.max()

            )


            # Clip

            df[IQR_COL] = (

                series.clip(

                    lower=lower_bound,

                    upper=upper_bound

                )

            )


            min_after = float(

                df[IQR_COL]

                .min()

            )


            max_after = float(

                df[IQR_COL]

                .max()

            )


            iqr_results["clip_column"] = (

                IQR_COL

            )


            iqr_results["clip_details"] = {

                "column":

                    IQR_COL,


                "q1":

                    round(q1, 4),


                "q3":

                    round(q3, 4),


                "iqr":

                    round(iqr, 4),


                "lower_bound":

                    round(
                        lower_bound,
                        4
                    ),


                "upper_bound":

                    round(
                        upper_bound,
                        4
                    ),


                "n_outliers":

                    number_of_outliers,


                "min_before":

                    round(
                        min_before,
                        4
                    ),


                "max_before":

                    round(
                        max_before,
                        4
                    ),


                "min_after":

                    round(
                        min_after,
                        4
                    ),


                "max_after":

                    round(
                        max_after,
                        4
                    )

            }


    results["iqr"] = (

        iqr_results

    )


    # ========================================================
    # TRAIN TEST SPLIT
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

            int(
                df.shape[0]
            ),


        "train_rows":

            int(
                train_df.shape[0]
            ),


        "test_rows":

            int(
                test_df.shape[0]
            ),


        "train_pct":

            round(

                len(train_df)

                /

                len(df)

                * 100,

                1

            ),


        "test_pct":

            round(

                len(test_df)

                /

                len(df)

                * 100,

                1

            )

    }


    # ========================================================
    # HANDLE MISSING NUMERICAL VALUES
    # MEDIAN IMPUTATION USING TRAIN DATA
    # ========================================================

    imputation_values = {}


    for col in numeric_cols:


        median_value = (

            train_df[col]

            .median()

        )


        # Safety fallback

        if pd.isna(median_value):

            median_value = 0


        imputation_values[col] = (

            float(
                median_value
            )

        )


        train_df[col] = (

            train_df[col]

            .fillna(
                median_value
            )

        )


        test_df[col] = (

            test_df[col]

            .fillna(
                median_value
            )

        )


    results["imputation_values"] = (

        imputation_values

    )


    # ========================================================
    # HANDLE CATEGORICAL MISSING VALUES
    # ========================================================

    for col in categorical_cols:

        train_df[col] = (

            train_df[col]

            .fillna("Missing")

            .astype(str)

        )


        test_df[col] = (

            test_df[col]

            .fillna("Missing")

            .astype(str)

        )


    # ========================================================
    # BEFORE SCALING
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
    # ONE-HOT ENCODING
    # ========================================================

    if categorical_cols:


        onehot_encoder = OneHotEncoder(

            handle_unknown="ignore",

            sparse_output=False

        )


        train_onehot = (

            onehot_encoder.fit_transform(

                train_df[
                    categorical_cols
                ]

            )

        )


        test_onehot = (

            onehot_encoder.transform(

                test_df[
                    categorical_cols
                ]

            )

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

                list(
                    onehot_feature_names
                ),


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

                len(
                    onehot_feature_names
                )

        }


    else:

        train_onehot_df = pd.DataFrame(

            index=train_df.index

        )


        test_onehot_df = pd.DataFrame(

            index=test_df.index

        )


        results["onehot"] = {

            "columns": [],

            "train_preview": [],

            "test_preview": [],

            "number_of_features": 0

        }


    # ========================================================
    # ORDINAL ENCODING
    # ========================================================

    ordinal_cols = [

        col

        for col

        in ORDINAL_COLS

        if col in df.columns

    ]


    if ordinal_cols:


        ordinal_encoder = OrdinalEncoder(

            handle_unknown="use_encoded_value",

            unknown_value=-1

        )


        train_ordinal = (

            ordinal_encoder.fit_transform(

                train_df[
                    ordinal_cols
                ]

                .fillna("Missing")

            )

        )


        test_ordinal = (

            ordinal_encoder.transform(

                test_df[
                    ordinal_cols
                ]

                .fillna("Missing")

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


    else:

        results["ordinal"] = {

            "columns": [],

            "train_preview": [],

            "test_preview": []

        }


    # ========================================================
    # TARGET ENCODING
    # ========================================================

    TARGET_ENCODING_TARGETS = [

        "explicit",

        "mode"

    ]


    target_results = {}


    for target in TARGET_ENCODING_TARGETS:


        if target not in train_df.columns:

            continue


        train_target = pd.to_numeric(

            train_df[target],

            errors="coerce"

        )


        global_mean = (

            train_target.mean()

        )


        target_results[target] = {}


        for col in categorical_cols:


            target_map = (

                train_df

                .assign(

                    _target=train_target

                )

                .groupby(col)[

                    "_target"

                ]

                .mean()

            )


            train_encoded = (

                train_df[col]

                .map(target_map)

                .fillna(
                    global_mean
                )

            )


            test_encoded = (

                test_df[col]

                .map(target_map)

                .fillna(
                    global_mean
                )

            )


            train_target_df = pd.DataFrame(

                {

                    col

                    + "_target_"

                    + target:

                    train_encoded

                }

            )


            test_target_df = pd.DataFrame(

                {

                    col

                    + "_target_"

                    + target:

                    test_encoded

                }

            )


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


    results["target_encoding"] = (

        target_results

    )


    # ========================================================
    # EMBEDDING ENCODING
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

                len(
                    category_to_id
                ),


            "embedding_output_dimension":

                3,


            "train_preview":

                pd.DataFrame(

                    {

                        col

                        + "_embedding_id":

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

                        col

                        + "_embedding_id":

                        test_ids

                    }

                )

                .head(5)

                .to_dict(
                    orient="records"
                )

        }


    results["embedding"] = (

        embedding_results

    )


    # ========================================================
    # MIN-MAX SCALING
    # ========================================================

    minmax_scaler = MinMaxScaler()


    train_mm = (

        minmax_scaler.fit_transform(

            train_df[
                numeric_cols
            ]

        )

    )


    test_mm = (

        minmax_scaler.transform(

            test_df[
                numeric_cols
            ]

        )

    )


    train_mm_df = pd.DataFrame(

        train_mm,

        columns=numeric_cols,

        index=train_df.index

    ).round(4)


    test_mm_df = pd.DataFrame(

        test_mm,

        columns=numeric_cols,

        index=test_df.index

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
    # STANDARD SCALING
    # ========================================================

    std_scaler = StandardScaler()


    train_std = (

        std_scaler.fit_transform(

            train_df[
                numeric_cols
            ]

        )

    )


    test_std = (

        std_scaler.transform(

            test_df[
                numeric_cols
            ]

        )

    )


    train_std_df = pd.DataFrame(

        train_std,

        columns=numeric_cols,

        index=train_df.index

    ).round(4)


    test_std_df = pd.DataFrame(

        test_std,

        columns=numeric_cols,

        index=test_df.index

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
    # CREATE FINAL PREPROCESSED DATASET
    # FINAL DATASET USES:
    # 1. Standard-scaled numerical features
    # 2. One-hot encoded categorical features
    # 3. Target column popularity
    # ========================================================

    train_target_final = (

        train_df[
            [TARGET_COL]
        ]

        .copy()

    )


    test_target_final = (

        test_df[
            [TARGET_COL]
        ]

        .copy()

    )


    # ========================================================
    # FINAL TRAIN DATASET
    # ========================================================

    final_train_dataset = pd.concat(

        [

            train_std_df,

            train_onehot_df,

            train_target_final

        ],

        axis=1

    )


    # ========================================================
    # FINAL TEST DATASET
    # ========================================================

    final_test_dataset = pd.concat(

        [

            test_std_df,

            test_onehot_df,

            test_target_final

        ],

        axis=1

    )


    # Remove duplicate columns

    final_train_dataset = (

        final_train_dataset.loc[

            :,

            ~

            final_train_dataset

            .columns

            .duplicated()

        ]

    )


    final_test_dataset = (

        final_test_dataset.loc[

            :,

            ~

            final_test_dataset

            .columns

            .duplicated()

        ]

    )


    # Reset indexes

    final_train_dataset = (

        final_train_dataset

        .reset_index(
            drop=True
        )

    )


    final_test_dataset = (

        final_test_dataset

        .reset_index(
            drop=True
        )

    )


    # ========================================================
    # SAVE TRAIN CSV
    # ========================================================

    train_csv_path = os.path.join(

        OUTPUT_DIR,

        "preprocessed_train.csv"

    )


    final_train_dataset.to_csv(

        train_csv_path,

        index=False

    )


    # ========================================================
    # SAVE TEST CSV
    # ========================================================

    test_csv_path = os.path.join(

        OUTPUT_DIR,

        "preprocessed_test.csv"

    )


    final_test_dataset.to_csv(

        test_csv_path,

        index=False

    )


    # ========================================================
    # CREATE COMBINED DATASET
    # ========================================================

    final_dataset = pd.concat(

        [

            final_train_dataset,

            final_test_dataset

        ],

        axis=0,

        ignore_index=True

    )


    # ========================================================
    # SAVE COMBINED CSV
    # ========================================================

    combined_csv_path = os.path.join(

        OUTPUT_DIR,

        "preprocessed_dataset.csv"

    )


    final_dataset.to_csv(

        combined_csv_path,

        index=False

    )


    # ========================================================
    # STORE CSV INFORMATION
    # ========================================================

    results["preprocessed_dataset"] = {

        "train_csv":

            train_csv_path,


        "test_csv":

            test_csv_path,


        "combined_csv":

            combined_csv_path,


        "train_shape":

            {

                "rows":

                    int(
                        final_train_dataset.shape[0]
                    ),


                "columns":

                    int(
                        final_train_dataset.shape[1]
                    )

            },


        "test_shape":

            {

                "rows":

                    int(
                        final_test_dataset.shape[0]
                    ),


                "columns":

                    int(
                        final_test_dataset.shape[1]
                    )

            },


        "combined_shape":

            {

                "rows":

                    int(
                        final_dataset.shape[0]
                    ),


                "columns":

                    int(
                        final_dataset.shape[1]
                    )

            },


        "columns":

            final_dataset

            .columns

            .tolist()

    }


    # ========================================================
    # SUCCESS MESSAGE
    # ========================================================

    print(

        "\n"

        + "=" * 60

    )


    print(

        "PREPROCESSING COMPLETED SUCCESSFULLY"

    )


    print(

        "=" * 60

    )


    print(

        f"\nOriginal Dataset Shape: "

        f"{df.shape}"

    )


    print(

        f"\nTrain Dataset Shape: "

        f"{final_train_dataset.shape}"

    )


    print(

        f"\nTest Dataset Shape: "

        f"{final_test_dataset.shape}"

    )


    print(

        f"\nCombined Dataset Shape: "

        f"{final_dataset.shape}"

    )


    print(

        "\n"

        + "=" * 60

    )


    print(

        "CSV FILES GENERATED"

    )


    print(

        "=" * 60

    )


    print(

        f"\nTrain CSV:\n"

        f"{train_csv_path}"

    )


    print(

        f"\nTest CSV:\n"

        f"{test_csv_path}"

    )


    print(

        f"\nCombined CSV:\n"

        f"{combined_csv_path}"

    )


    print(

        "\n"

        + "=" * 60

    )


    return results


# ============================================================
# RUN DIRECTLY
# ============================================================

if __name__ == "__main__":

    results = run_preprocessing()

    print(
        "\nPreprocessing Finished!"
    )

    print(
        results["preprocessed_dataset"]
    )