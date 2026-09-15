"""
preprocessing_data.py
=====================
Binary classification preprocessing pipeline.

Target  : explicit  (False->0 / True->1)
Task    : Classification
Dataset : SpotifyPlaylistRecords.csv  (20 columns)

Column breakdown
----------------
  Numerical  (13) : popularity, duration_ms, danceability, energy,
                    key, loudness, mode, speechiness, acousticness,
                    instrumentalness, liveness, valence, tempo
  Categorical (6) : track_id, artists, album_name, track_name,
                    time_signature, track_genre
  Target      (1) : explicit
  ─────────────────────────────────────
  Total           : 20 columns

Design rules
------------
• Split BEFORE fitting any transformer (no leakage)
• Median imputation on numerical columns
• StandardScaler on numerical columns
• OneHotEncoder (sparse) on LOW-cardinality cols only:
    time_signature, track_genre
• HIGH-cardinality cols (track_id, artists, album_name,
  track_name) get compact derived features — never one-hot
• IQR analysis on numerical features only
• Sparse outputs saved as .npz; X/y CSVs also saved
• Full preprocessing_report.json saved to processed_data/
"""

import os
import json
import warnings

import numpy as np
import pandas as pd
import scipy.sparse as sp

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split

from load_data import load_data

warnings.filterwarnings("ignore")

# ============================================================
# CONFIGURATION
# ============================================================

TARGET_COL = "explicit"

# 13 numerical features (popularity stays as a feature)
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
]

# 6 categorical input columns
CATEGORICAL_COLS = [
    "track_id",
    "artists",
    "album_name",
    "track_name",
    "time_signature",
    "track_genre",
]

# Only these two are low-cardinality → safe to one-hot encode
LOW_CARD_COLS = ["time_signature", "track_genre"]

# High-cardinality columns → compact derived features only
HIGH_CARD_ID_COL   = "track_id"
HIGH_CARD_TEXT_COLS = ["artists", "album_name", "track_name"]

# IQR outlier analysis runs on the same 13 numerical columns
IQR_NUMERIC_COLS = NUM_COLS

# Output directory
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "processed_data")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# HELPER — convert explicit to binary 0/1
# ============================================================

def _encode_target(series: pd.Series) -> pd.Series:
    """
    Convert explicit column to integer 0/1.
    Handles: True/False booleans, "True"/"False" strings,
             "true"/"false", "TRUE"/"FALSE", 1/0 integers.
    Returns an int64 Series with NaN preserved as NaN.
    """
    def _to_int(val):
        if pd.isna(val):
            return np.nan
        if isinstance(val, bool):
            return int(val)
        s = str(val).strip().lower()
        if s in ("true", "1", "yes"):
            return 1
        if s in ("false", "0", "no"):
            return 0
        return np.nan

    return series.map(_to_int)


# ============================================================
# HELPER — compact high-cardinality features
# ============================================================

def _derive_compact_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replace high-cardinality text columns with low-dimensional
    numeric proxies.  The original columns are kept in the df
    for reference but these new columns are added.
    """
    out = df.copy()

    # artists -> count of artists split on ";"
    if "artists" in out.columns:
        out["artist_count"] = (
            out["artists"]
            .fillna("")
            .astype(str)
            .apply(lambda x: len([a for a in x.split(";") if a.strip()]))
        )

    # album_name -> character length
    if "album_name" in out.columns:
        out["album_name_length"] = (
            out["album_name"].fillna("").astype(str).str.len()
        )

    # track_name -> character length + word count
    if "track_name" in out.columns:
        out["track_name_length"] = (
            out["track_name"].fillna("").astype(str).str.len()
        )
        out["track_name_word_count"] = (
            out["track_name"]
            .fillna("")
            .astype(str)
            .apply(lambda x: len(x.split()))
        )

    return out


# ============================================================
# HELPER — missing value analysis (report only, not pipeline)
# ============================================================

def _missing_value_analysis(df: pd.DataFrame, numeric_cols: list) -> dict:
    total_rows = len(df)
    col_info   = {}

    for col in df.columns:
        n = int(df[col].isnull().sum())
        col_info[col] = {
            "missing": n,
            "pct":     round(n / total_rows * 100, 4) if total_rows else 0.0,
            "dtype":   str(df[col].dtype),
        }

    missing_cols        = {c: v for c, v in col_info.items() if v["missing"] > 0}
    total_missing_cells = sum(v["missing"] for v in col_info.values())
    rows_with_missing   = int(df.isnull().any(axis=1).sum())

    # --- row-wise deletion assessment ---
    df_dropped    = df.dropna()
    rows_dropped  = total_rows - len(df_dropped)
    pct_dropped   = round(rows_dropped / total_rows * 100, 4) if total_rows else 0.0
    row_applicable = pct_dropped < 5.0

    rowdrop = {
        "applicable":   row_applicable,
        "rows_before":  total_rows,
        "rows_after":   int(len(df_dropped)),
        "rows_dropped": rows_dropped,
        "pct_dropped":  pct_dropped,
        "reason": (
            f"Only {rows_dropped} row(s) ({pct_dropped}%) contain missing values. "
            "Dropping them causes negligible data loss — row-wise deletion is safe here."
            if row_applicable else
            f"{rows_dropped} rows ({pct_dropped}%) would be lost — too much data loss."
        ),
    }

    # --- column-wise deletion assessment ---
    THRESHOLD = 30.0
    cols_to_drop   = [c for c, v in col_info.items() if v["missing"] > 0 and v["pct"] > THRESHOLD]
    per_col_reasons = {}
    for c, v in col_info.items():
        if v["missing"] == 0:
            continue
        if v["pct"] > THRESHOLD:
            per_col_reasons[c] = {"drop": True,
                                   "reason": f"{v['pct']}% missing — exceeds {THRESHOLD}% threshold. Column should be dropped."}
        else:
            per_col_reasons[c] = {"drop": False,
                                   "reason": f"{v['pct']}% missing ({v['missing']} value(s)) — below threshold. Retain and impute."}

    coldrop = {
        "applicable":      len(cols_to_drop) > 0,
        "threshold_pct":   THRESHOLD,
        "cols_dropped":    cols_to_drop,
        "cols_retained":   [c for c in df.columns if c not in cols_to_drop],
        "per_col_reasons": per_col_reasons,
        "reason": (
            f"Columns with >{THRESHOLD}% missing: {cols_to_drop}. These should be dropped."
            if cols_to_drop else
            f"No column exceeds the {THRESHOLD}% missing threshold. "
            "Column-wise deletion is NOT applied — all columns retained."
        ),
    }

    # --- mean imputation report ---
    mean_results = []
    for col in numeric_cols:
        n = int(df[col].isnull().sum()) if col in df.columns else 0
        if n > 0:
            mv = round(float(df[col].mean()), 4)
            mean_results.append({"column": col, "n_filled": n, "fill_value": mv,
                                  "applicable": True,
                                  "reason": f"Numeric column — {n} missing replaced with mean ({mv})."})
    for c, v in col_info.items():
        if v["missing"] > 0 and c not in numeric_cols:
            mean_results.append({"column": c, "n_filled": v["missing"], "fill_value": None,
                                  "applicable": False,
                                  "reason": f"Non-numeric (dtype: {v['dtype']}) — mean imputation not applicable."})

    # --- median imputation report ---
    median_results = []
    for col in numeric_cols:
        n = int(df[col].isnull().sum()) if col in df.columns else 0
        if n > 0:
            mv = round(float(df[col].median()), 4)
            median_results.append({"column": col, "n_filled": n, "fill_value": mv,
                                    "applicable": True,
                                    "reason": f"Numeric column — {n} missing replaced with median ({mv})."})
    for c, v in col_info.items():
        if v["missing"] > 0 and c not in numeric_cols:
            median_results.append({"column": c, "n_filled": v["missing"], "fill_value": None,
                                    "applicable": False,
                                    "reason": f"Non-numeric (dtype: {v['dtype']}) — median imputation not applicable."})

    return {
        "total_rows":          total_rows,
        "total_missing_cells": total_missing_cells,
        "rows_with_missing":   rows_with_missing,
        "col_info":            col_info,
        "missing_cols":        missing_cols,
        "rowdrop":             rowdrop,
        "coldrop":             coldrop,
        "mean_imputation":     {"results": mean_results},
        "median_imputation":   {"results": median_results},
    }


# ============================================================
# HELPER — IQR outlier analysis (report only)
# ============================================================

def _iqr_outlier_analysis(df: pd.DataFrame, numeric_cols: list) -> dict:
    column_stats   = {}
    outlier_masks  = []

    for col in numeric_cols:
        if col not in df.columns:
            continue
        s     = pd.to_numeric(df[col], errors="coerce")
        valid = s.dropna()
        if len(valid) == 0:
            column_stats[col] = {"n_outliers": 0, "q1": None, "q3": None,
                                  "iqr": None, "lower": None, "upper": None}
            outlier_masks.append(pd.Series(False, index=df.index))
            continue

        q1    = float(valid.quantile(0.25))
        q3    = float(valid.quantile(0.75))
        iqr   = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        mask  = (s < lower) | (s > upper)

        column_stats[col] = {
            "q1":          round(q1, 4),
            "q3":          round(q3, 4),
            "iqr":         round(iqr, 4),
            "lower_bound": round(lower, 4),
            "upper_bound": round(upper, 4),
            "n_outliers":  int(mask.sum()),
            "min_before":  round(float(valid.min()), 4),
            "max_before":  round(float(valid.max()), 4),
        }
        outlier_masks.append(mask.fillna(False))

    n_outliers_total = sum(v["n_outliers"] for v in column_stats.values())
    n_rows_with_any  = 0
    if outlier_masks:
        combined        = np.column_stack([m.values for m in outlier_masks])
        n_rows_with_any = int(combined.any(axis=1).sum())

    # For the dashboard IQR card — use danceability detail if present
    clip_col    = "danceability"
    clip_detail = column_stats.get(clip_col, {})

    # column_outliers: flat {col: count} — what the template iterates
    column_outliers = {col: v["n_outliers"] for col, v in column_stats.items()}
    max_out         = max(column_outliers.values(), default=0)

    # clip_details: nested dict the template accesses via results.iqr.clip_details.*
    clip_details = None
    if clip_detail:
        clip_details = {
            "column":      clip_col,
            "q1":          clip_detail.get("q1"),
            "q3":          clip_detail.get("q3"),
            "iqr":         clip_detail.get("iqr"),
            "lower_bound": clip_detail.get("lower_bound"),
            "upper_bound": clip_detail.get("upper_bound"),
            "n_outliers":  clip_detail.get("n_outliers"),
            "min_before":  clip_detail.get("min_before"),
            "max_before":  clip_detail.get("max_before"),
            "min_after":   clip_detail.get("lower_bound"),
            "max_after":   clip_detail.get("upper_bound"),
        }

    return {
        # keys the template uses directly
        "column_outliers":    column_outliers,
        "n_outliers":         n_outliers_total,
        "n_rows_with_outlier": n_rows_with_any,
        "max_outliers":       max_out,
        "n_numeric_columns":  len(column_stats),
        "clip_details":       clip_details,
        # also keep full stats for report
        "column_stats":       column_stats,
    }


# ============================================================
# MAIN FUNCTION
# ============================================================

def run_preprocessing() -> dict:
    results = {}

    # --------------------------------------------------------
    # 1. LOAD & CLEAN
    # --------------------------------------------------------
    df = load_data().copy()

    # Drop unnamed index columns added by CSV re-saves
    df.drop(columns=[c for c in df.columns if str(c).lower().startswith("unnamed")],
            inplace=True, errors="ignore")

    results["n_rows"]      = int(df.shape[0])
    results["n_columns"]   = int(df.shape[1])
    results["all_columns"] = df.columns.tolist()

    # --------------------------------------------------------
    # 2. VALIDATE COLUMNS
    # --------------------------------------------------------
    numeric_cols   = [c for c in NUM_COLS       if c in df.columns]
    categoric_cols = [c for c in CATEGORICAL_COLS if c in df.columns]
    low_card_cols  = [c for c in LOW_CARD_COLS   if c in df.columns]

    results["numeric_cols"]          = numeric_cols
    results["categorical_cols"]      = categoric_cols
    results["n_numeric_features"]    = len(numeric_cols)
    results["n_categorical_features"] = len(categoric_cols)

    # --------------------------------------------------------
    # 3. CAST NUMERICAL COLUMNS
    # --------------------------------------------------------
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # --------------------------------------------------------
    # 4. ENCODE TARGET  (explicit -> 0/1)
    # --------------------------------------------------------
    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found in dataset.")

    df[TARGET_COL] = _encode_target(df[TARGET_COL])

    # Remove rows where target is missing (cannot impute classification label)
    rows_before = len(df)
    df = df.dropna(subset=[TARGET_COL])
    df[TARGET_COL] = df[TARGET_COL].astype(int)
    results["target_missing_rows_removed"] = rows_before - len(df)
    results["target_class_counts"] = df[TARGET_COL].value_counts().to_dict()

    # --------------------------------------------------------
    # 5. MISSING VALUE ANALYSIS  (on raw data, before split)
    # --------------------------------------------------------
    results["missing_analysis"] = _missing_value_analysis(df.copy(), numeric_cols)

    # --------------------------------------------------------
    # 6. IQR OUTLIER ANALYSIS  (on raw data, numerical only)
    # --------------------------------------------------------
    iqr_cols         = [c for c in IQR_NUMERIC_COLS if c in df.columns]
    iqr_full         = _iqr_outlier_analysis(df, iqr_cols)
    results["iqr"]   = iqr_full

    # Clip danceability outliers (example justified clipping)
    if "danceability" in df.columns:
        stats = iqr_full["column_stats"].get("danceability", {})
        if stats.get("lower_bound") is not None:
            df["danceability"] = df["danceability"].clip(
                lower=stats["lower_bound"],
                upper=stats["upper_bound"]
            )

    # --------------------------------------------------------
    # 7. DERIVE COMPACT FEATURES for high-cardinality cols
    # --------------------------------------------------------
    df = _derive_compact_features(df)

    compact_feature_cols = []
    if "artists"    in df.columns: compact_feature_cols.append("artist_count")
    if "album_name" in df.columns: compact_feature_cols += ["album_name_length"]
    if "track_name" in df.columns: compact_feature_cols += ["track_name_length", "track_name_word_count"]

    results["compact_features"] = compact_feature_cols

    # --------------------------------------------------------
    # 8. STRATIFIED TRAIN / TEST SPLIT  (split BEFORE fitting)
    # --------------------------------------------------------
    train_df, test_df = train_test_split(
        df, test_size=0.30, random_state=42, stratify=df[TARGET_COL]
    )
    train_df = train_df.copy()
    test_df  = test_df.copy()

    results["split"] = {
        "total":      int(df.shape[0]),
        "train_rows": int(train_df.shape[0]),
        "test_rows":  int(test_df.shape[0]),
        "train_pct":  round(len(train_df) / len(df) * 100, 1),
        "test_pct":   round(len(test_df)  / len(df) * 100, 1),
    }

    # --------------------------------------------------------
    # 9. MEDIAN IMPUTATION  (fit on train only)
    # --------------------------------------------------------
    imputation_values = {}
    all_numeric = numeric_cols + compact_feature_cols

    for col in all_numeric:
        if col not in train_df.columns:
            continue
        median_val = train_df[col].median()
        if pd.isna(median_val):
            median_val = 0.0
        imputation_values[col] = float(median_val)
        train_df[col] = train_df[col].fillna(median_val)
        test_df[col]  = test_df[col].fillna(median_val)

    results["imputation_values"] = imputation_values

    # Fill categorical missing with "Missing"
    for col in low_card_cols:
        train_df[col] = train_df[col].fillna("Missing").astype(str)
        test_df[col]  = test_df[col].fillna("Missing").astype(str)

    # --------------------------------------------------------
    # 10. STANDARD SCALER  (fit on train only)
    # --------------------------------------------------------
    valid_numeric = [c for c in all_numeric if c in train_df.columns]

    results["before_scaling"] = (
        train_df[valid_numeric].head(5).round(4).to_dict(orient="records")
    )

    scaler    = StandardScaler()
    X_train_num = scaler.fit_transform(train_df[valid_numeric])
    X_test_num  = scaler.transform(test_df[valid_numeric])

    train_scaled_df = pd.DataFrame(X_train_num, columns=valid_numeric,
                                   index=train_df.index).round(4)
    test_scaled_df  = pd.DataFrame(X_test_num,  columns=valid_numeric,
                                   index=test_df.index).round(4)

    results["standard"] = {
        "train_preview": train_scaled_df.head(5).to_dict(orient="records"),
        "test_preview":  test_scaled_df.head(5).to_dict(orient="records"),
        "train_stats":   train_scaled_df.describe().round(4).to_dict(),
        "test_stats":    test_scaled_df.describe().round(4).to_dict(),
    }

    # Provide minmax alias so the preprocessing.html tab still works
    results["minmax"] = results["standard"]

    results["numeric_cols"] = valid_numeric

    # --------------------------------------------------------
    # 11. SPARSE ONE-HOT ENCODING  (low-cardinality only)
    #     sparse_output=True  ← critical, prevents 110 GiB alloc
    # --------------------------------------------------------
    valid_low_card = [c for c in low_card_cols if c in train_df.columns]

    if valid_low_card:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
        X_train_ohe = ohe.fit_transform(train_df[valid_low_card])
        X_test_ohe  = ohe.transform(test_df[valid_low_card])
        ohe_feature_names = list(ohe.get_feature_names_out(valid_low_card))

        # Dense preview (small slice only — safe)
        preview_dense = X_train_ohe[:5].toarray()
        ohe_preview_df = pd.DataFrame(
            preview_dense,
            columns=ohe_feature_names[:preview_dense.shape[1]]
        )

        results["onehot"] = {
            "columns":          ohe_feature_names[:8],   # first 8 for display
            "train_preview":    ohe_preview_df.to_dict(orient="records"),
            "test_preview":     [],
            "number_of_features": len(ohe_feature_names),
        }
    else:
        ohe_feature_names = []
        X_train_ohe = sp.csr_matrix((len(train_df), 0))
        X_test_ohe  = sp.csr_matrix((len(test_df),  0))
        results["onehot"] = {
            "columns": [], "train_preview": [], "test_preview": [], "number_of_features": 0
        }

    # Remove legacy keys that old template sections expect
    results["target_encoding"] = {}
    results["embedding"]       = {}
    results["ordinal"]         = {"columns": [], "train_preview": [], "test_preview": []}

    # --------------------------------------------------------
    # 12. ASSEMBLE FINAL X, y
    # --------------------------------------------------------
    X_train_num_sp = sp.csr_matrix(X_train_num)
    X_test_num_sp  = sp.csr_matrix(X_test_num)

    X_train = sp.hstack([X_train_num_sp, X_train_ohe], format="csr")
    X_test  = sp.hstack([X_test_num_sp,  X_test_ohe],  format="csr")

    y_train = train_df[TARGET_COL].astype(int).reset_index(drop=True)
    y_test  = test_df[TARGET_COL].astype(int).reset_index(drop=True)

    total_features = X_train.shape[1]
    all_feature_names = valid_numeric + ohe_feature_names

    results["final_feature_count"] = total_features
    results["final_feature_names"] = all_feature_names[:50]  # first 50 for display

    # --------------------------------------------------------
    # 13. SAVE OUTPUTS
    # --------------------------------------------------------
    # Sparse matrices
    sp.save_npz(os.path.join(OUTPUT_DIR, "X_train.npz"), X_train)
    sp.save_npz(os.path.join(OUTPUT_DIR, "X_test.npz"),  X_test)

    # y as CSV
    y_train_path = os.path.join(OUTPUT_DIR, "y_train.csv")
    y_test_path  = os.path.join(OUTPUT_DIR, "y_test.csv")
    y_train.to_csv(y_train_path, index=False, header=["explicit"])
    y_test.to_csv(y_test_path,   index=False, header=["explicit"])

    # X as CSV (dense numerical part only — manageable size)
    x_train_csv = os.path.join(OUTPUT_DIR, "X_train.csv")
    x_test_csv  = os.path.join(OUTPUT_DIR, "X_test.csv")
    train_scaled_df.to_csv(x_train_csv, index=False)
    test_scaled_df.to_csv(x_test_csv,   index=False)

    # track_id reference
    track_id_path = os.path.join(OUTPUT_DIR, "track_id_reference.csv")
    if "track_id" in df.columns:
        df[["track_id"]].reset_index(drop=True).to_csv(track_id_path, index=False)

    # Backward-compat aliases for existing CSV paths
    train_csv = os.path.join(OUTPUT_DIR, "preprocessed_train.csv")
    test_csv  = os.path.join(OUTPUT_DIR, "preprocessed_test.csv")
    train_scaled_df.assign(**{TARGET_COL: y_train.values}).to_csv(train_csv, index=False)
    test_scaled_df.assign(**{TARGET_COL: y_test.values}).to_csv(test_csv, index=False)

    results["preprocessed_dataset"] = {
        "train_csv":    train_csv,
        "test_csv":     test_csv,
        "X_train_npz":  os.path.join(OUTPUT_DIR, "X_train.npz"),
        "X_test_npz":   os.path.join(OUTPUT_DIR, "X_test.npz"),
        "y_train_csv":  y_train_path,
        "y_test_csv":   y_test_path,
        "train_shape":  {"rows": int(X_train.shape[0]), "columns": int(X_train.shape[1])},
        "test_shape":   {"rows": int(X_test.shape[0]),  "columns": int(X_test.shape[1])},
        "columns":      all_feature_names,
    }

    # --------------------------------------------------------
    # 14. PREPROCESSING REPORT (JSON)
    # --------------------------------------------------------
    report = {
        "task":                  "binary_classification",
        "target_column":         TARGET_COL,
        "original_shape":        {"rows": results["n_rows"], "columns": results["n_columns"]},
        "numerical_columns":     numeric_cols,
        "compact_feature_cols":  compact_feature_cols,
        "categorical_columns":   categoric_cols,
        "low_cardinality_ohe":   valid_low_card,
        "high_cardinality_skip": [HIGH_CARD_ID_COL] + HIGH_CARD_TEXT_COLS,
        "target_class_counts":   {str(k): int(v) for k, v in results["target_class_counts"].items()},
        "target_missing_removed": int(results["target_missing_rows_removed"]),
        "split": results["split"],
        "imputation_strategy":   "median",
        "imputation_values":     {k: round(v, 4) for k, v in imputation_values.items()},
        "scaling":               "StandardScaler (fit on train only)",
        "encoding_ohe_sparse":   True,
        "total_features_final":  int(total_features),
        "ohe_feature_count":     len(ohe_feature_names),
        "numerical_feature_count": len(valid_numeric),
        "missing_summary": {
            "total_missing_cells":  results["missing_analysis"]["total_missing_cells"],
            "rows_with_missing":    results["missing_analysis"]["rows_with_missing"],
        },
        "iqr_summary": {
            "n_numeric_columns_checked": iqr_full["n_numeric_columns"],
            "total_outliers_found":      iqr_full["n_outliers"],
            "rows_with_any_outlier":     iqr_full["n_rows_with_outlier"],
        },
        "output_files": {
            "X_train_npz": os.path.join(OUTPUT_DIR, "X_train.npz"),
            "X_test_npz":  os.path.join(OUTPUT_DIR, "X_test.npz"),
            "y_train_csv": y_train_path,
            "y_test_csv":  y_test_path,
            "X_train_csv": x_train_csv,
            "X_test_csv":  x_test_csv,
        },
    }

    report_path = os.path.join(OUTPUT_DIR, "preprocessing_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    results["preprocessing_report_path"] = report_path

    print("\n" + "=" * 60)
    print("PREPROCESSING COMPLETED SUCCESSFULLY")
    print("=" * 60)
    print(f"  Task           : Binary classification  (target = explicit)")
    print(f"  Original shape : {df.shape}")
    print(f"  Train rows     : {X_train.shape[0]}")
    print(f"  Test rows      : {X_test.shape[0]}")
    print(f"  Total features : {X_train.shape[1]}  "
          f"({len(valid_numeric)} numeric + {len(ohe_feature_names)} one-hot)")
    print(f"  y_train 0/1    : {dict(y_train.value_counts().sort_index())}")
    print(f"  y_test  0/1    : {dict(y_test.value_counts().sort_index())}")
    print(f"  Report         : {report_path}")
    print("=" * 60)

    return results


# ============================================================
# RUN DIRECTLY
# ============================================================
if __name__ == "__main__":
    r = run_preprocessing()
    print("\nDone. Keys returned:", list(r.keys()))
