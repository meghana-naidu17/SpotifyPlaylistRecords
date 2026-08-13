import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split

# ============================================================
# 1. LOAD DATASET
# ============================================================

file_path = r"/mnt/data/a33898d3-7417-4970-817b-0e06a24150fe.csv"

df = pd.read_csv(file_path)

print("=" * 88)
print("ORIGINAL DATASET")
print("=" * 88)

print(df)

print("\nOriginal Dataset Shape:")
print(df.shape)


# ============================================================
# 2. IQR OUTLIER DETECTION
# ============================================================

# Column selected for outlier detection
features = ["danceability"]

print("\n" + "=" * 88)
print("ORIGINAL STATISTICS - DANCEABILITY")
print("=" * 88)

print(df[features].describe())


# Calculate Q1, Q3 and IQR
Q1 = df[features].quantile(0.25)
Q3 = df[features].quantile(0.75)

IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR


print("\nQ1:")
print(Q1)

print("\nQ3:")
print(Q3)

print("\nIQR:")
print(IQR)

print("\nLower Bound:")
print(lower_bound)

print("\nUpper Bound:")
print(upper_bound)


# ============================================================
# 3. FIND OUTLIERS
# ============================================================

outliers = df[
    (df["danceability"] < lower_bound["danceability"]) |
    (df["danceability"] > upper_bound["danceability"])
]

print("\n" + "=" * 88)
print("OUTLIER INFORMATION")
print("=" * 88)

print("\nNumber of Outliers:")
print(len(outliers))

print("\nOutliers:")
print(outliers[["danceability"]])


# ============================================================
# 4. CLIP OUTLIERS
# ============================================================

df["danceability_clipped"] = df["danceability"].clip(
    lower=lower_bound["danceability"],
    upper=upper_bound["danceability"]
)

print("\n" + "=" * 88)
print("BEFORE AND AFTER CLIPPING")
print("=" * 88)

print("\nMinimum BEFORE Clipping:")
print(df["danceability"].min())

print("\nMaximum BEFORE Clipping:")
print(df["danceability"].max())

print("\nMinimum AFTER Clipping:")
print(df["danceability_clipped"].min())

print("\nMaximum AFTER Clipping:")
print(df["danceability_clipped"].max())


# ============================================================
# 5. SELECT NUMERICAL COLUMNS
# ============================================================

num_cols = [
    "popularity",
    "duration_ms",
    "explicit",
    "danceability_clipped",
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


# ============================================================
# 6. TRAIN-TEST SPLIT
# ============================================================

# This dataset has no PlacementStatus column,
# so a normal random train-test split is used.

train_df, test_df = train_test_split(
    df,
    test_size=0.30,
    random_state=42
)

print("\n" + "=" * 88)
print("TRAIN-TEST SPLIT")
print("=" * 88)

print("\nOriginal Dataset Shape:")
print(df.shape)

print("\nTraining Dataset Shape:")
print(train_df.shape)

print("\nTesting Dataset Shape:")
print(test_df.shape)


# ============================================================
# 7. DATA BEFORE SCALING
# ============================================================

print("\n" + "=" * 88)
print("TRAINING DATA BEFORE SCALING")
print("=" * 88)

print(train_df[num_cols].head())


print("\n" + "=" * 88)
print("TESTING DATA BEFORE SCALING")
print("=" * 88)

print(test_df[num_cols].head())


# ============================================================
# 8. MIN-MAX SCALING
# ============================================================

minmax_scaler = MinMaxScaler()

# Fit ONLY on training data
train_minmax = minmax_scaler.fit_transform(
    train_df[num_cols]
)

# Transform test data using the same scaler
test_minmax = minmax_scaler.transform(
    test_df[num_cols]
)


# Convert arrays back to DataFrames
train_minmax_df = pd.DataFrame(
    train_minmax,
    columns=num_cols,
    index=train_df.index
)

test_minmax_df = pd.DataFrame(
    test_minmax,
    columns=num_cols,
    index=test_df.index
)


print("\n" + "=" * 88)
print("MIN-MAX SCALING")
print("=" * 88)

print("\nTraining Data AFTER Min-Max Scaling:")
print(train_minmax_df.head())

print("\nTesting Data AFTER Min-Max Scaling:")
print(test_minmax_df.head())


# ============================================================
# 9. STANDARD SCALING
# ============================================================

standard_scaler = StandardScaler()

# Fit ONLY on training data
train_standard = standard_scaler.fit_transform(
    train_df[num_cols]
)

# Transform test data using the same scaler
test_standard = standard_scaler.transform(
    test_df[num_cols]
)


# Convert arrays back to DataFrames
train_standard_df = pd.DataFrame(
    train_standard,
    columns=num_cols,
    index=train_df.index
)

test_standard_df = pd.DataFrame(
    test_standard,
    columns=num_cols,
    index=test_df.index
)


print("\n" + "=" * 88)
print("STANDARD SCALING")
print("=" * 88)

print("\nTraining Data AFTER Standard Scaling:")
print(train_standard_df.head())

print("\nTesting Data AFTER Standard Scaling:")
print(test_standard_df.head())


# ============================================================
# 10. FINAL DATASET INFORMATION
# ============================================================

print("\n" + "=" * 88)
print("FINAL DATASET INFORMATION")
print("=" * 88)

print("\nOriginal Dataset Shape:")
print(df.shape)

print("\nTraining Dataset Shape:")
print(train_df.shape)

print("\nTesting Dataset Shape:")
print(test_df.shape)

print("\nNumber of Numerical Columns:")
print(len(num_cols))


# ============================================================
# 11. SCALING COMPARISON
# ============================================================

print("\n" + "=" * 88)
print("SCALING COMPARISON - FIRST 5 TRAINING ROWS")
print("=" * 88)

print("\nOriginal:")
print(train_df[num_cols].head())

print("\nMin-Max Scaled:")
print(train_minmax_df.head())

print("\nStandard Scaled:")
print(train_standard_df.head())


print("\n" + "=" * 88)
print("PREPROCESSING COMPLETED SUCCESSFULLY")
print("=" * 88)