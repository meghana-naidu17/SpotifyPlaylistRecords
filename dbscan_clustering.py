import os
import numpy as np
import pandas as pd

from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

PREPROCESSED_PATH = os.path.join(
    BASE_DIR,
    "processed_data",
    "preprocessed_dataset.csv"
)


def _load_preprocessed():

    if not os.path.exists(PREPROCESSED_PATH):
        raise FileNotFoundError(
            "Preprocessed dataset not found: "
            "processed_data/preprocessed_dataset.csv"
        )

    df = pd.read_csv(PREPROCESSED_PATH)

    # Remove target/ID columns if they exist.
    drop_columns = [
        "explicit",
        "track_id",
        "id"
    ]

    df = df.drop(
        columns=[
            col for col in drop_columns
            if col in df.columns
        ],
        errors="ignore"
    )

    # DBSCAN requires numerical data.
    numeric = df.select_dtypes(
        include=np.number
    ).copy()

    if numeric.empty:
        raise ValueError(
            "No numeric features were found in the preprocessed dataset."
        )

    # Replace infinity values.
    numeric = numeric.replace(
        [np.inf, -np.inf],
        np.nan
    )

    # Fill missing values using median.
    numeric = numeric.fillna(
        numeric.median(numeric_only=True)
    )

    numeric = numeric.fillna(0)

    # Remove constant columns.
    varying_columns = numeric.columns[
        numeric.nunique(dropna=False) > 1
    ]

    numeric = numeric[varying_columns]

    if numeric.empty:
        raise ValueError(
            "No varying numeric features are available for DBSCAN."
        )

    return numeric


def run_dbscan(
    eps=0.75,
    min_samples=5,
    max_rows=1200
):

    numeric = _load_preprocessed()

    original_rows = len(numeric)

    # DBSCAN can become extremely expensive with
    # very large datasets.
    #
    # Your dataset contains many thousands of rows,
    # so we use a reproducible sample for clustering.
    if len(numeric) > max_rows:

        sample = numeric.sample(
            n=max_rows,
            random_state=42
        )

    else:

        sample = numeric.copy()

    # Standardize the preprocessed features.
    scaler = StandardScaler()

    X = scaler.fit_transform(
        sample.to_numpy(dtype=float)
    )

    # Run DBSCAN.
    model = DBSCAN(
        eps=float(eps),
        min_samples=int(min_samples),
        n_jobs=-1
    )

    labels = model.fit_predict(X)

    # PCA is ONLY used for visualization.
    pca = PCA(
        n_components=2,
        random_state=42
    )

    reduced = pca.fit_transform(X)

    # Count clusters.
    unique_labels = set(labels)

    cluster_count = len(
        unique_labels -
        {-1}
    )

    noise_count = int(
        np.sum(labels == -1)
    )

    # Cluster distribution.
    counts = (
        pd.Series(labels)
        .value_counts()
        .sort_index()
    )

    rows = []

    for cluster_id, count in counts.items():

        cluster_id = int(cluster_id)

        if cluster_id == -1:

            name = "Noise (-1)"

        else:

            name = f"Cluster {cluster_id}"

        rows.append({
            "cluster": name,
            "records": int(count)
        })

    # Points for browser visualization.
    points = []

    for point, label in zip(
        reduced,
        labels
    ):

        points.append({
            "x": float(point[0]),
            "y": float(point[1]),
            "label": int(label)
        })

    return {

        "eps": float(eps),

        "min_samples": int(
            min_samples
        ),

        "clusters": int(
            cluster_count
        ),

        "noise": int(
            noise_count
        ),

        "sample_size": int(
            len(sample)
        ),

        "dataset_records": int(
            original_rows
        ),

        "feature_count": int(
            numeric.shape[1]
        ),

        "rows": rows,

        "points": points
    }