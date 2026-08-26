import os

import pandas as pd

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)
from sklearn.preprocessing import StandardScaler, MinMaxScaler


# =========================================================
# PATHS
# =========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# IMPORTANT:
# Your actual file is SpotifyPlaylistRecords.csv
DATASET_PATH = os.path.join(
    BASE_DIR,
    "SpotifyPlaylistRecords.csv"
)

CHART_DIR = os.path.join(
    BASE_DIR,
    "static",
    "charts"
)

os.makedirs(CHART_DIR, exist_ok=True)


# =========================================================
# LOAD DATA
# =========================================================

def load_data():

    # -----------------------------------------------------
    # Check dataset exists
    # -----------------------------------------------------

    if not os.path.exists(DATASET_PATH):

        raise FileNotFoundError(
            f"Dataset not found: {DATASET_PATH}"
        )

    # -----------------------------------------------------
    # Read CSV
    # -----------------------------------------------------

    data = pd.read_csv(DATASET_PATH)

    # -----------------------------------------------------
    # Required features
    # -----------------------------------------------------

    features = [
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

    # -----------------------------------------------------
    # Check features
    # -----------------------------------------------------

    missing_features = [
        col
        for col in features
        if col not in data.columns
    ]

    if missing_features:

        raise ValueError(
            "Missing feature columns: "
            + ", ".join(missing_features)
        )

    # -----------------------------------------------------
    # Check target
    # -----------------------------------------------------

    if "explicit" not in data.columns:

        raise ValueError(
            "The dataset must contain an 'explicit' column."
        )

    # -----------------------------------------------------
    # Features
    # -----------------------------------------------------

    X = data[features].copy()

    # -----------------------------------------------------
    # Target
    # -----------------------------------------------------

    y = data["explicit"].copy()

    # -----------------------------------------------------
    # Convert target to 0 / 1
    # -----------------------------------------------------

    if y.dtype == "object":

        y = (
            y.astype(str)
            .str.strip()
            .str.lower()
            .map({
                "true": 1,
                "false": 0,
                "yes": 1,
                "no": 0,
                "1": 1,
                "0": 0,
                "explicit": 1,
                "not explicit": 0
            })
        )

    else:

        y = pd.to_numeric(
            y,
            errors="coerce"
        )

    # -----------------------------------------------------
    # Convert feature columns to numeric
    # -----------------------------------------------------

    for col in features:

        X[col] = pd.to_numeric(
            X[col],
            errors="coerce"
        )

    # -----------------------------------------------------
    # Combine and remove missing rows
    # -----------------------------------------------------

    combined = pd.concat(
        [
            X,
            y.rename("explicit")
        ],
        axis=1
    )

    combined = combined.dropna()

    # -----------------------------------------------------
    # Recreate X and y
    # -----------------------------------------------------

    X = combined[features].copy()

    y = combined["explicit"].astype(int)

    # -----------------------------------------------------
    # Validate target
    # -----------------------------------------------------

    unique_target_values = sorted(
        y.unique().tolist()
    )

    if not set(unique_target_values).issubset({0, 1}):

        raise ValueError(
            "The explicit target must contain only 0/1 "
            "or recognized True/False values."
        )

    if y.nunique() < 2:

        raise ValueError(
            "The explicit target must contain two classes."
        )

    return X, y


# =========================================================
# TRAIN MODEL
# =========================================================

def train_model(
    X_train,
    X_test,
    y_train,
    y_test
):

    model = LogisticRegression(
        max_iter=2000,
        random_state=42
    )

    # -----------------------------------------------------
    # Train
    # -----------------------------------------------------

    model.fit(
        X_train,
        y_train
    )

    # -----------------------------------------------------
    # Predictions
    # -----------------------------------------------------

    train_predictions = model.predict(
        X_train
    )

    test_predictions = model.predict(
        X_test
    )

    # -----------------------------------------------------
    # Accuracy
    # -----------------------------------------------------

    train_accuracy = accuracy_score(
        y_train,
        train_predictions
    )

    test_accuracy = accuracy_score(
        y_test,
        test_predictions
    )

    return (
        model,
        train_accuracy,
        test_accuracy,
        test_predictions
    )


# =========================================================
# CREATE SCALING COMPARISON CHART
# =========================================================

def create_scaling_chart(
    methods,
    train_scores,
    test_scores
):

    plt.figure(
        figsize=(10, 6)
    )

    x = list(
        range(len(methods))
    )

    width = 0.35

    # Training accuracy
    plt.bar(
        [
            i - width / 2
            for i in x
        ],
        train_scores,
        width=width,
        label="Training Accuracy"
    )

    # Testing accuracy
    plt.bar(
        [
            i + width / 2
            for i in x
        ],
        test_scores,
        width=width,
        label="Testing Accuracy"
    )

    plt.xticks(
        x,
        methods
    )

    plt.ylim(
        0,
        1.05
    )

    plt.xlabel(
        "Scaling Method"
    )

    plt.ylabel(
        "Accuracy"
    )

    plt.title(
        "Logistic Regression Scaling Comparison"
    )

    plt.legend()

    plt.tight_layout()

    chart_path = os.path.join(
        CHART_DIR,
        "logistic_scaling_comparison.png"
    )

    plt.savefig(
        chart_path,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()

    return chart_path


# =========================================================
# CREATE CONFUSION MATRIX
# =========================================================

def create_confusion_matrix_chart(
    y_test,
    predictions
):

    cm = confusion_matrix(
        y_test,
        predictions
    )

    plt.figure(
        figsize=(7, 6)
    )

    plt.imshow(
        cm
    )

    plt.title(
        "Logistic Regression Confusion Matrix"
    )

    plt.xlabel(
        "Predicted"
    )

    plt.ylabel(
        "Actual"
    )

    plt.xticks(
        [0, 1],
        [
            "Not Explicit",
            "Explicit"
        ]
    )

    plt.yticks(
        [0, 1],
        [
            "Not Explicit",
            "Explicit"
        ]
    )

    # -----------------------------------------------------
    # Display values
    # -----------------------------------------------------

    max_value = cm.max()

    for i in range(cm.shape[0]):

        for j in range(cm.shape[1]):

            # Safe text contrast
            if max_value > 0 and cm[i, j] < max_value * 0.7:

                text_color = "white"

            else:

                text_color = "black"

            plt.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                color=text_color,
                fontsize=12,
                fontweight="bold"
            )

    plt.tight_layout()

    chart_path = os.path.join(
        CHART_DIR,
        "logistic_confusion_matrix.png"
    )

    plt.savefig(
        chart_path,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()

    return chart_path


# =========================================================
# MAIN LOGISTIC REGRESSION
# =========================================================

def run_logistic_regression():

    # -----------------------------------------------------
    # Load
    # -----------------------------------------------------

    X, y = load_data()

    # -----------------------------------------------------
    # Train-test split
    # -----------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(

        X,
        y,

        test_size=0.20,

        stratify=y,

        random_state=42
    )

    # =====================================================
    # UNSCALED MODEL
    # =====================================================

    (
        unscaled_model,
        unscaled_train,
        unscaled_test,
        unscaled_predictions
    ) = train_model(

        X_train,
        X_test,
        y_train,
        y_test
    )

    # =====================================================
    # STANDARD SCALER
    # =====================================================

    standard_scaler = StandardScaler()

    X_train_std = standard_scaler.fit_transform(
        X_train
    )

    X_test_std = standard_scaler.transform(
        X_test
    )

    (
        standard_model,
        standard_train,
        standard_test,
        standard_predictions
    ) = train_model(

        X_train_std,
        X_test_std,
        y_train,
        y_test
    )

    # =====================================================
    # MIN-MAX SCALER
    # =====================================================

    minmax_scaler = MinMaxScaler()

    X_train_mm = minmax_scaler.fit_transform(
        X_train
    )

    X_test_mm = minmax_scaler.transform(
        X_test
    )

    (
        minmax_model,
        minmax_train,
        minmax_test,
        minmax_predictions
    ) = train_model(

        X_train_mm,
        X_test_mm,
        y_train,
        y_test
    )

    # =====================================================
    # CLASSIFICATION REPORT
    # =====================================================

    report = classification_report(
        y_test,
        standard_predictions,
        target_names=[
            "Not Explicit",
            "Explicit"
        ],
        output_dict=True,
        zero_division=0
    )

    # =====================================================
    # CONFUSION MATRIX
    # =====================================================

    confusion_chart = create_confusion_matrix_chart(
        y_test,
        standard_predictions
    )

    # =====================================================
    # SCALING COMPARISON
    # =====================================================

    methods = [
        "Unscaled",
        "StandardScaler",
        "MinMaxScaler"
    ]

    train_scores = [
        unscaled_train,
        standard_train,
        minmax_train
    ]

    test_scores = [
        unscaled_test,
        standard_test,
        minmax_test
    ]

    scaling_chart = create_scaling_chart(
        methods,
        train_scores,
        test_scores
    )

    # =====================================================
    # RETURN RESULTS
    # =====================================================

    return {

        "train_accuracy": round(
            standard_train * 100,
            2
        ),

        "test_accuracy": round(
            standard_test * 100,
            2
        ),

        "training_records": len(
            X_train
        ),

        "testing_records": len(
            X_test
        ),

        "feature_count": len(
            X.columns
        ),

        "features": list(
            X.columns
        ),

        "not_explicit_count": int(
            (y == 0).sum()
        ),

        "explicit_count": int(
            (y == 1).sum()
        ),

        "classification_report": report,

        "confusion_matrix": [
            row.tolist()
            for row in confusion_matrix(
                y_test,
                standard_predictions
            )
        ],

        "scaling_comparison": [

            {
                "method": "Unscaled",

                "training_accuracy": round(
                    unscaled_train * 100,
                    2
                ),

                "testing_accuracy": round(
                    unscaled_test * 100,
                    2
                )
            },

            {
                "method": "StandardScaler",

                "training_accuracy": round(
                    standard_train * 100,
                    2
                ),

                "testing_accuracy": round(
                    standard_test * 100,
                    2
                )
            },

            {
                "method": "MinMaxScaler",

                "training_accuracy": round(
                    minmax_train * 100,
                    2
                ),

                "testing_accuracy": round(
                    minmax_test * 100,
                    2
                )
            }
        ],

        "scaling_chart":
            "charts/logistic_scaling_comparison.png",

        "confusion_chart":
            "charts/logistic_confusion_matrix.png"
    }


# =========================================================
# RUN DIRECTLY
# =========================================================

if __name__ == "__main__":

    result = run_logistic_regression()

    print()
    print("==========================================")
    print("       LOGISTIC REGRESSION")
    print("==========================================")
    print()

    print(
        "Dataset:",
        DATASET_PATH
    )

    print(
        "Target: explicit"
    )

    print(
        "Training Accuracy:",
        result["train_accuracy"],
        "%"
    )

    print(
        "Testing Accuracy:",
        result["test_accuracy"],
        "%"
    )

    print(
        "Training Records:",
        result["training_records"]
    )

    print(
        "Testing Records:",
        result["testing_records"]
    )

    print(
        "Number of Features:",
        result["feature_count"]
    )

    print(
        "Not Explicit:",
        result["not_explicit_count"]
    )

    print(
        "Explicit:",
        result["explicit_count"]
    )

    print()

    print("Features:")

    for feature in result["features"]:
        print("-", feature)

    print()

    print("Scaling Comparison:")

    for item in result["scaling_comparison"]:

        print(
            item["method"],
            "| Training:",
            item["training_accuracy"],
            "%",
            "| Testing:",
            item["testing_accuracy"],
            "%"
        )

    print()

    print(
        "Scaling Chart:",
        result["scaling_chart"]
    )

    print(
        "Confusion Matrix:",
        result["confusion_chart"]
    )

    print()