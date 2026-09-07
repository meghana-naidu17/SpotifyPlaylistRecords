from flask import Flask, render_template, request

from load_data import get_data_summary
from playlist_eda import run_eda
from preprocessing_data import run_preprocessing
from logistic_regression import run_logistic_regression, VALID_PENALTIES
from linear_regression import run_linear_regression



# =========================================================
# FLASK APP
# =========================================================

app = Flask(__name__)


# =========================================================
# HOME PAGE
# =========================================================

@app.route("/")
def index():

    return render_template(
        "index.html",
        active="none"
    )


# =========================================================
# DATASET SUMMARY
# =========================================================

@app.route("/data-loading")
def data_loading():

    try:

        summary = get_data_summary()

        return render_template(
            "index.html",
            active="data-loading",
            summary=summary,
            error=None
        )

    except Exception as e:

        return render_template(
            "index.html",
            active="data-loading",
            summary=None,
            error=str(e)
        )


# =========================================================
# EXPLORATORY DATA ANALYSIS
# =========================================================

@app.route("/eda")
def eda_page():

    try:

        results = run_eda()

        return render_template(
            "eda.html",
            active="eda",
            results=results,
            error=None
        )

    except Exception as e:

        return render_template(
            "eda.html",
            active="eda",
            results=None,
            error=str(e)
        )


# =========================================================
# DATA PREPROCESSING
# =========================================================

@app.route("/preprocessing")
def preprocessing_page():

    try:

        results = run_preprocessing()

        return render_template(
            "preprocessing.html",
            active="preprocessing",
            results=results,
            error=None
        )

    except Exception as e:

        return render_template(
            "preprocessing.html",
            active="preprocessing",
            results=None,
            error=str(e)
        )


# =========================================================
# LOGISTIC REGRESSION
# =========================================================

@app.route("/logistic-regression")
def logistic_regression_page():

    # -----------------------------------------------------
    # Read the regularization choice from the dropdown.
    # Falls back to "l2" and is validated against the
    # allowed set so a bad/missing query param can't
    # break the model call.
    # -----------------------------------------------------

    penalty = request.args.get("penalty", "l2").lower().strip()

    if penalty not in VALID_PENALTIES:
        penalty = "l2"

    try:

        results = run_logistic_regression(penalty=penalty)

        return render_template(
            "logistic_regression.html",
            active="logistic-regression",
            results=results,
            selected_penalty=penalty,
            error=None
        )

    except Exception as e:

        return render_template(
            "logistic_regression.html",
            active="logistic-regression",
            results=None,
            selected_penalty=penalty,
            error=str(e)
        )


# =========================================================
# LINEAR REGRESSION
# =========================================================

@app.route("/linear-regression")
def linear_regression_page():

    try:

        results = run_linear_regression()

        return render_template(
            "linear_regression.html",
            active="linear-regression",
            results=results,
            error=None
        )

    except Exception as e:

        return render_template(
            "linear_regression.html",
            active="linear-regression",
            results=None,
            error=str(e)
        )


# =========================================================
# RUN APPLICATION
# =========================================================

if __name__ == "__main__":

    app.run(
        debug=True
    )