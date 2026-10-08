from flask import Flask, render_template, request

from load_data import get_data_summary
from playlist_eda import run_eda
from preprocessing_data import run_preprocessing
from logistic_regression import run_logistic_regression, VALID_PENALTIES
from linear_regression import run_linear_regression
from ml_models import run_classifier, run_kmeans
from hierarchical_clustering import run_hierarchical_clustering
from dbscan_clustering import run_dbscan
from pca_model import run_pca
from tsne_umap_model import run_tsne_umap
from anomaly_detection import run_anomaly_detection
from model_evaluation import (
    run_data_splitting_evaluation,
    run_cross_validation_evaluation,
    run_classification_metrics_evaluation,
    run_regression_metrics_evaluation,
    run_calibration_evaluation,
    run_hyperparameter_search_evaluation,
    run_learning_curves_evaluation,
    run_significance_testing_evaluation
)


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

    penalty = request.args.get(
        "penalty",
        "l2"
    ).lower().strip()

    if penalty not in VALID_PENALTIES:
        penalty = "l2"

    try:

        results = run_logistic_regression(
            penalty=penalty
        )

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

    penalty = request.args.get(
        "penalty",
        "l2"
    ).lower().strip()

    if penalty not in VALID_PENALTIES:
        penalty = "l2"

    try:

        results = run_linear_regression(
            penalty=penalty
        )

        return render_template(
            "linear_regression.html",
            active="linear-regression",
            results=results,
            selected_penalty=penalty,
            error=None
        )

    except Exception as e:

        return render_template(
            "linear_regression.html",
            active="linear-regression",
            results=None,
            selected_penalty=penalty,
            error=str(e)
        )


# =========================================================
# DECISION TREE
# =========================================================

@app.route("/decision-tree/<algorithm>")
def decision_tree_page(algorithm):

    try:

        return render_template(
            "model_results.html",
            active="decision-tree",
            results=run_classifier(
                "tree",
                algorithm
            ),
            error=None
        )

    except Exception as e:

        return render_template(
            "model_results.html",
            active="decision-tree",
            results=None,
            error=str(e)
        )


# =========================================================
# ENSEMBLE MODELS
# =========================================================

@app.route("/ensemble/<family>/<algorithm>")
def ensemble_page(family, algorithm):

    try:

        return render_template(
            "model_results.html",
            active="ensemble",
            results=run_classifier(
                family,
                algorithm
            ),
            error=None
        )

    except Exception as e:

        return render_template(
            "model_results.html",
            active="ensemble",
            results=None,
            error=str(e)
        )


# =========================================================
# HIERARCHICAL CLUSTERING
# =========================================================

@app.route("/unsupervised/hierarchical")
def hierarchical_page():

    method = request.args.get(
        "method",
        "ward"
    )

    k_param = request.args.get("k", None)

    try:

        results = run_hierarchical_clustering(
            method=method,
            k=k_param
        )

        return render_template(
            "hierarchical.html",
            active="hierarchical",
            results=results,
            error=None
        )

    except Exception as e:

        return render_template(
            "hierarchical.html",
            active="hierarchical",
            results=None,
            error=str(e)
        )


# =========================================================
# K-MEANS CLUSTERING
# =========================================================

@app.route("/unsupervised/kmeans")
def kmeans_page():

    method = request.args.get(
        "method",
        "elbow"
    )

    try:

        k = int(
            request.args.get(
                "k",
                3
            )
        )

    except (ValueError, TypeError):

        k = 3

    try:

        return render_template(
            "kmeans.html",
            active="unsupervised",
            results=run_kmeans(
                method,
                k
            ),
            error=None
        )

    except Exception as e:

        return render_template(
            "kmeans.html",
            active="unsupervised",
            results=None,
            error=str(e)
        )


# =========================================================
# DBSCAN CLUSTERING
# =========================================================

@app.route("/unsupervised/dbscan")
def dbscan_page():

    try:

        eps = float(
            request.args.get(
                "eps",
                1.6
            )
        )

    except (ValueError, TypeError):

        eps = 1.6

    try:

        min_samples = int(
            request.args.get(
                "min_samples",
                5
            )
        )

    except (ValueError, TypeError):

        min_samples = 5

    # Prevent invalid values.

    eps = max(
        0.05,
        eps
    )

    min_samples = max(
        2,
        min_samples
    )

    try:

        results = run_dbscan(
            eps=eps,
            min_samples=min_samples
        )

        return render_template(
            "dbscan.html",
            active="dbscan",
            results=results,
            error=None
        )

    except Exception as e:

        return render_template(
            "dbscan.html",
            active="dbscan",
            results=None,
            error=str(e)
        )


# =========================================================
# ANOMALY DETECTION
# =========================================================

@app.route("/unsupervised/anomaly-detection")
def anomaly_detection_page():

    try:
        contamination = float(request.args.get("contamination", 0.05))
    except (ValueError, TypeError):
        contamination = 0.05

    try:
        n_estimators = int(request.args.get("n_estimators", 100))
    except (ValueError, TypeError):
        n_estimators = 100

    try:
        nu = float(request.args.get("nu", 0.05))
    except (ValueError, TypeError):
        nu = 0.05

    kernel = request.args.get("kernel", "rbf")

    try:
        hidden_dim = int(request.args.get("hidden_dim", 32))
    except (ValueError, TypeError):
        hidden_dim = 32

    try:
        latent_dim = int(request.args.get("latent_dim", 4))
    except (ValueError, TypeError):
        latent_dim = 4

    try:
        epochs = int(request.args.get("epochs", 80))
    except (ValueError, TypeError):
        epochs = 80

    try:
        results = run_anomaly_detection(
            contamination=contamination,
            n_estimators=n_estimators,
            nu=nu,
            kernel=kernel,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            epochs=epochs,
        )
        return render_template(
            "anomaly_detection.html",
            active="anomaly-detection",
            results=results,
            error=None,
        )
    except Exception as e:
        return render_template(
            "anomaly_detection.html",
            active="anomaly-detection",
            results=None,
            error=str(e),
        )


# =========================================================
# PRINCIPAL COMPONENT ANALYSIS (PCA)
# =========================================================

@app.route("/unsupervised/pca")
def pca_page():

    try:

        results = run_pca()

        return render_template(
            "pca.html",
            active="pca",
            results=results,
            error=None
        )

    except Exception as e:

        return render_template(
            "pca.html",
            active="pca",
            results=None,
            error=str(e)
        )


# =========================================================
# t-SNE & UMAP MANIFOLD LEARNING
# =========================================================

@app.route("/unsupervised/tsne-umap")
def tsne_umap_page():

    method = request.args.get("method", "both")
    perplexity = request.args.get("perplexity", 30)
    learning_rate = request.args.get("learning_rate", 200)
    n_iter = request.args.get("n_iter", 1000)
    n_neighbors = request.args.get("n_neighbors", 15)
    min_dist = request.args.get("min_dist", 0.1)
    metric = request.args.get("metric", "euclidean")

    try:

        results = run_tsne_umap(
            method=method,
            perplexity=perplexity,
            learning_rate=learning_rate,
            n_iter=n_iter,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            metric=metric
        )

        return render_template(
            "tsne_umap.html",
            active="tsne-umap",
            results=results,
            error=None
        )

    except Exception as e:

        return render_template(
            "tsne_umap.html",
            active="tsne-umap",
            results=None,
            error=str(e)
        )


# =========================================================
# MODEL EVALUATION ROUTES
# =========================================================

@app.route("/model-evaluation")
@app.route("/model-evaluation/data-splitting")
def eval_data_splitting():
    force_refresh = request.args.get("refresh", "false").lower() == "true"
    try:
        results = run_data_splitting_evaluation(force_refresh=force_refresh)
        return render_template("model_evaluation.html", active="eval-splitting", results=results, error=None)
    except Exception as e:
        return render_template("model_evaluation.html", active="eval-splitting", results=None, error=str(e))


@app.route("/model-evaluation/cross-validation")
def eval_cross_validation():
    force_refresh = request.args.get("refresh", "false").lower() == "true"
    try:
        results = run_cross_validation_evaluation(force_refresh=force_refresh)
        return render_template("model_evaluation.html", active="eval-cv", results=results, error=None)
    except Exception as e:
        return render_template("model_evaluation.html", active="eval-cv", results=None, error=str(e))


@app.route("/model-evaluation/classification-metrics")
def eval_classification_metrics():
    force_refresh = request.args.get("refresh", "false").lower() == "true"
    try:
        results = run_classification_metrics_evaluation(force_refresh=force_refresh)
        return render_template("model_evaluation.html", active="eval-classification", results=results, error=None)
    except Exception as e:
        return render_template("model_evaluation.html", active="eval-classification", results=None, error=str(e))


@app.route("/model-evaluation/regression-metrics")
def eval_regression_metrics():
    force_refresh = request.args.get("refresh", "false").lower() == "true"
    try:
        results = run_regression_metrics_evaluation(force_refresh=force_refresh)
        return render_template("model_evaluation.html", active="eval-regression", results=results, error=None)
    except Exception as e:
        return render_template("model_evaluation.html", active="eval-regression", results=None, error=str(e))


@app.route("/model-evaluation/calibration")
def eval_calibration():
    force_refresh = request.args.get("refresh", "false").lower() == "true"
    try:
        results = run_calibration_evaluation(force_refresh=force_refresh)
        return render_template("model_evaluation.html", active="eval-calibration", results=results, error=None)
    except Exception as e:
        return render_template("model_evaluation.html", active="eval-calibration", results=None, error=str(e))


@app.route("/model-evaluation/hyperparameter-search")
def eval_hyperparameter_search():
    force_refresh = request.args.get("refresh", "false").lower() == "true"
    try:
        results = run_hyperparameter_search_evaluation(force_refresh=force_refresh)
        return render_template("model_evaluation.html", active="eval-hyperparam", results=results, error=None)
    except Exception as e:
        return render_template("model_evaluation.html", active="eval-hyperparam", results=None, error=str(e))


@app.route("/model-evaluation/learning-curves")
def eval_learning_curves():
    force_refresh = request.args.get("refresh", "false").lower() == "true"
    try:
        results = run_learning_curves_evaluation(force_refresh=force_refresh)
        return render_template("model_evaluation.html", active="eval-curves", results=results, error=None)
    except Exception as e:
        return render_template("model_evaluation.html", active="eval-curves", results=None, error=str(e))


@app.route("/model-evaluation/significance-testing")
def eval_significance_testing():
    force_refresh = request.args.get("refresh", "false").lower() == "true"
    try:
        results = run_significance_testing_evaluation(force_refresh=force_refresh)
        return render_template("model_evaluation.html", active="eval-significance", results=results, error=None)
    except Exception as e:
        return render_template("model_evaluation.html", active="eval-significance", results=None, error=str(e))


# =========================================================
# RUN APPLICATION
# =========================================================

if __name__ == "__main__":

    app.run(
        debug=True
    )