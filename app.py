from flask import Flask, render_template, request

from load_data import get_data_summary
from playlist_eda import run_eda
from preprocessing_data import run_preprocessing
from logistic_regression import run_logistic_regression, VALID_PENALTIES
from linear_regression import run_linear_regression
from ml_models import run_classifier, run_kmeans
from hierarchical_clustering import run_hierarchical_clustering
from dbscan_clustering import run_dbscan
from autoencoder_model import run_autoencoder
from pca_model import run_pca
from tsne_umap_model import run_tsne_umap


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
# AUTOENCODER REPRESENTATION LEARNING
# =========================================================

@app.route("/unsupervised/autoencoder")
def autoencoder_page():

    latent_dim = request.args.get("latent_dim", 2)
    hidden_dim = request.args.get("hidden_dim", 32)
    activation = request.args.get("activation", "relu")
    epochs = request.args.get("epochs", 80)
    alpha = request.args.get("alpha", 0.0001)

    try:

        results = run_autoencoder(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            activation=activation,
            epochs=epochs,
            alpha=alpha
        )

        return render_template(
            "autoencoder.html",
            active="autoencoder",
            results=results,
            error=None
        )

    except Exception as e:

        return render_template(
            "autoencoder.html",
            active="autoencoder",
            results=None,
            error=str(e)
        )


# =========================================================
# PRINCIPAL COMPONENT ANALYSIS (PCA)
# =========================================================

@app.route("/unsupervised/pca")
def pca_page():

    n_components = request.args.get("n_components", 5)
    whiten = request.args.get("whiten", "false")
    svd_solver = request.args.get("svd_solver", "auto")

    try:

        results = run_pca(
            n_components=n_components,
            whiten=whiten,
            svd_solver=svd_solver
        )

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
# RUN APPLICATION
# =========================================================

if __name__ == "__main__":

    app.run(
        debug=True
    )