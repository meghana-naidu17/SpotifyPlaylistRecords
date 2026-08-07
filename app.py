from flask import Flask, render_template
from load_data import get_data_summary
from playlist_eda import run_eda

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html", active="none")

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

if __name__ == "__main__":
    app.run(debug=True)