from flask import Flask, render_template, request  # redirect, url_for
from flask import Response
import plotly
import json

from src.create_figures_strategy import figures_strategy_for_webapp
from src.create_figures_optimization import figures_optimization_for_webapp
from src.create_figures_backtest import figures_backesting_for_webapp

app = Flask("__name__", template_folder='app/templates', static_folder='app/static')


@app.route("/")
@app.route("/strategy")
def strategy_metrics():
    #figures = return_figures()
    figures = figures_strategy_for_webapp()

    # plot ids for the html id tag
    ids = ['figure-{}'.format(i) for i, _ in enumerate(figures)]

    # Convert the plotly figures to JSON for javascript in html template
    figuresJSON = json.dumps(figures, cls=plotly.utils.PlotlyJSONEncoder)

    return render_template("strategy.html",
                           ids=ids,
                           figuresJSON=figuresJSON)


@app.route("/portfolio_optimization")
def portfolio_optimization():
    #figures = return_figures()
    figures = figures_optimization_for_webapp()

    # plot ids for the html id tag
    ids = ['figure-{}'.format(i) for i, _ in enumerate(figures)]

    # Convert the plotly figures to JSON for javascript in html template
    figuresJSON = json.dumps(figures, cls=plotly.utils.PlotlyJSONEncoder)

    return render_template("portfolio_optimization.html",
                           ids=ids,
                           figuresJSON=figuresJSON)


@app.route("/backtest")
def backtest():
    #return render_template("backtest.html")
    #figures = return_figures()
    figures = figures_backesting_for_webapp()

    # plot ids for the html id tag
    ids = ['figure-{}'.format(i) for i, _ in enumerate(figures)]

    # Convert the plotly figures to JSON for javascript in html template
    figuresJSON = json.dumps(figures, cls=plotly.utils.PlotlyJSONEncoder)

    return render_template("backtest.html",
                           ids=ids,
                           figuresJSON=figuresJSON)


@app.route("/description")
def description():
    return render_template("description.html")


# @app.route("/code")
# def code_viewer():
#     option = request.args.get("option", "strategy")  # Default to 'strategy'
    
#     code_map = {
#         "strategy": "src/create_figures_strategy.py",
#         "optimization": "src/create_figures_optimization.py",
#         "backtesting": "src/create_figures_backtest.py"
#     }
    
#     selected_file = code_map.get(option, code_map["strategy"])

#     try:
#         with open(selected_file, "r") as f:
#             code_content = f.read()
#     except FileNotFoundError:
#         code_content = "# Error: File not found."

#     return render_template("code.html", code=code_content, selected=option)

@app.route("/code")
def code_home():
    return render_template("code.html")


@app.route("/code/<section>")
def serve_code(section):
    files = {
        "strategy": "src/strategy_performance.py",
        "optimization": "src/portfolio_optimization.py",
        "backtesting": "src/backtest_optimizer.py"
    }

    if section not in files:
        return Response("Invalid section", status=404)

    try:
        with open(files[section], "r") as f:
            code = f.read()
        return Response(code, mimetype="text/plain")
    except FileNotFoundError:
        return Response("Code file not found", status=404)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5001, use_reloader=False)

