from flask import Flask, render_template, redirect, url_for
import plotly, json

from src.create_figures_strategy import figures_strategy_for_webapp
from src.create_figures_optimization import figures_optimization_for_webapp
from src.create_figures_backtest import figures_backesting_for_webapp

app = Flask("__name__")

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


if __name__ == "__main__":
    app.run(debug=True)

