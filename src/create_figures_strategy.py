from src import  utils
#import utils
import plotly.graph_objects as go
import os
import numpy as np
from scipy.stats import norm
import pandas as pd
from src import aws_s3bucket_load_data

path = '../data/stock_prices.h5'
base_path = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(base_path, path)


def figures_strategy_for_webapp():

    """
        We create the figures provided by the trading strategy and its performance evaluation in file "strategy_performance.py"
        These graphs will summarize our trading stratgy performance and will be used to design the web app dashboard. 

        Input:
         Relevant modules from "strategy_performance.py"
        Output:
        List of graphs to be used for the web app dashboard
    """

    # First graph: Portfolio performance metrics
    first_graph = []
                               
    # aws s3 version
    portfolio_performance = (aws_s3bucket_load_data.
                             load_csv_from_aws_s3("strategy_optimization_2019_2023/strategy_performance_metrics.csv", 
                                                  index_type="plain")
                             )
                                        
    for metric in portfolio_performance.columns:
        first_graph.append(go.Bar(
            x=portfolio_performance.index,
            y=portfolio_performance[metric],
            name=metric
        ))

    # Customize Layout
    layout_one = dict(
        title=dict(text="Holding Periods (Days): Sharpe Ratio vs. Sortino Ratio vs Calmar Ratio", 
                        x=0.5,
                        y=0.9,
                        font=dict(size=13),
                        xanchor='center',
                        yanchor='top'),
        xaxis=dict(title="Holding Periods", tickangle=-45),
        yaxis=dict(title="Values"),
        barmode='group',
        legend=dict(title="Legend:", orientation="h", y=1.1, yanchor='top')
    )
    # Secon graph: Strategy's return distribution
    # aws s3 version
    factor_returns = aws_s3bucket_load_data.load_csv_from_aws_s3("strategy_optimization_2019_2023/factor_returns.csv",
                                                                 index_type="date")

    second_graph = []
    second_graph.append(go.Histogram(
        x=factor_returns['21D'],
        nbinsx=30,
        histnorm='probability density',
        opacity=0.7,
        marker=dict(color='green'),
        name='Histogram'
    ))
    mean = factor_returns['21D'].mean()
    std = factor_returns['21D'].std()

    N = 1000
    x = np.linspace(factor_returns['21D'].min(), factor_returns['21D'].max(), N)
    pdf = norm.pdf(x, loc=mean, scale=std)
    # Create the normal distribution line plot
    second_graph.append(go.Scatter(
        x=x,
        y=pdf,
        mode='lines',
        line=dict(color='red', width=2.5),
        name='Normal Distribution'
    ))

    # Add a vertical line for the mean
    second_graph.append(go.Scatter(
        x=[mean, mean],
        y=[0, max(pdf)],
        mode='lines',
        line=dict(color='black', width=3.5, dash='dash'),
        name='Mean Line'
    ))

    # Update layout for title, labels, and legend
    layout_two = dict(
        title=dict(text="21D Holding Portfolio Returns: Histogram vs. Normal Distribution", 
                   x=0.5,
                   y=0.9,
                   font=dict(size=13),
                   xanchor='center',
                   yanchor='top'),
        xaxis=dict(title="Returns"),
        yaxis=dict(title="Density"),
        legend=dict(x=0.7, y=0.95),
        # template='ggplot2',
    
    )

    # Combine the plots into a figure
    # aws s3 version
    mean_ret_by_quantile = aws_s3bucket_load_data.load_csv_from_aws_s3("strategy_optimization_2019_2023/mean_ret_by_quantile.csv",
                                                                       index_type="plain")

    third_graph = []
    third_graph.append(go.Bar(
                x=mean_ret_by_quantile['21D'].index,
                y=mean_ret_by_quantile['21D'],
                name='metric',
                marker=dict(color='red')

                ))

    # Customize Layout
    layout_three = dict(
        title=dict(text="21D Holdings: Mean Forward Return By Quantile",
                   xanchor='center',
                   yanchor='top',
                   font=dict(size=13),
                   x=0.5,
                   y=0.9),
        xaxis=dict(title="Qantiles"),
        yaxis=dict(title="Values"),
        barmode='group',
        # template='ggplot2'
    )

    
    # Fourth graph: Stragy vs. S&P500 Benchmark performance comparison 
    fourth_graph = []
    # test sp500 vs the strategy performance
    forward_21d_strtgy = factor_returns[['21D']]
    # aws s3 version
    forward_21d_sp500 = aws_s3bucket_load_data.load_csv_from_aws_s3("strategy_optimization_2019_2023/fwd_ret_21d_sp500.csv",
                                                                    index_type="date")

    forward_21d_sp500 = forward_21d_sp500.loc[forward_21d_strtgy.index, :]
    sp500_strtgy_df = pd.concat([forward_21d_sp500, forward_21d_strtgy], axis=1)
    sp500_vs_strtgy_metrics = utils.performance_metrics(sp500_strtgy_df)
    sp500_vs_strtgy_metrics.rename(index={'21D': 'strategy'}, inplace=True)

    dict_benchmark_and_strategy = {}

    list_columns = sp500_vs_strtgy_metrics.index.tolist()
    metrics = sp500_vs_strtgy_metrics.columns.tolist()

    for i in range(len(list_columns)):
        dict_benchmark_and_strategy[list_columns[i]] = sp500_vs_strtgy_metrics.loc[list_columns[i], :].tolist()

    # fig = go.Figure()

    for col in dict_benchmark_and_strategy.keys():
        fourth_graph.append(go.Bar(
            x=metrics,
            y=dict_benchmark_and_strategy[col],
            name=col,
        ))

    # Create the layout
    layout_four = dict(
        title=dict(text="Performance Metrics: S&P500 vs Strategy", 
                   x=0.5,
                   y=0.9,
                   font=dict(size=13, family="bold"),
                   xanchor='center',
                   yanchor='top'),
        xaxis=dict(title="Metrics"),
        yaxis=dict(title="Values"),
        barmode='group',  # Grouped bars
        # legend=dict(title="Legend:", orientation="h", y=1.15, xanchor="center"),
        legend=dict(title="Legend:", orientation="h", y=1.15),

        titlefont=dict(size=14),
    )

    figures = []
    figures.append(dict(data=first_graph, layout=layout_one))
    figures.append(dict(data=second_graph, layout=layout_two))
    figures.append(dict(data=third_graph, layout=layout_three))
    figures.append(dict(data=fourth_graph, layout=layout_four))

    return figures


if __name__ == "__main__":
    print("Inplace Calling")
