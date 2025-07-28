#from src import strategy_performance, utils
from src import utils
#import utils #utils
#from src import portfolio_optimization
import plotly.graph_objects as go
#import os
import numpy as np
#from scipy.stats import norm
import pandas as pd
from src import aws_s3bucket_load_data


# # portfolio constructio initialization

hold_period = 21
periods = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 14, 21)

aws_df = aws_s3bucket_load_data.load_csv_from_aws_s3("stock_prices.csv",
                                                     index_type="multi")

list_tickers, data = utils.get_universe_from_aws_data(aws_df)

ret = data.pct_change(hold_period).dropna()
ret_df = pd.DataFrame(ret.unstack(), columns=[f'{hold_period}_d_ret'])

# csv version
#ff_df = pd.read_hdf("./data/FF_risk_factors.h5", key='df')
# ff_df = pd.read_csv("data/factors/ff5_factors.csv",
#                     index_col=[0],
#                     parse_dates=[0])
# aws s3 version
ff_df = aws_s3bucket_load_data.load_csv_from_aws_s3("factors/ff5_factors.csv",
                                                    index_type="date")
ff_df.drop(['RF'], axis=1, inplace=True)
# aws version
ff_and_ret_df = ff_df.join(ret_df)


def figures_optimization_for_webapp():

    """
    We create the figures from the portfolio optimization results in  "portfolio_optimization.py"
    These graphs will summarize the optimal portfolio contruction from the underlying alpha strategy

    Input:
        
    Output:
    List of plotly graphs to be used for the web app dashboard
    """

    # factor betas graph
    first_graph = []
    
    # aws s3 version
    betas_df = aws_s3bucket_load_data.load_csv_from_aws_s3("strategy_optimization_2019_2023/po_betas_df.csv",
                                                           index_type="plain")
    factor_names = betas_df.columns.tolist()
    ticker_list = betas_df.index.tolist()

    data = betas_df.values.tolist()
    # fig = go.Figure()

    for ticker, values in zip(ticker_list, data):
        # print(f"{ticker}: {values}")
        first_graph.append(
            go.Bar(
                x=factor_names,
                y=values,
                name=ticker,
            )
        )

    layout_one = dict(
        title=dict(text="Estimated Factor Exposures"),
        xaxis=dict(title="Risk Factors", tickangle=-45),
        yaxis=dict(title="Exposure Level"),
        #width=600,
        #height=600,
        barmode='group',
        legend=dict(title='Legend', orientation='h', y=1.09, yanchor='top'),
        bargap=.2
        
    )
    
    # next optimal weights graph
    second_graph = []
    
    # aws s3 version
    alpha_vector = aws_s3bucket_load_data.load_csv_from_aws_s3("strategy_optimization_2019_2023/po_alpha_vector.csv",
                                                               index_type="plain")
    # aws s3 version 
    optimal_weights = aws_s3bucket_load_data.load_csv_from_aws_s3("strategy_optimization_2019_2023/po_optimal_weights_df.csv",
                                                                  index_type="plain")
    data = optimal_weights
    second_graph.append(go.Bar(
        x=data.index.tolist(),
        y=data.values.flatten().tolist(),
        name='holdings',
        marker=dict(color='red'),
        )
    )

    layout_two = dict(
        title=dict(text="Optimal Weights (in percentage) with Risk Tolerance 12 %", 
                   x=0.5, y=0.95, xanchor='center', yanchor='top'),
        xaxis=dict(title="Assets", tickangle=-45),
        yaxis=dict(title="Optimal Values"),
        legend=dict(title="Legend:", orientation="h", y=1.15),
        bargap=2.95

    )
    
    # next factor exposures graph
    third_graph = []
    port_risk_exposure = betas_df.T.dot(optimal_weights)
    port_risk_exposure = port_risk_exposure.rename(columns={'optimal_weights': 'risk_exposure'})

    data = port_risk_exposure
    third_graph.append(go.Bar(
        x=data.index.tolist(),
        y=data.values.flatten().tolist(),
        name="Portfolio risk exposure",
        marker=dict(color="orange")
        )
    )

    layout_three = dict(
        title=dict(text="Portfolio: Optimal Risk Exposure", x=0.5, y=0.95, xanchor='center', yanchor='top'),
        xaxis=dict(title="Risk Factors", tickangle=-45),
        yaxis=dict(title="Exposure Level"),
        bargap=0.6
    )

    # next transfer coefficient graph
    fourth_graph = []
    ## Tranfer coefficient
    alpha_vector_vs_optimal_weights_df = alpha_vector.join(optimal_weights)
    # Data for the charts
    assets = alpha_vector_vs_optimal_weights_df.index.tolist()
    alpha_vec = np.round(alpha_vector_vs_optimal_weights_df['factor'].values.flatten().tolist(), 3)
    opt_weights = alpha_vector_vs_optimal_weights_df['optimal_weights'].values.flatten().tolist()
    
    # Define data for each chart
    fourth_graph.append(go.Bar(
        x=assets,
        y=opt_weights, 
        name="Optimal Weights",
        marker_color="red",
        yaxis="y1",  # Map to the first y-axis
        xaxis="x1"   # Map to the first x-axis
        )
    )

    fourth_graph.append(go.Bar(
        x=assets,
        y=alpha_vec,
        name="Alpha Vector",
        marker_color="green",
        yaxis="y2",  # Map to the second y-axis
        xaxis="x2"   # Map to the second x-axis
        )
    )

    # Define layout
    layout_four = dict(
        title=dict(text="(Predictive) Alpha Vector vs. (Estimated) Optimal Weights", font_size=18),
        template="plotly_white",
        xaxis=dict(title="",
                   tickangle=-45,
                   domain=[0.0, 1.0],
                   anchor="y1",
                   showticklabels=False,
                   ),  # Top x-axis
        xaxis2=dict(title="",
                    tickangle=-45,
                    domain=[0.0, 1.0],
                    anchor="y2"
                    ),  # Bottom x-axis
        yaxis=dict(title="Optimal Weights", domain=[0.55, 1.0]),  # Top y-axis
        yaxis2=dict(title="Alpha Vector", domain=[0.0, 0.45]),  # Bottom y-axis
        showlegend=False  # Legend not needed, subplot titles explain the data
        )

    # Define annotations for subplot titles
    annotations = [
        dict(
            text="Optimal Weights",
            x=0.5, y=1.05,
            showarrow=False,
            xref="paper", yref="paper",
            font=dict(size=14)
        ),
        dict(
            text="Alpha Vector",
            x=0.5, y=0.47,
            showarrow=False,
            xref="paper", yref="paper",
            font=dict(size=14)
        )
    ]

    # Add annotations to layout
    layout_four["annotations"] = annotations

    figures = []
    figures.append(dict(data=second_graph, layout=layout_two))
    figures.append(dict(data=fourth_graph, layout=layout_four))
    figures.append(dict(data=third_graph, layout=layout_three))
    figures.append(dict(data=first_graph, layout=layout_one))

    return figures


if __name__ == "__main__":
    print("module called")
