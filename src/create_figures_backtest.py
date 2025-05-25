import plotly.graph_objects as go
import pandas as pd
import numpy as np
import os


def figures_backesting_for_webapp():
    """
    We create the figures from the portfolio optimization Backtesing results from "backtest_optimizer.py"
    These graphs will summarize the optimal portfolios from the backtest of the nderlying alpha strategy

    Input: void
        
    Output:
    List of plotly graphs to be used for the web app dashboard
    """
    path1 = "../data/attrib_df.h5"
    path2 = "../data/p_charac_df.h5"
    base_path = os.path.dirname(os.path.abspath(__file__))
    file_path1 = os.path.join(base_path, path1)
    file_path2 = os.path.join(base_path, path2)
    #port_attribution_df = pd.read_hdf("data_maip/data/attrib.h5", key='df')
    port_attribution_df = pd.read_hdf(file_path1)
    #port_charac_df = pd.read_hdf("data_manip/data/port_charac.h5", key='df')
    port_charac_df = pd.read_hdf(file_path2)

    # First graph: PnL Distribution
    pnl = port_attribution_df['daily_PnL']
    first_graph = []

    first_graph.append(
        go.Histogram(
            x=pnl,
            nbinsx=50,
            histnorm='probability density',
            opacity=0.7,
            marker=dict(color='Tomato'),
            name='Histogram')
        )

    layout_one = dict(
            title=dict(text="PnL Distribution", xanchor='center', yanchor='top', x=0.5),
            xaxis=dict(title="PnL"),
            yaxis=dict(title="Frequency"),
            showlegend=True,
            # legend=dict(x=0.7, y=0.95),
            legend=dict(x=0.7, y=0.95),
            margin=dict(t=60),
            #template='plotly_white',
            )

    # Second graph: PnL distrib and VaR + CVaR
    confidence_level = .95
    VaR = - np.percentile(pnl, 100*(1-confidence_level))
    # print(f"Value at Risk {VaR}")
    # expected shortfall
    ieval = pnl > VaR
    ES = np.sum(pnl * ieval) / np.sum(ieval)
    # print(f"Expected shortfall: {ES}")

    second_graph = []
    counts, bins = np.histogram(pnl, bins=50)
    hist_ymax = counts.max()
    offset = 7e-2 * (max(pnl) - min(pnl))

    # Histogram of PnL
    second_graph.append(
        go.Histogram(
            x=pnl,
            nbinsx=50,
            name='PnL Distribution',
            opacity=0.7,
            marker=dict(color='Tomato')
        )
    )

    # VaR vertical line
    second_graph.append(
        go.Scatter(
            x=[-VaR, -VaR],
            y=[0, hist_ymax],  # placeholder for y, will scale later
            mode='lines',
            name=f'VaR 95% ({VaR:.2f})',
            line=dict(color='red', dash='dash', width=5)
            )
    )

    # CVaR vertical line
    second_graph.append(
        go.Scatter(
            # x=[-ES + offset, -ES + offset],
            x=[-ES, -ES],
            y=[0, hist_ymax],  # placeholder for y, will scale later
            mode='lines',
            name=f'CVaR 95% ({ES:.2f})',
            line=dict(color='purple', dash='dash', width=5)
            )
            )

    layout_two = dict(
        title=dict(
            text="PnL Distribution with VaR and CVaR (95%)",
            x=0.5,
            xanchor='center'
        ),
        xaxis=dict(title='PnL'),
        yaxis=dict(title='Frequency'),
        showlegend=True,
        # bargap=0.05,
        margin=dict(t=60),
        #template='plotly_white'
    )

    # Third graph: Portfolio Characteristics
    dfs = port_charac_df.reset_index('date')
   
    third_graph = [
                go.Scatter(x=dfs['date'], y=dfs['long'], name='long', line=dict(color='red')),
                go.Scatter(x=dfs['date'], y=dfs['short'], name='short', line=dict(color='blue')),
                go.Scatter(x=dfs['date'], y=dfs['net'], name='net', line=dict(color='mediumpurple')),
                go.Scatter(x=dfs['date'], y=dfs['gmv'], name='gmv', line=dict(color='gray')),
                go.Scatter(x=dfs['date'], y=dfs['traded'], name='traded', line=dict(color='orange'))
                ]

    # Layout
    layout_three = dict(
        title=dict(text="Portfolio Characteristics (traded-long-short-gross-net)", x=0.5),
        xaxis=dict(title="Date"),
        yaxis=dict(title="Quantity ($)"),
        template="plotly_white",
        legend=dict(orientation="v", x=0.01, y=0.99),
        margin=dict(t=60)
    )

    # Fourth graph: Correlation PnL and and PnL attributed to Alpha
    alpha_att = port_attribution_df['pnl.alpha.attribution']
    fourth_graph = []
    fourth_graph.append(
        go.Scatter(
            x=alpha_att,
            y=pnl,
            mode="markers",
            name='scatter',
            marker=dict(
                    color='red',
                    size=20,
                    symbol='arrow',
                    angle=45,
                    line=dict(
                        color='green',
                        width=2
                    )
                ),
        )
    )

    layout_four = dict(
        title=dict(text="PnL Distribution vs. PnL Attributed to Strategy", x=0.5,  xanchor='center', yanchor='top'),
        xaxis=dict(title='Strategy PnL Attribution'),
        yaxis=dict(title="PnL Distribution"),
        # legend=dict(x=0.7, y=0.95),
        #template='plotly_white',
    )

    figures = []
    figures.append(dict(data=first_graph, layout=layout_one))
    figures.append(dict(data=second_graph, layout=layout_two))
    figures.append(dict(data=third_graph, layout=layout_three))
    figures.append(dict(data=fourth_graph, layout=layout_four))

    return figures


if __name__ == "__main__":
    df = figures_backesting_for_webapp()
    print(df)
