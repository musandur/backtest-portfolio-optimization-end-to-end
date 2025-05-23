import pandas as pd
import plotly.graph_objs as go
import statsmodels.api as sm
import os
import numpy as np
# import alphalens as al
# import matplotlib.pyplot as plt
from scipy.stats import norm

# import utils module
# from data_manip import utils
from data_manip import utils
#import utils

idx = pd.IndexSlice
window_size = 21   # number days: lookback periods


class AlgoStrategy:
    """ Build and Evaluate the algorithmic strategy
    """
    def __init__(self, tickers, universe_df):
        """ Initialize parameters

        Args:
            universe_df (DataFrame: datetime as index, and tickers as columns): The trading universe chosen to 
                                    built the trading strategy
            list_tickers (list): list of assets that compose our universe
        """
        self.universe_df = universe_df
        self.list_tickers = tickers

    # def get_universe(self, start_date='2019-01-01', end_date='2022-12-31'):
    #     '''
    #         extract the trading universe from the data
    #     '''
    #     #base_path = os.path.dirname(os.path.abspath(__file__))
    #     #file_path = os.path.join(base_path, self.path_to_data)

    #     data_prices = pd.read_hdf(self.path_to_data, key="df")

    #     df = data_prices.loc[:, idx[start_date:end_date], :]
    #     # select 21 most traded assets in terms of highest volume
    #     selected_assets = df['Adj Close'].mul(df.Volume)\
    #         .groupby('Date')\
    #         .rank(ascending=True)\
    #         .unstack()\
    #         .dropna()\
    #         .mean(axis=1)\
    #         .nlargest(21)\
    #         .index\
    #         .tolist()
    #     # remove GOOGL ticker to avoid duplicates with GOOG
    #     selected_assets = [x for x in selected_assets if x != 'GOOGL']
    #     universe_df = df.loc[selected_assets, :]['Adj Close'].unstack('Ticker')

    #     return selected_assets, universe_df
    
    def stock_polynomial_regression(self, ticker_name, rounding_error=6):

        # _, df = self.get_universe()

        ticker_prices = self.universe_df[ticker_name]
        N = len(ticker_prices)

        t = 1 + np.arange(window_size)
        ts = t**2
    
        X = np.array([t, ts]).T

        poly_reg_coefs = pd.DataFrame(columns=['const', 'gain', 'acc'], index=ticker_prices.index)

        for start in range(N - window_size + 1):
            end = start + window_size

            y = ticker_prices[start:end].values
            model = sm.OLS(endog=y, exog=sm.add_constant(X)).fit()

            ticker_date = ticker_prices[start:end].index[-1]

            poly_reg_coefs.loc[ticker_date, :] = np.round(model.params, rounding_error).tolist()

        return poly_reg_coefs.dropna()

    def momentum_alpha_factor_construction(self):

        list_tickers = self.list_tickers
        # df = self.universe_df

        list_df = []

        for ticker in list_tickers:
            # coefs_df = pd.DataFrame()
            coefs_df = self.stock_polynomial_regression(ticker_name=ticker)
            # print(coefs_df)
            
            coefs_df['Ticker'] = ticker
            coefs_df.set_index('Ticker', append=True, inplace=True)

            list_df.append(coefs_df)

        result = pd.concat(list_df, axis=0)

        result = result.groupby('Date').apply(lambda x: x).droplevel(1)

        # construct the alpha factor
        # ranking_func = lambda x: x.rank(ascending=True)
        ranked_gain = result.groupby('Date')['gain'].transform(lambda x: x.rank(ascending=True))
        ranked_acc = result.groupby('Date')['acc'].transform(lambda x: x.rank(ascending=True))
        result['alpha_factor'] = pd.DataFrame(ranked_gain*ranked_acc)#.rename(columns={0:'ranked_alpha'})

        return result.dropna()
    
    def alpha_factors_and_forward_returns(self, periods):
        alpha_fac = self.momentum_alpha_factor_construction()
        # factor = pd.DataFrame(alpha_fac['alpha_factor'])
        close_prices = self.universe_df
        # periods = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 14, 21)
        
        forward_returns = utils.fixed_compute_forward_returns(alpha_fac, close_prices, periods)

        factor_data = utils.fixed_get_clean_factor(alpha_fac['alpha_factor'], forward_returns)

        return factor_data
    
    # Use the fixed function in your pipeline

    def factor_returns_df(self, alpha_data):
        """
           Input:
            alpha_data (df): alpha factors and forward returns DataFrame

        Returns:
             factor returns DataFrame
        """
        grouper = [alpha_data.index.get_level_values('date')]

        factor_weights = alpha_data.groupby(grouper)['factor'].transform(utils.build_weights_from_ranked_factors)

        weighted_returns = alpha_data[alpha_data.columns[:-2]].mul(factor_weights, axis=0)
        factor_returns = weighted_returns.groupby(level='date').sum()

        return factor_returns

    def strategy_performance_metrics(self, alpha_data):
        """
            calculate performance metrics such as:  Sharpe Ratio, Sortino Ratio, Calmar Ratio

            Input: 
                    alpha_data: alpha factors and forward returns DataFrame
            Output:
                    DataFrame of the performance metrics
        """
        factor_returns = self.factor_returns_df(alpha_data)
        portfolio_performance_metrics = utils.performance_metrics(factor_returns)

        return portfolio_performance_metrics
    
    def mean_return_by_quantile(self, alpha_data):
        """
            Group the forward returns by quantile and calculate their mean
           Input:
            alpha_data (df): forward retunrs alpha factors and their quantile

        Output:
              DataFrame of mean return of the forward returns by quantile
        """
        demeaned_fwd_ret = pd.merge(alpha_data[alpha_data.columns[:-2]].groupby('date').transform(utils.demean), 
                                    alpha_data[alpha_data.columns[-2:]],
                                    left_index=True, right_index=True,
                                    how='inner')
        
        mean_ret_by_quantile = demeaned_fwd_ret.groupby('factor_quantile')[demeaned_fwd_ret.columns[:-2]].mean()

        return mean_ret_by_quantile


if __name__ == "__main__":
    # path_to_data = 'data/stock_prices.h5'
    # base_path = os.path.dirname(os.path.abspath(__file__))
    # file_path = os.path.join(base_path, path_to_data)
    # df = pd.read_hdf(file_path, key='df')
    # print(df.head())
    # path_to_data = '/Users/mndour/Documents/webapp_dashboard/data_manip/data/stock_prices.h5'
    path = 'data/stock_prices.h5'
    base_path = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(base_path, path)
    periods = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 14, 21)

    tickers, stock_universe_df = utils.get_universe(file_path)

    strtgy = AlgoStrategy(tickers, stock_universe_df)

    alpha_data = strtgy.alpha_factors_and_forward_returns(periods)
    print(f"Alpha factors: \n {alpha_data}")
    df = strtgy.strategy_performance_metrics(alpha_data)
    # save here
    df.to_hdf("data/portfolio_performance.h5", key='df')
    print(f"Alpha factors: \n {df}")
    mean_ret_df = strtgy.mean_return_by_quantile(alpha_data)
    # save here
    mean_ret_df.to_hdf("data/mean_ret_by_quantile.h5", key='df')
    print(f"Alpha factors: \n {mean_ret_df}")

    fact_ret = strtgy.factor_returns_df(alpha_data)
    # save here
    fact_ret.to_hdf("data/factor_returns.h5", key='df')
    forward_21d_strtgy = fact_ret[['21D']]

    # test sp500 vs the strategy performance
    sp500_dataset_df = pd.read_hdf(file_path, key='df')
    sp500_data = sp500_dataset_df['Adj Close'].unstack('Ticker')
    forward_21d_sp500 = pd.DataFrame(sp500_data.pct_change(21, fill_method=None).shift(-21).mean(axis=1).dropna(), columns=["S&P500"])
    forward_21d_sp500 = forward_21d_sp500.loc[forward_21d_strtgy.index, :]
    sp500_strtgy_df = pd.concat([forward_21d_sp500, forward_21d_strtgy], axis=1)
    sp500_vs_strtgy_metrics = utils.performance_metrics(sp500_strtgy_df)
    sp500_vs_strtgy_metrics.rename(index={'21D': 'strategy'}, inplace=True)

    dict_benchmark_and_strategy = {}

    list_columns = sp500_vs_strtgy_metrics.index.tolist()
    metrics = sp500_vs_strtgy_metrics.columns.tolist()

    for i in range(len(list_columns)):
        dict_benchmark_and_strategy[list_columns[i]] = sp500_vs_strtgy_metrics.loc[list_columns[i], :].tolist()

    fig = go.Figure()

    for col in dict_benchmark_and_strategy.keys():
        fig.add_trace(go.Bar(
            x=metrics,
            y=dict_benchmark_and_strategy[col],
            name=col,
        ))

    # Create the layout
    fig.update_layout(
        title=dict(text="Performance Metrics: S&P500 vs Strategy", x=0.5, y=0.9, xanchor='center', yanchor='top'),
        xaxis=dict(title="Metrics"),
        yaxis=dict(title="Values"),
        barmode='group',  # Grouped bars
        #legend=dict(title="Legend:", orientation="h", y=1.15, xanchor="center"),
        legend=dict(title="Legend:", orientation="h", y=1.15),

        titlefont=dict(size=14),
    )
    # Create the figure
    # fig = go.Figure(data=[sp500_bar, strategy_bar], layout=layout)
    # Show the figure
    fig.show()




    # Create the histogram plot
    histogram = go.Histogram(
        x=fact_ret['21D'],
        nbinsx=30,
        histnorm='probability density',
        opacity=0.7,
        marker=dict(color='green'),
        name='Histogram'
    )
    mean = fact_ret['21D'].mean()
    std = fact_ret['21D'].std()

    N = 1000
    x = np.linspace(fact_ret['21D'].min(), fact_ret['21D'].max(), N)
    pdf = norm.pdf(x, loc=mean, scale=std)
    # Create the normal distribution line plot
    normal_dist = go.Scatter(
        x=x,
        y=pdf,
        mode='lines',
        line=dict(color='red', width=2.5),
        name='Normal Distribution'
    )

    # Add a vertical line for the mean
    mean_line = go.Scatter(
        x=[mean, mean],
        y=[0, max(pdf)],
        mode='lines',
        line=dict(color='black', width=3.5, dash='dash'),
        name='Mean Line'
    )

    # Update layout for title, labels, and legend
    layout = go.Layout(
        title=dict(text="21D Holding Portfolio Returns: Histogram vs. Normal Distribution", x=0.5, y=0.95),
        xaxis=dict(title="Returns"),
        yaxis=dict(title="Density"),
        legend=dict(x=0.7, y=0.95),
        #template='ggplot2',
    
    )

    # Combine the plots into a figure
    fig = go.Figure(data=[histogram, normal_dist, mean_line], layout=layout)
    fig.show()

    # alpha_fac = strtgy.momentum_alpha_factor_construction()
    # alpha_df = pd.DataFrame(alpha_fac['alpha_factor'])
    # _, prices = strtgy.get_universe()
    # fwd = strtgy.fixed_compute_forward_returns(alpha_fac, prices, [1, 2, 3])
    # gcf = strtgy.get_clean_factor(alpha_fac['alpha_factor'], fwd)

    # print(f"Alpha factors: \n {alpha_fac['alpha_factor']}")
    # print(f"Alpha factors: \n {fwd}")
    
    # print(f"alpha data: \n {gcf}")





    # test get_clean_factor()
    # factor = alpha_fac['alpha_factor']
    # initial_amount = float(len(factor.index))

    # factor_copy = factor.copy()
    # factor_copy.index = factor_copy.index.rename(['date', 'asset'])
    # factor_copy = factor_copy[np.isfinite(factor_copy)]
    # merged_data = fwd.copy()
    # merged_data['factor'] = factor_copy
    # merged_data = merged_data.dropna()
    # print(f"merged_data:\n {merged_data}")
    # fwdret_amount = float(len(merged_data.index))
    # max_loss = 0.35
    # no_raise = False if max_loss == 0 else True
    # quantiles = 5
    # bins = None
    # binning_by_group = False
    # zero_aware = False
    # quantile_data = al.utils.quantize_factor(merged_data, quantiles, bins, binning_by_group, no_raise, zero_aware)
    # print(f"quantile_data:\n {quantile_data}")
    # quantile_data.index = quantile_data.index.droplevel(0)
    # print(f"quantile_data:\n {quantile_data}")




    #gcf = strtgy.get_clean_factor(alpha_fac['alpha_factor'], fwd)
    #alpha_data = strtgy.alpha_factors_and_forward_returns()

    #print(f"Alpha factors: \n {alpha_fac['alpha_factor']}")
    #print(f"Alpha factors: \n {fwd}")
    
    #print(f"alpha data: \n {alpha_data.head(23)}")




# ## wrangling module for all four graphs


# def cleandata(dataset, keepcolumns=['Country Name', '1990', '2015'], value_variables=['1990', '2015']):
#     """Clean world bank data for a visualizaiton dashboard

#     Keeps data range of dates in keep_columns variable and data for the top 10 economies
#     Reorients the columns into a year, country and value
#     Saves the results to a csv file

#     Args:
#         dataset (str): name of the csv data file

#     Returns:
#         None

#     """   
 
#     base_path = os.path.dirname(os.path.abspath(__file__))  # Get the directory of the current script
#     file_path = os.path.join(base_path, dataset)  #

#     df = pd.read_csv(file_path, skiprows=4)

#     #df = pd.read_csv(dataset, skiprows=4)

#     # Keep only the columns of interest (years and country name)
#     df = df[keepcolumns]

#     top10country = ['United States', 'China', 'Japan', 'Germany', 'United Kingdom', 'India', 'France', 'Brazil', 'Italy', 'Canada']
#     df = df[df['Country Name'].isin(top10country)]

#     # melt year columns  and convert year to date time
#     df_melt = df.melt(id_vars='Country Name', value_vars = value_variables)
#     df_melt.columns = ['country','year', 'variable']
#     df_melt['year'] = df_melt['year'].astype('datetime64[ns]').dt.year

#     # output clean csv file
#     return df_melt


# def return_figures():
#     """Creates four plotly visualizations

#     Args:
#         None

#     Returns:
#         list (dict): list containing the four plotly visualizations
#     """

#     # first chart plots arable land from 1990 to 2015 in top 10 economies 
#     # as a line chart
#     graph_one = []
#     df = cleandata('data/API_AG.LND.ARBL.HA.PC_DS2_en_csv_v2.csv')
#     df.columns = ['country', 'year', 'hectaresarablelandperperson']
#     df.sort_values('hectaresarablelandperperson', ascending=False, inplace=True)
#     countrylist = df.country.unique().tolist()

#     for country in countrylist:
#         x_val = df[df['country'] == country].year.tolist()
#         y_val = df[df['country'] == country].hectaresarablelandperperson.tolist()
#         graph_one.append(
#             go.Scatter(
#                 x=x_val,
#                 y=y_val,
#                 mode='lines',
#                 name=country
#             )
#         )

#     layout_one = dict(
#         title='Change in Hectares Arable Land <br> per Person 1990 to 2015',
#         xaxis=dict(title='Year', autotick=False, tick0=1990, dtick=25),
#         yaxis=dict(title='Hectares'),
#     )

#     # second chart plots arable land for 2015 as a bar chart    
#     graph_two = []
#     df = cleandata('data/API_AG.LND.ARBL.HA.PC_DS2_en_csv_v2.csv')
#     df.columns = ['country', 'year', 'hectaresarablelandperperson']
#     df.sort_values('hectaresarablelandperperson', ascending=False, inplace=True)
#     df = df[df['year'] == 2015]

#     graph_two.append(
#         go.Bar(
#             x=df.country.tolist(),
#             y=df.hectaresarablelandperperson.tolist(),
#         )
#     )

#     layout_two = dict(
#         title='Hectares Arable Land per Person in 2015',
#         xaxis=dict(title='Country'),
#         yaxis=dict(title='Hectares per person'),
#     )

#     # third chart plots percent of population that is rural from 1990 to 2015
#     graph_three = []
#     df = cleandata('data/API_SP.RUR.TOTL.ZS_DS2_en_csv_v2_9948275.csv')
#     df.columns = ['country', 'year', 'percentrural']
#     df.sort_values('percentrural', ascending=False, inplace=True)
    
#     for country in countrylist:
#         x_val = df[df['country'] == country].year.tolist()
#         y_val = df[df['country'] == country].percentrural.tolist()
#         graph_three.append(
#             go.Scatter(
#                 x=x_val,
#                 y=y_val,
#                 mode='lines',
#                 name=country
#             )
#         )

#     layout_three = dict(
#         title='Change in Rural Population <br> (Percent of Total Population)',
#         xaxis=dict(title='Year', autotick=False, tick0=1990, dtick=25),
#         yaxis=dict(title='Percent'),
#     )

#     # fourth chart shows rural population vs arable land
#     graph_four = []

#     valuevariables = [str(x) for x in range(1995, 2016)]
#     keepcolumns = [str(x) for x in range(1995, 2016)]
#     keepcolumns.insert(0, 'Country Name')

#     df_one = cleandata('data/API_SP.RUR.TOTL_DS2_en_csv_v2_9914824.csv', keepcolumns, valuevariables)
#     df_two = cleandata('data/API_AG.LND.FRST.K2_DS2_en_csv_v2_9910393.csv', keepcolumns, valuevariables)

#     df_one.columns = ['country', 'year', 'variable']
#     df_two.columns = ['country', 'year', 'variable']

#     df = df_one.merge(df_two, on=['country', 'year'])

#     for country in countrylist:
#         x_val = df[df['country'] == country].variable_x.tolist()
#         y_val = df[df['country'] == country].variable_y.tolist()
#         year = df[df['country'] == country].year.tolist()
#         country_label = df[df['country'] == country].country.tolist()

#         text = []
#         for country, year in zip(country_label, year):
#             text.append(str(country) + ' ' + str(year))

#         graph_four.append(
#             go.Scatter(
#                 x=x_val,
#                 y=y_val,
#                 mode='markers',
#                 text=text,
#                 name=country,
#                 textposition='top right'
#             )
#         )

#     layout_four = dict(
#         title='Rural Population versus <br> Forested Area (Square Km) 1990-2015',
#         xaxis=dict(title='Rural Population'),
#         yaxis=dict(title='Forest Area (square km)'),
#     )

#     # append all charts to the figures list
#     figures = []
#     figures.append(dict(data=graph_one, layout=layout_one))
#     figures.append(dict(data=graph_two, layout=layout_two))
#     figures.append(dict(data=graph_three, layout=layout_three))
#     figures.append(dict(data=graph_four, layout=layout_four))

#     return figures

