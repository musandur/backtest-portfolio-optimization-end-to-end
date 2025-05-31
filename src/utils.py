import numpy as np
import pandas as pd
import alphalens as al
from scipy.stats import mode
# from datetime import timedelta
import pandas_market_calendars as pd_mcal

idx = pd.IndexSlice


class NonMatchingTimezoneError(Exception):
    pass


class MaxLossExceededError(Exception):
    pass


def build_weights_from_ranked_factors(group, _demeaned=True, _equal_weight=False):
    """
    Inputs:
        group (_type_): _description_
        _demeaned (bool, optional): suggest to demean (mean=0) an equal weighted alpha vector. 
                                    Defaults to True.
        _equal_weight (bool, optional): whether we are having an equal weighted alpha factor 
                                        or not (i.e. +1 (long) or -1 (short)). 
                                        Defaults to False.
    Output:
        DataFrame: The demeaned alpha factor
    """

    if _equal_weight:
        group = group.copy()

        if _demeaned:
            # top assets positive weights, bottom ones negative
            group = group - group.median()

        negative_mask = group < 0
        group[negative_mask] = -1.0
        positive_mask = group > 0
        group[positive_mask] = 1.0

        if _demeaned:
            # positive weights must equal negative weights
            if negative_mask.any():
                group[negative_mask] /= negative_mask.sum()
            if positive_mask.any():
                group[positive_mask] /= positive_mask.sum()

    elif _demeaned:
        group = group - group.mean()

    return group / group.abs().sum()


def demean(x):

    return x - x.mean()


def demean_and_normalize(x):
    return demean(x) / abs(demean(x)).sum()


def performance_metrics(returns_df):
    """
            calculate performance metrics such as Sharpe Ratio, Sortino Ratio, Calmar Ratio

            Input: 
                    returns: returns DataFrame
            Output:
                    DataFrame of the performance metrics
    """
    annual_mean_ret = returns_df.mean() * 252
    annual_vol = returns_df.std() * np.sqrt(252)
    #skewness = factor_returns.skew()
    #kurtosis = factor_returns.kurtosis()

    # sharpe ratio
    sharpe_ratio = (annual_mean_ret).div(annual_vol)

    sharpe_ratio = pd.DataFrame(sharpe_ratio, columns=['sharpe_ratio'])

    # sortino ratio
    sortino_ratio = pd.DataFrame(
        (annual_mean_ret).div(np.sqrt(252)*(returns_df[returns_df < 0].std())),
        columns=['sortino_ratio']
                                )

    # calmar ratio
    cumulative_ret = (1 + returns_df).cumprod()
    running_max = cumulative_ret.cummax()
    drawdown = (cumulative_ret - running_max).div(running_max)
    max_drawdown = drawdown.min()
    calmar_ratio = pd.DataFrame((annual_mean_ret).div(-max_drawdown), columns=['calmar_ratio'])

    portfolio_performance_metrics = pd.concat([sharpe_ratio, sortino_ratio, calmar_ratio], axis=1)

    return portfolio_performance_metrics


def fixed_compute_forward_returns(factor, prices, periods, filter_zscore=None, cumulative_returns=True):
    """
        Custom fixed function based on Alphalens' compute_forward_returns, 
        but ensuring `mode(..., keepdims=True).mode[0]` is used.
    """

    factor_dateindex = factor.index.levels[0]
    if factor_dateindex.tz != prices.index.tz:
        raise NonMatchingTimezoneError("The timezone of 'factor' is not the "
                                    "same as the timezone of 'prices'. See "
                                    "the pandas methods tz_localize and "
                                    "tz_convert.")

    freq = al.utils.infer_trading_calendar(factor_dateindex, prices.index)

    factor_dateindex = factor_dateindex.intersection(prices.index)

    if len(factor_dateindex) == 0:
        raise ValueError("Factor and prices indices don't match: make sure "
                        "they have the same convention in terms of datetimes "
                        "and symbol-names")

    prices = prices.filter(items=factor.index.levels[1])

    raw_values_dict = {}
    column_list = []

    for period in sorted(periods):
        if cumulative_returns:
            returns = prices.pct_change(period)
        else:
            returns = prices.pct_change()

        forward_returns = returns.shift(-period).reindex(factor_dateindex)

        if filter_zscore is not None:
            mask = abs(
                forward_returns - forward_returns.mean()
            ) > (filter_zscore * forward_returns.std())
            forward_returns[mask] = np.nan

        days_diffs = []
        for i in range(30):
            if i >= len(forward_returns.index):
                break
            p_idx = prices.index.get_loc(forward_returns.index[i])
            if p_idx is None or p_idx < 0 or (
                    p_idx + period) >= len(prices.index):
                continue
            start = prices.index[p_idx]
            end = prices.index[p_idx + period]
            period_len = al.utils.diff_custom_calendar_timedeltas(start, end, freq)
            days_diffs.append(period_len.components.days)

        delta_days = period_len.components.days - mode(days_diffs, keepdims=True).mode[0]
        period_len -= pd.Timedelta(days=delta_days)
        label = al.utils.timedelta_to_string(period_len)

        column_list.append(label)

        raw_values_dict[label] = np.concatenate(forward_returns.values)

    df = pd.DataFrame.from_dict(raw_values_dict)
    df.set_index(pd.MultiIndex.from_product([factor_dateindex, prices.columns], names=['date', 'asset']), inplace=True)
    df = df.reindex(factor.index)

    # now set the columns correctly
    df = df[column_list]
    df.index.levels[0].freq = freq
    df.index.set_names(['date', 'asset'], inplace=True)

    return df
    

def fixed_get_clean_factor(factor, forward_returns, groupby=None, binning_by_group=False, 
                           quantiles=5, bins=None, groupby_labels=None, max_loss=0.35, zero_aware=False):
    """
        Custom fixed function based on Alphalens' get_clean_factor, 
        but ensuring   quantile_data.index = quantile_data.index.droplevel(0) is used.
        It merges the forward returns, the factors and the factor quantile.
    """

    initial_amount = float(len(factor.index))

    factor_copy = factor.copy()
    factor_copy.index = factor_copy.index.rename(['date', 'asset'])
    factor_copy = factor_copy[np.isfinite(factor_copy)]

    merged_data = forward_returns.copy()
    merged_data['factor'] = factor_copy

    if groupby is not None:
        if isinstance(groupby, dict):
            diff = set(factor_copy.index.get_level_values(
                'asset')) - set(groupby.keys())
            if len(diff) > 0:
                raise KeyError(
                    "Assets {} not in group mapping".format(
                        list(diff)))

            ss = pd.Series(groupby)
            groupby = pd.Series(index=factor_copy.index,
                                data=ss[factor_copy.index.get_level_values(
                                    'asset')].values)

        if groupby_labels is not None:
            diff = set(groupby.values) - set(groupby_labels.keys())
            if len(diff) > 0:
                raise KeyError(
                    "groups {} not in passed group names".format(
                        list(diff)))

            sn = pd.Series(groupby_labels)
            groupby = pd.Series(index=groupby.index,
                                data=sn[groupby.values].values)

        merged_data['group'] = groupby.astype('category')

    merged_data = merged_data.dropna()

    fwdret_amount = float(len(merged_data.index))

    no_raise = False if max_loss == 0 else True
    quantile_data = al.utils.quantize_factor(merged_data, quantiles, bins, binning_by_group, no_raise, zero_aware)
    # print(f"inspect the index of quantile_data:\n {quantile_data.index}")
    if len(quantile_data.index.names) == 3:
        quantile_data.index = quantile_data.index.droplevel(0)
    # print(f"inspect the index of quantile_data:\n {quantile_data.index}")
    merged_data['factor_quantile'] = quantile_data

    merged_data = merged_data.dropna()

    binning_amount = float(len(merged_data.index))

    tot_loss = (initial_amount - binning_amount) / initial_amount
    fwdret_loss = (initial_amount - fwdret_amount) / initial_amount
    bin_loss = tot_loss - fwdret_loss

    print("Dropped %.1f%% entries from factor data: %.1f%% in forward "
        "returns computation and %.1f%% in binning phase "
        "(set max_loss=0 to see potentially suppressed Exceptions)." %
        (tot_loss * 100, fwdret_loss * 100, bin_loss * 100))

    if tot_loss > max_loss:
        message = ("max_loss (%.1f%%) exceeded %.1f%%, consider increasing it."
                % (max_loss * 100, tot_loss * 100))
        raise MaxLossExceededError(message)
    else:
        print("max_loss is %.1f%%, not exceeded: OK!" % (max_loss * 100))

    return merged_data


def get_universe(path_to_data, start_date='2019-01-01', end_date='2022-12-31'):
    '''
        extract the trading universe from the data and the selcted tickers
    '''
    # base_path = os.path.dirname(os.path.abspath(__file__))
    # file_path = os.path.join(base_path, self.path_to_data)

    # data_prices = pd.read_hdf(path_to_data, key="df") # for the h5 version
    data_prices = pd.read_csv(path_to_data,
                              index_col=[0, 1],
                              parse_dates=[1])  # for the csv version

    df = data_prices.loc[:, idx[start_date:end_date], :]
    # select 21 most traded assets in terms of highest volume
    selected_assets = df['Adj Close'].mul(df.Volume)\
        .groupby('Date')\
        .rank(ascending=True)\
        .unstack()\
        .dropna()\
        .mean(axis=1)\
        .nlargest(21)\
        .index\
        .tolist()
    # remove GOOGL ticker to avoid duplicates with GOOG
    selected_assets = [x for x in selected_assets if x != 'GOOGL']
    universe_df = df.loc[selected_assets, :]['Adj Close'].unstack('Ticker')

    return selected_assets, universe_df


def get_universe_from_aws_data(data, start_date='2019-01-01', end_date='2022-12-31'):
    '''
        extract the trading universe from the data imported from aws 
        and the select corresponding tickers
    '''
    # base_path = os.path.dirname(os.path.abspath(__file__))
    # file_path = os.path.join(base_path, self.path_to_data)

    # data_prices = pd.read_hdf(path_to_data, key="df") # for the h5 version
    # data_prices = pd.read_csv(path_to_data,
    #                           index_col=[0, 1],
    #                           parse_dates=[1])  # for the csv version

    df = data.loc[:, idx[start_date:end_date], :]
    # select 21 most traded assets in terms of highest volume
    selected_assets = df['Adj Close'].mul(df.Volume)\
        .groupby('Date')\
        .rank(ascending=True)\
        .unstack()\
        .dropna()\
        .mean(axis=1)\
        .nlargest(21)\
        .index\
        .tolist()
    # remove GOOGL ticker to avoid duplicates with GOOG
    selected_assets = [x for x in selected_assets if x != 'GOOGL']
    universe_df = df.loc[selected_assets, :]['Adj Close'].unstack('Ticker')

    return selected_assets, universe_df


def fetch_backward_dates_and_concate(alpha_dates_index, num_days_offset=365):

    ''' 
        We want to collect the overall historical trading dates we will need
        during backtesting.
         
        INPUT.S:
        First we will offset `num_year_offset` in order to be able to 
        have enough data to esimate risk exposures for every trading date
        this helps in finding optimal weights.

        `num_year_offset`: number of years to lookback starting from the initial trading date

        `alpha_dates_index`: index date of the alpha factor time series

        RETURN:
        Total historical dates required to complete the backtesting process
    '''
    # get the first backward trading date
    end_backward_date = alpha_dates_index[0]
    # get the final backward trading date
    start_backward_date = end_backward_date - pd.DateOffset(days=num_days_offset)

    # start_backward_date = end_backward_date - pd.DateOffset(days=365)

    # get the NYSE trading calendar
    nyse = pd_mcal.get_calendar('NYSE')

    # fetch valid trading days within the range from years back until the first trading date
    schedule = nyse.valid_days(start_date=start_backward_date, end_date=end_backward_date)
    
    # Convert to the naive (no timezone) index for simplicity and without loss of generality
    schedule = schedule.tz_localize(None)

    # print(schedule)

    # now concatenate the lookback years dates with the trading dates to get the whole historical dates
    historical_backtest_dates = schedule.union(alpha_dates_index[1:])

    return historical_backtest_dates


class VolumeFiller:
    def __init__(self, nan_value=None, zero_value=None, negative_value=None):

        """
        The average dollar volume should not hold null values, 
        NaN values, or negative values. This class fills those unwanted values 
        in case they exist with custom values.

        Initializes the object with custom replacement values.
        - nan_value: what to replace NaN with (None = do not replace)
        - zero_value: what to replace 0 with (None = do not replace)
        - negative_value: what to replace negative values with (None = do not replace)
        """
         
        self.nan_val = nan_value
        self.zero_val = zero_value
        self.negative_val = negative_value
    
    def transform(self, series):
        """
        Applies the defined replacements to a pandas Series.
        """
        if self.nan_val is not None:
            series = series.fillna(self.nan_val)
        if self.zero_val is not None:
            series = series.replace(0, self.zero_val)
        if self.negative_val is not None:
            series = series.apply(lambda x: self.negative_val if x < 0 else x)

        return series


if __name__ == "__main__":
    print("utils.py Module called!")

