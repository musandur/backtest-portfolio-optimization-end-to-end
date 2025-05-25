import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import cvxpy as cvx
import statsmodels.api as sm
from abc import ABC, abstractmethod
from src import strategy_performance, utils
#import strategy_performance
#import utils
# import utils


class AbstractClassOptimalPortfolio(ABC):
    """ 
      abract class interface for portfolio convex optimization 
    """

    @abstractmethod
    def get_objective_func(self, weights, alpha_vector):
        raise NotImplementedError
    
    @abstractmethod
    def get_constraints(self, weights, portfolio_risk):
        raise NotImplementedError
    
    def get_portfolio_risk(self, weights, factor_betas_df, idiosync_var_mat, risk_fac_cov_mat):
        """" 
        Portfolio variance estimation in terms of risk factors and beta exposures
        Inputs:
            factor_betas_df: estimated factor exposures
            idiosync_var_mat: estimated idiosyncratic returns variance
            risk_fac_cov_mat: estimated risk factors covariance matrix
        Outputs:
            Estimated annualized portfolio variance
        """

        X = weights
        B = factor_betas_df.T.values @ X
        S = idiosync_var_mat
        F = risk_fac_cov_mat

        portfolio_risk = cvx.quad_form(B, F) + cvx.quad_form(X, S)

        return portfolio_risk

    def solve_optimal_holdings(self, alpha_vector, factor_betas_df, idiosync_var_mat, risk_fac_cov_mat):
        """
            Solve the  portfolio optimization 
        Inputs:
            alpha_vector (pd Series): predicted alpha value for each stock
            ff_and_ret_df (multi index df): Fama-French 5-risk-factors and returns for each stock

        Returns:
           df: DataFrame of optimal weights for each stock
        """

        weights = cvx.Variable(len(alpha_vector))
        portfolio_risk = self.get_portfolio_risk(weights, factor_betas_df, idiosync_var_mat, risk_fac_cov_mat)

        objective_func = self.get_objective_func(weights, alpha_vector)
        constraints = self.get_constraints(weights, factor_betas_df, portfolio_risk)

        set_problem = cvx.Problem(objective_func, constraints)
        set_problem.solve()

        optimal_holdings = np.asarray(weights.value).flatten()

        optimal_holdings_df = pd.DataFrame(data=optimal_holdings, index=alpha_vector.index)

        return optimal_holdings_df 


class OptimalHoldings(AbstractClassOptimalPortfolio):
    """ Specify the optimization technique: optimization with regularization

    Args:
        AbstractClassOptimalPortfolio (abstract class): yields opportinuty for various optimization technique
    """
    def __init__(self, lambda_reg=0.005, risk_cap=0.120, factor_max=5.0, factor_min=-5.0, weights_max=0.955, weights_min=-0.955):
        self.risk_cap = risk_cap
        self.factor_max = factor_max
        self.factor_min = factor_min
        self.weights_max = weights_max
        self.weights_min = weights_min
        self.lambda_reg = lambda_reg
    
    def get_objective_func(self, weights, alpha_vector):

        assert len(alpha_vector.columns)==1, "solve for one alpha factor"
        obj_func = cvx.Minimize(-alpha_vector.values.flatten() @ weights + self.lambda_reg * cvx.norm2(weights))
        
        return obj_func
    
    def get_constraints(self, weights, factor_betas_df, portfolio_risk):
        portfolio_exposure = factor_betas_df.values.T @ weights
        constraints = [
            weights >= self.weights_min,
            weights <= self.weights_max,
            sum(weights) == 0.0,
            sum(cvx.abs(weights)) <= 1.0,
            portfolio_exposure <= self.factor_max,
            portfolio_exposure >= self.factor_min,
            portfolio_risk <= self.risk_cap**2
        ]

        return constraints
    
    
# Optimization function

def get_beta_factors(ff_and_ret_df):
    """ 
    Estimates the factor betas exposures
    INPUTS:
        fama_fac_and_return: Multi-Index DataFrame of the Fama-French 5-risk-factors columns  and the asset returns column.
        factor_data: Fama-French 5-risk-factors DataFrame.
    OUTPUT:
        betas: estimated factor betas DataFrame.
    """

    betas = ff_and_ret_df.groupby('Ticker', group_keys=False).apply(lambda x: sm.OLS(endog=x[ff_and_ret_df.columns[-1]],
                                                                    exog=sm.add_constant(x[ff_and_ret_df.columns[:-1]]))
                                                                    .fit()
                                                                    .params)
    if 'const' in betas.columns:
        betas.drop(['const'], axis=1, inplace=True)

    factor_betas_df = betas.reindex(ff_and_ret_df.index.levels[0].tolist())

    return factor_betas_df


def get_idiosyncratic_var(ff_and_ret_df, factor_betas_df, ff_df):
    """Estimate the specific return's variance matrix

    Args:
        ff_and_ret_df (df): Multi-Index DataFrame of the Fama-French 5-risk-factors columns  and the asset returns column. 
        factor_betas_df (df): Estimated Beta exposures
        ff_df (df): Fama-French 5-risk-factors DataFrame

    Returns:
        Matrix: annualized variance matrix of size number of assets
    """

    true_ret = ff_and_ret_df[ff_and_ret_df.columns[-1]].unstack(0)
    date_idx = true_ret.index
    estimated_ret = ff_df.loc[date_idx, :].dot(factor_betas_df.T)

    specific_ret_df = true_ret.subtract(estimated_ret)
    idiosyncratic_var = np.diag(252 * specific_ret_df.var(ddof=1))

    return idiosyncratic_var


def get_risk_factor_cov_mat(ff_df, ff_and_ret_df):

    """Estimate the risk factors cconvariance matrix

    Args:
        ff_and_ret_df (df): Multi-Index DataFrame of the Fama-French 5-factors columns  and the asset returns column. 
        ff_df (df): risk factors DataFrame

    Returns:
        Matrix: Annualized covariance matrix of size number of risk factors
    """
    true_ret = ff_and_ret_df[ff_and_ret_df.columns[-1]].unstack(0)
    date_idx = true_ret.index

    risk_factors = ff_df.loc[date_idx, :]
    cov_mat = np.cov(risk_factors.T, ddof=1)
    risk_factors_cov = np.sqrt(252) * cov_mat
    
    return risk_factors_cov


def demean(x):
    return x - x.mean()


def demean_and_normalize(x):
    return demean(x) / abs(demean(x)).sum()


# take out the function
# def get_optimal_weights(alpha_vector, fama_fac_and_return, factor_data):
#     '''
#     Function finding the optimal weights for an optimal portfolio
    
#     INPUTS:
#         alpha_vector: alpha values for  a selected day.
#         fama_fac_and_return: Multi-Index DataFrame of the Fama-French 5-factors and the 10-day asset returns.
#         factor_data: Fama-French 5-factors DataFrame.
#     RETURN:
#         opt_weights: Single column DataFrame of the optimal weights with underlying  as index.
#     '''

#     optimal_weigths = OptimalHoldings()

#     factor_betas, specific_returns = factor_betas_and_specific_return(fama_fac_and_return, factor_data) 
#     date_idx = fama_fac_and_return['return'].index.unique('date')
#     factor_cov_matrix = np.sqrt(252)*np.cov(factor_data.loc[date_idx, :].T, ddof=1)
#     idiosynchratic_var_vector = 252*specific_returns.var(ddof=1)

#     opt_wights_df = optimal_weigths.solve_optimal_holdings(alpha_vector, factor_betas_df, factor_cov_matrix, idiosynchratic_var_vector)
#     opt_wights_df.rename(columns={0:'optimal_weights'}, inplace=True)

#     return opt_wights_df


if __name__ == '__main__':
    print("Inplace Calling. No output expected!")
    hold_period = 21
    periods = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 14, 21)

    path = "data/stock_prices.h5"
    base_path = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(base_path, path)
    list_tickers, data = utils.get_universe("data/stock_prices.h5")
    #list_tickers, data = utils.get_universe("data_manip/data/stock_prices.h5")
    straj_obj = strategy_performance.AlgoStrategy(list_tickers, data)

    ret = data.pct_change(hold_period).dropna()
    ret_df = pd.DataFrame(ret.unstack(), columns=[f'{hold_period}_d_ret'])

    #ff_df = pd.read_hdf("data_manip/data/FF_risk_factors.h5", key='df')
    ff_df = pd.read_hdf("data/FF_risk_factors.h5", key='df')

    ff_df.drop(['RF'], axis=1, inplace=True)
    ff_and_ret_df = ff_df.join(ret_df)

    betas_df = get_beta_factors(ff_and_ret_df)
    F = get_risk_factor_cov_mat(ff_df, ff_and_ret_df)
    S = get_idiosyncratic_var(ff_and_ret_df, betas_df, ff_df)
    betas_df.to_hdf("data/po_betas_df.h5", key='df')  

    alpha_data = straj_obj.alpha_factors_and_forward_returns(periods)
    alpha_vector = alpha_data[['factor']].loc[alpha_data.index.unique('date')[-1]]
    alpha_vector = alpha_vector.transform(utils.demean_and_normalize)
    alpha_vector.to_hdf("data/po_alpha_vector.h5", key='df')

    portfolio_obj = OptimalHoldings()
    optimal_weights = portfolio_obj.solve_optimal_holdings(alpha_vector, betas_df, S, F)
    optimal_weights.rename(columns={0: 'optimal_weights'}, inplace=True)
    optimal_weights.to_hdf("data/po_optimal_weights_df.h5", key='df')
    print(optimal_weights)

    # optimal_weights.plot.bar()
    # plt.show()

    # alpha_vector.plot.bar()
    # plt.show()
    # betas_df.T.plot.bar()
    # plt.show()
    factor_names = betas_df.columns.tolist()
    ticker_list = list(betas_df.index)
    fig, ax = plt.subplots(figsize=(16, 5))
    num_tickers = len(ticker_list)
    num_factors = len(factor_names)
    bar_width = 0.03
    indices = np.arange(num_factors)

    for i, ticker in enumerate(ticker_list):
        bar_positions = indices + i * bar_width
        ax.bar(bar_positions, betas_df.loc[ticker], width=bar_width, label=ticker)

    ax.set_ylabel('Exposure Level', fontsize=10)
    ax.set_xticks(indices + bar_width * (num_factors))
    ax.set_xticklabels(factor_names)              

    ax.set_title('Estimated Factor Exposures', fontsize=13, fontweight='bold')
    ax.legend(ncol=len(ticker_list), fontsize=5.8)
    plt.xticks(rotation=45)
    plt.show()

    # plot the beta exposures with plotly
    fig = go.Figure()

    # for ticker, values in zip(ticker_list, data):



    ## Now test the plotting with pyplot



