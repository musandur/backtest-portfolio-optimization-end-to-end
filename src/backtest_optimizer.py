
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import scipy
from tqdm import tqdm
from strategy_performance import AlgoStrategy
import portfolio_optimization
import utils
from utils import VolumeFiller
idx = pd.IndexSlice

nan_val = 1.0e4
zero_val = 1.0e4
neg_val = 1.0e4
v_filler = VolumeFiller(nan_val, zero_val, neg_val)


class BacktestData(AlgoStrategy):

    # static variables
    df = pd.read_hdf("data/stock_prices.h5", key="df")

    # ff5f = pd.read_hdf("data_manip/data/FF_risk_factors.h5")
    # ff5f.drop(['RF'], axis=1, inplace=True)
    # ff5f = ff5f.loc[all_dates, :]
    # ff5f.index.name = 'Date'

    def __init__(self, tickers, backtest_universe_df, backtest_dates, num_lookback_days, holding_period, risk_aversion):
        super().__init__(tickers, backtest_universe_df)
        self.backtest_dates = backtest_dates
        self.num_offset_days = num_lookback_days
        self.all_dates = utils.fetch_backward_dates_and_concate(self.backtest_dates, self.num_offset_days)
        self.ticker_list = tickers
        self.holding_period = holding_period
        self.risk_aversion = risk_aversion
        self.universe = BacktestData.df.loc[idx[tickers, self.all_dates], :]

    def set_return(self):
        num_days = self.holding_period[0] * 2
        ret_dates = utils.fetch_backward_dates_and_concate(self.all_dates, num_days_offset=num_days)
        
        ret_t = BacktestData.df.loc[idx[self.ticker_list, ret_dates], :].pct_change(self.holding_period[0]).dropna().loc[idx[self.ticker_list, self.all_dates], :]["Adj Close"]  # .unstack('Ticker')
        ret_t.name = 'return'
        self.universe = self.universe.join(ret_t)

    def set_adv_and_lambda(self):

        df_adv_30 = pd.DataFrame(self.universe.groupby("Ticker")["Volume"]
                                 .rolling(window=30)
                                 .mean())\
                                 .reset_index(level=0, drop=True)\
                                 .rename(columns={"Volume": "ADV"})
        
        df_adv_30 = df_adv_30.groupby('Ticker')['ADV'].transform(v_filler.transform)
        Lambda = 0.1 / df_adv_30
        Lambda.name = 'Lambda'

        self.universe = self.universe.join(df_adv_30)
        self.universe = self.universe.join(Lambda)

    def get_factor_data(self):

        fac_data = AlgoStrategy.alpha_factors_and_forward_returns(self, self.holding_period)
        fac_data = fac_data[fac_data.columns[:-1]]
        fac_data['alpha_factor_normalize'] = fac_data['factor'].groupby('date').transform(utils.demean_and_normalize)
        fac_data.rename(columns={'21D': 'lookforward_ret'}, inplace=True)

        return fac_data

    def get_estimate_factor_matrices(self):
        """
        Estimate the historic factor exposures, risk factor covariance matrices and idosyncratic 
        variance matrices for each backtest date

        Arguments: 
            backtest_dates: all trading dates considered for the backtesting process
            all_dates: combine the historical dates and the bactesting dates

        Returns:
            List of dictionaries: - estimated factor exposures (value) for each backtesting date (key)
                                  - estimated risk factor covariance matrix (value) for each backtest date (key)
                                  - estimated idosyncratic variance matrix (value) for each backtest date (key) 
        """ 
        
        ff5f = pd.read_hdf("data/FF_risk_factors.h5")
        ff5f.drop(['RF'], axis=1, inplace=True)
        ff5f = ff5f.loc[self.all_dates, :]
        ff5f.index.name = 'Date'
        
        factor_betas_dict = {}
        risk_cov_mat_dict = {}
        idiosyncratic_var_mat_dict = {}
        dict_factor_matrices = {}
        factor_returns_dict = {}

        ret = self.universe['return'].sort_index()
        fac_data = self.get_factor_data()

        for past_date, actual_date in zip(self.all_dates, self.backtest_dates):
            backtest_date = actual_date.strftime("%Y%m%d")
            #print(ret_df)
            #period_ret = ret_sort.loc[:, past_date:actual_date]

            period_ret = ret.loc[:, past_date:actual_date]
            period_ret = period_ret.loc[self.ticker_list, :]
            #period_ret = ret_df.loc[idx[ticker_list, past_date:actual_date], :]
            period_ff_risk = ff5f.loc[past_date:actual_date, :]
            period_ff_and_ret = period_ff_risk.join(period_ret)
            #print(period_ff_and_ret)

            betas_df = portfolio_optimization.get_beta_factors(period_ff_and_ret).dropna()
            betas_df = betas_df.loc[self.ticker_list, :]
            #print(betas_df)
            
            #betas_df.loc[:, "ALPHA"] = fac_data.loc[actual_date, 'alpha_factor_normalize']
            #betas_df.loc[:, "return"] = fac_data.loc[actual_date, 'lookforward_ret']

            factor_betas_dict[backtest_date] = betas_df.copy()
            factor_betas_dict[backtest_date].loc[:, "ALPHA"] = fac_data.loc[actual_date, 'alpha_factor_normalize']
            factor_betas_dict[backtest_date].loc[:, "fwd_return"] = fac_data.loc[actual_date, 'lookforward_ret']

            # now get the factor returns 
            model = sm.OLS(
                endog=factor_betas_dict[backtest_date].iloc[:, -1],
                exog=factor_betas_dict[backtest_date].iloc[:, :-1]
            ).fit()
            factor_returns_dict[backtest_date] = pd.DataFrame(model.params, columns=['factor_returns'])

            factor_betas_dict[backtest_date].loc[:, 'idiosync_ret'] = model.resid
            
            # end 

            # print(factor_betas_dict)
            #print("\n")
            # print(betas_df)

            S = portfolio_optimization.get_idiosyncratic_var(period_ff_and_ret, betas_df, period_ff_risk)  
            idiosyncratic_var_mat_dict[backtest_date] = S
            

            F = portfolio_optimization.get_risk_factor_cov_mat(period_ff_risk, period_ff_and_ret)  
            risk_cov_mat_dict[backtest_date] = np.diag(np.diag(F))
        
        dict_factor_matrices['betas'] = factor_betas_dict
        dict_factor_matrices['risks'] = risk_cov_mat_dict
        dict_factor_matrices['idiosyncratic'] = idiosyncratic_var_mat_dict
        dict_factor_matrices['f_returns'] = factor_returns_dict
                                                                
        return dict_factor_matrices  # [factor_betas_dict, risk_cov_mat_dict, idiosyncratic_var_mat_dict]
    
    def get_actual_lambda(self, actual_date):
        assert 'Lambda' in self.universe.columns, "run 'set_adv_and_lambda' first"
        
        lambda_df = self.universe\
                        .swaplevel()\
                        .groupby('Date')\
                        .apply(lambda x: x)\
                        .reset_index(level=0, drop=True)\
                        .loc[pd.to_datetime(actual_date, format='%Y%m%d'), ['Lambda']]
        return lambda_df
        
    # def get_actual_alpha(self, actual_date, fac_and_fwd_data):
    #     assert 'alpha_factor_normalize' in fac_and_fwd_data.columns, "run 'get_factor_data' first"
        
    #     alpha_df = fac_and_fwd_data.loc[pd.to_datetime(actual_date, format='%Y%m%d'), ['alpha_factor_normalize']]
                                          
    #     return alpha_df
    def get_objective_function(self, h0, Q, specific_var, alpha_vec, Lambda):

        def objective_func(h):
            f = 0
            f += 0.5*self.risk_aversion*np.sum(np.matmul(Q, h)**2)
            f += 0.5*self.risk_aversion*np.dot(h**2, specific_var)
            f -= np.dot(h, alpha_vec)
            f += np.dot((h-h0)**2, Lambda)

            return f
        
        return objective_func
    
    def get_gradient_function(self, h0, QT, Q, specific_var, alpha_vec, Lambda):
        
        def gradient_func(h):
            g = self.risk_aversion*(np.matmul(QT, np.matmul(Q, h)) + (specific_var*h)) - alpha_vec + 2*(h-h0)*Lambda
            return np.asarray(g)

        return gradient_func

    def get_portfolio_risk_exposure(self, BT, h_star):
        return pd.DataFrame(np.matmul(BT, h_star), columns=['portfolio.beta.exposure'])

    def get_total_transaction_costs(self, h0, h_star, Lambda):
        return np.dot((h_star-h0)**2, Lambda.values.flatten())
                                
    def get_portfolio_alpha_exposure(self, alpha_vec, h_star):
        return np.matmul(alpha_vec.values.flatten().T, h_star)

    def optimized_portfolio(self, actual_date,
                            previous_holdings,
                            factor_betas,
                            risk_factors,
                            idiosyncratic_var_matrices, 
                            ):
    
        B = factor_betas[actual_date].iloc[:, :-3]
        F = risk_factors[actual_date]
        S = idiosyncratic_var_matrices[actual_date]
        S = np.diag(S)

        # build matrix for efficient multiplication
        G = scipy.linalg.sqrtm(F)
        BT = B.transpose()
        Q = np.matmul(G, BT)
        QT = Q.transpose()
        Lambda = self.get_actual_lambda(actual_date)
        #alpha_vec = bd.get_actual_alpha(actual_date, fac_data)
        alpha_vec = factor_betas[actual_date][['ALPHA']]

        h0 = previous_holdings.values.flatten()

        # now run the optimization
        obj_func = self.get_objective_function(h0, Q, S, alpha_vec.values.flatten(), Lambda.values.flatten())
        grad_func = self.get_gradient_function(previous_holdings["h.opt.previous"].values, QT, Q, S, alpha_vec.values.flatten(), Lambda['Lambda'])

        actual_holdings = scipy.optimize.fmin_l_bfgs_b(obj_func, h0, grad_func)
        h_opt = actual_holdings[0]
        h_opt_df = pd.DataFrame({'h.opt': h_opt}, index=self.ticker_list)
        h_opt_df.index.name = 'Ticker'
        h = h_opt_df.values.flatten()

        portfolio_risk_exposures = self.get_portfolio_risk_exposure(BT, h)
        portfolio_alpha_exposures = self.get_portfolio_alpha_exposure(alpha_vec, h)
        portfolio_transaction_cost = self.get_total_transaction_costs(h0, h, Lambda)

        return {
            "opt.portfolio": h_opt_df,
            "risk.exposure": portfolio_risk_exposures,
            "alpha.exposures": portfolio_alpha_exposures,
            "total.cost": portfolio_transaction_cost
        }

    # update holdings
    def update_previous_to_current_holding(self, dict_opt_results):

        prev = dict_opt_results['opt.portfolio']
        prev = prev.rename(index=str, columns={'h.opt': 'h.opt.previous'}, copy=True, inplace=False)

        return prev

    # build a tradelist
    def trade_list(self, prev_holdings, dict_opt_results):

        merged_holdings_df = prev_holdings.merge(dict_opt_results['opt.portfolio'], how='outer', on='Ticker')
        merged_holdings_df['h.opt.previous'] = np.nan_to_num(merged_holdings_df['h.opt.previous'])
        merged_holdings_df['h.opt'] = np.nan_to_num(merged_holdings_df['h.opt'])

        return merged_holdings_df
        
    def run_backtest_optimizer(self,
                               prev_holdings,
                               factor_matrices):

        backtest_results_dict = {}
        holdings_trades_dict = {}

        for dt in tqdm(self.backtest_dates, desc='Optimizing Portfolio', unit='day'):
            date = dt.strftime("%Y%m%d")
            result = self.optimized_portfolio(date,
                                              prev_holdings,
                                              factor_matrices['betas'],
                                              factor_matrices['risks'],
                                              factor_matrices['idiosyncratic'],
                                              )
        
            backtest_results_dict[date] = result
            holdings_trades_dict[date] = self.trade_list(prev_holdings, result)
            prev_holdings = self.update_previous_to_current_holding(result)

        return backtest_results_dict, holdings_trades_dict
    
    def partial_dot_product(self, v, w):
        common_idx = v.index.intersection(w.index)
        return np.sum(v.loc[common_idx].values.flatten() * w.loc[common_idx].values.flatten())
        
    def PnL_attribution(self, backtest_results_dict, factor_matrices):
        attribution_df = pd.DataFrame(index=self.backtest_dates)

        for dt in self.backtest_dates:
            date = dt.strftime('%Y%m%d')

            #optimal_port = backtest_results_dict[date]

            holding = backtest_results_dict[date]['opt.portfolio']
            fwd_ret = factor_matrices['betas'][date][['fwd_return']]
            attribution_df.at[dt, 'daily_PnL'] = np.dot(holding.values.flatten(), fwd_ret.values.flatten())

            # PnL attribution
            ## risks
            e = backtest_results_dict[date]['risk.exposure']
            fr = factor_matrices['f_returns'][date]
            attribution_df.at[dt, 'pnl.risk.attribution'] = self.partial_dot_product(e, fr)

            ## alpha
            a = backtest_results_dict[date]['alpha.exposures']
            idiosyncratic_ret = factor_matrices['betas'][date][['idiosync_ret']]
            attribution_df.at[dt, 'pnl.alpha.attribution'] = self.partial_dot_product(idiosyncratic_ret, holding) + a * fr.loc['ALPHA'].values[0]   
            
            attribution_df.at[dt, 'pnl.cost.attribution'] = backtest_results_dict[date]['total.cost']

        return attribution_df
    
    def build_portfolio_characteristics(self, backtest_results_dict, holdings_traded_dict): 
        df = pd.DataFrame(index=self.backtest_dates)
        
        for dt in self.backtest_dates:
            date = dt.strftime('%Y%m%d')
    
            p = backtest_results_dict[date]
            tradelist = holdings_traded_dict[date]
            h = p['opt.portfolio']['h.opt']
            
            # TODO: Implement
            
            df.at[dt, "long"] = sum(h[h > 0])
            df.at[dt, "short"] = sum(h[h < 0])
            df.at[dt, "net"] = sum(h)
            df.at[dt, "gmv"] = sum(np.abs(h))
            df.at[dt, "traded"] = sum(tradelist['h.opt.previous'] - tradelist['h.opt'])
            
        return df
        

if __name__ == '__main__':
    print("Inplace Calling\n")

    ticker_list, _ = utils.get_universe("data/stock_prices.h5", start_date='2023-12-01', end_date='2023-12-31')
    df = pd.read_hdf("data/stock_prices.h5", key="df")

    one_year_2024_df = df.loc[idx[:, '2024'], :]
    one_year_2024_df = one_year_2024_df['Adj Close'].unstack('Ticker')
    one_year_2024_df = one_year_2024_df.loc[:, ticker_list]
    print(one_year_2024_df.head())

    print(one_year_2024_df.tail())

    obj = AlgoStrategy(ticker_list, one_year_2024_df)

    forward_period = [21]
    fac_data = obj.alpha_factors_and_forward_returns(forward_period)
    fac_data = fac_data[fac_data.columns[:-1]]
    fac_data['alpha_factor_normalize'] = fac_data['factor'].groupby('date').transform(utils.demean_and_normalize)
    fac_data.rename(columns={'21D': 'lookforward_ret'}, inplace=True)
    # print(fac_data)
    
    backtest_dates = fac_data.index.unique('date')

    bd = BacktestData(ticker_list,
                      one_year_2024_df,
                      backtest_dates,
                      num_lookback_days=3*365,
                      holding_period=[21],
                      risk_aversion=1.0e-5)
    
    # set up inputs for the backtesting
    bd.set_return()
    bd.set_adv_and_lambda()
    factor_mats = bd.get_estimate_factor_matrices()

    # run the backtest optimizer
    prev_holdings = pd.DataFrame({'h.opt.previous': np.zeros(len(ticker_list))}, index=ticker_list)
    prev_holdings.index.name = 'Ticker'

    backtest_results, trades = bd.run_backtest_optimizer(prev_holdings, factor_mats)

    attribution_df = bd.PnL_attribution(backtest_results, factor_mats)
    portfolio_characs = bd.build_portfolio_characteristics(backtest_results, trades)
    attribution_df.to_hdf("data/attrib_df.h5", key='df')
    portfolio_characs.to_hdf("data/p_charac_df.h5", key='df')

    print(attribution_df)
    print("\n")
    print(portfolio_characs)

    pnl = attribution_df['daily_PnL']
    
    plt.figure(figsize=(10, 6))
    plt.plot(np.cumsum(pnl))
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.hist(pnl, bins=50, alpha=0.7, label='PnL Distribution')
    plt.legend()
    # plt.axvline(np.percentile(pnl, 100 * (1 - confidence_level)), color='red', linestyle='dashed', label='VaR (95%)')
    plt.title("PnL Distribution")
    plt.xlabel("PnL")
    plt.ylabel("Frequency")
    # plt.legend()
    plt.show()

    confidence_level = .95
    VaR = - np.percentile(pnl, 100*(1-confidence_level))
    print(f"Value at Risk {VaR}")
    # expected shortfall
    ieval = pnl > VaR
    ES = np.sum(pnl * ieval) / np.sum(ieval)
    print(f"Expected shortfall: {ES}")
    plt.figure(figsize=(10, 6))
    plt.hist(pnl, bins=50, alpha=0.7, label='PnL Distribution')
    plt.axvline(-VaR, color='red', linestyle='--', linewidth=2, label=f'VaR 95% ({VaR:.2f})')
    plt.axvline(-ES, color='purple', linestyle='--', linewidth=2, label=f'CVaR 95% ({ES:.2f})')
    plt.title("PnL Distribution with VaR and CVaR (95%)")
    plt.xlabel("PnL")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    attribution_df.plot()
    plt.show()







#bd = BacktestData(ticker_list, one_year_2024_df, backtest_dates, 1) 

# fd = bd.get_factor_data()<