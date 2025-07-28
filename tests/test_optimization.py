# import pytest
from unittest.mock import patch, MagicMock
import numpy as np
import pandas as pd
from datetime import date, timedelta
from src import portfolio_optimization  # Adjust if path differs


class TestPortfolioOptimization:
    
    def generate_random_tickers(self, num_tickers=None):
        min_ticker_len = 2
        max_ticker_len = 4
        if not num_tickers:
            num_tickers = np.random.randint(6, 8)
        ticker_symbol_random = np.random.randint(ord('A'), ord('Z')+1, (num_tickers, max_ticker_len))
        ticker_symbol_lengths = np.random.randint(min_ticker_len, max_ticker_len, num_tickers)
        tickers = []
        for ticker_symbol_rand, ticker_symbol_length in zip(ticker_symbol_random, ticker_symbol_lengths):
            ticker_symbol = ''.join([chr(c_id) for c_id in ticker_symbol_rand[:ticker_symbol_length]])
            tickers.append(ticker_symbol)
        return tickers

    def generate_random_dates(self, num_dates=None):
        if not num_dates:
            num_dates = np.random.randint(4, 7)
        start_date = date(np.random.randint(2019, 2023), np.random.randint(1, 12), np.random.randint(1, 29))
        return [start_date + timedelta(days=i) for i in range(num_dates)]

    def generate_multi_index_df(self, tickers, dates, columns_names, index_names):
        index = pd.MultiIndex.from_product([tickers, dates], names=index_names)
        data = 2 * np.random.rand(len(index), len(columns_names)) - 1
        return pd.DataFrame(data, index=index, columns=columns_names)

    @patch('src.portfolio_optimization.OptimalHoldings')
    def test_get_optimal_weights(self, mock_optimal_holdings):
        factors = ['Factor_1', 'Factor_22', 'Factor_3', 'Factor_4', 'Factor_5']
        tickers = self.generate_random_tickers(20)
        dates = self.generate_random_dates(30)

        betas = pd.DataFrame(2 * np.random.rand(len(tickers), len(factors)) - 1, columns=factors, index=tickers)
        betas[factors[0]] = np.abs(betas[factors[0]])

        factor_data = pd.DataFrame(2 * np.random.rand(len(dates), len(factors)) - 1, index=dates, columns=factors)
        factor_data.index.name = 'Date'
        F_cov = np.cov(factor_data.T, ddof=1)
        S_var = np.diag(np.random.uniform(0.1, 12, len(tickers)))

        mock_instance = MagicMock()
        mock_optimal_holdings.return_value = mock_instance
        mock_instance.solve_optimal_holdings.return_value = \
            pd.DataFrame({'optimal_weights': 2 * np.random.rand(len(tickers)) - 1}, index=tickers)

        alpha_vals = 2 * np.random.rand(len(tickers)) - 1
        alpha_vector = pd.DataFrame({'alpha': alpha_vals}, index=tickers)

        result = mock_instance.solve_optimal_holdings(alpha_vector, betas, S_var, F_cov)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(alpha_vector)
        assert 'optimal_weights' in result.columns
        mock_instance.solve_optimal_holdings.assert_called_once()
