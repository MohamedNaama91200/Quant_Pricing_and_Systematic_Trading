#Utils functions
import pandas as pd
from statsmodels.api import OLS, add_constant
import numpy as np

def calculate_spread(prices_df,symbol_y="TTF",symbol_x="EUA"):
    """
    Calculates the spread between symbol_y and symbol_x.
    """
    # Regression to calcule the spread without linear regression
    prices_df["Spread"] = prices_df[symbol_y] - prices_df[symbol_x]

    # Compute rolling statistics for Z-score calculation
    rolling_window = 126  # One-year lookback period
    prices_df["Spread_Mean"] = prices_df["Spread"].rolling(rolling_window).mean()
    prices_df["Spread_Std"] = prices_df["Spread"].rolling(rolling_window).std()
    prices_df["Spread_Z"] = (prices_df["Spread"] - prices_df["Spread_Mean"]) / prices_df["Spread_Std"]

    prices_df.dropna(inplace=True)

    return prices_df

def calculate_metrics(prices_df,tickers,positions,initial_capital,transaction_cost) :

    # --- Compute Daily Returns ---
    returns = prices_df[tickers].pct_change().fillna(0)
    daily_portfolio_return = (positions.shift(1) * returns).sum(axis=1)

    # --- Account for Transaction Costs ---
    turnover = positions.diff().abs().sum(axis=1)
    daily_tc = transaction_cost * turnover
    net_daily_return = daily_portfolio_return - daily_tc

    # --- Calculate Portfolio Value & Performance Metrics ---
    cumulative_returns = (1 + net_daily_return).cumprod() * initial_capital
    pnl = cumulative_returns.diff().fillna(0)

    # Annualized Sharpe ratio
    sharpe_ratio = (net_daily_return.mean() / net_daily_return.std()) * np.sqrt(252) if net_daily_return.std() != 0 else np.nan

    # Calculate drawdown
    cum_max = cumulative_returns.cummax()
    drawdown = (cum_max - cumulative_returns) / cum_max
    max_drawdown = drawdown.max()


    return cumulative_returns,pnl,max_drawdown,sharpe_ratio,net_daily_return

if __name__ == "__main__":

    d = {'a': "1", 'b': "2", 'c': "3"}
    rolling_dd = pd.Series(data=d, index=['a', 'b', 'c'])

    for date in rolling_dd.index :

        print(rolling_dd.loc[date])

        rolling_dd.iloc[-1] = f"epepep_{date}"




