#Utils functions
from statsmodels.api import OLS, add_constant
import numpy as np

def calculate_spread(prices_df,symbol_y="TTF",symbol_x="EUA"):
    """
    Calculates the spread between symbol_y and symbol_x using a linear regression model.
    """

    # Regression to calculate hedge ratio
    X = add_constant(prices_df[symbol_x])
    model = OLS(prices_df[symbol_y], X).fit()
    beta, intercept = model.params[symbol_x], model.params["const"]

    # Compute the spread
    prices_df["Spread"] = prices_df[symbol_y] - (beta * prices_df[symbol_x] + intercept)

    # Compute rolling statistics for Z-score calculation
    rolling_window = 252  # One-year lookback period
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





