#Utils functions
from statsmodels.api import OLS, add_constant

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


