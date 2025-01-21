import pandas as pd
import numpy as np
from visualization.graphics import Graphics

# Example data for convergence graph
mu_k = np.cumsum(np.random.normal(0, 1, 1000)) / np.arange(1, 1001)
lower_bound = mu_k - 1.96 / np.sqrt(np.arange(1, 1001))
upper_bound = mu_k + 1.96 / np.sqrt(np.arange(1, 1001))

# Example historical stock data
historical_prices_df = pd.DataFrame({
    'Date': pd.date_range(start='2020-01-01', periods=100, freq='D'),
    'Close': np.cumsum(np.random.normal(0, 1, 100)) + 100
})
historical_prices_df.set_index('Date', inplace=True)

# Initialize Graphics class
graphics = Graphics()

# Plot convergence graph
graphics.print_convergence_graph(mu_k, lower_bound, upper_bound)

# Plot stock chart
graphics.plot_stock_chart(historical_prices_df, ticker="AAPL")