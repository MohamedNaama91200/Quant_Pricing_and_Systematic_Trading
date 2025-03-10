import numpy as np
import matplotlib.pyplot as plt
from utils import calculate_spread,calculate_metrics
import pandas as pd
import itertools

def backtest_strategy_mean_reversion(prices_df,
                                     symbol_y="TTF",
                                     symbol_x="EUA",
                                     transaction_cost=0.001,
                                     z_entry=2,
                                     z_exit=0.5,
                                     initial_capital=100000,
                                     extreme_multiplier=1.5,
                                     plotting=True):
    """
    Backtests a mean-reversion strategy using Z-score signals .

    Parameters:
      prices_df: DataFrame with DateTimeIndex and columns for symbol_y & symbol_x.
      symbol_y: Dependent asset (spread numerator).
      symbol_x: Independent asset (spread denominator).
      transaction_cost: Cost per trade as a fraction (default 0.001).
      z_entry: Z-score threshold for entering a trade (default 2).
      z_exit: Z-score threshold for closing a trade (default 0.5).
      initial_capital: Starting portfolio value.
      plotting: Whether to visualize cumulative PnL.

    Returns:
      DataFrame with portfolio value, daily return, PnL, and drawdown.
    """

    # --- Compute Spread & Z-score ---
    spread_prices_df = calculate_spread(prices_df, symbol_y=symbol_y, symbol_x=symbol_x)

    # Define an "extreme" condition: if the absolute spread is greater than extreme_multiplier * rolling std
    spread_prices_df["Extreme"] = abs(spread_prices_df["Spread"]) > abs(spread_prices_df["Spread_Mean"]) + extreme_multiplier * spread_prices_df["Spread_Std"]

    # Define trading signals based on Z-score, but only take trades if not in an extreme spread condition
    spread_prices_df["Long_Signal"] = (spread_prices_df["Spread_Z"] < -z_entry) & (~spread_prices_df["Extreme"])
    spread_prices_df["Short_Signal"] = (spread_prices_df["Spread_Z"] > z_entry) & (~spread_prices_df["Extreme"])
    spread_prices_df["Close_Signal"] = (spread_prices_df["Spread_Z"] > -z_exit) & (
                spread_prices_df["Spread_Z"] < z_exit)

    # --- Initialize Positions ---
    spread_prices_df["Position"] = 0.0
    for i in range(1, len(spread_prices_df)):
        if spread_prices_df["Long_Signal"].iloc[i]:
            spread_prices_df.loc[spread_prices_df.index[i], "Position"] = 1  # Long spread
        elif spread_prices_df["Short_Signal"].iloc[i]:
            spread_prices_df.loc[spread_prices_df.index[i], "Position"] = -1  # Short spread
        elif spread_prices_df["Close_Signal"].iloc[i]:
            spread_prices_df.loc[spread_prices_df.index[i], "Position"] = 0  # Close position
        else:
            spread_prices_df.loc[spread_prices_df.index[i], "Position"] = spread_prices_df.loc[
                spread_prices_df.index[i - 1], "Position"]

    # --- Compute Cumulative Returns ---

    positions = pd.DataFrame({
        symbol_y: spread_prices_df["Position"],  # Long sur symbol_y quand Position = 1, short quand -1
        symbol_x: -spread_prices_df["Position"]  # L'inverse pour symbol_x
    }, index=spread_prices_df.index)

    # --- Compute Final Performance Metrics ---
    cumulative_returns, pnl, max_drawdown, sharpe_ratio, net_daily_return = calculate_metrics(
        prices_df, [symbol_y, symbol_x], positions, initial_capital, transaction_cost
    )

    # Print performance metrics
    print(f"Total Net Return: {(cumulative_returns.iloc[-1] / initial_capital - 1) * 100:.2f} %")
    print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
    print(f"Maximum Drawdown: {max_drawdown:.2%}")

    # --- Visualization ---
    if plotting:
        plt.figure(figsize=(12, 6))
        plt.plot(cumulative_returns.index, cumulative_returns, label="Portfolio Value", color='blue')
        plt.axhline(y=initial_capital, color='black', linestyle='--', linewidth=0.8)
        plt.title("Mean Reversion Strategy with Brent and WTI spread")
        plt.xlabel("Date")
        plt.ylabel("Portfolio Value")
        plt.legend()
        plt.grid(True)
        plt.show()

    # --- Return Results ---
    results = pd.DataFrame({
        "Portfolio Value": cumulative_returns,
        "Daily Return": net_daily_return,
        "PnL": pnl,
        "Drawdown": max_drawdown
    })
    return results







def backtest_asset_class_trend(prices_df,
                               tickers=["SPY", "EFA", "IEF", "VNQ", "GSG"],
                               sma_period=125,
                               balancing_freq='monthly',
                               transaction_cost=0.001,
                               initial_capital=100000,
                               factor_weighting=True,
                               factor_weighting_method="momentum_carry",
                               top_n_assets=3,allocation_type='vol_adj_weights',exposure_reduction=0.5,max_drawdown_threshold=0.1,plotting=True):
    """
    Backtests an asset class trend following strategy with Factor-Based Asset Selection.
    - Uses a moving average (SMA) trend-following rule.
    - Uses momentum (6M/12M returns) & carry (rolling yield) to rank assets.
    - Allocates capital only to the top N ranked assets.
    - Applies volatility-adjusted weighting.

    Parameters:
      prices_df: DataFrame with a DateTimeIndex and columns for each ticker.
      tickers: List of asset tickers.
      sma_period: Window for the SMA (default 125 days).
      balancing_freq: 'weekly' or 'monthly' rebalancing.
      transaction_cost: Daily transaction cost as a fraction (default 0.001).
      initial_capital: Starting portfolio value.
      factor_weighting: Whether to rank assets based on momentum & carry (default True).
      factor_weighting_method: Factor ranking method ('momentum_carry' or 'momentum_only').
      top_n_assets: Number of top-ranked assets to select per rebalancing.
      max_drawdown_threshold : the maximum drawdown in a day that we accept, before gradually de-risking
      exposure_reduction : the exposure reduction, meaning if in date d we exceed the drawdown threshold, we gradually reduce the exposure until exposure_reduction

    Returns:
      DataFrame with portfolio value, daily return, PnL, and drawdown.
    """

    prices_df = prices_df.sort_index()

    # --- Compute SMA and Trend Signal ---
    for ticker in tickers:
        sma_col = f"{ticker}_SMA"
        signal_col = f"{ticker}_signal"
        prices_df[sma_col] = prices_df[ticker].rolling(window=sma_period, min_periods=sma_period).mean()
        prices_df[signal_col] = prices_df[ticker] > prices_df[sma_col]

    # --- Compute Factor-Based Ranking ---
    # 6-month (126-day) and 12-month (252-day) momentum
    momentum_6M = prices_df[tickers].pct_change(126)
    momentum_12M = prices_df[tickers].pct_change(252)

    # Carry proxy: rolling 21-day moving average price / current price
    carry = prices_df[tickers].rolling(21).mean() / prices_df[tickers]

    # Normalize factors (z-score transformation)
    momentum_6M = (momentum_6M - momentum_6M.mean()) / momentum_6M.std()
    momentum_12M = (momentum_12M - momentum_12M.mean()) / momentum_12M.std()
    carry = (carry - carry.mean()) / carry.std()

    # Combine factors based on user selection
    if factor_weighting_method == "momentum_carry":
        factor_score = 0.5 * momentum_6M + 0.5 * momentum_12M + carry
    elif factor_weighting_method == "momentum_only":
        factor_score = 0.5 * momentum_6M + 0.5 * momentum_12M
    elif factor_weighting_method == "momentum_6M":
        factor_score = 1 * momentum_6M
    elif factor_weighting_method == "momentum_12M":
        factor_score = 1 * momentum_12M

    else:
        raise ValueError("Invalid factor weighting method")

    # --- Set up Positions ---
    positions = pd.DataFrame(index=prices_df.index, columns=tickers)
    positions[:] = np.nan

    # Identify rebalancing dates: weekly or monthly
    if balancing_freq == 'weekly':
        rebal_dates = prices_df.groupby([
            prices_df.index.isocalendar().year,
            prices_df.index.isocalendar().week
        ]).head(1).index
    elif balancing_freq == 'monthly':
        rebal_dates = prices_df.groupby([
            prices_df.index.year,
            prices_df.index.month
        ]).head(1).index
    else:
        raise ValueError('Balancing frequency must be : "monthly" or "weekly" ')

    # Compute 20-day realized volatility for each asset
    realized_vol = prices_df[tickers].pct_change().rolling(20).std()


    # --- Rebalancing Process ---


    for date in rebal_dates:

        # Identify trend-following assets
        active_assets = [ticker for ticker in tickers if prices_df.loc[date, f"{ticker}_signal"]]
        if active_assets:
            if factor_weighting:
                # Rank active assets by factor score and select top N
                ranked_assets = factor_score.loc[date, active_assets].nlargest(top_n_assets).index.tolist()
            else:
                ranked_assets = active_assets

            # Compute volatility-adjusted weights for selected assets
            inv_vol = 1 / realized_vol.loc[date, ranked_assets]
            vol_adj_weights = inv_vol / inv_vol.sum()
            #Compute equal-weights
            equal_weights = 1 / len(active_assets)

            # Assign weights to each ranked asset
            for ticker in tickers:
                if allocation_type == 'vol_adj_weights' :
                    #Positions are change if the delta between the positions at the previous date and the new positions is greater than 0.1
                    positions.loc[date, ticker] = vol_adj_weights[ticker] if ticker in ranked_assets else 0.0
                    if abs(positions.loc[date, ticker] - positions.loc[
                        rebal_dates[rebal_dates.get_loc(date) - 1], ticker]) < 0.1:

                        positions.loc[date, ticker] = positions.loc[
                        rebal_dates[rebal_dates.get_loc(date) - 1], ticker]

                elif allocation_type == 'equal_weights' :
                    positions.loc[date, ticker] = equal_weights if ticker in ranked_assets else 0.0
        else:
            # If no asset is trending, stay in cash (all weights zero)
            positions.loc[date] = 0.0

    # Forward-fill positions for days between rebalancing dates.
    positions = positions.ffill().fillna(0.0)


    cumulative_returns = calculate_metrics(prices_df,tickers,positions,initial_capital,transaction_cost)[0]

    # --- Rolling Drawdown-Based De-Risking ---

    rolling_peak = cumulative_returns.rolling(window=120, min_periods=1).max()  # 6-month rolling peak
    rolling_dd = (rolling_peak - cumulative_returns) / rolling_peak

    processed_dates = set()  # Keep track of modified dates, goal is to not reduce the exposure twice in the backtest
    for date in rolling_dd.index:
        if rolling_dd.loc[date] >= max_drawdown_threshold:  # Drawdown limit exceeded
            next_idx = rolling_dd.index.get_loc(date)
            reduction_days = 2  # Reduce exposure over 3 days
            for i in range(1, reduction_days + 1):
                future_idx = next_idx + i
                if future_idx < len(rolling_dd.index):
                    future_date = rolling_dd.index[future_idx]
                    # Check if future_date has already been modified
                    if future_date not in processed_dates:
                        positions.loc[future_date] *= exposure_reduction * (i / reduction_days)
                        processed_dates.add(future_date)  # Mark as modified
            # After finishing the x-day reduction, forward-fill until next rebal date
            last_reduction_idx = next_idx + reduction_days
            if last_reduction_idx < len(rolling_dd.index):
                final_reduction_day = rolling_dd.index[last_reduction_idx]

                # Find the next rebal date after final_reduction_day
                rebal_after = rebal_dates[rebal_dates > final_reduction_day]
                if len(rebal_after) > 0:
                    next_rebal_day = rebal_after[0]
                    # Forward-fill the final reduced positions up to (but not including) next_rebal_day
                    positions.loc[final_reduction_day:next_rebal_day] = positions.loc[final_reduction_day].values

            #Recomputing cum returns
            cumulative_returns = calculate_metrics(prices_df, tickers, positions, initial_capital, transaction_cost)[0]
            # Immediately update rolling drawdown after new cumulative returns
            rolling_peak = cumulative_returns.rolling(window=120, min_periods=1).max()
            rolling_dd = (rolling_peak - cumulative_returns) / rolling_peak



    #Recompute metrics with new positions after rolling drawdown de-risking

    # Annualized Sharpe ratio
    cumulative_returns,pnl,max_drawdown,sharpe_ratio,net_daily_return = calculate_metrics(
        prices_df, tickers, positions, initial_capital, transaction_cost
    )

    # Calculate drawdown
    cum_max = cumulative_returns.cummax()
    drawdown = (cum_max - cumulative_returns) / cum_max
    max_drawdown = drawdown.max()

    # Print performance metrics
    print(f"Total Net Return: {(cumulative_returns.iloc[-1] / initial_capital - 1) * 100:.2f} %")
    print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
    print(f"Maximum Drawdown: {max_drawdown:.2%}")

    if plotting :
        # --- Visualization ---
        plt.figure(figsize=(12, 6))
        plt.plot(cumulative_returns.index, cumulative_returns, label="Portfolio Value", color='blue')
        plt.axhline(y=initial_capital, color='black', linestyle='--', linewidth=0.8)
        plt.title("Asset Class Trend Following with Factor-Based Selection")
        plt.xlabel("Date")
        plt.ylabel("Portfolio Value")
        plt.legend()
        plt.grid(True)
        plt.show()

    # --- Return Results ---
    results = pd.DataFrame({
        "Portfolio Value": cumulative_returns,
        "Daily Return": net_daily_return,
        "PnL": pnl,
        "Drawdown": drawdown
    })
    return results




def backtest_static_equal_weighted_benchmark(prices_df,
                                             tickers=["SPY", "EFA", "IEF", "VNQ", "GSG"],
                                             transaction_cost=0.001,
                                             initial_capital=100000,
                                             plotting=True):
    """
    Backtests a simple static equal-weighted portfolio benchmark (no rebalancing).

    Parameters:
        prices_df: DataFrame with asset price data.
        tickers: List of asset tickers.
        transaction_cost: Cost per trade.
        initial_capital: Starting capital.
        plotting: Whether to plot the portfolio value.

    Returns:
        DataFrame with portfolio value, daily return, and drawdown.
    """

    prices_df = prices_df.sort_index()


    # Allocate capital equally at the start (no rebalancing)
    num_assets = len(tickers)
    initial_allocation = initial_capital / num_assets  # Capital per asset

    # Compute the initial number of shares for each asset
    initial_prices = prices_df.iloc[0][tickers]  # First available prices
    num_shares = initial_allocation / initial_prices  # Shares bought at start

    # Compute portfolio value over time
    portfolio_value = (num_shares * prices_df[tickers]).sum(axis=1)

    # Compute daily portfolio returns
    daily_portfolio_return = portfolio_value.pct_change().fillna(0)

    # Apply transaction costs at the start only
    initial_transaction_cost = transaction_cost * initial_capital  # One-time cost
    portfolio_value -= initial_transaction_cost  # Deduct initial costs

    # Compute cumulative PnL
    pnl = portfolio_value.diff().fillna(0)
    #total_return = portfolio_value.iloc[-1] - initial_capital

    # Compute Sharpe Ratio
    sharpe_ratio = (daily_portfolio_return.mean() / daily_portfolio_return.std()) * np.sqrt(
        252) if daily_portfolio_return.std() != 0 else np.nan

    # Compute Drawdown
    cum_max = portfolio_value.cummax()
    drawdown = (cum_max - portfolio_value) / cum_max
    max_drawdown = drawdown.max()

    # Print Performance Metrics
    print(f"Total Net Return: {(portfolio_value.iloc[-1] / initial_capital - 1)*100:.2f} %")
    print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
    print(f"Maximum Drawdown: {max_drawdown:.2%}")

    if plotting:
        # --- Visualization ---
        plt.figure(figsize=(12, 6))
        plt.plot(portfolio_value.index, portfolio_value, label="Static Equal-Weighted Portfolio", color='blue')
        plt.axhline(y=initial_capital, color='black', linestyle='--', linewidth=0.8)
        plt.title("Static Equal-Weighted Portfolio Benchmark")
        plt.xlabel("Date")
        plt.ylabel("Portfolio Value")
        plt.legend()
        plt.grid(True)
        plt.show()


    # --- Return Results ---
    results = pd.DataFrame({
        "Portfolio Value": portfolio_value,
        "Daily Return": daily_portfolio_return,
        "PnL": pnl,
        "Drawdown": drawdown
    })

    return results




if __name__ == "__main__":
    from loading_market_data import loading_market_data

    symbols_ttf_eua = {"TTF": "TTF=F", "EUA": "CO2.L"}
    symbols_brent_wti = {"Brent": "BZ=F", "WTI": "CL=F"}
    symbols_index = {"SPY":"SPY", "EFA":"EFA", "IEF":"IEF", "VNQ":"VNQ", "GSG":"GSG"}
    spy = {"SPY":"SPY"}

    prices_df = loading_market_data(symbols=symbols_brent_wti,start_date="2015-01-01", end_date="2025-02-01")

    #Plot the different tickers prices over time on the same plot


    #spy_prices = loading_market_data(symbols=spy,start_date="2015-01-01", end_date="2025-02-01")

    # Plot the different tickers prices over time on the same plot
    """""
    prices_df.plot(figsize=(12, 6))
    plt.title("Prices of Different Tickers Over Time")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend(loc="best")
    plt.grid(True)
    plt.show()
    """

    #Backtesting PnL

    spread_prices_df = backtest_strategy_mean_reversion(prices_df=prices_df, symbol_y="Brent", symbol_x="WTI")
    #trend_pairs = backtest_strategy_trend_pairs_trading(prices_df=prices_df,symbol_y="TTF",symbol_x="EUA")
    #trend_symbol = backtest_strategy_trend(prices_df=prices_df,symbol_x="EUA",window=15)

    if False :
        # Define parameter ranges
        sma_periods = [100, 125, 150, 175, 200]#[100, 125, 150, 175, 200]  # Different SMA periods
        balancing_frequencies = ['weekly', 'monthly']  # Rebalancing frequency balancing_frequencies = ['weekly', 'monthly']  # Rebalancing frequency
        factor_methods = ["momentum_only", "momentum_carry","momentum_6M","momentum_12M"]  # Factor weighting ["momentum_only", "momentum_carry","momentum_6M",]
        top_n_assets_list = [1,2, 3, 4, 5]  # Number of top assets selected [1,2, 3, 4, 5]
        allocation_type_list = ['vol_adj_weights']#,'equal_weights']
        max_drawdown_list = [0.1]#[0.05,0.1,0.15,0.2,0.3]
        exposure_list = [0.5] #[0,0.1,0.2,0.5,0.75,1]

        # Store results
        results_list = []

        # Iterate over all parameter combinations
        for sma_period, balancing_freq, factor_method, top_n_assets,allocation_type,max_dr,expo in itertools.product(
                sma_periods, balancing_frequencies, factor_methods, top_n_assets_list,allocation_type_list,max_drawdown_list,exposure_list):
            print(f"Testing: SMA={sma_period}, Freq={balancing_freq}, Factor={factor_method}, TopN={top_n_assets}")

            # Run backtest
            result = backtest_asset_class_trend(
                prices_df=prices_df,
                tickers=[key for key in symbols_index.keys()],
                sma_period=sma_period,
                balancing_freq=balancing_freq,
                factor_weighting_method=factor_method,
                transaction_cost=0.001,
                top_n_assets=top_n_assets,
                initial_capital=100000,
                allocation_type=allocation_type,
                max_drawdown_threshold=max_dr,
                exposure_reduction=expo,
                plotting=False
            )

            # Extract Sharpe Ratio and Max Drawdown
            sharpe_ratio = (result["Daily Return"].mean() / result["Daily Return"].std()) * np.sqrt(252)
            max_drawdown = result["Drawdown"].max()

            # Save results
            results_list.append({
                "SMA Period": sma_period,
                "Balancing Frequency": balancing_freq,
                "Factor Method": factor_method,
                "Top N Assets": top_n_assets,
                "Sharpe Ratio": sharpe_ratio,
                "Max Drawdown": max_drawdown,
                "Alloc Type": allocation_type,
                "Exposure":expo,
                "Drawdown" : max_dr
            })

        # Convert results to DataFrame
        results_df = pd.DataFrame(results_list)

        # Sort by Sharpe Ratio (descending) and Max Drawdown (ascending)
        best_results = results_df.sort_values(by=["Sharpe Ratio", "Max Drawdown"], ascending=[False, True])

        print(best_results.head(1).to_string())



    sma_strat = backtest_asset_class_trend(prices_df,tickers=[key for key in symbols_index.keys()],sma_period=175,balancing_freq='weekly',factor_weighting=True,
                                             factor_weighting_method="momentum_12M",transaction_cost=0.001,top_n_assets=2,
                                         initial_capital=100000,allocation_type='vol_adj_weights',exposure_reduction=0.5,max_drawdown_threshold=0.1)


    benchmark = backtest_static_equal_weighted_benchmark(prices_df=prices_df,tickers=[key for key in symbols_brent_wti.keys()],transaction_cost=0.001,initial_capital=100000)



