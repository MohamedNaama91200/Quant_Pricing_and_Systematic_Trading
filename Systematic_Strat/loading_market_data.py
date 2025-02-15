import yfinance as yf
import pandas as pd


def loading_market_data(start_date="2015-01-01", end_date="2025-02-01",symbols = {"TTF": "TTF=F","EUA": "CO2.L" }):
    """
    Fetches prices data from yfinance.
    """

    data = {
        key: yf.download(symbol, start=start_date, end=end_date)[["Close"]].rename(columns={"Close": key})
        for key, symbol in symbols.items()
    }

    prices_df = pd.concat(data.values(), axis=1).dropna()

    prices_df.columns = [key for key in symbols.keys()]
    prices_df = prices_df.reset_index()
    prices_df = prices_df.set_index("Date")

    return prices_df


if __name__ == "__main__":
    prices_df = loading_market_data()
    print(prices_df)
    print(prices_df.columns)