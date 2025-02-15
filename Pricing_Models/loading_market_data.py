import yfinance as yf
import pandas as pd
from datetime import datetime
import numpy as np
from functools import lru_cache


@lru_cache(maxsize=None)
def get_latest_price(ticker) :
    stock = yf.Ticker(ticker)
    hist = stock.history(period="1d")
    return round(hist['Close'].iloc[-1],4)

@lru_cache(maxsize=None)
def get_options_dataframe(ticker,option_type='call'):
    # Récupérer les informations du ticker
    stock = yf.Ticker(ticker)
    option_dates = stock.options  # Dates d'expiration disponibles

    options_data = []

    # Récupérer les données pour chaque date d'expiration
    for option_date in option_dates:
        options_chain = stock.option_chain(option_date)
        #Call ou put :
        if option_type == 'call' :
            option_df = options_chain.calls
            option_df['maturity'] = option_date
            options_data.append(option_df)
        elif option_type == 'put' :
            option_df = options_chain.puts
            option_df['maturity'] = option_date
            options_data.append(option_df)

    # Combiner toutes les données
    all_options = pd.concat(options_data, ignore_index=True)

    # TTM en années
    today = datetime.today()

    all_options['TTM'] = all_options['maturity'].apply(
        lambda x: (datetime.strptime(x, "%Y-%m-%d").replace(hour=22,minute=00) - today).total_seconds() /(365*24*3600)
    ) #22h = 16h à NY donc close du Nasdaq

    all_calls = all_options[['TTM', 'strike', 'lastPrice','impliedVolatility']]

    return all_calls
@lru_cache(maxsize=None)
def get_risk_free_rate():

    # Récupérer les données des Treasury Bills à 13 semaines (^IRX)
    tbill = yf.Ticker("^IRX")
    tbill_data = tbill.history(period="1d")

    # Si les données des T-Bills ne sont pas disponibles, utiliser les Treasury Bonds à 10 ans (^TNX)
    if tbill_data.empty:
        tbond = yf.Ticker("^TNX")
        tbond_data = tbond.history(period="1d")
        if tbond_data.empty:
            raise ValueError("Impossible de récupérer le taux sans risque.")
        risk_free_rate = tbond_data['Close'].iloc[-1] / 100
    else:
        risk_free_rate = tbill_data['Close'].iloc[-1] / 100

    return round(risk_free_rate,4)
@lru_cache(maxsize=None)
def get_realized_vol(ticker,period="6mo") :
    ticker = yf.Ticker(ticker)
    hist = ticker.history(period=period)
    prices = hist['Close']
    #Realized vol is on the log returns
    log_returns = np.log(prices / prices.shift(1)).dropna()

    return round(log_returns.std() * np.sqrt(252),4)

@lru_cache(maxsize=None)
def get_dividend_yield(ticker) :
    ticker = yf.Ticker(ticker)
    dividend_yield = ticker.info.get('dividendYield', None)
    if dividend_yield is not None :
        return dividend_yield
    else:
        return 0




@lru_cache(maxsize=None)
def get_futures_options_data(ticker='ES=F'):
    # Get the futures ticker data using yfinance.
    tk = yf.Ticker(ticker)

    # Retrieve the latest available futures price.
    try:
        hist = tk.history(period="1d")
        if not hist.empty:
            underlying_price = hist['Close'].iloc[-1]
        else:
            underlying_price = None
    except Exception as e:
        underlying_price = None

    # Set today's date for time-to-expiry calculations.
    today = datetime.today().date()

    data_list = []
    # For demonstration purposes, process a subset of available expiration dates.
    available_expirations = tk.options
    for exp in available_expirations[:2]:
        try:
            # Get the option chain for the expiration date (using the calls side).
            opt_chain = tk.option_chain(exp)
        except Exception as e:
            continue  # Skip this expiration if there is an issue.

        df_calls = opt_chain.calls.copy()
        if df_calls.empty:
            continue

        # Calculate time-to-expiry (in years)
        try:
            exp_date = datetime.strptime(exp, '%Y-%m-%d').date()
        except Exception as e:
            continue  # Skip if the date format is unexpected.
        TTM = (exp_date - today).days / 365.25

        # If the implied volatility field is missing, assign a default value (e.g., 0.25)
        if 'impliedVolatility' not in df_calls.columns:
            df_calls['impliedVolatility'] = 0.25

        # Add columns for expiry and underlying futures price.
        df_calls['expiry'] = TTM
        df_calls['underlying_price'] = underlying_price

        # Keep only the required columns.
        df_calls = df_calls[['expiry', 'strike', 'impliedVolatility', 'underlying_price']]
        df_calls.rename(columns={'impliedVolatility': 'implied_volatility'}, inplace=True)

        data_list.append(df_calls)

    if len(data_list) == 0:
        return pd.DataFrame()

    futures_options_df = pd.concat(data_list, ignore_index=True)
    return futures_options_df


# Example usage:
if __name__ == '__main__':
    df = get_futures_options_data(ticker='CL=F')
    print(df.head())




