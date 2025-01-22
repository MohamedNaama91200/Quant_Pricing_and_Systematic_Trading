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

    all_calls = all_options[['TTM', 'strike', 'lastPrice']]

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





