import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime

def get_options_dataframe(ticker):
    """
    Récupère et traite les données d'options d'un ticker donné, 
    retourne un DataFrame avec les colonnes TTM, Strike et imp_vol.

    Paramètres :
    ticker : str - Le symbole boursier de l'action (par exemple, "AAPL").

    Retourne :
    pd.DataFrame - Un DataFrame contenant les colonnes TTM, Strike, et imp_vol.
    """
    # Récupérer les informations du ticker
    stock = yf.Ticker(ticker)
    option_dates = stock.options  # Dates d'expiration disponibles

    options_data = []

    # Récupérer les données pour chaque date d'expiration
    for option_date in option_dates:
        options_chain = stock.option_chain(option_date)
        calls = options_chain.calls
        calls['maturity'] = option_date  # Ajouter la maturité
        options_data.append(calls)

    # Combiner toutes les données
    all_calls = pd.concat(options_data, ignore_index=True)

    # Calcul de TTM (time to maturity en années)
    today = datetime.today()
    all_calls['TTM'] = all_calls['maturity'].apply(
        lambda x: (datetime.strptime(x, "%Y-%m-%d") - today).days / 365
    )

    # Renommer les colonnes nécessaires
    all_calls = all_calls.rename(columns={
        'strike': 'Strike',
        'lastPrice': 'imp_vol'  # Approximation de la volatilité implicite par le dernier prix
    })

    # Ne conserver que les colonnes pertinentes
    all_calls = all_calls[['TTM', 'Strike', 'imp_vol']]

    return all_calls

