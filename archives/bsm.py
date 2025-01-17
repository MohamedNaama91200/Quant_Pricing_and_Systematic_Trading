import math
from scipy.stats import norm
import numpy as np

def call_bsm(So, K, r, T, option_type, sig):
    """
    Calcul du prix des options européennes avec le modèle Black-Scholes-Merton.
    
    Arguments:
    So : float - Prix actuel de l'actif sous-jacent
    K : float - Prix d'exercice de l'option
    r : float - Taux d'intérêt sans risque (en décimal, ex : 0.05 pour 5%)
    T : float - Temps jusqu'à l'échéance (en années)
    option_type : str - Type d'option ("Call" ou "Put")
    sig : float - Volatilité de l'actif sous-jacent (en décimal)
    
    Retourne:
    float - Prix de l'option
    """
    d1 = (math.log(So / K) + (r + (sig ** 2) / 2) * T) / (sig * math.sqrt(T))
    d2 = d1 - sig * math.sqrt(T)
    
    if option_type == "Call":
        price = So * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
        return price
    elif option_type == "Put":
        price = -So * norm.cdf(-d1) + K * math.exp(-r * T) * norm.cdf(-d2)
        return price
    else:
        raise ValueError("Le type d'option doit être 'Call' ou 'Put'.")
    

def implied_vol(So, K, T, r, market_price, option_type, tol=1e-4, max_iter=10000):
    """
    Calcule la volatilité implicite en utilisant la méthode de bissection.

    Paramètres :
    So : float - Prix actuel de l'actif sous-jacent
    K : float - Prix d'exercice de l'option
    T : float - Temps jusqu'à l'échéance (en années)
    r : float - Taux d'intérêt sans risque (en décimal)
    market_price : float - Prix de marché de l'option
    option_type : str - Type d'option ("Call" ou "Put")
    tol : float - Tolérance pour la convergence
    max_iter : int - Nombre maximal d'itérations

    Retourne :
    float - Volatilité implicite estimée
    """
    a, b = 0.0001, 1.0
    for _ in range(max_iter):
        sigma = (a + b) / 2
        price = call_bsm(So, K, r, T, option_type, sigma)
        diff = price - market_price
        
        if abs(diff) < tol:
            return sigma
        elif diff > 0:
            b = sigma
        else:
            a = sigma
    
    raise RuntimeError("La méthode de bissection n'a pas convergé.")



def calculate_local_volatility(df, So, r, increment=1.5):
    """
    Calcule la volatilité locale en utilisant la formule de Dupire.

    Paramètres :
    df : DataFrame - Contient les colonnes 'TTM', 'Strike' et 'imp_vol'
    So : float - Prix actuel de l'actif sous-jacent
    r : float - Taux d'intérêt sans risque (en décimal)
    increment : float - Incrément utilisé pour les dérivées numériques

    Retourne :
    DataFrame - DataFrame avec une colonne supplémentaire 'local_vol' représentant la volatilité locale
    """
    local_volatility = []
    
    for i in range(len(df)):
        T = df.loc[i, 'TTM']
        K = df.loc[i, 'Strike']
        sig = df.loc[i, 'imp_vol']
        
        # Dérivée partielle par rapport au temps
        price_T_plus = call_bsm(So, K, r, T + increment, "Call", sig)
        price_T = call_bsm(So, K, r, T, "Call", sig)
        deltac_by_deltat = (price_T_plus - price_T) / increment
        
        # Dérivée première par rapport au strike
        price_K_plus = call_bsm(So, K + increment, r, T, "Call", sig)
        price_K = call_bsm(So, K, r, T, "Call", sig)
        deltac_by_deltak = (price_K_plus - price_K) / increment
        
        # Dérivée seconde par rapport au strike
        price_K_minus = call_bsm(So, K - increment, r, T, "Call", sig)
        delta2c_by_deltak2 = (price_K_plus - 2 * price_K + price_K_minus) / (increment ** 2)
        
        # Calcul de la volatilité locale
        numerator = deltac_by_deltat + r * K * deltac_by_deltak
        denominator = 0.5 * K ** 2 * delta2c_by_deltak2
        if denominator != 0:
            local_vol = np.sqrt(numerator / denominator)
        else:
            local_vol = np.nan  # Évite la division par zéro
        
        local_volatility.append(local_vol)
    
    df['local_vol'] = local_volatility
    return df