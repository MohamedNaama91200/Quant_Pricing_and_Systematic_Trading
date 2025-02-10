import numpy as np
from scipy.stats import norm
import math



def black_scholes_pricing(S, K, T, r, sigma,Option_Type):


    # Calcul des paramètres d1 et d2
    d1 = (math.log(S / K) + (r + (sigma ** 2) / 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)

    if Option_Type == "call":
        # Prix du call
        price = S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
    elif Option_Type == "put":
        # Prix du put
        price = K * math.exp(-r * T) * norm.cdf(-d2) - S* norm.cdf(-d1)
    else:
        raise ValueError("option_type doit être 'call' ou 'put'")

    return price
