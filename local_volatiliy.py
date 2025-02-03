import numpy as np
from scipy.interpolate import CloughTocher2DInterpolator

class LocalVol:
    def __init__(self, market_prices, r):
        self.market_prices = market_prices
        if market_prices.empty:
            raise ValueError("Le df market_prices est vide.")

        required_columns = {'TTM', 'strike', 'lastPrice'}
        if not required_columns.issubset(market_prices.columns):
            raise ValueError(f"Le DataFrame doit contenir les colonnes {required_columns}.")

        market_prices = market_prices.bfill()

        self.market_prices = market_prices
        self.strikes = market_prices['strike']#.unique()
        self.maturities = market_prices['TTM']#.unique()
        self.r = r
        self.local_vol_surface = None
    @staticmethod
    def C(T, K,interpolator):
        if interpolator(T, K) == interpolator(T, K) :
            return interpolator(T, K)

        else :
            return 0


    def calibrate_local_vol(self,T,K):

        TTM = self.maturities
        Strike = self.strikes
        C_market = self.market_prices['lastPrice'].values
        delta = 10**-1

        # Création de l'interpolateur 2D
        interpolator = CloughTocher2DInterpolator((TTM, Strike), C_market)

        # Calcul de la volatilité locale pour chaque (T, K)
#        sigma_values = []
      #   for in zip(TTM, Strike):
        # Première dérivée par rapport à T

        dC_dT = (LocalVol.C(T + delta, K,interpolator) - LocalVol.C(T - delta, K,interpolator)) / (2 * delta)

        # Première dérivée par rapport à
        dC_dK = (LocalVol.C(T, K + delta,interpolator) - LocalVol.C(T, K - delta,interpolator)) / (2 * delta)

        # Deuxième dérivée par rapport à K
        d2C_dK2 = (LocalVol.C(T, K + delta,interpolator) - 2 * LocalVol.C(T, K,interpolator) + LocalVol.C(T, K - delta,interpolator)) / (delta**2)

        # Calcul final de σ(T, K) local
        numerator = dC_dT + self.r * K * dC_dK
        denominator = K**2 * d2C_dK2
        sigma_value = np.sqrt(2 * numerator / denominator)
        #sigma_values.append(sigma_value)

        self.local_vol_surface = sigma_value

    def get_local_vol(self,T,K):

        self.calibrate_local_vol(T,K)
        if self.local_vol_surface is None:
            raise ValueError("La surface de volatilité locale doit être calibrée avant d'obtenir la volatilité.")
        return self.local_vol_surface
