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
        print(interpolator(T, K))
        if interpolator(T, K) == interpolator(T, K) :
            return interpolator(T, K)

        else :
            return 0


    def calibrate_local_vol(self,T,K):
        # Calcul de la volatilité locale pour (T, K)

        TTM = self.maturities
        Strike = self.strikes
        C_market = self.market_prices['lastPrice'].values

        T_min_data = np.min(TTM)
        T_max_data = np.max(TTM)
        K_min_data = np.min(Strike)
        K_max_data = np.max(Strike)

        # Clamp T and K to be within the data boundaries (convexity problem of the interpolator)
        T = np.clip(T, T_min_data, T_max_data)
        K = np.clip(K, K_min_data, K_max_data)

        delta = 1e-1  # finite difference step

        points = np.column_stack((TTM, Strike))
        interpolator = CloughTocher2DInterpolator(points, C_market)

        # For the T-derivative, check if T ± delta remain within bounds.
        if T - delta < T_min_data:
            dC_dT = (LocalVol.C(T + delta, K, interpolator) - LocalVol.C(T, K, interpolator)) / delta
        elif T + delta > T_max_data:
            dC_dT = (LocalVol.C(T, K, interpolator) - LocalVol.C(T - delta, K, interpolator)) / delta
        else:
            dC_dT = (LocalVol.C(T + delta, K, interpolator) - LocalVol.C(T - delta, K, interpolator)) / (2 * delta)

        # For the K-derivative, do similar clamping.
        if K - delta < K_min_data:
            dC_dK = (LocalVol.C(T, K + delta, interpolator) - LocalVol.C(T, K, interpolator)) / delta
        elif K + delta > K_max_data:
            dC_dK = (LocalVol.C(T, K, interpolator) - LocalVol.C(T, K - delta, interpolator)) / delta
        else:
            dC_dK = (LocalVol.C(T, K + delta, interpolator) - LocalVol.C(T, K - delta, interpolator)) / (2 * delta)

        # For the second derivative in K, if near a boundary, you might not be able to get a reliable value.
        if K - delta < K_min_data or K + delta > K_max_data:
            d2C_dK2 = np.nan  # or use a fallback method
        else:
            d2C_dK2 = (LocalVol.C(T, K + delta, interpolator) - 2 * LocalVol.C(T, K, interpolator) + LocalVol.C(T, K - delta, interpolator)) / (delta ** 2)

        # Calcul final de σ(T, K) local
        numerator = dC_dT + self.r * K * dC_dK
        denominator = K**2 * d2C_dK2

        if denominator <= 0:
            raise ValueError("Non-positive denominator in local volatility computation. "
                             "Check input parameters or market data.")

        # Dupire's formula numerator and variance calculation:
        local_variance = 2 * numerator / denominator

        if local_variance < 0:
            raise ValueError("Negative local variance computed, cannot take square root.")

        sigma_value = np.sqrt(local_variance)
        #sigma_values.append(sigma_value)

        self.local_vol_surface = sigma_value


    def get_local_vol(self,T,K):

        self.calibrate_local_vol(T,K)
        if self.local_vol_surface is None:
            raise ValueError("La surface de volatilité locale doit être calibrée avant d'obtenir la volatilité.")
        return self.local_vol_surface
