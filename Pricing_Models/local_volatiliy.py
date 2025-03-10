import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CloughTocher2DInterpolator
from scipy.ndimage import gaussian_filter


class LocalVolatility:
    def __init__(self, market_data, r, q=0):
        """
        Initialise la classe LocalVolatility avec les prix de marché et les paramètres financiers.

        :param market_data: DataFrame contenant 'TTM', 'strike', 'impliedVolatility', 'lastPrice'
        :param r: Taux sans risque
        :param q: Rendement du dividende (optionnel, par défaut 0)
        """
        if market_data.empty:
            raise ValueError("Le DataFrame market_data est vide.")

        self.market_data = market_data.dropna().sort_values(by=['TTM', 'strike'])
        self.r = r
        self.q = q
        self._fit_interpolators()

    def _fit_interpolators(self):
        """
        Crée des interpolateurs pour la surface de volatilité implicite et la surface de prix.
        """
        points = np.column_stack((self.market_data['TTM'], self.market_data['strike']))
        implied_vols = self.market_data['impliedVolatility'].values
        prices = self.market_data['lastPrice'].values

        # Interpolation de la volatilité implicite
        self.vol_interpolator = CloughTocher2DInterpolator(points, implied_vols)

        # Interpolation des prix des options (lissage avec un filtre Gaussien si bruit)
        smoothed_prices = gaussian_filter(prices, sigma=1)
        self.price_interpolator = CloughTocher2DInterpolator(points, smoothed_prices)

    def get_price(self, T, K):
        """ Renvoie le prix interpolé de l'option. """
        return self.price_interpolator((T, K))

    def get_volatility(self, T, K):
        """ Renvoie la volatilité implicite interpolée. """
        return self.vol_interpolator((T, K))

    def compute_local_vol(self, T, K, delta=1e-4):
        """
        Calcule la volatilité locale en utilisant la formule de Dupire.
        """
        C_TK = self.get_price(T, K)

        # Dérivée temporelle dC/dT (différences finies centrées)
        dC_dT = (self.get_price(T + delta, K) - self.get_price(T, K)) / delta

        # Dérivée première en strike dC/dK
        dC_dK = (self.get_price(T, K + delta) - self.get_price(T, K - delta)) / (2 * delta)

        # Dérivée seconde en strike d²C/dK²
        d2C_dK2 = (self.get_price(T, K + delta) - 2 * C_TK + self.get_price(T, K - delta)) / (delta ** 2)
        d2C_dK2 = max(d2C_dK2, 1e-6)  # Éviter les valeurs négatives

        # Application de la formule de Dupire
        numerator = dC_dT + (self.r - self.q) * K * dC_dK + self.q * C_TK
        numerator = max(numerator, 1e-6)  # Stabilité numérique
        denominator = K ** 2 * d2C_dK2
        denominator = max(denominator, 1e-6)

        local_variance = 2 * numerator / denominator
        return np.sqrt(local_variance) if local_variance > 0 else np.nan

    def fit_local_vol_surface(self):
        """
        Calcule la surface complète de volatilité locale.
        """
        self.market_data['localVolatility'] = self.market_data.apply(
            lambda row: self.compute_local_vol(row['TTM'], row['strike']), axis=1
        )
        return self.market_data

    def plot_local_vs_implied_vol(self):
        """
        Trace la comparaison entre la volatilité implicite et la volatilité locale calibrée.
        """
        self.fit_local_vol_surface()
        calibrated_lv_results = {}

        for TTM, group in self.market_data.groupby('TTM'):
            strikes = group['strike'].values
            market_vols = group['impliedVolatility'].values
            model_vols = group['localVolatility'].values

            plt.figure(figsize=(8, 5))
            plt.plot(strikes, market_vols, 'bo', label='Market Implied Vol')
            plt.plot(strikes, model_vols, 'r-', label='Local Vol Model')
            plt.xlabel("Strike")
            plt.ylabel("Volatility")
            plt.title(f"Local Volatility vs Market Implied Vol for TTM = {TTM} years")
            plt.legend()
            plt.grid(True)
            plt.show()

            calibrated_lv_results[TTM] = model_vols

        return calibrated_lv_results


# Exemple d'utilisation
def main():
    from loading_market_data import (get_options_dataframe, get_risk_free_rate,
                                     get_dividend_yield)

    # Chargement des données de marché
    option_data_df = get_options_dataframe(ticker='AAPL', option_type='put')
    r_risk_free = get_risk_free_rate()
    dividend_yield = get_dividend_yield(ticker='AAPL')

    # Création de la surface de volatilité locale
    lv_surface = LocalVolatility(market_data=option_data_df, r=r_risk_free, q=dividend_yield)

    # Affichage des courbes
    lv_surface.plot_local_vs_implied_vol()


if __name__ == "__main__":
    main()
