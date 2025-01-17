import numpy as np
from scipy.interpolate import interp1d

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
        self.strikes = market_prices['strike'].unique()
        self.maturities = market_prices['TTM'].unique()
        self.r = r
        self.local_vol_surface = None

    def calibrate_local_vol(self):

        local_vols = np.zeros((len(self.maturities), len(self.strikes)))  # (n_maturities, n_strikes)

        for i, T in enumerate(self.maturities):
            for j, K in enumerate(self.strikes):
                # Initialiser les dérivées à 0 par défaut
                dC_dT, dC_dK, d2C_dK2 = 0.0, 0.0, 0.0

                # Récupérer le prix actuel
                current_price_data = self.market_prices.loc[
                    (self.market_prices['TTM'] == T) & (self.market_prices['strike'] == K),
                    'lastPrice'
                ]
                if not current_price_data.empty:
                    current_price = current_price_data.values[0]
                else:
                    # Si le prix actuel n'existe pas, passer à l'itération suivante
                    local_vols[i, j] = 0.2  # Valeur par défaut
                    continue

                # Calcul de dC_dT
                if i < len(self.maturities) - 1:
                    next_maturity = self.maturities[i + 1]
                    next_price_data = self.market_prices.loc[
                        (self.market_prices['TTM'] == next_maturity) & (self.market_prices['strike'] == K),
                        'lastPrice'
                    ]
                    if not next_price_data.empty:
                        next_price = next_price_data.values[0]
                        dC_dT = (next_price - current_price) / (next_maturity - T)

                # Calcul de dC_dK
                if j < len(self.strikes) - 1:
                    next_strike = self.strikes[j + 1]
                    next_price_data = self.market_prices.loc[
                        (self.market_prices['TTM'] == T) & (self.market_prices['strike'] == next_strike),
                        'lastPrice'
                    ]
                    if not next_price_data.empty:
                        next_price = next_price_data.values[0]
                        dC_dK = (next_price - current_price) / (next_strike - K)

                # Calcul de d2C_dK2
                if 0 < j < len(self.strikes) - 1:
                    prev_strike = self.strikes[j - 1]
                    next_strike = self.strikes[j + 1]

                    prev_price_data = self.market_prices.loc[
                        (self.market_prices['TTM'] == T) & (self.market_prices['strike'] == prev_strike),
                        'lastPrice'
                    ]
                    next_price_data = self.market_prices.loc[
                        (self.market_prices['TTM'] == T) & (self.market_prices['strike'] == next_strike),
                        'lastPrice'
                    ]

                    if not prev_price_data.empty and not next_price_data.empty:
                        prev_price = prev_price_data.values[0]
                        next_price = next_price_data.values[0]
                        d2C_dK2 = (next_price - 2 * current_price + prev_price) / ((next_strike - K) ** 2)

                # Formule de Dupire pour la volatilité locale
                if d2C_dK2 != 0:
                    local_vols[i, j] = np.sqrt((dC_dT + self.r * K * dC_dK) / (0.5 * K ** 2 * d2C_dK2))
                else:
                    local_vols[i, j] = 0.2  # Valeur par défaut si la dérivée seconde est nulle

        # Interpolation de la surface de volatilité locale
        self.local_vol_surface = interp1d(self.strikes, local_vols, axis=1, fill_value="extrapolate").y

    def get_local_vol(self):

        self.calibrate_local_vol()
        if self.local_vol_surface is None:
            raise ValueError("La surface de volatilité locale doit être calibrée avant d'obtenir la volatilité.")
        return self.local_vol_surface
