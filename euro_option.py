import math
from scipy.stats import norm
import numpy as np
from local_volatiliy import LocalVol
class EuropeanOption:

    def __init__(self,S,K,T,r,sigma,Option_Type,
                 N_Heston=252,v0=0.04,kappa=2.0,v_bar=0.04,gamma=0.5,
                 rho=-0.7,M_Heston=10000,market_df=False):
        self.S = S #Spot price
        self.K = K #Strike
        self.T = T #Time to Maturity
        self.r = r #Interest rate
        self.sigma = sigma #vol
        self.Option_Type = Option_Type
        #Heston Parameters
        self.N_Heston = N_Heston
        self.v0 = v0
        self.kappa =kappa #"= 2.0  # Vitesse de retour à la moyenne
        self.v_bar = v_bar #= 0.04  # Variance à long terme
        self.gamma = gamma #"= 0.5  # Volatilité de la variance
        self.rho = rho #"-0.7  # Corrélation entre les processus
        self.N_Heston = N_Heston# 252  # Nombre de pas (1 par jour pour 1 an)
        self.M_Heston = M_Heston #10000  # Nombre de simulations Monte-Carlo

    def black_scholes_pricing(self,vol='constant'):

        if vol == 'constant' :
            vol = self.sigma

        # Calcul des paramètres d1 et d2
        d1 = (math.log(self.S / self.K) + (self.r + (vol ** 2) / 2) * self.T) / (vol * math.sqrt(self.T))
        d2 = d1 - vol * math.sqrt(self.T)

        if self.Option_Type == "call":
            # Prix du call
            price = self.S * norm.cdf(d1) - self.K * math.exp(-self.r * self.T) * norm.cdf(d2)
        elif self.Option_Type == "put":
            # Prix du put
            price = self.K * math.exp(-self.r * self.T) * norm.cdf(-d2) - self.S* norm.cdf(-d1)
        else:
            raise ValueError("option_type doit être 'call' ou 'put'")

        return price

    def heston_monte_carlo_pricing(self):

        dt = self.T / self.N_Heston  # Pas de temps

        # Initialisation des matrices pour S et v
        S = np.zeros((self.M_Heston, self.N_Heston + 1))
        v = np.zeros((self.M_Heston, self.N_Heston + 1))
        S[:, 0] = self.S
        v[:, 0] = self.v0

        # Génération des variables normales
        for i in range(self.N_Heston):
            Z1 = np.random.normal(size=self.M_Heston)
            Z2 = np.random.normal(size=self.M_Heston)
            W1 = Z1
            W2 = self.rho * Z1 + np.sqrt(1 - self.rho ** 2) * Z2

            # Mise à jour de v (variance)
            v[:, i + 1] = v[:, i]
            + self.kappa * (self.v_bar - v[:, i]) * dt + self.gamma * np.sqrt(np.maximum(v[:, i] * dt, 0)) * W2

            # Mise à jour de S (prix de l'actif)
            S[:, i + 1] = S[:, i] * np.exp(
                (self.r - 0.5 * v[:, i]) * dt + np.sqrt(np.maximum(v[:, i] * dt, 0)) * W1
            )

        # Calcul des payoffs
        payoffs = np.maximum(S[:, -1] - self.K, 0)

        # Estimation du prix de l'option
        C = np.exp(-self.r * self.T) * np.mean(payoffs)

        return C

    def price_with_local_vol(self,market_df):
        #Dupire Formula
        #The dataframe containing : TTM, Strikes, Prices. False by default
        local_vol = LocalVol(market_prices=market_df,r=self.r)

        # Obtention de la volatilité locale pour le strike et la maturité de l'option
        sigma_local = local_vol.get_local_vol()

        # Calcul du prix de l'option avec Black-Scholes et la volatilité locale
        return self.black_scholes_pricing(vol=sigma_local)



