import numpy as np
import black_scholes
from local_volatiliy import LocalVol
from sabr import SABRModel
class EuropeanOption:

    def __init__(self,S,K,T,r,sigma,Option_Type,
                 N_Heston=252,v0=0.04,kappa=2.0,v_bar=0.04,gamma=0.5,
                 rho=-0.7,M_Heston=10000,dividend_yield=0):
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
        self.dividend_yield = dividend_yield


    def black_scholes_pricing(self,vol=10000000000):

        if vol == 10000000000 :
            vol = self.sigma

        return black_scholes.black_scholes_pricing(S=self.S,K=self.K,T=self.T,r=self.r,sigma=vol,Option_Type=self.Option_Type)

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

        #The dataframe containing : TTM, Strikes, Prices. False by default
        lv_surface = LocalVol(market_prices=market_df,r=self.r)
        sigma_local = lv_surface.get_local_vol(T=self.T,K=self.K)

        """

        dt = self.T / num_steps
        S_paths = np.zeros((num_paths, num_steps + 1))
        S_paths[:, 0] = self.S


        # Monte-Carlo to simulate the underlying using the SDE with sigma_local
        for i in range(num_steps):
            t = i * dt
            # Compute the local volatility for each path given the current asset price.
            sigma_local = np.array([lv_surface.get_local_vol(t, s) for s in S_paths[:, i]])
            # If the local volatility cannot be computed (NaN), use the constant sigma as a fallback.
            sigma_local = np.where(np.isnan(sigma_local), self.sigma, sigma_local)

            # Generate Brownian increments.
            dW = np.random.normal(0, np.sqrt(dt), size=num_paths)
            # Use the Euler-Maruyama scheme with the log-Euler discretization.
            S_paths[:, i + 1] = S_paths[:, i] * np.exp((self.r - 0.5 * sigma_local ** 2) * dt + sigma_local * dW)
            print(f"Steps {i} / {num_steps-1} completed")


        # Compute the payoff at maturity.
        if self.Option_Type == 'call':
            payoffs = np.maximum(S_paths[:, -1] - self.K, 0)
        else:
            payoffs = np.maximum(self.K - S_paths[:, -1], 0)

        # Discount the average payoff back to present value.
        price = np.exp(-self.r * self.T) * np.mean(payoffs)
        """

        return self.black_scholes_pricing(vol=sigma_local),sigma_local

    def price_with_sabr(self):





        sabr = SABRModel()



