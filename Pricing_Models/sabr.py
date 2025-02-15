import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import black_scholes

class SABRModel:
    def __init__(self, F, T, alpha, beta, rho, nu):
        """
        Initialize the SABR model with the given parameters.

        Parameters:
            F     : Forward price/rate.
            T     : Time to maturity.
            alpha : Volatility level.
            beta  : Elasticity parameter (0 <= beta <= 1).
            rho   : Correlation between asset and volatility.
            nu    : Volatility of volatility.
        """
        self.F = F
        self.T = T
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.nu = nu

    def implied_vol(self, K):

        F = self.F
        T = self.T
        alpha = self.alpha
        beta = self.beta
        rho = self.rho
        nu = self.nu

        # ATM case: when F is nearly equal to K.
        if np.isclose(F, K):
            term1 = ((1 - beta) ** 2 / 24) * (alpha ** 2) / (F ** (2 - 2 * beta))
            term2 = (1 / 4) * (rho * beta * nu * alpha) / (F ** (1 - beta))
            term3 = ((2 - 3 * rho ** 2) / 24) * nu ** 2
            vol_atm = alpha / (F ** (1 - beta)) * (1 + (term1 + term2 + term3) * T)
            return vol_atm
        else:
            logFK = np.log(F / K)
            FK_avg = (F * K) ** ((1 - beta) / 2)
            z = (nu / alpha) * FK_avg * logFK

            # For small z, revert to the ATM expansion.
            if np.abs(z) < 1e-07:
                return self.implied_vol(F)

            numerator = np.sqrt(1 - 2 * rho * z + z ** 2) + z - rho
            denominator = 1 - rho  # valid since rho < 1
            x_z = np.log(numerator / denominator)

            term1 = alpha / FK_avg * (z / x_z)
            correction = ((1 - beta) ** 2 / 24) * (alpha ** 2) / ((F * K) ** (1 - beta)) \
                         + (rho * beta * nu * alpha) / (4 * (F * K) ** ((1 - beta) / 2)) \
                         + ((2 - 3 * rho ** 2) / 24) * nu ** 2
            term2 = 1 + T * correction
            return term1 * term2


    def calibrate(self, strikes, market_vols, initial_guess, bounds):

        def objective(params):
            # Unpack parameters and update the model temporarily.
            alpha, beta, rho, nu = params
            self.alpha, self.beta, self.rho, self.nu = alpha, beta, rho, nu
            model_vols = np.array([self.implied_vol(K) for K in strikes])
            return np.sum((model_vols - market_vols) ** 2)

        result = minimize(objective, initial_guess, bounds=bounds, method='L-BFGS-B')
        self.alpha, self.beta, self.rho, self.nu = result.x
        return result.x

    def plot_volatility(self, strikes, market_vols):

        model_vols = np.array([self.implied_vol(K) for K in strikes])
        plt.figure(figsize=(8, 5))
        plt.plot(strikes, market_vols, 'bo', label='Market Implied Vols')
        plt.plot(strikes, model_vols, 'r-', label='SABR Model Vols')
        plt.xlabel("Strike")
        plt.ylabel("Implied Volatility")
        plt.title("SABR Model Calibration")
        plt.legend()
        plt.grid(True)
        plt.show()

    def price_option(self, K, option_type="call", risk_free_rate=0.0):

        # Retrieve the SABR-implied volatility for the given strike.
        sigma_sabr = self.implied_vol(K)
        discount = np.exp(-risk_free_rate * self.T)

        if sigma_sabr <= 0:
            if option_type.lower() == "call":
                return discount * max(F - K, 0)
            elif option_type.lower() == "put":
                return discount * max(K - F, 0)
            else:
                raise ValueError("option_type must be either 'call' or 'put'")


        return black_scholes.black_scholes_pricing(S=self.F, K=K, T=self.T, r=risk_free_rate, sigma=sigma_sabr,Option_Type=option_type)


if __name__ == '__main__':
    # For each unique time-to-maturity, we will calibrate the SABR model.
    calibrated_results = {}  # To store calibrated SABR parameters by TTM

    from loading_market_data import (get_options_dataframe, get_risk_free_rate, get_latest_price, get_realized_vol,get_dividend_yield)
    #Loading Data
    option_data_df = get_options_dataframe(ticker='AAPL', option_type='call')
    spot_price = get_latest_price(ticker='AAPL')
    r_risk_free = 0.0423#get_risk_free_rate()
    sigma = get_realized_vol(ticker='AAPL')
    dividend_yield = get_dividend_yield(ticker='AAPL')



    for TTM, group in option_data_df.groupby('TTM'):
        strikes = group['strike'].values
        market_vols = group['impliedVolatility'].values

        # Compute the forward price for this TTM using:
        F = spot_price * np.exp((r_risk_free - dividend_yield) * TTM)

        print(f"\nProcessing TTM = {TTM} years with computed forward price: {F:.2f}")

        # Create a SABRModel instance with an initial guess for [alpha, beta, rho, nu]
        initial_params = [0.2, 0.5, 0.0, 0.2]  # random initialization
        sabr_model = SABRModel(F, TTM, *initial_params)

        # Define parameter bounds:
        bounds = [(1e-6, None), (0, 1), (-0.999, 0.999), (1e-6, None)]

        calibrated_params = sabr_model.calibrate(strikes, market_vols, initial_params, bounds)
        calibrated_results[TTM] = calibrated_params
        print(f"Calibrated parameters for TTM={TTM}: alpha={calibrated_params[0]:.5f}, "
              f"beta={calibrated_params[1]:.5f}, rho={calibrated_params[2]:.5f}, nu={calibrated_params[3]:.5f}")

        model_vols = [sabr_model.implied_vol(K) for K in strikes]


        atm_strike = F  # At-the-money forward
        call_price = sabr_model.price_option(atm_strike, option_type="call", risk_free_rate=r_risk_free)
        print(f"ATM Call Price for TTM {TTM}: {call_price:.4f}")

        # Plot the SABR-implied volatility smile vs. the market implied volatilities.
        plt.figure(figsize=(8, 5))
        plt.plot(strikes, market_vols, 'bo', label='Market Implied Vol')
        plt.plot(strikes, model_vols, 'r-', label='SABR Model Vol')
        plt.xlabel("Strike")
        plt.ylabel("Implied Volatility")
        plt.title(f"SABR Calibration for TTM = {TTM}")
        plt.legend()
        plt.grid(True)
        plt.show()


