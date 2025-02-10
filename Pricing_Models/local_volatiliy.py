import numpy as np
from scipy.interpolate import CloughTocher2DInterpolator,NearestNDInterpolator


class LocalVol:
    def __init__(self, market_prices, r):
        if market_prices.empty:
            raise ValueError("Le df market_prices est vide.")

        required_columns = {'TTM', 'strike', 'lastPrice'}
        if not required_columns.issubset(market_prices.columns):
            raise ValueError(f"Le DataFrame doit contenir les colonnes {required_columns}.")

        market_prices = market_prices.bfill()

        self.market_prices = market_prices['lastPrice'].values
        self.strikes = market_prices['strike'].values
        self.maturities = market_prices['TTM'].values
        self.r = r
        self.local_vol_surface = None

    @staticmethod
    def C(T_val, K_val, interpolator,nearest_interpolator):
        """
        Computes the interpolated price vol for a given (T,K)
        """
        # The interpolator is built over (T, K) points for the market prices. If the couple is not on the convex hull, then we use nearest interolator
        price_interpo = interpolator((T_val, K_val))
        if price_interpo == price_interpo :

            return price_interpo
        else:
            return nearest_interpolator(T_val, K_val)


    def calibrate_local_vol(self, T, K):
        """
        Calibrates the local volatility σ_local(T, K) using Dupire's formula.
        """
        # Extract data arrays for maturities, strikes, and market prices:
        TTM = self.maturities
        Strike = self.strikes
        lastPrice = self.market_prices

        # Data boundaries (to ensure interpolation stays within the convex hull)
        T_min_data = np.min(TTM)
        T_max_data = np.max(TTM)
        K_min_data = np.min(Strike)
        K_max_data = np.max(Strike)

        # Clamp T and K to be within the data boundaries
        T = np.clip(T, T_min_data, T_max_data)
        K = np.clip(K, K_min_data, K_max_data)

        delta = 1e-2  # finite difference step

        # Create a 2D interpolator for the implied volatility surface
        points = np.column_stack((TTM, Strike))
        interpolator = CloughTocher2DInterpolator(points, lastPrice)
        #for points outside the convex hull
        nearest_interp = NearestNDInterpolator(points, lastPrice)

        # Compute finite difference approximations for the partial derivatives

        # Partial derivative with respect to T:
        if T - delta < T_min_data:
            dC_dT = (LocalVol.C(T + delta, K,interpolator,nearest_interp) - LocalVol.C(T, K,interpolator,nearest_interp)) / delta
        elif T + delta > T_max_data:
            dC_dT = (LocalVol.C(T, K,interpolator,nearest_interp) - LocalVol.C(T - delta, K,interpolator,nearest_interp)) / delta
        else:
            dC_dT = (LocalVol.C(T + delta, K,interpolator,nearest_interp) - LocalVol.C(T - delta, K,interpolator,nearest_interp)) / (2 * delta)

        # Partial derivative with respect to K:
        if K - delta < K_min_data:
            dC_dK = (LocalVol.C(T, K + delta,interpolator,nearest_interp) - LocalVol.C(T, K,interpolator,nearest_interp)) / delta
        elif K + delta > K_max_data:
            dC_dK = (LocalVol.C(T, K,interpolator,nearest_interp) - LocalVol.C(T, K - delta,interpolator,nearest_interp)) / delta
        else:
            dC_dK = (LocalVol.C(T, K + delta,interpolator,nearest_interp) - LocalVol.C(T, K - delta,interpolator,nearest_interp)) / (2 * delta)

        # Second partial derivative with respect to K:
        #if K - delta < K_min_data or K + delta > K_max_data:
         #   d2C_dK2 = np.nan
        #else:
        d2C_dK2 = (LocalVol.C(T, K + delta,interpolator,nearest_interp) - 2 * LocalVol.C(T, K,interpolator,nearest_interp) + LocalVol.C(T, K - delta,interpolator,nearest_interp)) / (delta ** 2)
        #d2C_dK2 = max(d2C_dK2, 1e-6) #forcing strike convexity (i.e no arbitrage condition
        # Apply Dupire's formula:
        numerator = dC_dT + self.r * K * dC_dK
        #numerator = max(numerator, 1e-6) #forcing numerator positivity (i.e no arbitrage condition)
        denominator = K**2 * d2C_dK2

        # Compute local variance and then the local volatility
        local_variance = 2 * numerator / denominator

        if local_variance < 0:
            # Negative variance is not meaningful; here we set it to NaN.
            sigma_local = np.nan
        else:
            sigma_local = np.sqrt(local_variance)

        self.local_vol_surface = sigma_local


    def get_local_vol(self,T,K):

        self.calibrate_local_vol(T,K)
        if self.local_vol_surface is None:
            raise ValueError("La surface de volatilité locale doit être calibrée avant d'obtenir la volatilité.")
        return self.local_vol_surface


if __name__ == '__main__' :
    from loading_market_data import (get_options_dataframe, get_risk_free_rate, get_latest_price, get_realized_vol,get_dividend_yield)
    import matplotlib.pyplot as plt

    # Loading Data
    option_data_df = get_options_dataframe(ticker='AAPL', option_type='put')
    spot_price = get_latest_price(ticker='AAPL')
    r_risk_free = get_risk_free_rate()
    sigma = get_realized_vol(ticker='AAPL')
    dividend_yield = get_dividend_yield(ticker='AAPL')


    # Implied Vol

    lv_surface = LocalVol(market_prices=option_data_df, r=r_risk_free)
    #option_data_df['localVolatility'] = option_data_df.apply(
     #   lambda row: lv_surface.get_local_vol(row['TTM'], row['strike']),
     #   axis=1
    #)
    calibrated_lv_results = {}
    for TTM, group in option_data_df.groupby('TTM'):
        strikes = group['strike'].values
        market_vols = group['impliedVolatility'].values

        # Compute the forward price for this TTM using:
        #     F = S * exp((r - q) * TTM)
        F = spot_price * np.exp((r_risk_free - dividend_yield) * TTM)
        print(f"\nProcessing TTM = {TTM} years with computed forward price: {F:.2f}")

        # Compute the model's local volatility for each strike.
        model_vols = [lv_surface.get_local_vol(TTM, K) for K in strikes]

        # Store the computed local volatilities (optional)
        calibrated_lv_results[TTM] = model_vols

        # Plot the market implied volatility vs. the local volatility model values.
        plt.figure(figsize=(8, 5))
        plt.plot(strikes, market_vols, 'bo', label='Market Implied Vol')
        plt.plot(strikes, model_vols, 'r-', label='Local Vol Model')
        plt.xlabel("Strike")
        plt.ylabel("Volatility")
        plt.title(f"Local Volatility vs Market Implied Vol for TTM = {TTM} years")
        plt.legend()
        plt.grid(True)
        plt.show()

    """
    print(option_data_df[['strike', 'TTM', 'impliedVolatility', 'localVolatility']].head(10))

    # -------------------------------------------
    # 1. Market Implied Volatility (Left 3D plot)
    # -------------------------------------------
    fig = plt.figure(figsize=(14, 6))

    ax1 = fig.add_subplot(1, 2, 1, projection='3d')

    # We'll interpret X=Strike, Y=TTM, Z=Vol, and color the points by vol as well.
    x1 = option_data_df['strike']
    y1 = option_data_df['TTM']
    z1 = option_data_df['impliedVolatility']

    # Create a 3D scatter; the `c=z1` means color is based on the Z-value
    sc1 = ax1.scatter(x1, y1, z1, c=z1, cmap='viridis', marker='o', edgecolor='k')

    ax1.set_xlabel('Strike (USD)')
    ax1.set_ylabel('Time to Maturity (Years)')
    ax1.set_zlabel('Implied Volatility')
    ax1.set_title('Market Implied Volatility Surface')

    # Add a colorbar for the left plot
    cb1 = fig.colorbar(sc1, ax=ax1, pad=0.1)
    cb1.set_label('Implied Volatility')

    # -------------------------------------------
    # 2. Local Volatility (Right 3D plot)
    # -------------------------------------------
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')

    # We'll interpret X=Strike, Y=TTM, Z=LocalVol, and color the points by local vol as well.
    x2 = option_data_df['strike']
    y2 = option_data_df['TTM']
    z2 = option_data_df['localVolatility']

    sc2 = ax2.scatter(x2, y2, z2, c=z2, cmap='viridis', marker='o', edgecolor='k')

    ax2.set_xlabel('Strike (USD)')
    ax2.set_ylabel('Time to Maturity (Years)')
    ax2.set_zlabel('Local Volatility')
    ax2.set_title('Local Volatility (Dupire) Surface')

    cb2 = fig.colorbar(sc2, ax=ax2, pad=0.1)
    cb2.set_label('Local Volatility')

    plt.tight_layout()
    plt.show()
    """