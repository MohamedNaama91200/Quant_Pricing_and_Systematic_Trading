from loading_market_data import (get_options_dataframe,get_risk_free_rate,get_latest_price,get_realized_vol)
from euro_option import EuropeanOption

#Loading Data
option_data_df = get_options_dataframe(ticker='AAPL',option_type='call')
S0 = get_latest_price(ticker='AAPL')
r_risk_free = get_risk_free_rate()
sigma = get_realized_vol(ticker='AAPL')

print(S0)

#Instanciate an European Option Call

option_data_df_test = option_data_df.iloc[-500]
CE_Option = EuropeanOption(S=S0,K=option_data_df_test['strike'],T=option_data_df_test['TTM'],r=r_risk_free,sigma=sigma,Option_Type='call')
print(option_data_df_test)
#Pricing Using BS


price_bs = CE_Option.black_scholes_pricing()
print(f"Price BS : {price_bs}")

#Pricing Using Local Volatility

price_dupire,sigma_local= CE_Option.price_with_local_vol(market_df=option_data_df)
print(f"Price Dupire : {price_dupire}")
print(sigma_local)
print(sigma)
