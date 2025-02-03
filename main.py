from loading_market_data import (get_options_dataframe,get_risk_free_rate,get_latest_price,get_realized_vol)
from euro_option import EuropeanOption

#Loading Data
option_data_df = get_options_dataframe(ticker='AAPL',option_type='call')
S0 = get_latest_price(ticker='AAPL')
r_risk_free = get_risk_free_rate()
sigma = get_realized_vol(ticker='AAPL')

print(S0)

#Instanciate an European Option Call

CE_Option = EuropeanOption(S=S0,K=option_data_df['strike'].iloc[-501],T=option_data_df['TTM'].iloc[-501],r=r_risk_free,sigma=sigma,Option_Type='call')
print(option_data_df['strike'].iloc[-501])
print(option_data_df['TTM'].iloc[-501])
print(option_data_df['lastPrice'].iloc[-501])
#Pricing Using BS


price_bs = CE_Option.black_scholes_pricing()
print(price_bs)

#Pricing Using Local Volatility

price_dupire = CE_Option.price_with_local_vol(market_df=option_data_df)
print(price_dupire)
