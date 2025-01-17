from loading_market_data import (get_options_dataframe,get_risk_free_rate,get_latest_price,get_realized_vol)
from euro_option import EuropeanOption

#Loading Data
option_data_df = get_options_dataframe(ticker='AAPL',option_type='call')
S0 = get_latest_price(ticker='AAPL')
r_risk_free = get_risk_free_rate()
sigma = get_realized_vol(ticker='AAPL')

print(option_data_df)
print(S0)
print(r_risk_free)
print(sigma)

#Instanciate an European Option Call

CE_Option = EuropeanOption(S=S0,K=229.145,T=0.019178,r=r_risk_free,sigma=sigma,Option_Type='call')

#Pricing Using BS


#price_bs = CE_Option.black_scholes_pricing()
#print(option_data_df.iloc[100:120])
#print(price_bs)

#Pricing Using Local Volatility

price_dupire = CE_Option.price_with_local_vol(market_df=option_data_df)
print(price_dupire)
