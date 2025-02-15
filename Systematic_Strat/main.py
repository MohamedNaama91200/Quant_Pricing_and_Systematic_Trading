import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from loading_market_data import loading_market_data

#TTF and EUA Prices Loading (First Line)

symbols_ttf_eua = {"TTF": "TTF=F","EUA": "CO2.L" }
prices_df = loading_market_data()

