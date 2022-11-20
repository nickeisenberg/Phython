from sys import exit
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
# from compiler import *

#-yf.Ticker.history--------------------------------
# period : 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max
# inteval : 1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo 
# start : yyyy-mm-dd
# end : yyyy-mm-dd
# prepost : (Default False) pre and post market data
# auto_adjust : (Default True)
# actions : (Default True) download dividend and split information 

#-yf.download-:-returns-a-df-with-OHCO-for-multiple-tickers
# same inputs as yf.Ticker.history plus extras
# group_by : (Default='columns') other option is 'ticker'
# threads : use threads for mass downloading? (True/False/Integer)
# proxy : (Default=None) proxy URL if you want to use a proxy server
#         for dowlnloading the data (optional)

# example : data = yf.download('AMZN AAPL GOOG', period=...)

#-yf.Ticker.option_chain---------------------------
# date : yyyy-mm-dd, epiry date. If None return all data
# yf.Ticker.options will return all expiry dates
# yf.Ticker.option_chain.calls or yf.Ticker.option_chain.puts for calls and puts
# to create a data frame, need to do calls and puts separately
# pd.DataFrame(yf.Ticker.option_chain('GOOG').calls(.puts)) works nicely

#--------------------------------------------------
gme = yf.Ticker('GME')

gme_df = gme.history(period='2y',
                     interval='1h',
                     prepost=False,
                     actions=False)

gme_open = gme_df['Open'].values
plt.plot(gme_open)
plt.show()

exit()
gme_open = gme_df['Open'].values.reshape((-1,1))
Mm = MinMaxScaler(feature_range=(0,1))
gme_open_scaled = Mm.fit_transform(gme_open)

vec1 = gme_open[len(gme_open)-100 :].reshape(-1)
vec2 = gme_open[:100].reshape(-1)

vec1_scale = gme_open_scaled[len(gme_open)-100 :].reshape(-1)
vec2_scale = gme_open_scaled[:100].reshape(-1)

print(pearson_corr(vec1, vec2))
print(pearson_corr(vec1_scale, vec2_scale))

print(norm_dot(vec1, vec2))
print(norm_dot(vec1_scale, vec2_scale))
# time = np.linspace(0, 1, len(gme_open))
# plt.plot(time, gme_open)
# plt.show()
