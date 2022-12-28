import numpy as np
import pandas as pd
import polygon as pg
import matplotlib.pyplot as plt
from datetime import date, datetime, timedelta
from scipy.stats import norm
import tensorflow as tf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sys import exit

# Read the data frame for stock prices
path = '/Users/nickeisenberg/GitRepos/Phython/yfDataFrames/'

amzn_df = pd.read_csv(path + 'amzn_december_1m.csv', index_col=0)
#--------------------------------------------------

# Get the dates, OHLC values, and volumn
time = amzn_df.index.values

amzn = {}
for type in ['Open', 'Close', 'High', 'Low', 'Volume']:
    amzn[type] = amzn_df[type].values.reshape(-1)
#--------------------------------------------------

# View the stock price
fig = make_subplots(specs=[[{'secondary_y': True}]])
fig.add_trace(
        go.Scatter(x=time, y=amzn['Volume']),
        secondary_y=True,
        )
fig.add_trace(
        go.Scatter(x=time, y =amzn['Open']),
        secondary_y=False
        )
fig.show()
#--------------------------------------------------

# use polygon to get the option price
def call_price(ticker, strike, expire, days_to_expire):
    option_client = pg.OptionsClient('wEnDK0VciFtxJ1kThvfJXbiSKbkbZhfK')
    call = pg.build_option_symbol(ticker,
                                  expire,
                                  'call',
                                  strike,
                                  _format='tradier')
    call_dic = option_client.get_daily_open_close(
            call,
            expire - timedelta(days=days_to_expire)
            )
    return call_dic
#--------------------------------------------------

# formulize the black-scholes framework
# S = stock price
# T = time to expire
# K = strike
# s = stadard deviation of log(S_t)
# r = risk free rate
# B = bond price

def f(S, T, K, s, r):
    g = (np.log(S / K) + (r + (.5 * s ** 2)) * T) / (s * np.sqrt(T))
    h = g - s * np.sqrt(T)
    return (S * norm.cdf(g)) - (K * np.exp(-r * T) * norm.cdf(h))

def f_x(S, T, K, s, r):
    with tf.GradientTape() as g:
        g.watch(S)
        y = f(S, T, K, s, r)
    return tf.keras.backend.get_value(df_dx = g.gradient(y, S))

def self_financing(t, B, S, T, K, s, r):
    a_t = f_x(S, T - t, K, s, r)
    b_t = (f(S, T - t, K, s, r) - (a_t * S)) / B
    return [a_t, b_t]
#--------------------------------------------------


