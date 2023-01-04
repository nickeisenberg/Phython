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

# Read the data frame for stock prices
path = '/Users/nickeisenberg/GitRepos/Phython/yfDataFrames/amzn/'
name = 'dec2022_1m'

amzn_df = pd.read_csv(f'{path}{name}.csv', index_col=0)
#--------------------------------------------------

# Get the dates, OHLC values, and volumn
time = amzn_df.index.values

print(time)

time = np.array([t[:-6] for t in time])

print(time)

amzn_df.set_index(keys=time, drop=True, inplace=True)

amzn = {}
for i in ['Open', 'Close', 'High', 'Low', 'Volume']:
    amzn[i] = amzn_df[i].values.reshape(-1)
#--------------------------------------------------

# View the stock price
fig = make_subplots(specs=[[{'secondary_y': True}]])
_ = fig.add_trace(
        go.Scatter(x=time, y =amzn['Open'], name='Open Price'),
        secondary_y=False
        )
_ = fig.add_trace(
        go.Scatter(x=time, y=amzn['Volume'], name='Volume'),
        secondary_y=True,
        )
_ = fig.update_yaxes(title={'text': 'Price'},
                     secondary_y=False)
_ = fig.update_yaxes(title={'text': 'Volume'},
                     secondary_y=True)
fig.show()
#--------------------------------------------------

# define black scholes variables
expire = datetime(2022, 12 ,23)
amzn221223_C83 = call_price('AMZN',
                            83,
                            expire,
                            days_to_expire=4)
                                
prev_week_start = str(expire - timedelta(days=11))
prev_week_end = str(expire - timedelta(days=6))
prev_week_start_bol = amzn_df.index > prev_week_start
prev_week_end_bol = amzn_df.index < prev_week_end
prev_week_bol = np.multiply(prev_week_start_bol, prev_week_end_bol)
prev_week = amzn_df.loc[prev_week_bol]

S_0 = prev_week.loc['2022-12-12 09:30:00']['Open']

sigma = np.log(prev_week['Open'].values).std()

