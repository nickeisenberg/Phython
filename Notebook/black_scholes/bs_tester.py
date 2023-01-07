from matplotlib.projections import geo
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from datetime import date, timedelta, datetime
import yfinance as yf
from black_scholes_funcs import *
import tensorflow as tf


path = '/Users/nickeisenberg/GitRepos/Phython/yfDataFrames/amzn/'
name = 'dec2022_1m'

amzn_df = pd.read_csv(f'{path}{name}.csv', index_col=0)
#--------------------------------------------------

# Get the dates, OHLC values, and volumn
time = amzn_df.index.values
time = np.array([t[:-6] for t in time])  # remove the GMT offset

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

# define black scholes variables
expire = datetime(2022, 12 ,23)

amzn221223_C83_4 = call_price('AMZN',
                              83,
                              expire,
                              days_to_expire=4)

amzn221223_C83_0 = call_price('AMZN',
                              83,
                              expire,
                              days_to_expire=0)

print(amzn221223_C83_4)

print(amzn221223_C83_0)


prev_week_start = str(expire - timedelta(days=11))
prev_week_end = str(expire - timedelta(days=6))
prev_week_start_bol = amzn_df.index > prev_week_start
prev_week_end_bol = amzn_df.index < prev_week_end
prev_week_bol = np.multiply(prev_week_start_bol, prev_week_end_bol)
prev_week = amzn_df.loc[prev_week_bol]

prev_week_log = np.log(prev_week['Open'].values)
prev_week_log_returns = np.diff(prev_week_log) / prev_week_log[:-1] 

week_of_expire_bol = amzn_df.index >= '2022-12-19 09:30:00'
week_of_expire_bol *= amzn_df.index <= '2022-12-23 16:00:00'
week_of_expire = amzn_df.loc[week_of_expire_bol]
week_of_expire_price = week_of_expire['Open'].values

partition = np.linspace(0, 1, week_of_expire.shape[0])
delta = partition[1] - partition[0]

mu_sample = prev_week_log_returns.mean()
sigma_sample = prev_week_log_returns.std()
mu = mu_sample / delta + .5 * sigma_sample ** 2
sigma = sigma_sample * np.sqrt(partition.shape[0])

print(mu)
print(sigma)

S_0 = amzn_df.loc['2022-12-19 09:30:00']['Open']
rate = .00441 / 52  # in weeks
t_to_exp = 1
K = 83
fair_price = f(S_0, t_to_exp, K, sigma, rate)

print(fair_price)

for s in range(10):
    plt.plot(geo_b_motion(partition, S_0, mu, sigma))
plt.plot(week_of_expire_price)
plt.show()

plt.plot(geo_b_motion(partition, S_0, mu, sigma))
plt.plot(week_of_expire_price)
plt.show()

fig = make_subplots(specs=[[{'secondary_y': True}]])
_ = fig.add_trace(
        go.Scatter(x=partition,
                   y =week_of_expire['Open'],
                   name='Open Price'),
        secondary_y=False
        )
_ = fig.add_trace(
        go.Scatter(x=partition,
                   y=xxx,
                   name=f'Geometric brownian motion simulated price'),
        secondary_y=True
        )
_ = fig.update_yaxes(title={'text': 'Price'},
                     secondary_y=False)
_ = fig.update_layout(
        title={
    'text': "A lucky simultation of AMZN from 12-19-22 to 12-23-22",
    'x': .5
    },
        )
fig.show()
