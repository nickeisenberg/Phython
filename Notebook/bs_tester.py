import pandas as pd
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from datetime import date, timedelta, datetime
import yfinance as yf

from black_scholes_funcs import *


path = '/Users/nickeisenberg/GitRepos/Phython/yfDataFrames/amzn/'
name = 'dec2022_1m'

amzn_df = pd.read_csv(f'{path}{name}.csv', index_col=0)
#--------------------------------------------------

# Get the dates, OHLC values, and volumn
time = amzn_df.index.values

print(time)

time = np.array([t[:-6] for t in time])  # remove the GMT offset

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

prev_week_start = str(expire - timedelta(days=11))
prev_week_end = str(expire - timedelta(days=6))
prev_week_start_bol = amzn_df.index > prev_week_start
prev_week_end_bol = amzn_df.index < prev_week_end
prev_week_bol = np.multiply(prev_week_start_bol, prev_week_end_bol)
prev_week = amzn_df.loc[prev_week_bol]

S_0 = prev_week.loc['2022-12-12 09:30:00']['Open']
sigma = np.log(prev_week['Open'].values).std()
rate = .00441 / 52  # in weeks
t_to_exp = 1
K = 83

fair_price = f(S_0, t_to_exp, K, sigma, rate)
print(fair_price)

# self financing stratgey







