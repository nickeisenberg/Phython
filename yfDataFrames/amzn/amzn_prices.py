# yfinance only allows for minute data up to 30 days prior so to
# get to get the full month of decemeber, you need to do it in pieces.

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots

path = '/Users/nickeisenberg/GitRepos/Phython/yfDataFrames/amzn/'
name = 'dec2022_1m'

dfs = []

# for the first four weeks of december
for i in range(4):
    start = datetime(2022, 11, 28, 4 - 3) + timedelta(days=7 * i)
    end = start + timedelta(days=5, hours=24)
    amzn = yf.download('AMZN',
                       start=start,
                       end=end,
                       interval='1m',
                       prepost=True)
    dfs.append(amzn)

df = pd.concat(dfs)

# commenting to avoid accidental overwrite
# df.to_csv(f'{path}{name}.csv')

#--------------------------------------------------

# for the last week of december
df = pd.read_csv(f'{path}{name}.csv',
                 index_col=0)

print(df.head())
print(df.tail())

start = datetime(2022, 12, 26, 4 - 3)
end = start + timedelta(days=5, hours=24)

amzn = yf.download('AMZN',
                   start=start,
                   end=end,
                   interval='1m',
                   prepost=True)

df_total = pd.concat([df, amzn])

# commenting this to avoid accidental overwrite
# df_total.to_csv(f'{path}{name}.csv')

#--------------------------------------------------

# load the csv
amzn_dec = pd.read_csv(f'{path}{name}.csv',
                       index_col=0)

dates = amzn_dec.index.values

open = amzn_dec['Open'].values

vol = amzn_dec['Volume'].values

fig = make_subplots(specs=[[{'secondary_y': True}]])
_ = fig.add_trace(
        go.Scatter(x=dates, y=open,
                   name='AMZN open price'),
        secondary_y=False)
_ = fig.add_trace(
        go.Scatter(x=dates, y=vol,
                   name='Volume'),
        secondary_y=True)
_ = fig.update_yaxes(title={'text':'price'},
                     secondary_y=False)
_ = fig.update_yaxes(title={'text':'volume'},
                     secondary_y=True)
_ = fig.update_layout(
        title={'text': 'AMZN December',
               'x': .5}
        )
fig.show()
#--------------------------------------------------
