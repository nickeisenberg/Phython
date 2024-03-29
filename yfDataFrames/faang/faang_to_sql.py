import plotly.graph_objects as go
import matplotlib.pyplot as plt
import yfinance as yf
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from scipy.ndimage import median_filter
from copy import deepcopy
from sklearn.preprocessing import MinMaxScaler

# Tickers to get info for
faang = ['AMZN', 'META', 'GOOG', 'NFLX', 'AAPL']
#--------------------------------------------------

# Get the info and interpolate the missing minute data
# need to acount for the three hour offset from lv to ny
# also batch the days into 7 day groups. This is because
# yfinance works better this way.
start = datetime(2023, 5, 1, 4 - 3, 0, 0)
end = datetime(2023, 5, 11, 4 - 3, 0, 0)

no_weeks = int(np.ceil(((end - start).days + 1) / 7))

data = []
for i in np.arange(0, no_weeks, 1):
    start_ = start + timedelta(days=7 * int(i))
    if i == np.arange(0, no_weeks, 1)[-1]:
        end_ = end
    else:
        end_ = start + timedelta(days=6 * (int(i) + 1))
    data.append(
            yf.download(
                tickers=faang,
                start=start_,
                end=end_,
                interval='1m',
                prepost=True)
            )
#--------------------------------------------------

# Split up the multiindex so that the the dataframes can be easily stored
types = ['Open', 'High', 'Low', 'Close', 'Volume']
faang_dfs_pre = {type_: [] for type_ in types}
for type_ in types:
    for d in data:
        faang_dfs_pre[type_].append(d[type_])
for type_ in types:
    df = pd.concat(faang_dfs_pre[type_])
    df.rename({fn: f'{fn}_{type_}' for fn in faang}, axis=1, inplace=True)
    faang_dfs_pre[type_] = df

# Check to see if things look right
# faang_dfs_pre['Close'].head()
# faang_dfs_pre['Close'].tail()
#--------------------------------------------------

# Figure out the max number of missing prices in a row
def consec_missing(arr):
    index = np.linspace(0, arr.size - 1, arr.size)
    index_nan = index[np.isnan(arr)]
    chain_lens = []
    chain_len = 1
    for i, (y0, y1) in enumerate(zip(index_nan[:-1], index_nan[1:])):
        if y0 + 1 == y1 and i != y[: -1].size - 1:
            chain_len +=1
        elif y0 + 1 == y1 and i == y[: -1].size - 1:
            chain_len += 1
            chain_lens.append(chain_len)
        elif y0 + 1 != y1 and i == y[: -1].size - 1:
            chain_lens.append(chain_len)
            chain_lens.append(1)
        else:
            chain_lens.append(chain_len)
            chain_len = 1
    max_no_consec_nan = np.array(chain_lens).max()
    return max_no_consec_nan

nan_in_a_row = {}
for tick in faang:
    xxx = faang_dfs_pre['Open'][f'{tick}_Open'].values
    nan_in_a_row[tick] = consec_missing(xxx)
for k, v in nan_in_a_row.items():
    print(k)
    print(v)
    print('')
#--------------------------------------------------

# interpolate the prices
faang_dfs = deepcopy(faang_dfs_pre)
for k, v in faang_dfs.items():
    faang_dfs[k] = v.interpolate(axis=0)
#--------------------------------------------------

# Filter out the spikes that appear. Not sure what caused the spikes.
# Probably an after hours or premarket market order. 
faang_filt_dfs = deepcopy(faang_dfs)
for type_ in types:
    for col in faang_dfs[type_].columns:
        faang_filt_dfs[type_][col] = median_filter(
                input=faang_filt_dfs[type_][col],
                size=4,
                mode='nearest')
#--------------------------------------------------

# plot and view
faang_scaled_dfs = deepcopy(faang_filt_dfs['Open'])
Mm = MinMaxScaler()
faang_scaled_dfs[faang_scaled_dfs.columns] = Mm.fit_transform(
        faang_scaled_dfs[faang_scaled_dfs.columns])

fig = go.Figure()
for col in faang_scaled_dfs.columns:
    _ = fig.add_trace(
            go.Scatter(# x=faang_scaled_dfs[tick].index,
                       y=faang_scaled_dfs[col].values,
                       name=f'{col}')
            )
fig.show()
#--------------------------------------------------

# Store the dataframes
path = '/Users/nickeisenberg/GitRepos/Phython/yfDataFrames/faang/'

for type_ in types:
    faang_dfs[type_].to_csv(f'{path}/unfilt/{type_}_23_05_01-23_05_05.csv')

for type_ in types:
    faang_filt_dfs[type_].to_csv(f'{path}/filt/{type_}_23_05_01-23_05_05.csv')
#--------------------------------------------------

# read the csv files, scale and plot them
reload = {}
for type_ in types:
    reload[type_] = pd.read_csv(
            f'{path}filt/{type_}_23_05_01-23_05_05.csv', index_col=0
            )

reload['Open'].index

reload['Open']

scaled = deepcopy(reload['Open'])
Mm = MinMaxScaler()
scaled[scaled.columns] = Mm.fit_transform(scaled[scaled.columns])

fig = go.Figure()
for tick in faang:
    _ = fig.add_trace(
            go.Scatter(y=scaled[tick+'_Open'].values, name=f'{tick}')
            )
fig.show()

