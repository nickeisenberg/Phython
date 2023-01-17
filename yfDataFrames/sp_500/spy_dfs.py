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
sp_500 = ['SPY']
#--------------------------------------------------

# Get the info and interpolate the missing minute data
start = datetime(2022, 12, 19, 4 - 3, 0, 0)
end = start + timedelta(days=5, hours=16)

data = []
for i in range(4):
    start_ = start + timedelta(days=7 * i)
    end_ = end + timedelta(days=7 * i)
    data.append(
            yf.download(
                tickers=sp_500,
                start=start_,
                end=end_,
                interval='1m',
                prepost=True)
            )
#--------------------------------------------------

# Split up the multiindex so that the the dataframes can be easily stored
types = ['Open', 'High', 'Low', 'Close', 'Volume']
sp_500_dfs_pre = {}
for type_ in types:
    sp_500_dfs_pre[type_] = []
for type_ in types:
    for d in data:
        sp_500_dfs_pre[type_].append(d[type_])
for type_ in types:
    sp_500_dfs_pre[type_] = pd.concat(sp_500_dfs_pre[type_])
#--------------------------------------------------

# Figure out the max number of missing prices in a row
bad_inds = {}
for tick in sp_500:
    bad_inds[tick] = []

nan_in_a_row = {}
for tick in sp_500:
    xxx = sp_500_dfs_pre['Open'][tick].values
    index = np.linspace(0, xxx.shape[0] - 1, xxx.shape[0])
    isnan = index[np.isnan(xxx)]
    isnan_diff = np.diff(isnan)
    count = 0
    thresh = 2 
    for i in range(isnan.shape[0] - thresh):
        ind = isnan[i]
        next = isnan[i: i + thresh]
        bol = []
        for j in range(thresh):
            bol.append(ind + j == next[j])
        if np.array(bol).sum() == thresh:
            count += 1
            bad_inds[tick].append(ind)
    nan_in_a_row[tick] = count

for k, v in nan_in_a_row.items():
    print(k)
    print(v)
    print('')
#--------------------------------------------------

# interpolate the prices
sp_500_dfs = deepcopy(sp_500_dfs_pre)

# for k, v in sp_500_dfs.items():  # SPY had no missing values
#     sp_500_dfs[k] = v.interpolate(axis=0)
#--------------------------------------------------

# Filter out the spikes that appear. Not sure what caused the spikes.
# Probably an after hours or premarket market order. 
sp_500_filt_dfs = deepcopy(sp_500_dfs)
for type_ in types:
    for col in sp_500_dfs[type_].columns:
        sp_500_filt_dfs[type_][col] = median_filter(
                input=sp_500_filt_dfs[type_][col],
                size=3,
                mode='nearest')
#--------------------------------------------------

# plot and view
fig = go.Figure()
for tick in sp_500:
    _ = fig.add_trace(
            go.Scatter(# x=sp_500_scaled_dfs[tick].index,
                       y=sp_500_filt_dfs['Open'][tick].values,
                       name=f'{tick}')
            )
    _ = fig.add_trace(
            go.Scatter(# x=sp_500_scaled_dfs[tick].index,
                       y=sp_500_dfs['Open'][tick].values,
                       name=f'{tick}')
            )
fig.show()

sp_500_scaled_dfs = deepcopy(sp_500_filt_dfs['Open'])
Mm = MinMaxScaler()
sp_500_scaled_dfs[sp_500_scaled_dfs.columns] = Mm.fit_transform(
        sp_500_scaled_dfs[sp_500_scaled_dfs.columns])

fig = go.Figure()
for tick in sp_500:
    _ = fig.add_trace(
            go.Scatter(# x=sp_500_scaled_dfs[tick].index,
                       y=sp_500_scaled_dfs[tick].values,
                       name=f'{tick}')
            )
fig.show()
#--------------------------------------------------


# Store the dataframes
path = '/Users/nickeisenberg/GitRepos/Phython/yfDataFrames/sp_500/'

for type_ in types:
    sp_500_dfs[type_].to_csv(f'{path}/unfiltered/sp_500_{type_}.csv')

for type_ in types:
    sp_500_filt_dfs[type_].to_csv(f'{path}/filtered/sp_500_{type_}.csv')
#--------------------------------------------------

# read the csv files, scale and plot them
reload = {}
for type_ in types:
    reload[type_] = pd.read_csv(
            f'{path}/filtered/sp_500_{type_}.csv', index_col=0
            )

scaled = deepcopy(reload['Open'])
Mm = MinMaxScaler()
scaled[scaled.columns] = Mm.fit_transform(scaled[scaled.columns])

fig = go.Figure()
for tick in sp_500:
    _ = fig.add_trace(
            go.Scatter(y=scaled[tick].values, name=f'{tick}')
            )
fig.show()

